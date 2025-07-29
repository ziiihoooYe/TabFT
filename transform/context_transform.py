import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from typing import List, Tuple, Optional
from transform.base import BaseTransform
import copy

############################################################################
###                          .    Encoders                               ###
############################################################################

class BaseEncoder:
    """
    Subclasses **must** implement:
        - `parameters(self) -> dict | None`
        - `encode(self, X_num: np.ndarray, X_cat: Optional[np.ndarray] = None, device: Optional[str] = None) -> np.ndarray`
    """
    def __init__(self):
        super().__init__()

    def parameters(self):
        raise NotImplementedError("Subclasses must return trainable parameters.")

    def encode(self, X_num, X_cat=None):
        raise NotImplementedError("Subclasses must implement `encode` method.")


class _MLP(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 8, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        hidden_dims = hidden_dims or [32]

        layers: List[nn.Module] = []
        cur = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.GELU())
            cur = h
        layers.append(nn.Linear(cur, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPMarginalEncoder(BaseEncoder, nn.Module):
    def __init__(self, n_features: int, output_dim_per_feature: int, hidden_dims_per_feature: Optional[List[int]] = None):
        BaseEncoder.__init__(self)
        nn.Module.__init__(self)
        
        self.n_features = n_features
        self.output_dim_per_feature = output_dim_per_feature
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.encoders = nn.ModuleList(
            [_MLP(input_dim=1, output_dim=output_dim_per_feature, hidden_dims=hidden_dims_per_feature)
             for _ in range(n_features)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, n_features)
        # encoded_features = [
        #     torch.cat((encoder(feature_col), x[:, i:i+1]), dim=-1) for i, (encoder, feature_col) in enumerate(zip(self.encoders, x.split(1, dim=1)))
        # ]
        encoded_features = [
            encoder(feature_col) for encoder, feature_col in zip(self.encoders, x.split(1, dim=1))
        ]

        return torch.cat(encoded_features, dim=1)

    def parameters(self):
        return self.encoders.parameters()

    def encode(self,
               X_num: np.ndarray,
               X_cat: Optional[np.ndarray] = None):
        self.to(self.device)
        encoded_output = self.forward(X_num)
        return encoded_output, X_cat


class IdentityEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()

    def parameters(self):
        return None

    def encode(self, X_num, X_cat=None):
        return X_num, X_cat

    def train(self):
        pass

    def eval(self):
        pass

############################################################################
###              .        Encoder Trainer (Optimized)                 .  ###
############################################################################
class MarginalEncoderTrainer:
    def __init__(self,
                 encoder: nn.Module,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 1024,
                 patience: int = 10,
                 delta: float = 0,
                 clip_grad_norm: Optional[float] = 1.0,
                 lr_scheduler_patience: int = 5,
                 device: str | None = None):
        self.encoder = encoder
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.patience = patience
        self.delta = delta
        self.clip_grad_norm = clip_grad_norm
        self.lr_scheduler_patience = lr_scheduler_patience

    def _evaluate(self,
                  X_num: np.ndarray,
                  Y_target_np: np.ndarray) -> float:
        """Return mean MSE on a validation split (no grad)."""
        self.encoder.eval()
        ds = TensorDataset(
            torch.from_numpy(X_num.astype(np.float32)),
            torch.from_numpy(Y_target_np.astype(np.float32))
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        loss_fn = nn.MSELoss()

        total_loss = 0.0
        n_obs = 0
        with torch.no_grad():
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                pred = self.encoder(bx)
                total_loss += loss_fn(pred, by).item() * bx.size(0)
                n_obs += bx.size(0)
        return total_loss / max(n_obs, 1)

    def train(self,
              X_num: np.ndarray,
              X_cat: Optional[np.ndarray],
              Y_target_np: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              C_val: Optional[np.ndarray] = None,
              Y_val: Optional[np.ndarray] = None):
        self.encoder.to(self.device)
        
        ds = TensorDataset(
            torch.from_numpy(X_num.astype(np.float32)),
            torch.from_numpy(Y_target_np.astype(np.float32))
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = optim.Adam(self.encoder.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'min', patience=self.lr_scheduler_patience, factor=0.5
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        patience_ctr = 0

        for epoch in range(self.epochs):
            self.encoder.train()
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                
                opt.zero_grad(set_to_none=True)
                pred = self.encoder(bx)
                loss = loss_fn(pred, by)
                loss.backward()

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad_norm)

                opt.step()

            # ----- validation & early‑stopping -----
            if X_val is not None and Y_val is not None:
                val_loss = self._evaluate(X_val, Y_val)
                
                scheduler.step(val_loss)

                if val_loss + self.delta < best_val_loss:
                    best_val_loss = val_loss
                    patience_ctr = 0
                    best_state = copy.deepcopy(self.encoder.state_dict())
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}. "
                              f"Best val MSE: {best_val_loss:.6f}")
                        break
        if best_state is not None:
            self.encoder.load_state_dict(best_state)
        print("Trainer finished.")


class ManifoldRegularizationBatchTrainer:
    def __init__(self,
                 encoder: nn.Module,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 1024,
                 gamma: float = 1.0,
                 lambda_cov: float = 0.1,
                 patience: int = 10,
                 delta: float = 0,
                 clip_grad_norm: Optional[float] = 1.0,
                 lr_scheduler_patience: int = 5,
                 device: str | None = None):
        self.encoder = encoder
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.lambda_cov = lambda_cov
        self.patience = patience
        self.delta = delta
        self.clip_grad_norm = clip_grad_norm
        self.lr_scheduler_patience = lr_scheduler_patience


    def _calculate_batch_loss(self, bx: torch.Tensor) -> torch.Tensor:
        full_embeddings = self.encoder(bx)
        total_loss = 0
        n_features = self.encoder.n_features
        output_dim = self.encoder.output_dim_per_feature

        for j in range(n_features):
            feature_col = bx[:, j:j+1]
            emb_j = full_embeddings[:, j*output_dim : (j+1)*output_dim]
            
            dist_sq = torch.cdist(feature_col, feature_col, p=2).pow(2)
            affinity = torch.exp(-self.gamma * dist_sq)
            emb_dist_sq = torch.cdist(emb_j, emb_j, p=2).pow(2)
            lap_loss = torch.sum(affinity * emb_dist_sq)
            
            emb_j_centered = emb_j - emb_j.mean(dim=0, keepdim=True)
            cov = (emb_j_centered.T @ emb_j_centered) / (emb_j.size(0) - 1)
            cov_loss = (cov - torch.eye(cov.size(0), device=self.device)).pow(2).sum()
            
            total_loss += lap_loss + self.lambda_cov * cov_loss
        return total_loss

    def _evaluate(self, X_val: np.ndarray) -> float:
        self.encoder.eval()
        ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        
        total_loss = 0.0
        n_obs = 0
        with torch.no_grad():
            for batch in dl:
                bx = batch[0].to(self.device)
                total_loss += self._calculate_batch_loss(bx).item() * bx.size(0)
                n_obs += bx.size(0)
        return total_loss / max(n_obs, 1)

    def train(self, X_num: np.ndarray, X_cat: Optional[np.ndarray], X_val: Optional[np.ndarray], C_val: Optional[np.ndarray]):
        self.encoder.to(self.device)
        ds = TensorDataset(torch.from_numpy(X_num.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = optim.Adam(self.encoder.parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'min', patience=self.lr_scheduler_patience, factor=0.5
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        patience_ctr = 0

        for epoch in range(self.epochs):
            self.encoder.train()
            for batch in dl:
                bx = batch[0].to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = self._calculate_batch_loss(bx)
                loss.backward()
                
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad_norm)

                opt.step()

            # ----- validation & early‑stopping ----- # 
            if X_val is not None:
                val_loss = self._evaluate(X_val)
                
                scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Val Loss: {val_loss:.6f}")

                if val_loss + self.delta < best_val_loss:
                    best_val_loss = val_loss
                    patience_ctr = 0
                    best_state = copy.deepcopy(self.encoder.state_dict())
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}. "
                              f"Best val loss: {best_val_loss:.6f}")
                        break
        
        if best_state is not None:
            self.encoder.load_state_dict(best_state)
        print("Trainer finished.")

        
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy
from typing import Optional

class JointManifoldTrainer:
    """
    一个新的训练器，它联合优化三个目标：
    1. 特征内流形结构保持 (Laplacian Loss)
    2. 特征内嵌入维度解耦 (Covariance Loss)
    3. 特征间关系保持 (Inter-feature Kernel Loss)
    
    这个类将替换 ManifoldRegularizationBatchTrainer，并以高效的向量化方式实现。
    """
    def __init__(self,
                 encoder: nn.Module,
                 # --- 损失权重超参数 ---
                 lambda_lap: float = 1.0,      # 权重：特征内拉普拉斯损失
                 lambda_cov: float = 0.1,      # 权重：特征内协方差损失
                 lambda_inter: float = 1.0,    # 权重：特征间核损失 (新增)
                 # --- 其他训练参数 ---
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 1024,
                 gamma: float = 1.0,           # RBF核的参数 (用于拉普拉斯损失)
                 patience: int = 10,
                 delta: float = 0.0,
                 clip_grad_norm: Optional[float] = 1.0,
                 lr_scheduler_patience: int = 5,
                 device: str | None = None):
        
        self.encoder = encoder
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 损失函数权重
        self.lambda_lap = lambda_lap
        self.lambda_cov = lambda_cov
        self.lambda_inter = lambda_inter
        
        self.gamma = gamma
        self.patience = patience
        self.delta = delta
        self.clip_grad_norm = clip_grad_norm
        self.lr_scheduler_patience = lr_scheduler_patience
        
        self.K_c = None # 用于存储预计算的特征间核矩阵

    def _precompute_feature_kernel(self, X: np.ndarray):
        """
        在训练前预计算特征间的关系核矩阵 K_c。
        这里使用皮尔逊相关系数的绝对值作为例子。你可以根据需要替换为其他关系度量。
        """
        print("Pre-computing feature correlation matrix (Column Kernel)...")
        # 使用 PyTorch 在目标设备上计算，避免不必要的数据传输
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        # torch.corrcoef 需要的输入形状是 (features, observations)
        # 我们的 X_tensor 是 (observations, features)，所以需要转置 .T
        corr_matrix = torch.corrcoef(X_tensor.T)
        
        # 处理可能因方差为0而产生的NaN值
        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
        
        # 我们关心的是关系的强度，而不是方向，所以取绝对值
        self.K_c = corr_matrix.abs()
        
        # 对角线是特征与自身的相关性（恒为1），在损失计算中没有意义，设为0
        self.K_c.fill_diagonal_(0)
        print("Feature kernel K_c computed.")
            
    def _evaluate(self, X_val: np.ndarray) -> float:
        """在验证集上评估总损失 (无梯度)"""
        self.encoder.eval()
        ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        
        total_loss = 0.0
        n_obs = 0
        with torch.no_grad():
            for batch in dl:
                bx = batch[0].to(self.device)
                total_loss += self._calculate_joint_loss(bx).item() * bx.size(0)
                n_obs += bx.size(0)
        return total_loss / max(n_obs, 1)

    def _calculate_joint_loss(self, bx: torch.Tensor) -> torch.Tensor:
        """
        高效的向量化损失计算，包含全部三个部分。
        """
        n_samples = bx.size(0)
        # 这些属性需要你的encoder提供
        n_features = self.encoder.n_features
        output_dim = self.encoder.output_dim_per_feature
        
        # --- 准备工作: 获取并重塑嵌入 ---
        # 1. 获取总嵌入: (n_samples, n_features * output_dim)
        full_embeddings = self.encoder(bx)
        # 2. 重塑以分离特征维度: (n_samples, n_features, output_dim)
        #    这是实现高效向量化的关键步骤。
        embeddings = full_embeddings.view(n_samples, n_features, output_dim)
        
        # --- Loss 1: 特征内拉普拉斯损失 (Intra-feature Laplacian Loss) ---
        # 目标：保持每个特征内部的数据点局部邻域结构。
        
        # 为了对每个特征独立计算样本间距离，我们需要转置，将特征维度作为批次维度
        # bx_t: (n_features, n_samples)
        # emb_t: (n_features, n_samples, output_dim)
        bx_t = bx.transpose(0, 1)
        emb_t = embeddings.transpose(0, 1)
        
        # 计算原始空间中样本间的距离平方，形状: (n_features, n_samples, n_samples)
        dist_sq_features = torch.cdist(bx_t.unsqueeze(-1), bx_t.unsqueeze(-1), p=2).pow(2)
        affinity = torch.exp(-self.gamma * dist_sq_features)
        
        # 计算嵌入空间中样本间的距离平方，形状: (n_features, n_samples, n_samples)
        dist_sq_embeddings = torch.cdist(emb_t, emb_t, p=2).pow(2)

        # 加权求和得到总的拉普拉斯损失
        lap_loss = torch.sum(affinity * dist_sq_embeddings)

        # --- Loss 2: 协方差损失 (Covariance Loss) ---
        # 目标：使每个特征的嵌入向量的各个维度尽可能解耦。
        
        # `emb_t` 已经是 (n_features, n_samples, out_dim)
        # 计算每个特征的均值，形状: (n_features, 1, out_dim)
        mean_embeddings_per_feature = emb_t.mean(dim=1, keepdim=True)
        centered_embeddings = emb_t - mean_embeddings_per_feature
        
        # 使用批量矩阵乘法 (bmm) 高效计算所有特征的协方差矩阵
        # (n_features, out_dim, n_samples) @ (n_features, n_samples, out_dim) -> (n_features, out_dim, out_dim)
        cov_matrices = torch.bmm(centered_embeddings.transpose(1, 2), centered_embeddings) / (n_samples - 1)
        
        # 创建一个批量的单位矩阵用于计算差值
        eye_batch = torch.eye(output_dim, device=self.device).expand_as(cov_matrices)
        
        # 计算所有协方差矩阵与单位矩阵的差值的Frobenius范数平方和
        cov_loss = (cov_matrices - eye_batch).pow(2).sum()
        
        # --- Loss 3: 特征间流形损失 (Inter-feature Manifold Loss) ---
        # 目标：使特征嵌入的中心点之间的距离关系，与预定义的特征核 K_c 一致。
        
        # `mean_embeddings_per_feature` 形状是 (n_features, 1, output_dim)，去掉多余维度
        feature_centroids = mean_embeddings_per_feature.squeeze(1) # 形状: (n_features, output_dim)
        
        # 计算所有特征中心点两两之间的距离平方，得到一个 (n_features, n_features) 的距离矩阵
        center_dist_sq = torch.cdist(feature_centroids, feature_centroids, p=2).pow(2)
        
        # 将距离矩阵与预先计算好的相关性核 K_c 进行元素级相乘后求和
        # 高相关性 (K_c 大) 乘以大距离，会产生大损失，从而迫使模型拉近它们的中心点。
        # 除以 2 是因为 cdist 计算了 (j, k) 和 (k, j) 两次，而损失应该是对称的。
        inter_feature_loss = torch.sum(self.K_c * center_dist_sq) / 2.0
        
        # --- 合并总损失 ---
        total_loss = (self.lambda_lap * lap_loss +
                      self.lambda_cov * cov_loss +
                      self.lambda_inter * inter_feature_loss)
                    
        return total_loss

    def train(self, X_num: np.ndarray, X_cat: Optional[np.ndarray] = None, X_val: Optional[np.ndarray] = None, C_val: Optional[np.ndarray] = None):
        # 步骤1: 在训练开始前，预计算特征核
        self._precompute_feature_kernel(X_num)
        
        self.encoder.to(self.device)
        ds = TensorDataset(torch.from_numpy(X_num.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True) # drop_last=True 避免协方差计算因batch size为1而出错
        opt = optim.Adam(self.encoder.parameters(), lr=self.lr)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'min', patience=self.lr_scheduler_patience, factor=0.5
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        patience_ctr = 0

        for epoch in range(self.epochs):
            self.encoder.train()
            for batch in dl:
                bx = batch[0].to(self.device)
                opt.zero_grad(set_to_none=True)
                
                # 步骤2: 计算联合损失
                loss = self._calculate_joint_loss(bx)
                
                loss.backward()
                
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad_norm)

                opt.step()

            # ----- validation & early‑stopping ----- # 
            if X_val is not None:
                val_loss = self._evaluate(X_val)
                scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Val Loss: {val_loss:.6f}")

                if val_loss + self.delta < best_val_loss:
                    best_val_loss = val_loss
                    patience_ctr = 0
                    best_state = copy.deepcopy(self.encoder.state_dict())
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}. "
                              f"Best val loss: {best_val_loss:.6f}")
                        break
        
        if best_state is not None:
            self.encoder.load_state_dict(best_state)
        print("Trainer finished.")

        

############################################################################
###             .        Marginal Context Transform                   .  ###
############################################################################
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh
from cuml.neighbors import NearestNeighbors
from cupyx.scipy.sparse.csgraph import connected_components
class RBFContextTransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.mode = args.get("mode", "explicit")
        if self.mode not in ["explicit", "implicit"]:
            raise ValueError("mode must be either 'explicit' or 'implicit'")

        self.output_dim = args.get("output_dim", 8)
        self.gamma = args.get("gamma", 1.0)
        self.encoder_ = None
        self.constant_cols_indices = [512, 512, 512, 512, 512]

        # --- explicit mode parameters ---
        self.k_neighbors = args.get("k_neighbors", None)
        
        # --- implicit mode parameters ---
        self.lambda_cov = args.get("lambda_cov", 0.1)

        # --- shared training parameters ---
        self.epochs = args.get("epochs", 1000)
        self.lr = args.get("lr", 1e-3)
        self.batch_size = args.get("batch_size", 1024)
        self.patience = args.get("patience", 10)

    def _spectral_embed_gpu(self, column_data_gpu: cp.ndarray) -> cp.ndarray:
        n_samples = column_data_gpu.shape[0]

        if column_data_gpu.min() == column_data_gpu.max():
            return cp.zeros((n_samples, self.output_dim), dtype=cp.float32)

        k_neighbors = self.k_neighbors or n_samples - 1
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nn.fit(column_data_gpu)
        A = nn.kneighbors_graph(mode='distance')
        A.data = cp.exp(-(A.data ** 2) * self.gamma)
        A = (A + A.T) / 2

        try:
            degree = cp.asarray(A.sum(axis=1)).flatten()
            D_inv_sqrt = 1.0 / cp.sqrt(cp.maximum(degree, 1e-12))
            A_norm = sp.diags(D_inv_sqrt) @ A @ sp.diags(D_inv_sqrt)
            
            eigenvalues, eigenvectors = eigsh(A_norm, k=self.output_dim + 1, which='LM', tol=1e-6)

            if cp.isnan(eigenvectors).any():
                print(f"  - Warning: Eigsh returned NaN on a connected graph. Returning zeros.")
                return cp.zeros((n_samples, self.output_dim), dtype=cp.float32)
            
            return eigenvectors[:, 1:]

        except Exception as e:
            print(f"  - Warning: Eigsh failed on a connected graph (Error: {e}). Returning zeros.")
            return cp.zeros((n_samples, self.output_dim), dtype=cp.float32)

    def fit(self, N_data, C_data=None, y_data=None, shared_state=None):
        if not (N_data and "train" in N_data):
            return self

        X_train = N_data.get("train")
        C_train = C_data.get("train") if C_data else None
        X_val = N_data.get("val")
        C_val = C_data.get("val") if C_data else None
        
        stds = np.std(X_train, axis=0)
        self.constant_cols_indices = np.where(stds == 0)[0]

        if len(self.constant_cols_indices) > 0:
            print(f"Found constant columns at indices: {self.constant_cols_indices}. Removing them.")
            X_train = np.delete(X_train, self.constant_cols_indices, axis=1)
            X_val = np.delete(X_val, self.constant_cols_indices, axis=1)

        n_features = X_train.shape[1]

        self.encoder_ = MLPMarginalEncoder(
            n_features=n_features,
            output_dim_per_feature=self.output_dim,
            hidden_dims_per_feature=[256, 256],
        )

        # ------------------------------------------------------------------
        # --- MODE 1: EXPLICIT (Original method) ---
        # ------------------------------------------------------------------
        if self.mode == 'explicit':
            print("Fitting RBFContextTransform in 'explicit' mode...")
            X_train_gpu = cp.asarray(X_train)
            X_val_gpu = cp.asarray(X_val) if X_val is not None else None
            n_val_samples = 0 if X_val_gpu is None else X_val_gpu.shape[0]

            target_embeddings_train: List[cp.ndarray] = []
            target_embeddings_val: List[cp.ndarray] = []

            for j in range(n_features):
                print(f"  - Pre-computing eigenfunctions for feature {j+1}/{n_features}")
                combined_col = cp.concatenate([X_train_gpu[:, j], X_val_gpu[:, j]]).reshape(-1, 1) if n_val_samples > 0 else X_train_gpu[:, j].reshape(-1, 1)
                Y_emb_gpu = self._spectral_embed_gpu(combined_col)
                target_embeddings_train.append(Y_emb_gpu[:X_train_gpu.shape[0]])
                if n_val_samples > 0:
                    target_embeddings_val.append(Y_emb_gpu[X_train_gpu.shape[0]:])

            Y_target_full = cp.concatenate(target_embeddings_train, axis=1).get()
            Y_target_full_val = cp.concatenate(target_embeddings_val, axis=1).get() if n_val_samples > 0 else None
            
            print(f"Training unified MarginalEncoder via supervised learning...")
            trainer = MarginalEncoderTrainer(
                encoder=self.encoder_, epochs=self.epochs, lr=self.lr,
                batch_size=self.batch_size, patience=self.patience
            )
            trainer.train(X_train, C_train, Y_target_full, X_val=X_val, C_val=C_val, Y_val=Y_target_full_val)

        # ------------------------------------------------------------------
        # --- MODE 2: IMPLICIT ---
        # ------------------------------------------------------------------
        elif self.mode == 'implicit':
            print("Fitting RBFContextTransform in 'implicit' mode...")
            # trainer = ManifoldRegularizationBatchTrainer(
            #     encoder=self.encoder_, epochs=self.epochs, lr=self.lr,
            #     batch_size=self.batch_size, gamma=self.gamma,
            #     lambda_cov=self.lambda_cov, patience=self.patience
            # )
            lambda_lap = 0
            lambda_inter = 0
            trainer = JointManifoldTrainer(
                            encoder=self.encoder_,
                            epochs=self.epochs,
                            lr=self.lr,
                            batch_size=128,
                            gamma=self.gamma,
                            patience=self.patience,
                            lambda_lap=lambda_lap,
                            lambda_cov=self.lambda_cov,
                            lambda_inter=lambda_inter 
                        )
            trainer.train(X_train, C_train, X_val=X_val, C_val=C_val)

        if shared_state is not None:
            feature_map_ = []
            for i in range(n_features-len(self.constant_cols_indices)):
                feature_map_.append({"orig_idx": i, "new_start": None, "size": self.output_dim})
            shared_state['encoder'] = self.encoder_
            shared_state['d_in'] = n_features * self.output_dim
            shared_state['feature_map_'] = feature_map_

        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if len(self.constant_cols_indices) > 0:
            print(f"Removing constant columns at indices: {self.constant_cols_indices} from transformed data.")
            for part in N_data:
                N_data[part] = np.delete(N_data[part], self.constant_cols_indices, axis=1)
        return N_data, C_data, y_data
    
