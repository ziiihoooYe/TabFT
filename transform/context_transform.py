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


class MLP(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 8, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        hidden_dims = hidden_dims or [32]

        layers: List[nn.Module] = []
        cur = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
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
            [MLP(input_dim=1, output_dim=output_dim_per_feature, hidden_dims=hidden_dims_per_feature)
             for _ in range(n_features)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, n_features)
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
###              .        Encoder Trainer                             .  ###
############################################################################
class MarginalEncoderTrainer:
    def __init__(self,
                 encoder: nn.Module,           # Expects a single encoder object
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 1024,
                 patience: int = 10,
                 delta: float = 1e-5,
                 device: str | None = None):
        self.encoder = encoder
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.patience = patience
        self.delta = delta

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
        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        patience_ctr = 0

        self.encoder.train()
        for epoch in range(self.epochs):
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                
                opt.zero_grad(set_to_none=True)
                
                pred = self.encoder(bx)
                
                loss = loss_fn(pred, by)
                loss.backward()
                opt.step()

            # ----- validation & early‑stopping -----
            if X_val is not None and Y_val is not None:
                val_loss = self._evaluate(X_val, Y_val)
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
        self.output_dim = args.get("output_dim", 8)
        self.gamma = args.get("gamma", 1.0)
        # k for k-Nearest-Neighbors graph, crucial for performance
        self.k_neighbors = args.get("k_neighbors", None) 
        self.encoder_ = None

    def _spectral_embed_gpu(self, column_data_gpu: cp.ndarray) -> cp.ndarray:
        """
        Performs spectral embedding robustly.
        If the graph is disconnected, it processes each connected component separately
        to preserve all structural information.
        """
        n_samples = column_data_gpu.shape[0]

        if column_data_gpu.min() == column_data_gpu.max():
            return cp.zeros((n_samples, self.output_dim), dtype=cp.float32)

        k_neighbors = self.k_neighbors or n_samples - 1
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nn.fit(column_data_gpu)
        A = nn.kneighbors_graph(mode='distance')
        A.data = cp.exp(-(A.data ** 2) * self.gamma)
        A = (A + A.T) / 2

        # 1. Find the connected components of the graph
        n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)

        # 2. If the graph is fully connected, use the fast, direct path
        if n_components == 1:
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

        # 3. If the graph is disconnected, process each component individually
        else:
            print(f"  - Info: Graph is disconnected with {n_components} components. Processing each individually.")
            # Initialize an empty result array
            Y_embedding = cp.zeros((n_samples, self.output_dim), dtype=cp.float32)

            # Loop over each "island"
            for i in range(n_components):
                # Find all samples belonging to the current component
                indices = cp.where(labels == i)[0]
                n_sub_samples = len(indices)

                # If a component is too small to compute the required embeddings, skip it.
                # Its embedding will remain zero.
                if n_sub_samples <= self.output_dim:
                    continue
                
                # Create a subgraph for this component
                sub_A = A[indices, :][:, indices]

                try:
                    # Run spectral embedding just on this component
                    degree = cp.asarray(sub_A.sum(axis=1)).flatten()
                    D_inv_sqrt = 1.0 / cp.sqrt(cp.maximum(degree, 1e-12))
                    sub_A_norm = sp.diags(D_inv_sqrt) @ sub_A @ sp.diags(D_inv_sqrt)

                    eigenvalues, sub_eigenvectors = eigsh(sub_A_norm, k=self.output_dim + 1, which='LM', tol=1e-6)
                    
                    if not cp.isnan(sub_eigenvectors).any():
                        # Place the resulting embeddings back into the correct rows
                        Y_embedding[indices, :] = sub_eigenvectors[:, 1:]
                
                except Exception:
                    # If even the subgraph fails, we leave its embeddings as zero and continue
                    continue
            
            return Y_embedding


    def fit(self, N_data, C_data=None, y_data=None, shared_state=None):
        if not (N_data and "train" in N_data):
            return self

        X_train_gpu = cp.asarray(N_data["train"])
        C_train_gpu = cp.asarray(C_data["train"]) if C_data is not None else None
        
        X_val_gpu = cp.asarray(N_data.get("val"))
        C_val_gpu = cp.asarray(C_data.get("val")) if C_data is not None else None
        n_val_samples = 0 if X_val_gpu is None else X_val_gpu.shape[0]

        n_samples, n_features = X_train_gpu.shape
        
        target_embeddings_train: List[cp.ndarray] = []
        target_embeddings_val: List[cp.ndarray] = []

        print(f"Fitting RBFContextTransform for {n_features} features...")

        for j in range(n_features):
            print(f"  - Processing feature {j+1}/{n_features}")
            if n_val_samples > 0:
                combined_col = cp.concatenate([X_train_gpu[:, j], X_val_gpu[:, j]]).reshape(-1, 1)
            else:
                combined_col = X_train_gpu[:, j].reshape(-1, 1)

            # Call our GPU-accelerated spectral embedding function
            Y_emb_gpu = self._spectral_embed_gpu(combined_col)

            # Split back into train / val parts
            target_embeddings_train.append(Y_emb_gpu[:n_samples])
            if n_val_samples > 0:
                target_embeddings_val.append(Y_emb_gpu[n_samples:])

        # Instantiate the encoder and move it to GPU
        self.encoder_ = MLPMarginalEncoder(
            n_features=n_features,
            output_dim_per_feature=self.output_dim,
            hidden_dims_per_feature=[32],
        ).to('cuda')
        
        # Concatenate embeddings on the GPU
        Y_target_full_gpu = cp.concatenate(target_embeddings_train, axis=1)
        Y_target_full_val_gpu = (
            None if n_val_samples == 0
            else cp.concatenate(target_embeddings_val, axis=1)
        )

        # Train the encoder on GPU data
        print(f"Training unified MarginalEncoder on GPU...")
        trainer = MarginalEncoderTrainer(
            encoder=self.encoder_,
            epochs=1000,
            lr=1e-3,
            batch_size=1024
        )
        trainer.train(
            X_train_gpu.get(),
            C_train_gpu.get() if C_train_gpu is not None else None,
            Y_target_full_gpu.get(),
            X_val=X_val_gpu.get(),
            C_val=C_val_gpu.get() if C_val_gpu is not None else None,
            Y_val=Y_target_full_val_gpu.get()
        )

        if shared_state is not None:
            shared_state['encoder'] = self.encoder_
            shared_state['d_in'] = n_features * self.output_dim

        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None): 
        # For a complete pipeline, this method should also use GPU
        return N_data, C_data, y_data