import os.path as osp
import time
import math
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from model.methods.base import Method
import torch
from tqdm import tqdm
from model.lib.data import Dataset
from model.utils import (
    Averager
)

class TabMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        self.const_N_cols = None
        self.const_C_cols = None
        
    def _precompute_laplacian_pe(self, N_train, C_train, d_lap_pe):
        print("Pre-computing Laplacian Positional Encodings...")
        device = N_train.device if N_train is not None else C_train.device
        
        # 1. Prepare the node matrix: one-hot categorical + numerical
        node_matrices = []
        if N_train is not None:
            node_matrices.append(N_train.double())

        if C_train is not None and self.categories:
            C_train_one_hot = torch.cat([
                F.one_hot(C_train[:, i], num_classes=self.categories[i]).to(device)
                for i in range(C_train.shape[1])
            ], dim=1)
            node_matrices.append(C_train_one_hot.double()) 

        if not node_matrices:
            print("No data available to compute LapPE. Skipping.")
            return None

        full_node_matrix = torch.cat(node_matrices, dim=1)
        
        affinity_matrix = torch.corrcoef(full_node_matrix.T)
        affinity_matrix = torch.nan_to_num(affinity_matrix, 0.0)

        A = F.relu(affinity_matrix)
        D = torch.diag(torch.sum(A, dim=1))
        D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0 # Handle isolated nodes
        L_norm = torch.eye(A.shape[0], device=device) - D_inv_sqrt @ A @ D_inv_sqrt
        
        _, eigenvectors = torch.linalg.eigh(L_norm)
        
        # 5. Select top `d_lap_pe` eigenvectors (corresponding to smallest eigenvalues)
        # The columns of `eigenvectors` are the eigenvectors.
        d_lap_pe = (d_lap_pe - 1)//self.args.config['model'].get('n_heads', 4) * self.args.config['model'].get('n_heads', 4)
        lap_pe = eigenvectors[:, 1 : d_lap_pe + 1].float() # Skip the first eigenvector (constant)
        
        print(f"Computed LapPE matrix of shape: {lap_pe.shape}")
        return lap_pe

    def construct_model(self, model_config = None, lap_pe: torch.Tensor = None):
        if model_config is None:
            model_config = self.args.config['model']
        self.feature_map = getattr(self, "feature_map_", None) or self.data_transform_pipeline.shared_state.get('feature_map_', None)
        
        n_cat_nodes = sum(self.categories) if self.categories else 0
        n_num_nodes = self.n_num_features if self.n_num_features is not None else 0
        self.num_groups = n_num_nodes + n_cat_nodes
        
        from model.models.Tab import Tab
        self.model = Tab(
            config=model_config,
            num_continuous=self.n_num_features or 0,
            categories=self.categories,
            d_out=self.d_out,
            lap_pe=lap_pe,
            x_num_train=self.N['train']
        ).to(self.args.device)

        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def delete_const_col(self, N: np.ndarray = None, C: np.ndarray = None):
        if N is None and C is None:
            return None
        if N is not None and isinstance(N, dict):
            for part in N:
                if part == 'train':
                    self.const_N_cols = np.std(N[part], axis=0) == 0
                    N[part] = N[part][:, ~self.const_N_cols]
                else:
                    N[part] = N[part][:, ~self.const_N_cols]
        if C is not None and isinstance(C, dict):
            for part in C:
                if part == 'train':
                    self.const_C_cols = np.std(C[part], axis=0) == 0
                    C[part] = C[part][:, ~self.const_C_cols]
                else:
                    C[part] = C[part][:, ~self.const_C_cols]
        return N, C
                
    def fit(self, data, info, train = True, config = None, tune = False):
        N,C,y = data 
        N, C = self.delete_const_col(N, C)
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features 
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)

        self.feature_map_ = self.shared_state.get('feature_map_', None)

        # Pre‑compute column‑level MI matrix (used by MI pre‑training)
        self.n_num_features = N['train'].shape[1] if N is not None else self.n_num_features
        
        # Pre‑compute Laplacian PE if enabled in config
        lap_pe = None
        d_lap_pe = self.args.config['model'].get('d_lap_pe', 0)
        if d_lap_pe > 0:
            N_train_tensor = self.N['train'].to(self.args.device) if self.N else None
            C_train_tensor = self.C['train'].to(self.args.device) if self.C else None
            lap_pe = self._precompute_laplacian_pe(N_train_tensor, C_train_tensor, d_lap_pe)
        
        self.construct_model(lap_pe=lap_pe)

        # self.construct_model()
        
        if self.shared_state.get('encoder', None) is not None:
            self.encoder = deepcopy(self.shared_state['encoder']) 
            params = [{'params': self.model.parameters()}, {'params': self.encoder.parameters()}]
        else:
            from transform.context_transform import IdentityEncoder
            self.encoder = IdentityEncoder()
            params = [{'params': self.model.parameters()}]  
        
        self.optimizer = torch.optim.AdamW(
            params, 
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return        

        # supervised learning
        time_cost = 0
        max_epoch = self.args.config['training'].get('max_epoch', None) or self.args.max_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epoch,
            eta_min=1e-6
        )
        for epoch in range(max_epoch):
            tic = time.time()
            self.train_epoch(epoch)
            self.validate(epoch)
            self.scheduler.step()
            elapsed = time.time() - tic
            time_cost += elapsed
            # print(f'Epoch: {epoch}, Time cost: {elapsed}')
            if not self.continue_training:
                break
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )
        return time_cost
    
    def validate(self, epoch):
        """
        Validate the model.

        :param epoch: int, the current epoch
        """
        # print('best epoch {}, best val res={:.4f}'.format(
        #     self.trlog['best_epoch'], 
        #     self.trlog['best_res']))
        
        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader), disable=True):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None                            

                pred = self.model(X_num, X_cat)

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()   

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)


        # print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        if measure(vres[0], self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres[0]
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 40:
                self.continue_training = False
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))   

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.

        :param epoch: int, the current epoch
        """
        self.model.train()
        tl = Averager()
        for i, (X, y) in enumerate(self.train_loader, 1):
            self.train_step = self.train_step + 1
            if self.N is not None and self.C is not None:
                X_num, X_cat = X[0], X[1]
            elif self.C is not None and self.N is None:
                X_num, X_cat = None, X
            else:
                X_num, X_cat = X, None

            loss = self.criterion(self.model(X_num, X_cat), y)

            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # if (i-1) % 50 == 0 or i == len(self.train_loader):
            #     print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
            #         epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)    

    def predict(self, data, info, model_name):
        """
        Predict the results of the data.

        :param data: tuple, (N, C, y)
        :param info: dict, information about the data
        :param model_name: str, name of the model
        :return: tuple, (loss, metric, metric_name, predictions)
        """
        N,C,y = data
        N, C = self.delete_const_col(N, C)
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        # print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        ## Evaluation Stage
        self.model.eval()
        if getattr(self, 'encoder', None) is not None:
            self.encoder.eval()

        self.data_format(False, N, C, y)
        
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.test_loader), disable=True):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None

                pred = self.model(*self.encoder.encode(X_num, X_cat))

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()     

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # print('Test: loss={:.4f}'.format(vl))
        # for name, res in zip(metric_name, vres):
        #     print('[{}]={:.4f}'.format(name, res)mean_std)

        
        return vl, vres, metric_name, test_logit