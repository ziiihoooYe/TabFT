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
        
    def construct_model(self):
        model_config = self.args.config['model']
        
        n_cat_nodes = sum(self.categories) if self.categories else 0
        n_num_nodes = self.n_num_features if self.n_num_features is not None else 0
        self.num_groups = n_num_nodes + n_cat_nodes
        
        from model.models.Tab import Tab
        self.model = Tab(
            config=model_config,
            num_continuous=self.n_num_features or 0,
            categories=self.categories,
            d_out=self.d_out,
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
        
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
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
            if not self.continue_training:
                break
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )
        return time_cost
    
    def validate(self, epoch): 
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

        measure = np.less_equal if self.is_regression else np.greater_equal
        
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

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
            
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)    

    def predict(self, data, info, model_name):
        N,C,y = data
        N, C = self.delete_const_col(N, C)
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        self.model.eval()
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

                pred = self.model(X_num, X_cat)
                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        
        return vl, vres, metric_name, test_logit