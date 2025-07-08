import os.path as osp
import time
import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from model.methods.base import Method
import torch
from tqdm import tqdm
import torch.nn.functional as F
from model.lib.data import Dataset
from model.utils import (
    Averager
)
from model.lib.data import data_loader_process

class TabMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        self.feature_map = getattr(self, "feature_map_", None)
        self.num_groups  = len(self.feature_map) if self.feature_map else self.n_num_features

        from model.models.Tab import Tab
        self.model = Tab(
            config=model_config,
            num_continuous=self.num_groups,
            categories=self.categories,
            d_out=self.d_out,
            is_regression=self.is_regression,
            feature_map=self.feature_map
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def fit(self, data, info, train = True, config = None, tune = False):
        """
        Fit the method to the data.

        :param data: tuple, (N, C, y)
        :param info: dict, information about the data
        :param train: bool, whether to train the method
        :param config: dict, configuration for the method
        :return: float, time cost
        """
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        N,C,y = data
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
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return

        # supervised learning
        time_cost = 0
        max_epoch = self.args.config['training'].get('max_epoch', None) or self.args.max_epoch
        for epoch in range(max_epoch):
            tic = time.time()
            self.train_epoch(epoch)
            self.validate(epoch)
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


        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
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
