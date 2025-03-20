from model.methods.base import Method
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from tqdm import tqdm
import os.path as osp

from model.utils import (
    Timer,
    Averager,
    set_seeds,
    get_device
)

from model.lib.data import Dataset

class GRANDEMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None):
        from model.models.grande import GRANDE
        if model_config is None:
            model_config = self.args.config['model']
        self.model = GRANDE(
                batch_size=self.args.batch_size,
                task_type='binclass' if self.is_binclass else 'multiclass' if self.is_multiclass else 'regression',
                **model_config
                ).to(self.args.device) 
        # print(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            N_data, C_data, y_data = self.D.N, self.D.C, self.D.y
            N_data, C_data, y_data = self.data_transform_pipeline.fit_transform(N_data, C_data, y_data)
            self.N, self.C, self.y = N_data, C_data, y_data
            self.y_info = self.data_transform_pipeline.shared_state.get('y_info', {'policy': 'none'})

        else:
            N_test, C_test, y_test = N, C, y
            N_test, C_test, y_test = self.data_transform_pipeline.transform(N_test, C_test, y_test)
            if N_test is not None and C_test is not None:
                self.N_test, self.C_test = N_test['test'], C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test, self.C_test = None, C_test['test']
            else:
                self.N_test, self.C_test = N_test['test'], None
            self.y_test = y_test['test']

    def fit(self, data, info, train = True, config = None):
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train=True)
        self.construct_model()
        
        if self.C is None:
            assert self.N is not None
            X_train = np.array(self.N['train'])
            X_val = np.array(self.N['val'])
            cat_idx = []
        elif self.N is None:
            assert self.C is not None
            X_train = np.array(self.C['train'])
            X_val = np.array(self.C['val'])
            cat_idx = list(range(self.C.shape[1]))
        else:
            assert self.C is not None and self.N is not None
            X_train = np.concatenate((np.array(self.C['train']), np.array(self.N['train'])), axis=1)
            X_val = np.concatenate((np.array(self.C['val']), np.array(self.N['val'])), axis=1)
            cat_idx = list(range(self.C['train'].shape[1]))
        
        y_train = np.array(self.y['train'])
        y_val = np.array(self.y['val'])
        
        self.model.cat_idx = cat_idx
        X_train, y_train, X_val, y_val = self.model.preprocess_data(X_train, y_train, X_val, y_val)
        self.model.number_of_variables = X_train.shape[1]
        if self.model.task_type in ['binclass', 'multiclass']:
            self.model.number_of_classes = len(np.unique(y_train))
        else:
            self.model.number_of_classes = 1

        self.model.build_model()
        
        self.model.to(self.args.device)
        self.model.double()

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32 if self.args.use_float else torch.float64), torch.tensor(y_train, dtype=torch.float32 if self.args.use_float else torch.float64))
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32 if self.args.use_float else torch.float64), torch.tensor(y_val, dtype=torch.float32 if self.args.use_float else torch.float64))
            self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        else:
            self.val_loader = None

        if self.model.task_type == 'multiclass':
            self.criterion = nn.CrossEntropyLoss()
        elif self.model.task_type == 'binclass':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.model.task_type == 'regression':
            self.criterion = nn.MSELoss()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return

        time_cost = 0
        for epoch in range(self.args.max_epoch):
            tic = time.time()
            self.train_epoch(epoch)
            self.validate(epoch)
            elapsed = time.time() - tic
            time_cost += elapsed
            print(f'Epoch: {epoch}, Time cost: {elapsed}')
            if not self.continue_training:
                break
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )
        return time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        ## Evaluation Stage
        self.model.eval()

        self.data_format(False, N, C, y)

        if self.C is None:
            assert self.N is not None
            X_test = np.array(self.N_test)
        elif self.N is None:
            assert self.C is not None
            X_test = np.array(self.C_test)
        else:
            assert self.C is not None and self.N is not None
            X_test = np.concatenate((np.array(self.C_test), np.array(self.N_test)), axis=1)
        
        y_test = np.array(self.y_test)
        
        # X_test = self.model.apply_preprocessing(X_test)
        
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.float64))
        self.test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, drop_last=False)
        
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.test_loader)):
                X, y = X.to(self.args.device), y.to(self.args.device)
                pred = self.model(X)
                test_logit.append(pred)
                test_label.append(y)
        
        # print(f'[DEBUG 318] {test_logit=}, {test_label=}')

        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        test_label__ = test_label.long() if self.is_multiclass else test_label.unsqueeze(1)
        vl = self.criterion(test_logit, test_label__).item()
        
        if self.is_binclass:
            test_logit = np.stack([-test_logit.cpu().squeeze(), test_logit.cpu().squeeze()], axis=-1)
        # elif self.is_regression:
        #     test_logit = test_logit * self.model.std + self.model.mean

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            # print('[{}]={:.4f}'.format(name, res))

        return vl, vres, metric_name, test_logit

    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        for i, (X, y) in enumerate(self.train_loader, 1):
            self.train_step = self.train_step + 1

            y = y.long() if self.is_multiclass else y.unsqueeze(1)
            X, y = X.to(self.args.device), y.to(self.args.device)
            loss = self.criterion(self.model(X), y)

            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i-1) % 50 == 0 or i == len(self.train_loader):
                print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
                    epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)    

    def validate(self, epoch):
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], 
            self.trlog['best_res']))
        
        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []

        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader)):
                X, y = X.to(self.args.device), y.to(self.args.device)
                pred = self.model(X)
                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        test_label__ = test_label.long() if self.is_multiclass else test_label.unsqueeze(1)
        vl = self.criterion(test_logit, test_label__).item()   

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        if self.is_binclass:
            test_logit = np.stack([-test_logit.cpu().squeeze(), test_logit.cpu().squeeze()], axis=-1)
        # elif self.is_regression:
        #     test_logit = test_logit * self.model.std + self.model.mean
            
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
            if self.val_count > 20:
                self.continue_training = False
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))  