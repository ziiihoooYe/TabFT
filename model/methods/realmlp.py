from model.methods.base import Method
import torch
import numpy as np
import torch.nn.functional as F
import time

from model.lib.data import (
    Dataset,
    get_categories
)

 
class RealMLPMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        

    def construct_model(self, model_config = None):
        from model.models.realmlp import RealMLP_TD_Classifier, RealMLP_TD_Regressor
        if model_config is None:
            model_config = self.args.config['model']
        if self.is_regression:
            self.model = RealMLP_TD_Regressor(
                device=self.args.device,
                random_state=self.args.seed,
                n_epochs=self.args.max_epoch,
                **model_config
            )
        else:
            self.model = RealMLP_TD_Classifier(
                device=self.args.device,
                random_state=self.args.seed,
                n_epochs=self.args.max_epoch,
                **model_config
            )


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            N_data, C_data, y_data = self.D.N, self.D.C, self.D.y
            N_data, C_data, y_data = self.data_transform_pipeline.fit_transform(N_data, C_data, y_data)
            self.N, self.C, self.y = N_data, C_data, y_data
            self.y_info = self.data_transform_pipeline.shared_state.get('y_info', {'policy': 'none'})

            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y['train']))
            self.d_in = 0 if self.N is None else self.N['train'].shape[1]
            self.categories = get_categories(self.C)
            self.criterion = (
                F.mse_loss
                if self.is_regression
                else F.cross_entropy
            )
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
        N, C, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        
        if config:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.construct_model()
        
        if self.C is None:
            assert self.N is not None
            X_train = np.array(self.N['train'])
            X_val = np.array(self.N['val'])
            cat_features = [0] * self.N['train'].shape[1]
        elif self.N is None:
            assert self.C is not None
            X_train = np.array(self.C['train'])
            X_val = np.array(self.C['val'])
            cat_features = [1] * self.C['train'].shape[1]
        else:
            assert self.C is not None and self.N is not None
            X_train = np.concatenate((np.array(self.C['train']), np.array(self.N['train'])), axis=1)
            X_val = np.concatenate((np.array(self.C['val']), np.array(self.N['val'])), axis=1)
            cat_features = [1] * self.C['train'].shape[1] + [0] * self.N['train'].shape[1]
        
        y_train = np.array(self.y['train'])
        y_val = np.array(self.y['val'])
        
        X_train_and_val = np.concatenate([X_train, X_val], axis=0)
        y_train_and_val = np.concatenate([y_train, y_val], axis=0)
        val_idxs = np.arange(X_train.shape[0], X_train.shape[0] + X_val.shape[0], dtype=np.int32)

        tic = time.time()
        self.model.fit(X_train_and_val, y_train_and_val, val_idxs=val_idxs, cat_features=cat_features)
        
        test_logit = self.model.predict(X_val)
        test_label = self.y['val']
        
        test_logit = torch.from_numpy(test_logit)
        test_label = torch.from_numpy(test_label)
        
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        time_cost = time.time() - tic
        self.trlog['best_res'] = vres[0]
        return time_cost


    def predict(self, data, info, model_name):
        N, C, y = data
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        
        self.data_format(False, N, C, y)
        
        if self.C_test is None:
            assert self.N_test is not None
            X_test = np.array(self.N_test)
        elif self.N_test is None:
            assert self.C_test is not None
            X_test = np.array(self.C_test)
        else:
            assert self.C_test is not None and self.N_test is not None
            X_test = np.concatenate((np.array(self.C_test), np.array(self.N_test)), axis=1)

        test_logit = self.model.predict(X_test)
        test_label = self.y_test
    
        test_logit = torch.from_numpy(test_logit)
        test_label = torch.from_numpy(test_label)
        
        vl = self.criterion(test_logit, test_label).item()

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # print('Test: loss={:.4f}'.format(vl))
        # for name, res in zip(metric_name, vres):
        #     print('[{}]={:.4f}'.format(name, res))

        return vl, vres, metric_name, test_logit