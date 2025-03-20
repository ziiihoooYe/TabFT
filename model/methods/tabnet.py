from model.methods.base import Method
import torch
import torch
import torch.nn.functional as F

from model.lib.data import Dataset
import time

class TabNetMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None):
        from model.models.tabnet import TabNetClassifier, TabNetRegressor
        if model_config is None:
            model_config = self.args.config['model']
        if self.is_regression:
            self.model = TabNetRegressor(
                gamma=model_config['gamma'],
                n_steps=model_config['n_steps'],
                n_independent=model_config['n_independent'],
                n_shared=model_config['n_shared'],
                momentum=model_config['momentum'],
                optimizer_params={'lr':model_config['lr']},
                seed=self.args.seed)
        else:
            self.model = TabNetClassifier(
                gamma=model_config['gamma'],
                n_steps=model_config['n_steps'],
                n_independent=model_config['n_independent'],
                n_shared=model_config['n_shared'],
                momentum=model_config['momentum'],
                optimizer_params={'lr':model_config['lr']},
                seed=self.args.seed)
            
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
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']
        self.criterion = F.cross_entropy if not self.is_regression else F.mse_loss

    def fit(self, data, info, train = True, config = None):
        N,C,y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.construct_model()
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        X_train = self.N['train']
        y_train = self.y['train']
        X_valid = self.N['val']
        y_valid = self.y['val']
        eval_metric = ['accuracy']
        if self.is_regression:
            y_train = y_train.reshape(-1, 1)
            y_valid = y_valid.reshape(-1, 1)
            eval_metric = ['rmse']
            task = "regression"
        elif self.is_binclass:
            task = "binclass"
        else:
            task = "multiclass"
        tic = time.time()
        loss, result, auc = self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=eval_metric,
            max_epochs=self.args.max_epoch, patience=20,
            batch_size=self.args.batch_size, virtual_batch_size=256,
            device=f'cuda:0',
            task=task
        )
        time_cost = time.time() - tic
        self.model.save_model(self.args.save_path)
        self.trlog['best_res'] = self.model.best_cost
        if self.is_regression:
            self.trlog['best_res'] = self.model.best_cost * self.y_info['std']
        return time_cost
    
    def predict(self, data, info, model_name):
        N,C,y = data
        self.model.load_model(self.args.save_path,self.args.seed)
        self.data_format(False, N, C, y)
        if self.is_regression:
            task_type = "regression"
            test_logit = self.model.predict(self.N_test)
        else:
            task_type = "classification"
            test_logit = self.model.predict_proba(self.N_test)
        test_label = self.y_test
        vl = self.criterion(torch.tensor(test_logit), torch.tensor(test_label)).item()     

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # print('Test: loss={:.4f}'.format(vl))
        # for name, res in zip(metric_name, vres):
        #     print('[{}]={:.4f}'.format(name, res))

        
        return vl, vres, metric_name, test_logit
