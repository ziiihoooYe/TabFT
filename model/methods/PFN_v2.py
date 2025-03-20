from model.methods.base import Method
import torch
import numpy as np
import torch
import torch.nn.functional as F

from model.lib.data import Dataset
import time

class TabPFNMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            N_data, C_data, y_data = self.D.N, self.D.C, self.D.y
            N_data, C_data, y_data = self.data_transform_pipeline.fit_transform(N_data, C_data, y_data)
            self.N, self.C, self.y = N_data, C_data, y_data
            self.y_info = self.data_transform_pipeline.shared_state.get('y_info', {'policy': 'none'})
            self.criterion = F.cross_entropy if  not self.is_regression else F.mse_loss
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

    def construct_model(self, model_config = None,cat_indices=[]):
        if self.is_regression:
            from model.models.PFN_v2 import TabPFNRegressor
            self.model = TabPFNRegressor(device = self.args.device,random_state = self.args.seed,n_estimators=4,ignore_pretraining_limits=True,categorical_features_indices=cat_indices)
        else:
            from model.models.PFN_v2 import TabPFNClassifier
            self.model = TabPFNClassifier(device = self.args.device,random_state = self.args.seed,n_estimators=4,ignore_pretraining_limits=True,categorical_features_indices=cat_indices)  

    def fit(self, data, info, train = True, config = None):
        N,C,y = data
        # if self.D is None:
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.data_format(is_train = True)
        

        sampled_Y = self.y['train']
        cat_indices = []
        if self.N is not None and self.C is not None:
            sampled_X = np.concatenate((self.N['train'],self.C['train']),axis=1)
            cat_indices = [i for i in range(self.N['train'].shape[1],self.N['train'].shape[1]+self.C['train'].shape[1])]
        elif self.N is None and self.C is not None:
            sampled_X = self.C['train']
            cat_indices = [i for i in range(self.C['train'].shape[1])]
        else:
            sampled_X = self.N['train']
        sample_size = self.args.config['general']['sample_size']
        self.sampled_X = sampled_X
        self.sampled_Y = sampled_Y
        tic = time.time()
        self.construct_model(cat_indices=cat_indices)
        self.model.fit(sampled_X,sampled_Y,sample_size)
        time_cost = time.time() - tic
        return time_cost
    
    def predict(self, data, info, model_name):
        import time
        start_time = time.time()
        N,C,y = data
        self.data_format(False, N, C, y)
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test,self.C_test),axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test
        
        if self.is_regression:
            test_logit = self.model.predict(Test_X)
        else:
            test_logit = self.model.predict_proba(Test_X)
        test_label = self.y_test
        vl = self.criterion(torch.tensor(test_logit),torch.tensor(test_label)).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        # print('Test: loss={:.4f}'.format(vl))
        # for name, res in zip(metric_name, vres):
        #     print('[{}]={:.4f}'.format(name, res))
        # print('Time cost: {:.4f}s'.format(time.time() - start_time))
        return vl, vres, metric_name, test_logit

