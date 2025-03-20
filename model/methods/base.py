import abc
import torch
import numpy as np
import time
import os.path as osp
from tqdm import tqdm
import sklearn.metrics as skm

from model.utils import (
    Timer,
    Averager,
    set_seeds,
    get_device
)

from model.lib.data import (
    Dataset,
    data_loader_process,
    get_categories
)

from transform.transform_pipeline import DataTransformPipeline

def check_softmax(logits):
    """
    Check if the logits are already probabilities, and if not, convert them to probabilities.
    
    :param logits: np.ndarray of shape (N, C) with logits
    :return: np.ndarray of shape (N, C) with probabilities
    """
    # Check if any values are outside the [0, 1] range and Ensure they sum to 1
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize by subtracting max
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        return logits

class Method(object, metaclass=abc.ABCMeta):
    def __init__(self, args, is_regression):
        """
        :param args: argparse object
        :param is_regression: bool, whether the task is regression or not
        """
        self.args = args
        self.is_regression = is_regression
        self.D = None

        self.train_step = 0
        self.val_count = 0
        self.continue_training = True
        self.timer = Timer()

        self.trlog = {}
        self.trlog['args'] = vars(args)
        self.trlog['train_loss'] = []
        self.trlog['best_epoch'] = 0
        if self.is_regression:
            self.trlog['best_res'] = 1e10
        else:
            self.trlog['best_res'] = 0 

        self.args.device = get_device()
        
        self.data_transform_pipeline = DataTransformPipeline(args.transform_list, args, is_regression)

    def reset_stats_withconfig(self, config):
        """
        Reset the training statistics with a new configuration.

        :param config: dict, new configuration
        """
        set_seeds(self.args.seed)
        self.train_step = 0
        self.val_count = 0
        self.continue_training = True
        self.timer = Timer()
        self.config = self.args.config = config
        
        # train statistics
        self.trlog = {}
        self.trlog['args'] = vars(self.args)
        self.trlog['train_loss'] = []
        self.trlog['best_epoch'] = 0
        if self.is_regression:
            self.trlog['best_res'] = 1e10
        else:
            self.trlog['best_res'] = 0

    def data_format(self, is_train = True, N = None, C = None, y = None):
        """
        Format the data for training or testing.

        :param is_train: bool, whether the data is for training or testing
        :param N: dict, numerical data
        :param C: dict, categorical data
        :param y: dict, labels
        """
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
            self.N, self.C, self.y, self.train_loader, self.val_loader, self.criterion = data_loader_process(self.is_regression, (self.N, self.C), self.y, self.y_info, self.args.device, self.args.batch_size, is_train = True,is_float=self.args.use_float)
        else:
            N_test, C_test, y_test = N, C, y
            N_test, C_test, y_test = self.data_transform_pipeline.transform(N_test, C_test, y_test)


            _, _, _, self.test_loader, _ =  data_loader_process(self.is_regression, (N_test, C_test), y_test, self.y_info, self.args.device, self.args.batch_size, is_train = False,is_float=self.args.use_float)                      
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']
    
    
    def fit(self, data, info, train = True, config = None):
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
        self.construct_model()
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
            # print(f'Epoch: {epoch}, Time cost: {elapsed}')
            if not self.continue_training:
                break
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )
        return time_cost

    def predict(self, data, info, model_name):
        """
        Predict the results of the data.

        :param data: tuple, (N, C, y)
        :param info: dict, information about the data
        :param model_name: str, name of the model
        :return: tuple, (loss, metric, metric_name, predictions)
        """
        N,C,y = data
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        # print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        ## Evaluation Stage
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

        # print('Test: loss={:.4f}'.format(vl))
        # for name, res in zip(metric_name, vres):
        #     print('[{}]={:.4f}'.format(name, res))

        
        return vl, vres, metric_name, test_logit

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
                    # epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)    

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
            if self.val_count > 20:
                self.continue_training = False
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))   

    def metric(self, predictions, labels, y_info):
        """
        Compute the evaluation metric.

        :param predictions: np.ndarray, predictions
        :param labels: np.ndarray, labels
        :param y_info: dict, information about the labels
        :return: tuple, (metric, metric_name)
        """
        if not isinstance(labels, np.ndarray):
            labels = labels.cpu().numpy()
        if not isinstance(predictions, np.ndarray):
            predictions = predictions.cpu().numpy()
        if self.is_regression:
            mae = skm.mean_absolute_error(labels, predictions)
            rmse = skm.mean_squared_error(labels, predictions) ** 0.5
            r2 = skm.r2_score(labels, predictions)
            if y_info['policy'] == 'mean_std':
                mae *= y_info['std']
                rmse *= y_info['std']
            return (mae,r2,rmse), ("MAE", "R2", "RMSE")
        elif self.is_binclass:
            # if not softmax, convert to probabilities
            predictions = check_softmax(predictions)
            accuracy = skm.accuracy_score(labels, predictions.argmax(axis=-1))
            avg_recall = skm.balanced_accuracy_score(labels, predictions.argmax(axis=-1))
            avg_precision = skm.precision_score(labels, predictions.argmax(axis=-1), average='macro')
            f1_score = skm.f1_score(labels, predictions.argmax(axis=-1), average='binary')
            log_loss = skm.log_loss(labels, predictions)
            auc = skm.roc_auc_score(labels, predictions[:, 1])
            return (accuracy, avg_recall, avg_precision, f1_score, log_loss, auc), ("Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC")
        elif self.is_multiclass:
            # if not softmax, convert to probabilities
            predictions = check_softmax(predictions)
            accuracy = skm.accuracy_score(labels, predictions.argmax(axis=-1))
            avg_recall = skm.balanced_accuracy_score(labels, predictions.argmax(axis=-1))
            avg_precision = skm.precision_score(labels, predictions.argmax(axis=-1), average='macro')
            f1_score = skm.f1_score(labels, predictions.argmax(axis=-1), average='macro')
            log_loss = skm.log_loss(labels, predictions)
            auc = skm.roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
            return (accuracy, avg_recall, avg_precision, f1_score, log_loss, auc), ("Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC")
        else:
            raise ValueError("Unknown tabular task type")