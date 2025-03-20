import abc
import numpy as np
import sklearn.metrics as skm
from model.utils import (
    set_seeds,
    get_device
)
# from sklearn.externals import joblib
from model.lib.data import Dataset
from transform.transform_pipeline import DataTransformPipeline

def check_softmax(logits):
    # Check if any values are outside the [0, 1] range and Ensure they sum to 1
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize by subtracting max
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        return logits
    
class classical_methods(object, metaclass=abc.ABCMeta):
    def __init__(self, args, is_regression):
        self.args = args
        self.is_regression = is_regression
        self.D = None
        self.args.device = get_device()
        self.trlog = {}
        self.data_transform_pipeline = DataTransformPipeline(args.transform_list, args, is_regression)

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
            self.n_num_features = self.N['train'].shape[1] if self.N is not None else 0
            self.n_cat_features = self.C['train'].shape[1] if self.C is not None else 0
            self.d_in = 0 if self.N is None else self.N['train'].shape[1]
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
            
    def construct_model(self, model_config = None):
        raise NotImplementedError

    def fit(self, data, info, train = True, config = None):
        N, C, y = data
        # if self.D is None:
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
          
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.construct_model()

        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        
    def reset_stats_withconfig(self, config):
        set_seeds(self.args.seed)
        self.config = self.args.config = config

    def metric(self, predictions, labels, y_info):
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
        
