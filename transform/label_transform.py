import numpy as np
from transform.base import BaseTransform
from sklearn.preprocessing import LabelEncoder

class LabelTransform(BaseTransform):
    """
    Handle label transformation: 
      - If regression, subtract mean, divide by std.
      - If classification, label-encode the classes.
    """

    def __init__(self, is_regression=False):
        super().__init__()
        self.is_regression = is_regression
        self.mean_ = None
        self.std_ = None
        self.label_encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if y_data is None:
            return self

        if shared_state is None:
            shared_state = {}

        if self.is_regression:
            # compute mean, std on train
            train_y = y_data['train']
            self.mean_ = np.mean(train_y)
            self.std_ = np.std(train_y)
            shared_state['y_info'] = {'policy': 'mean_std', 
                                             'mean': self.mean_, 
                                             'std': self.std_}
        else:
            # classification => label encode
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y_data['train'])
            shared_state['y_info'] = {'policy': 'none'}
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if y_data is None:
            return N_data, C_data, y_data

        if self.is_regression:
            for part, arr in y_data.items():
                y_data[part] = (arr - self.mean_) / self.std_ if self.std_ != 0 else (arr - self.mean_)
        else:
            for part, arr in y_data.items():
                y_data[part] = self.label_encoder_.transform(arr)
        return N_data, C_data, y_data