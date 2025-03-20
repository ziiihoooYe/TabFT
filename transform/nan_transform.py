import numpy as np
from transform.base import BaseTransform

class NanTransform(BaseTransform):
    """
    Fill numeric NaN with the mean/median of training data (per column).
    Fill categorical NaN with a new token '___null___'.
    """

    def __init__(self, args):
        super().__init__()
        self.num_policy = args.get('policy', 'mean')
        self.num_new_value = None
        self.cat_new_token = "___null___"
        assert self.num_policy in ['mean', 'median'], "Numeric Nan Policy must be 'mean' or 'median'."

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        # obtain numeric NaN value
        if N_data is not None and 'train' in N_data:
            # shape: (n_train, d_num)
            train_array = N_data['train']
            if self.num_policy == 'median':
                self.num_new_value = np.nanmedian(train_array, axis=0)
            else:
                self.num_new_value = np.nanmean(train_array, axis=0)

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        # numeric NaN
        if N_data is not None and self.num_new_value is not None:
            for part in N_data.keys():
                arr = N_data[part]
                nan_mask = np.isnan(arr)
                arr[nan_mask] = np.take(self.num_new_value, nan_mask.nonzero()[1])
        
        # categorical NaN
        if C_data is not None:
            for part, arr in C_data.items():
                if arr.dtype.kind in ['U','S','O']:  # string or object
                    mask = (arr == '') | (arr == 'nan') | (arr == 'NaN')
                else:
                    mask = np.isnan(arr)
                arr[mask] = self.cat_new_token

        return N_data, C_data, y_data

