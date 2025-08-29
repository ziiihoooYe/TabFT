import math
import numpy as np
import sklearn.preprocessing
import category_encoders
from transform.base import BaseTransform
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from typing import Dict, List, Any, Optional

from torch.utils.data import DataLoader, TensorDataset
from model.lib.num_embeddings import (
    PiecewiseLinearEncoding, UnaryEncoding, BinsEncoding, JohnsonEncoding, _check_bins
)


class BinningTransform(BaseTransform):
    def __init__(self, args, is_regression=False):
        super().__init__()
        self.method = args.get('method', 'Q')
        self.n_bins = args.get('n_bins', 2)
        self.tree_kwargs = args.get('tree_kwargs', {'min_samples_leaf': 64, 
                                                    'min_impurity_decrease': 1e-4})
        self.is_regression = is_regression

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        import torch
        from model.lib.num_embeddings import compute_bins

        if shared_state is None:
            shared_state = {}

        if N_data is not None and 'train' in N_data:
            train_t = torch.from_numpy(N_data['train']).float()
            if self.method == 'Q':
                if isinstance(self.n_bins, (list, tuple, np.ndarray)):
                    bins_ = []
                    for col_idx, nb in enumerate(self.n_bins):
                        col_t = train_t[:, col_idx : col_idx + 1]          # (N,1)
                        # compute_bins 返回 [edges]；取第 0 个
                        col_edges = compute_bins(col_t,
                                                 n_bins=int(nb),
                                                 tree_kwargs=None,
                                                 y=None,
                                                 regression=None)[0]
                        bins_.append(col_edges)
                else:
                    bins_ = compute_bins(train_t, n_bins=self.n_bins,
                                         tree_kwargs=None, y=None, regression=None)
            elif self.method == 'T':
                y_train = torch.from_numpy(y_data['train']) if y_data else None
                bins_ = compute_bins(train_t, 
                                     n_bins=self.n_bins,
                                     tree_kwargs=self.tree_kwargs,
                                     y=y_train,
                                     regression=self.is_regression)
            else:
                raise ValueError(f"Unknown binning method: {self.method}")

            # Store bins in context so other transforms can retrieve
            shared_state['bins_'] = bins_
            shared_state['feat_dim'] = train_t.shape[1]  # save feature dimension for later use

        return self

    def transform(self, N_data, C_data, y_data=None, context=None):
        # For demonstration, we do not directly transform numeric data here.
        # We just provided bins in the context. 
        return N_data, C_data, y_data


class PLETransform(BaseTransform):
    """
    A transform that reads 'bins_' from context and applies PiecewiseLinearEncoding.
    """
    def __init__(self, args):
        super().__init__()
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, context=None):
        # We might want to create the encoder once we know the bins.
        from model.lib.num_embeddings import PiecewiseLinearEncoding
        if context is None:
            context = {}
        bins_ = context.get('bins_')
        if bins_ is not None:
            self.encoder_ = PiecewiseLinearEncoding(bins_)
        return self

    def transform(self, N_data, C_data, y_data=None, context=None):
        import torch
        if self.encoder_ is None:
            # If we have no encoder, do nothing
            return N_data, C_data, y_data

        for partition in N_data.keys():
            arr_t = torch.from_numpy(N_data[partition])
            out_t = self.encoder_(arr_t)
            N_data[partition] = out_t.cpu().numpy()

        return N_data, C_data, y_data


class UnaryTransform(BaseTransform):
    """
    Unary encoding (Q_Unary, T_Unary). Also requires bins_ from context.
    """
    def __init__(self, args):
        super().__init__()
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if shared_state is None:
            shared_state = {}
        bins_ = shared_state.get('bins_')
        if bins_ is not None:
            self.encoder_ = UnaryEncoding(bins_)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not self.encoder_ or not N_data:
            return N_data, C_data, y_data

        for partition in N_data:
            arr_t = torch.from_numpy(N_data[partition])
            out_t = self.encoder_(arr_t)
            N_data[partition] = out_t.cpu().numpy()
        return N_data, C_data, y_data
    

class BinIndexTransform(BaseTransform):
    """
    Bins encoding (Q_bins, T_bins). Also needs bins_ from context.
    """
    def __init__(self, args):
        super().__init__()
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if shared_state is None:
            shared_state = {}
        bins_ = shared_state.get('bins_')
        if bins_ is not None:
            self.encoder_ = BinsEncoding(bins_)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not self.encoder_ or not N_data:
            return N_data, C_data, y_data

        for partition in N_data:
            arr_t = torch.from_numpy(N_data[partition])
            out_t = self.encoder_(arr_t)
            N_data[partition] = out_t.cpu().numpy()
        return N_data, C_data, y_data


class BinsTransform(BaseTransform):
    """
    Bins encoding (Q_bins, T_bins). Also needs bins_ from context.
    """
    def __init__(self, args):
        super().__init__()
        self.encoder_ = None
        self.bins_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if shared_state is None:
            shared_state = {}
        bins_ = shared_state.get('bins_')
        self.bins_ = bins_
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        import torch

        if not N_data:
            return N_data, C_data, y_data

        for partition in N_data:
            arr_t = torch.from_numpy(N_data[partition])

            bins_ = self.bins_
            if bins_ is None:
                N_data[partition] = arr_t.numpy()
                continue

            batch_size, num_features = arr_t.shape
            outputs = []

            for col_idx in range(num_features):
                edges = torch.tensor(bins_[col_idx], dtype=torch.float32, device=arr_t.device)
                midpoints = 0.5 * (edges[:-1] + edges[1:])

                col_values = arr_t[:, col_idx]
                bin_idx = torch.bucketize(col_values, edges, right=False) - 1
                bin_idx = bin_idx.clamp(0, len(midpoints) - 1)

                col_out = midpoints[bin_idx]

                outputs.append(col_out.unsqueeze(-1))

            out_t = torch.cat(outputs, dim=-1)
            N_data[partition] = out_t.cpu().numpy()

        return N_data, C_data, y_data


class JohnsonTransform(BaseTransform):

    """
    Johnson encoding (Q_Johnson, T_Johnson). Also needs bins_ from context.
    """
    def __init__(self, args):
        super().__init__()
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if shared_state is None:
            shared_state = {}
        bins_ = shared_state.get('bins_')
        if bins_ is not None:
            self.encoder_ = JohnsonEncoding(bins_)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not self.encoder_ or not N_data:
            return N_data, C_data, y_data

        for partition in N_data:
            arr_t = torch.from_numpy(N_data[partition])
            out_t = self.encoder_(arr_t)
            N_data[partition] = out_t.cpu().numpy()
        return N_data, C_data, y_data


class QuantileTransform(BaseTransform):
    """
    Apply a QuantileTransformer to each group of bin columns belonging to a single feature.

    Suppose the shape of numeric data is (N, feature_dim * bin_num).
    For each feature (each group of bin_num columns), we fit a separate QuantileTransformer.
    """

    def __init__(self, args):
        super().__init__()
        self.n_quantiles = args.get('n_quantiles', 1000)
        self.output_distribution = args.get('output_distribution', 'normal')  # 'uniform' or 'normal'
        self.random_state = args.get('random_state', 0)

        self.feature_dim = None
        self.bin_num = None

        self.transformers_ = []

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        """
        1. obtrain (n_samples, feature_dim * bin_num) matrix from N_data['train']
        2. extract bin_num columns for each feature in the matrix
        3. fit corresponding QuantileTransformer
        """
        from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

        if not N_data or 'train' not in N_data:
            return self

        train_array = N_data['train']  # numpy array
        if train_array.ndim != 2:
            raise ValueError("Expected a 2D array for the numeric data.")

        qt = _QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
            random_state=self.random_state
        )
        qt.fit(train_array)
        self.transformers_ = qt

        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        """
        Apply the fitted QuantileTransformers to the numeric data in N_data.
        """
        if not self.transformers_:
            return N_data, C_data, y_data

        for part in N_data:
            arr = N_data[part]
            if arr.ndim != 2:
                raise ValueError(f"N_data[{part}] must be a 2D array.")

            N_data[part] = self.transformers_.transform(arr)

        return N_data, C_data, y_data


# categorical encoding transforms
class OrdinalTransform(BaseTransform):
    """
    Replaces categorical values with integer codes. 
    Unknown values become a special code and then possibly replaced with a mode if needed.
    """
    def __init__(self, args):
        super().__init__()
        self.handle_unknown = args.get('handle_unknown', 'use_encoded_value')
        self.unknown_value = args.get('unknown_value', np.iinfo('int64').max - 3)
        self.dtype = args.get('dtype', 'int64')

        self.ord_encoder_ = None
        self.mode_values_ = None  # used for post-hoc unknown replacement if desired

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if C_data and 'train' in C_data:
            self.ord_encoder_ = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown=self.handle_unknown,
                unknown_value=self.unknown_value,
                dtype=self.dtype
            )
            self.ord_encoder_.fit(C_data['train'])

            # Optionally compute mode-values for test-time unknown replacement
            train_enc = self.ord_encoder_.transform(C_data['train'])
            self.mode_values_ = []
            for col_idx in range(train_enc.shape[1]):
                col_ints = train_enc[:, col_idx].astype(int)
                col_mode = np.argmax(np.bincount(col_ints[col_ints != self.unknown_value]))
                self.mode_values_.append(col_mode)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.ord_encoder_:
            return N_data, C_data, y_data

        for part in C_data:
            arr_enc = self.ord_encoder_.transform(C_data[part])
            # Replace unknown_value with per-column mode
            mask = (arr_enc == self.unknown_value)
            if mask.any() and self.mode_values_ is not None:
                for col_idx in range(arr_enc.shape[1]):
                    col_mask = mask[:, col_idx]
                    arr_enc[col_mask, col_idx] = self.mode_values_[col_idx]

            C_data[part] = arr_enc
        return N_data, C_data, y_data


class IndiceTransform(BaseTransform):
    """
    IndiceTransform maps categorical values to unique integer indices without implying any order.
    
    Example:
      Training data for one column: ['red', 'blue', 'red', 'green']
      Mapping will be: {'red': 0, 'blue': 1, 'green': 2}
      In transformation, if a new value like 'yellow' is encountered, it is set to -1.
    """
    def __init__(self, args):
        super().__init__()
        self.unknown_index = args.get('unknown_index', -1)
        self.mapping = {}  # key: column index, value: dict mapping category -> index

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if C_data and 'train' in C_data:
            train_data = C_data['train']
            # Ensure train_data is a 2D numpy array (n_samples, n_features)
            if len(train_data.shape) != 2:
                raise ValueError("C_data['train'] must be a 2D numpy array")
            n_features = train_data.shape[1]
            self.mapping = {}
            for col in range(n_features):
                unique_vals = []
                for val in train_data[:, col]:
                    if val not in unique_vals:
                        unique_vals.append(val)
                # Build mapping: category -> index (starting from 0)
                self.mapping[col] = {cat: i for i, cat in enumerate(unique_vals)}
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if C_data:
            for part in C_data:
                data = C_data[part]
                if len(data.shape) != 2:
                    raise ValueError("Each partition in C_data must be a 2D numpy array")
                n_samples, n_features = data.shape
                # Prepare an output array of the same shape, with integer type.
                transformed = np.empty((n_samples, n_features), dtype=np.int64)
                for col in range(n_features):
                    col_mapping = self.mapping.get(col, {})
                    for i in range(n_samples):
                        val = data[i, col]
                        transformed[i, col] = col_mapping.get(val, self.unknown_index)
                C_data[part] = transformed
        return N_data, C_data, y_data


class OneHotTransform(BaseTransform):
    """
    One-hot encode integer-coded categorical features (after OrdinalTransform).
    """
    def __init__(self, args):
        super().__init__()
        self.ohe_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if C_data and 'train' in C_data:
            self.ohe_ = sklearn.preprocessing.OneHotEncoder(
                handle_unknown='ignore',
                dtype='float64'
            )
            self.ohe_.fit(C_data['train'])
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.ohe_:
            return N_data, C_data, y_data

        for part in C_data:
            arr_enc = self.ohe_.transform(C_data[part])
            arr_enc = arr_enc.toarray()  # Convert sparse matrix to dense
            C_data[part] = arr_enc
        return N_data, C_data, y_data


class BinaryTransform(BaseTransform):
    """
    Binary encoding (similar to category_encoders.BinaryEncoder).
    """
    def __init__(self, args):
        super().__init__()
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if C_data and 'train' in C_data:
            self.encoder_ = category_encoders.BinaryEncoder(cols=None)
            # Convert numeric-coded categories to strings, if the encoder expects that.
            self.encoder_.fit(C_data['train'].astype(str))
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.encoder_:
            return N_data, C_data, y_data

        for part in C_data:
            arr = self.encoder_.transform(C_data[part].astype(str)).values
            C_data[part] = arr
        return N_data, C_data, y_data


class HashTransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.n_components = args.get('n_components', 8)
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if C_data and 'train' in C_data:
            self.encoder_ = category_encoders.HashingEncoder(n_components=self.n_components)
            self.encoder_.fit(C_data['train'].astype(str))
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.encoder_:
            return N_data, C_data, y_data

        for part in C_data:
            arr = self.encoder_.transform(C_data[part].astype(str)).values
            C_data[part] = arr
        return N_data, C_data, y_data


class LeaveOneOutTransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        # we typically need the target 'train' to do LOO
        if C_data and 'train' in C_data and y_data and 'train' in y_data:
            y_train = y_data['train']
            self.encoder_ = category_encoders.LeaveOneOutEncoder()
            self.encoder_.fit(C_data['train'].astype(str), y_train)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.encoder_:
            return N_data, C_data, y_data
        for part in C_data:
            arr = self.encoder_.transform(C_data[part].astype(str)).values
            C_data[part] = arr
        return N_data, C_data, y_data


class TargetTransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if C_data and 'train' in C_data and y_data and 'train' in y_data:
            y_train = y_data['train']
            self.encoder_ = category_encoders.TargetEncoder()
            self.encoder_.fit(C_data['train'].astype(str), y_train)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.encoder_:
            return N_data, C_data, y_data
        for part in C_data:
            arr = self.encoder_.transform(C_data[part].astype(str)).values
            C_data[part] = arr
        return N_data, C_data, y_data


class CatBoostTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.encoder_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if C_data and 'train' in C_data and y_data and 'train' in y_data:
            y_train = y_data['train']
            self.encoder_ = category_encoders.CatBoostEncoder()
            self.encoder_.fit(C_data['train'].astype(str), y_train)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.encoder_:
            return N_data, C_data, y_data
        for part in C_data:
            arr = self.encoder_.transform(C_data[part].astype(str)).values
            C_data[part] = arr
        return N_data, C_data, y_data 


class TargetRankingIndiceTransform(BaseTransform):
    
    """
    Indice transform that ranks categories based on their mean target values.
    """
    def __init__(self, args):
        super().__init__()
        self.unknown_index = args.get('unknown_index', -1)
        self.mapping_ = {}

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        """
        Have to use y_data to compute the mean of each category.
        """
        if C_data and 'train' in C_data and y_data and 'train' in y_data:
            X_train = C_data['train']  # shape = (n_samples, n_features)
            y_train = y_data['train']  # shape = (n_samples, )
            if len(X_train.shape) != 2:
                raise ValueError("C_data['train'] must be a 2D numpy array.")
            if len(y_train) != X_train.shape[0]:
                raise ValueError("X_train and y_train must have the same number of samples.")
            
            n_samples, n_features = X_train.shape
            self.mapping_ = {}

            for col_idx in range(n_features):
                cat_to_sum = {}
                cat_to_count = {}
                for row_idx in range(n_samples):
                    cat_val = X_train[row_idx, col_idx]
                    cat_to_sum[cat_val] = cat_to_sum.get(cat_val, 0.0) + y_train[row_idx]
                    cat_to_count[cat_val] = cat_to_count.get(cat_val, 0) + 1

                cat_means = []
                for cat_val, total_sum in cat_to_sum.items():
                    mean_val = total_sum / cat_to_count[cat_val]
                    cat_means.append((cat_val, mean_val))

                # Sort categories by their mean values
                cat_means.sort(key=lambda x: x[1])

                # Create a rank mapping
                rank_map = {}
                for rank, (cat_val, _) in enumerate(cat_means):
                    rank_map[cat_val] = rank

                self.mapping_[col_idx] = rank_map

        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.mapping_:
            return N_data, C_data, y_data

        for part_name, data in C_data.items():
            if len(data.shape) != 2:
                raise ValueError(f"C_data[{part_name}] must be a 2D numpy array")
            n_samples, n_features = data.shape

            transformed = np.empty((n_samples, n_features), dtype=np.int64)
            for col_idx in range(n_features):
                rank_map = self.mapping_.get(col_idx, {})
                for i in range(n_samples):
                    cat_val = data[i, col_idx]
                    transformed[i, col_idx] = rank_map.get(cat_val, self.unknown_index)

            C_data[part_name] = transformed
        
        return N_data, C_data, y_data


class RobustScaleTransform:
    def __init__(self, args):
        self.medians_ = None     # shape = (n_features,)
        self.scales_  = None     # shape = (n_features,)
    
    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        assert N_data and 'train' in N_data, "N_data['train'] is required for fitting."
        X = N_data['train']  # shape = (n_samples, n_features)
        q0   = np.min(X, axis=0)   # min
        q1_4 = np.quantile(X, 0.25, axis=0)
        q1_2 = np.quantile(X, 0.50, axis=0)  # median
        q3_4 = np.quantile(X, 0.75, axis=0)

        q1   = np.max(X, axis=0)   # max
        self.medians_ = q1_2
        
        scales = []
        for j in range(X.shape[1]):
            iqr = q3_4[j] - q1_4[j]   # inter-quartile range
            rng = q1[j] - q0[j]      # overall range

            if abs(iqr) > 1e-15:
                sj = 1.0 / iqr
            else:
                # iqr = 0
                if abs(rng) > 1e-15:
                    # min-max scaler
                    sj = 2.0 / rng
                else:
                    sj = 0.0
            scales.append(sj)
        
        self.scales_ = np.array(scales, dtype=float)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        for part_name, data in N_data.items():
            if len(data.shape) != 2:
                raise ValueError(f"N_data[{part_name}] must be a 2D numpy array")
            n_samples, n_features = data.shape
            assert n_features == len(self.medians_), \
                f"Data shape {data.shape} does not match fitted medians shape {self.medians_.shape}"
            assert n_features == len(self.scales_), \
                f"Data shape {data.shape} does not match fitted scales shape {self.scales_.shape}"
            transformed = np.empty((n_samples, n_features), dtype=np.float32)
            for j in range(n_features):
                transformed[:, j] = (data[:, j] - self.medians_[j]) * self.scales_[j]
            N_data[part_name] = transformed
        return N_data, C_data, y_data


class SmoothClipTransform:
    def __init__(self, args):
        pass

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        for part_name, data in N_data.items():
            if len(data.shape) != 2:
                raise ValueError(f"N_data[{part_name}] must be a 2D numpy array")
            n_samples, n_features = data.shape
            transformed = np.empty((n_samples, n_features), dtype=np.float32)
            for j in range(n_features):
                transformed[:, j] = self.smooth_clip(data[:, j])
            N_data[part_name] = transformed
        return N_data, C_data, y_data

    def smooth_clip(self, X):
        return X / np.sqrt(1.0 + (X / 3.0)**2)


##########################################################################
#                           Custom Transform                             #
##########################################################################
import numpy as np
from typing import Dict, List

class PLEUNITransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.n_bins = args.get("n_bins", 10)
        self.bin_edges_: list[np.ndarray] | None = None  # per‑feature edges

    def fit(self, N_data, C_data=None, y_data=None, shared_state=None):
        if not (N_data and "train" in N_data):
            return self

        X = N_data["train"]
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        n_samples, n_features = X.shape
        self.bin_edges_ = []

        if isinstance(self.n_bins, (list, tuple, np.ndarray)):
            if len(self.n_bins) != n_features:
                raise ValueError(
                    f"n_bins list length ({len(self.n_bins)}) "
                    f"≠ number of features ({n_features})"
                )

        for j in range(n_features):
            col = X[:, j]

            if isinstance(self.n_bins, (list, tuple, np.ndarray)):
                nb = int(max(1, self.n_bins[j]))
            else:
                nb = int(max(1, self.n_bins))

            nb = min(nb, n_samples - 1)

            edges = np.percentile(col, np.linspace(0, 100, nb + 1))
            edges[0], edges[-1] = col.min(), col.max()
            edges = np.unique(edges)
            if edges.size < 2:
                edges = np.array([0.0, 1.0], dtype=float)

            self.bin_edges_.append(edges.astype(np.float32))

        return self

    def transform(self, N_data, C_data=None, y_data=None, shared_state=None):
        if not self.bin_edges_:
            return N_data, C_data, y_data

        for part, X in N_data.items():
            n_samples, n_features = X.shape
            assert n_features == len(self.bin_edges_), (
                f"Feature mismatch: expected {len(self.bin_edges_)}, got {n_features}"
            )

            X_out = np.empty_like(X, dtype=np.float32)

            for j, edges in enumerate(self.bin_edges_):
                col = X[:, j]
                idx = np.searchsorted(edges, col, side="right") - 1
                idx = np.clip(idx, 0, edges.size - 2)

                denom = edges[idx + 1] - edges[idx]
                denom[denom == 0] = 1.0

                frac = (col - edges[idx]) / denom
                X_out[:, j] = (idx + frac) / (edges.size - 1)

            N_data[part] = X_out

        return N_data, C_data, y_data


# Inserted: AdaptiveBandwidthCdfTransform
class AdaptiveBandwidthCdfTransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.k = int(args.get("k", 5))
        self.n_ref = int(args.get("n_ref", 2048))
        self.min_sigma = float(args.get("min_sigma", 1e-6))
        self.random_state = int(args.get("random_state", 0))

        # Per-feature references: list of dicts with keys {'v': np.ndarray, 'sigma': np.ndarray}
        self.refs_ = []

    @staticmethod
    def _knn_sigma_sorted(xs: np.ndarray, k: int, min_sigma: float) -> np.ndarray:
        """
        Given a sorted 1D array xs, return per-point radius to the k-th neighbor
        (on either side), as a robust local bandwidth. Boundary points use the
        farthest available neighbor within k steps.
        """
        n = xs.size
        if n == 1:
            return np.array([1.0], dtype=float)
        idx = np.arange(n)
        left_idx  = np.maximum(idx - k, 0)
        right_idx = np.minimum(idx + k, n - 1)

        left_dist  = xs - xs[left_idx]
        right_dist = xs[right_idx] - xs
        radius = np.maximum(left_dist, right_dist)
        # Numerical floor
        radius = np.maximum(radius, min_sigma)
        return radius

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if not N_data or "train" not in N_data:
            return self

        X = N_data["train"]
        if X.ndim != 2:
            raise ValueError("Expected 2D numeric array for N_data['train'].")

        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        self.refs_ = []

        for j in range(d):
            col = X[:, j].astype(np.float64)
            # Handle degenerate/constant columns
            uniq = np.unique(col)
            if uniq.size <= 1:
                v_ref = np.array([uniq[0] if uniq.size == 1 else 0.0], dtype=np.float64)
                s_ref = np.array([1.0], dtype=np.float64)  # arbitrary nonzero
                self.refs_.append({"v": v_ref, "sigma": s_ref})
                continue

            # Sort and compute kNN bandwidths on the 1D axis
            order = np.argsort(col)
            xs = col[order]
            sig_all = self._knn_sigma_sorted(xs, self.k, self.min_sigma)
            sig_all = sig_all

            # Subsample reference pairs (value, sigma) to cap compute
            m = min(self.n_ref, xs.size)
            if m == xs.size:
                ref_idx = np.arange(xs.size)
            else:
                # Evenly spaced quantile subsampling to preserve coverage
                ref_idx = np.linspace(0, xs.size - 1, num=m).astype(int)

            v_ref = xs[ref_idx].astype(np.float64)
            s_ref = sig_all[ref_idx].astype(np.float64)

            self.refs_.append({"v": v_ref, "sigma": s_ref})

        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        # If not fitted or no numeric data, no-op
        if not self.refs_ or not N_data:
            return N_data, C_data, y_data

        for part, arr in N_data.items():
            if arr.ndim != 2:
                raise ValueError(f"N_data[{part}] must be a 2D array.")
            n, d = arr.shape
            if d != len(self.refs_):
                raise ValueError(f"Feature dimension mismatch: data has {d}, refs has {len(self.refs_)}.")

            cols_out = []
            for j in range(d):
                ref = self.refs_[j]
                v = ref["v"]      # (M,)
                s = ref["sigma"]  # (M,)
                x = arr[:, j].astype(np.float64)  # (n,)

                # Evaluate variable-bandwidth mixture CDF: mean_r Phi((x - v_r)/s_r)
                # Broadcasting: (n,1) - (1,M) divided by (1,M) -> (n,M)
                z = (x[:, None] - v[None, :]) / s[None, :]
                out = norm.cdf(z).mean(axis=1)  # (n,)

                cols_out.append(out.astype(np.float32)[:, None])

            N_data[part] = np.hstack(cols_out)

        return N_data, C_data, y_data


class OofSampleStretchTransform(BaseTransform):
    def __init__(self, args, is_regression: bool | None = None):
        super().__init__()
        self.oof_n_splits = int(args.get("oof_n_splits", 10))
        self.adaptive_k = int(args.get("k", 10))
        self.min_h = float(args.get("min_h", 1e-6))
        self.norm = str(args.get("norm", "l2")).lower()
        self.n_bins = int(args.get("n_bins", 1))
        self.min_unique = int(args.get("min_unique", 3))
        self.eps = float(args.get("eps", 1e-9))

        self.is_regression = bool(is_regression) if is_regression is not None else False
        self._feats_: list[dict] = []

    # ---------- helpers ----------
    @staticmethod
    def _silverman_bandwidth(x: np.ndarray) -> float:
        x = np.asarray(x, float).ravel()
        n = max(1, x.size)
        std = np.std(x) + 1e-12
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        a = min(std, iqr / 1.349) if iqr > 0 else std
        return 0.9 * a * n ** (-1/5)

    def _auto_k(self, n: int) -> int:
        """Adaptive k: ~2% of samples, clipped to [5, 200]."""
        return int(np.clip(round(0.02 * max(1, n)), 5, 200))

    def _local_bandwidth(self, x_tr_sorted: np.ndarray, x_ev: np.ndarray, k: int) -> np.ndarray:
        x_tr_sorted = np.asarray(x_tr_sorted, float).ravel()
        x_ev = np.asarray(x_ev, float).ravel()
        n = x_tr_sorted.size
        pos = np.searchsorted(x_tr_sorted, x_ev, side='left')
        left_idx = np.clip(pos - k, 0, n - 1)
        right_idx = np.clip(pos + k, 0, n - 1)
        left_dist = x_ev - x_tr_sorted[left_idx]
        right_dist = x_tr_sorted[right_idx] - x_ev
        h = np.maximum.reduce([left_dist, right_dist, np.full_like(left_dist, self.min_h)])
        return h

    def _nadaraya_watson_adaptive(self, x_tr: np.ndarray, Y_tr: np.ndarray, x_ev: np.ndarray, k: int) -> np.ndarray:
        order = np.argsort(x_tr, kind='mergesort')
        xs = x_tr[order]
        if Y_tr.ndim == 1:
            Ys = Y_tr[order][:, None]  # (n,1)
        else:
            Ys = Y_tr[order]           # (n,C)
        h = self._local_bandwidth(xs, x_ev, k)[:, None]  # (m,1)
        r = (x_ev[:, None] - xs[None, :]) / h            # (m,n)
        K = np.exp(-0.5 * r * r)
        W = np.maximum(K.sum(axis=1, keepdims=True), self.eps)
        num = K @ Ys
        out = num / W
        return out[:, 0] if out.shape[1] == 1 else out

    def _row_diffs(self, A: np.ndarray) -> np.ndarray:
        if A.ndim == 1:
            return np.abs(np.diff(A)).astype(float)
        D = np.diff(A, axis=0)
        if self.norm == 'l1':
            return np.abs(D).sum(axis=1).astype(float)
        return np.sqrt((D * D).sum(axis=1)).astype(float)

    # ---------- fit/transform ----------
    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if not (N_data and 'train' in N_data and y_data and 'train' in y_data):
            self._feats_ = []
            return self

        X = np.asarray(N_data['train'], float)
        ytr = np.asarray(y_data['train'])
        n, d = X.shape

        if not self.is_regression and ytr.ndim == 1:
            # Map labels (possibly non-consecutive) to [0..C-1]
            classes = np.unique(ytr)
            cls2idx = {c: i for i, c in enumerate(classes)}
            y_idx = np.vectorize(cls2idx.get)(ytr)
            C = len(classes)
            Yall = np.eye(C, dtype=float)[y_idx]
            y_for_strat = y_idx
            use_stratified = True
        elif not self.is_regression and ytr.ndim > 1:
            # Already one-hot / multi-target
            Yall = ytr.astype(float)
            y_for_strat = np.argmax(Yall, axis=1)
            use_stratified = True
        else:
            # Regression
            Yall = ytr.astype(float)
            y_for_strat = None
            use_stratified = False

        from sklearn.model_selection import KFold, StratifiedKFold
        if use_stratified:
            counts = np.bincount(y_for_strat.astype(int))
            max_splits = int(counts.min()) if counts.size else 2
            n_splits = int(min(max(2, self.oof_n_splits), max_splits))
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
            splits = list(kf.split(X, y_for_strat))
        else:
            n_splits = int(min(max(2, self.oof_n_splits), len(X)))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            splits = list(kf.split(X))

        self._feats_.clear()

        for j in range(d):
            xj = X[:, j]
            uniq = np.unique(xj)
            if uniq.size < self.min_unique or np.std(xj) == 0.0:
                self._feats_.append({
                    'identity': True,
                    'x_knots': np.array([float(xj.min()), float(xj.max())]),
                    't_knots': np.array([0.0, 1.0], float),
                })
                continue

            # OOF mbar(x) with either global or adaptive bandwidths
            mbar = np.zeros((n, Yall.shape[1]), dtype=float) if Yall.ndim == 2 else np.zeros(n, dtype=float)
            for tr_idx, ho_idx in splits:
                x_tr, x_ho = xj[tr_idx], xj[ho_idx]
                Y_tr = Yall[tr_idx]
                n_tr = int(tr_idx.size)
                k_use = max(int(self.adaptive_k), self._auto_k(n_tr))
                preds = self._nadaraya_watson_adaptive(x_tr, Y_tr, x_ho, k_use)
                mbar[ho_idx] = preds

            # Sort by x and bin on x directly
            ord_idx = np.argsort(xj, kind='mergesort')
            x_sorted = xj[ord_idx]
            m_sorted = mbar[ord_idx]

            B = max(1, int(self.n_bins))
            edges = np.quantile(x_sorted, np.linspace(0.0, 1.0, B + 1))
            edges[0], edges[-1] = float(x_sorted[0]), float(x_sorted[-1])
            edges = np.unique(edges)
            if edges.size < 2:
                self._feats_.append({
                    'identity': True,
                    'x_knots': np.array([float(x_sorted[0]), float(x_sorted[-1])]),
                    't_knots': np.array([0.0, 1.0], float),
                })
                continue

            # per-bin variation S_b over x
            S_list = []
            for b in range(edges.size - 1):
                l, r = edges[b], edges[b + 1]
                if b < edges.size - 2:
                    mask = (x_sorted >= l) & (x_sorted < r)
                else:
                    mask = (x_sorted >= l) & (x_sorted <= r)
                idxs = np.nonzero(mask)[0]
                if idxs.size < 2:
                    S_list.append(0.0)
                else:
                    S_list.append(self._row_diffs(m_sorted[idxs]).sum())
            S = np.asarray(S_list, dtype=float)

            # allocate bin lengths
            if S.sum() <= self.eps:
                t_knots = np.linspace(0.0, 1.0, edges.size, dtype=float)
            else:
                w = S / S.sum()
                t_knots = np.zeros(edges.size, dtype=float)
                t_knots[1:] = np.cumsum(w)
                t_knots[-1] = 1.0

            self._feats_.append({
                'identity': False,
                'x_knots': edges.astype(float),  # domain in raw x
                't_knots': t_knots.astype(float),
            })

        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not self._feats_ or not N_data:
            return N_data, C_data, y_data

        for part, X in N_data.items():
            X = np.asarray(X, float)
            n, d = X.shape
            X_out = X.copy()
            for j in range(d):
                st = self._feats_[j]
                if st['identity']:
                    continue
                xcol = X_out[:, j].astype(float)
                X_out[:, j] = np.interp(xcol, st['x_knots'], st['t_knots']).astype(np.float32)
            N_data[part] = X_out.astype(np.float32)
        return N_data, C_data, y_data