import math
import numpy as np
import sklearn.preprocessing
import category_encoders
from transform.base import BaseTransform
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional

from torch.utils.data import DataLoader, TensorDataset
from model.lib.num_embeddings import (
    PiecewiseLinearEncoding, UnaryEncoding, BinsEncoding, JohnsonEncoding, _check_bins
)
# ----------- added for cache support -----------
import os, json, hashlib


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
import pickle
import json
class CdfTransform(BaseTransform):
    def __init__(self, args: Dict, dataset=None):
        super().__init__()
        self.cdf_type  = args.get("cdf_type", "uniform").lower()
        self.cdf_path  = args.get("cache_path", None)
        self.dataset   = dataset

        # uniform
        self.binning_method   = args.get("binning_method", "quantile")
        self.linkage_method   = args.get("linkage_method", "ward")
        self.min_cluster_size = args.get("min_cluster_size", 10)

        # shared
        self.n_components = args.get("n_components", 100)
        self.min_sigma    = args.get("min_sigma", 1e-6)

        # gaussian
        self.weight_threshold = float(args.get("weight_threshold", 1e-4))
        self.weight_concentration_prior_type = args.get(
            "weight_concentration_prior_type", "dirichlet_process"
        )
        self.reg_covar = args.get("reg_covar", 1e-6)

        # dynamic
        self.dynamic_ks_thresh      = args.get("dynamic_ks_thresh", 0.05)
        self.feature_types_: List[str] = []     # per‑column decision

        # misc
        self.concat_raw = args.get("concat_raw", False)

        # learned
        self.bin_edges_, self.bin_stats_, self.bin_weights_ = [], [], []
        self.feature_map_: List[Dict] = []

        # --- config used for cache key ---
        self.config = (
            "dataset", "cdf_type", "binning_method", "n_components",
            "linkage_method", "min_cluster_size", "min_sigma"
        ) if self.cdf_type == "uniform" else (
            "dataset", "cdf_type", "n_components",
            "weight_threshold", "weight_concentration_prior_type",
            "reg_covar"
        )
        if self.cdf_type == "dynamic":
            self.config += ("dynamic_ks_thresh",)


    def _cfg(self):  # dict used for hashing
        return {k: getattr(self, k) for k in self.config}

    def _hash(self):
        return hashlib.md5(json.dumps(self._cfg(), sort_keys=True).encode()).hexdigest()[:16]

    def _cache_file(self):
        if not self.cdf_path:
            return None
        return os.path.join(self.cdf_path, f"{self._hash()}.pkl")

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if not N_data or "train" not in N_data:
            return self
        shared_state = shared_state or {}
        if N_data['train'].shape[0] <= self.n_components:
            self.n_components = N_data['train'].shape[0] - 1

        cfile = self._cache_file()
        if cfile and os.path.exists(cfile):
            state = pickle.load(open(cfile, "rb"))
            if state.get("config") == self._cfg():
                if "feature_types_" in state:
                    self.feature_types_ = state["feature_types_"]
                self.__dict__.update({k: state[k] for k in (
                    "bin_edges_", "bin_stats_", "bin_weights_", "feature_map_"
                )})
                shared_state["feature_map_"] = self.feature_map_
                return self

        X = N_data["train"]
        n, d = X.shape
        self.bin_edges_, self.bin_stats_, self.bin_weights_, self.feature_map_ = [], [], [], []
        if self.cdf_type == "dynamic":
            self.feature_types_.clear()

        for j in range(d):
            col = X[:, j]

            # --------------- gaussian path ---------------
            if self.cdf_type == "gaussian":
                self._fit_gaussian(col)
            # --------------- uniform path ----------------
            elif self.cdf_type == "uniform":
                self._fit_uniform(col)
            # --------------- dynamic path ----------------
            elif self.cdf_type == "dynamic":
                self._fit_dynamic(col)
            else:
                raise ValueError(f"Unknown cdf_type: {self.cdf_type}")

            # feature map bookkeeping
            size = len(self.bin_weights_[-1])
            if self.concat_raw:
                size += 1
            self.feature_map_.append({"orig_idx": j, "new_start": None, "size": size})

        # assign start indices
        offset = 0
        for meta in self.feature_map_:
            meta["new_start"] = offset
            offset += meta["size"]
        shared_state["feature_map_"] = self.feature_map_

        # ---------- save cache ----------
        if cfile:
            os.makedirs(os.path.dirname(cfile), exist_ok=True)
            pickle.dump(
                {
                    "config": self._cfg(),
                    "bin_edges_": self.bin_edges_,
                    "bin_stats_": self.bin_stats_,
                    "bin_weights_": self.bin_weights_,
                    "feature_map_": self.feature_map_,
                    "feature_types_": self.feature_types_,
                },
                open(cfile, "wb"),
            )
        return self

    # ============ helper sub‑fit methods =============
    def _fit_gaussian(self, col):
        bgmm = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior_type=self.weight_concentration_prior_type,
            random_state=0,
            reg_covar=max(col.var() * 1e-6, 1e-8),
        ).fit(col.reshape(-1, 1))
        w, m, c = bgmm.weights_, bgmm.means_.ravel(), bgmm.covariances_
        idx = np.where(w > self.weight_threshold)[0]
        if not idx.size:
            idx = np.array([np.argmax(w)])
        w, m, c = w[idx], m[idx], c[idx]
        order = np.argsort(m)
        m, w, c = m[order], w[order], c[order]
        s = np.sqrt(np.atleast_1d(c).ravel())

        self.bin_edges_.append(m.astype(float))
        self.bin_stats_.append(np.vstack([m, s]).T.astype(float))
        self.bin_weights_.append(w.astype(float))

    def _fit_uniform(self, col):
        edges = self._sort_bins(self._find_bins_1d(col))
        # edges = self._adjust_edges_max_samples(edges, col)
        cnt, _ = np.histogram(col, bins=edges)
        w = cnt / cnt.sum() if cnt.sum() else np.full(len(cnt), 1 / len(cnt))
        stats = []
        for k in range(len(edges) - 1):
            mask = (col >= edges[k]) & (col <= edges[k + 1])
            mu = col[mask].mean() if mask.any() else 0.5 * (edges[k] + edges[k + 1])
            sg = col[mask].std(ddof=0) if mask.any() else self.min_sigma
            stats.append((mu, max(sg, self.min_sigma)))

        self.bin_edges_.append(edges.astype(float))
        self.bin_stats_.append(np.asarray(stats, float))
        self.bin_weights_.append(w.astype(float))

    def _fit_dynamic(self, col):
        # single BGMM fit
        bgmm = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior_type=self.weight_concentration_prior_type,
            random_state=0,
            reg_covar=max(col.var() * 1e-6, 1e-8),
        ).fit(col.reshape(-1, 1))

        w_all, m_all, c_all = bgmm.weights_, bgmm.means_.ravel(), bgmm.covariances_
        idx = np.where(w_all > self.weight_threshold)[0]
        if not idx.size:
            idx = np.array([np.argmax(w_all)])
        w, m, c = w_all[idx], m_all[idx], c_all[idx]
        order = np.argsort(m)
        w, m, c = w[order], m[order], c[order]
        s = np.sqrt(np.atleast_1d(c).ravel())

        # KS distance
        xs = np.sort(col)
        emp_cdf = np.arange(1, len(xs) + 1) / len(xs)
        mix_cdf = (norm.cdf((xs[:, None] - m) / s) * w).sum(axis=1)
        ks_D = np.max(np.abs(emp_cdf - mix_cdf))

        if ks_D <= self.dynamic_ks_thresh:
            self.bin_edges_.append(m.astype(float))
            self.bin_stats_.append(np.vstack([m, s]).T.astype(float))
            self.bin_weights_.append(w.astype(float))
            self.feature_types_.append("gaussian")
        else:
            self._fit_uniform(col)
            self.feature_types_.append("uniform")

    # ---------------- transform ----------------------
    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not self.feature_map_:
            return N_data, C_data, y_data

        for part, arr in N_data.items():
            cols_out = []
            for j, meta in enumerate(self.feature_map_):
                v = arr[:, meta["orig_idx"]]

                is_gauss = (
                    self.cdf_type == "gaussian" or
                    (self.cdf_type == "dynamic" and self.feature_types_[j] == "gaussian")
                )
                if is_gauss:
                    m = self.bin_stats_[j][:, 0]
                    s = self.bin_stats_[j][:, 1]
                    w = self.bin_weights_[j]
                    z = (v[:, None] - m) / s
                    out = norm.cdf(z)
                else:
                    edges = self.bin_edges_[j]
                    denom = edges[1:] - edges[:-1]
                    denom[denom == 0] = 1
                    out = np.clip((v[:, None] - edges[:-1]) / denom, 0, 1)

                if self.concat_raw:
                    out = np.concatenate([out, v[:, None]], axis=1)
                cols_out.append(out.astype(np.float32))

            N_data[part] = np.hstack(cols_out)
        return N_data, C_data, y_data

    # ========== helper for uniform binning ==========
    def _find_bins_1d(self, x):
        x = x.ravel()
        if np.unique(x).size <= 2 or np.var(x) == 0:
            return np.array([x.min(), x.max()])
        if self.binning_method == "quantile":
            nb = self.n_components or 5
            return np.percentile(x, np.linspace(0, 100, nb + 1))
        if self.binning_method=="hdbscan":
            try:
                import hdbscan
                lab=hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size).fit_predict(x[:,None])
                if np.all(lab==-1): raise ModuleNotFoundError
                uniq=np.unique(lab[lab>=0])
                mins=np.array([x[lab==u].min() for u in uniq])
                maxs=np.array([x[lab==u].max() for u in uniq])
                order=np.argsort(mins); mins,maxs=mins[order],maxs[order]
                inner=[0.5*(maxs[i]+mins[i+1]) for i in range(len(maxs)-1)]
                return np.array([x.min(),*inner,x.max()])
            except ModuleNotFoundError: self.binning_method="hierarchical"
        # hierarchical fallback
        from scipy.cluster.hierarchy import linkage,fcluster
        Z=linkage(x[:,None],method=self.linkage_method)
        labels=fcluster(Z,t=self.n_components,criterion="maxclust")
        uniq=np.unique(labels)
        mins=np.array([x[labels==u].min() for u in uniq])
        maxs=np.array([x[labels==u].max() for u in uniq])
        order=np.argsort(mins); mins,maxs=mins[order],maxs[order]
        inner=[0.5*(maxs[i]+mins[i+1]) for i in range(len(maxs)-1)]
        return np.array([x.min(),*inner,x.max()])
    
    def _adjust_edges_max_samples(self, edges, col):
        if self.binning_method == "quantile":
            return edges
        # ---------------- STEP-0 : installation -----------------
        col = col.ravel()
        edges = np.sort(np.unique(edges.astype(float)))
        if edges.size < 2:
            v = float(col[0]) if edges.size else float(np.mean(col))
            edges = np.array([v - 1e-6, v + 1e-6], dtype=float)

        quota  = int(math.ceil(len(col) / self.n_components) * 2)
        min_sz = int(math.ceil(len(col) / self.n_components))

        # ---------------- STEP-1 : split ----------------
        extra_edges = []
        for k in range(len(edges) - 1):
            l, r = edges[k], edges[k + 1]
            mask = (col >= l) & (col <= r)
            cnt  = int(mask.sum())
            if cnt <= quota:
                continue

            xs_bin = col[mask]
            uniq_vals = np.unique(xs_bin)
            ucnt = uniq_vals.size

            if ucnt == 1:
                continue

            if ucnt == 2:
                extra_edges.append(0.5 * (uniq_vals[0] + uniq_vals[1]))
                continue

            nb = int(math.ceil(cnt / quota))
            if cnt / nb < min_sz:
                nb = max(1, cnt // min_sz)
            nb = min(nb, ucnt - 1)
            if nb <= 1:
                continue

            sub_edges = np.percentile(xs_bin,
                                      np.linspace(0, 100, nb + 1))[1:-1]
            extra_edges.extend(sub_edges)

        if extra_edges:
            edges = np.sort(np.unique(np.concatenate([edges, extra_edges])))

        # ---------------- STEP-2 : beyond cap → merge ----------------
        counts, _ = np.histogram(col, bins=edges)
        while (len(edges) - 1) > self.n_components or counts.min() < min_sz:
            counts, _ = np.histogram(col, bins=edges)
            idx_merge = int(np.argmin(counts[:-1] + counts[1:]))  # 最小相邻和
            edges = np.delete(edges, idx_merge + 1)

        return edges.astype(float)
    
    @staticmethod
    def _sort_bins(edges):
        edges = np.sort(edges)
        return edges


class UniPiecewiseCDFTransform(BaseTransform):
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


class SlopeEqualizeStretchTransform(BaseTransform):
    def __init__(self, args, is_regression: bool | None = None):
        super().__init__()
        self.norm       = args.get("norm", "l1").lower()
        self.eps        = float(args.get("eps", 1e-12))

        self.lambda_ = float(args.get("lambda_", 1.0))
        self.lambda_ = max(0.0, min(self.lambda_, 1.0))

        self.is_regression = is_regression

        self.n_bins = args.get("n_bins", 1)

        self.maps_: list[tuple[np.ndarray, np.ndarray]] = []   # (xs_unique, f_vals)

    # ---------- helper ----------
    def _vec_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.norm == "l2":
            return float(np.linalg.norm(a - b))
        # default L1
        return float(np.abs(a - b).sum())

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if not (N_data and "train" in N_data and y_data and "train" in y_data):
            return self

        X = N_data["train"]
        y = y_data["train"].ravel()
        uniq = np.unique(y)

        if self.is_regression is True:
            mode = "regression"
        elif self.is_regression is False:
            mode = "binary" if uniq.size == 2 else "multiclass"
        else:
            if np.issubdtype(y.dtype, np.floating) and uniq.size > 2:
                mode = "regression"
            elif uniq.size == 2:
                mode = "binary"
            else:
                mode = "multiclass"

        if mode == "multiclass":
            C = int(uniq.max()) + 1

        self.maps_.clear()
        n_samples, n_features = X.shape

        for j in range(n_features):
            x_col = X[:, j]
            xs_valid, y_valid = x_col, y
            uniq_vals = np.unique(xs_valid)
            uniq_cnt  = uniq_vals.size
            n_bins = min(self.n_bins, uniq_cnt - 1)

            if n_bins <= 1:
                x_min, x_max = xs_valid.min(), xs_valid.max()
                if x_max - x_min < self.eps:
                    self.maps_.append((np.array([x_min], dtype=float),
                                       np.array([0.0], dtype=float)))
                else:
                    self.maps_.append((np.array([x_min, x_max], dtype=float),
                                       np.array([0.0, 1.0], dtype=float)))
                continue

            if 1 < n_bins < uniq_cnt - 1:
                edges = np.percentile(xs_valid, np.linspace(0, 100, n_bins + 1))
                edges[0], edges[-1] = xs_valid.min(), xs_valid.max()
                edges = np.unique(edges)

                xs_u = uniq_vals
                inv  = np.searchsorted(xs_u, xs_valid)

                # --- expected target value per unique x ---
                if mode == "multiclass":
                    gs = np.zeros((len(xs_u), C), dtype=float)
                    for idx_u in range(len(xs_u)):
                        labs = y_valid[inv == idx_u].astype(int)
                        if labs.size:
                            cnts = np.bincount(labs, minlength=C)
                            gs[idx_u] = cnts / cnts.sum()
                else:
                    sums = np.bincount(inv, weights=y_valid, minlength=len(xs_u))
                    cnts = np.bincount(inv,              minlength=len(xs_u)).astype(float)
                    gs   = sums / np.maximum(cnts, 1.0)

                if len(xs_u) == 1:
                    self.maps_.append((xs_u.astype(float),
                                       np.zeros_like(xs_u, dtype=float)))
                    continue

                if mode == "multiclass":
                    neigh = np.array([self._vec_dist(gs[k + 1], gs[k])
                                      for k in range(len(xs_u) - 1)], dtype=float)
                else:
                    neigh = np.abs(np.diff(gs)).astype(float)

                interval_bins = np.searchsorted(edges, xs_u[:-1], side='right') - 1
                num_bins = edges.size - 1
                S = np.zeros(num_bins)
                for b in range(num_bins):
                    mask = interval_bins == b
                    if mask.any():
                        S[b] = neigh[mask].sum()

                total_S = float(S.sum())
                if total_S < self.eps:
                    span = xs_u[-1] - xs_u[0]
                    f_vals = np.zeros_like(xs_u, dtype=float) if span < self.eps \
                                else (xs_u - xs_u[0]) / span
                    self.maps_.append((xs_u.astype(float), f_vals.astype(float)))
                    continue

                alpha   = S / total_S
                s_vals  = np.zeros(edges.size, dtype=float)
                for b in range(num_bins):
                    s_vals[b + 1] = s_vals[b] + alpha[b]

                linear = np.linspace(0.0, 1.0, s_vals.size, dtype=float)
                s_vals = (1.0 - self.lambda_) * linear + self.lambda_ * s_vals

                self.maps_.append((edges.astype(float), s_vals.astype(float)))
                continue

            xs_u, inv = np.unique(xs_valid, return_inverse=True)

            if mode == "multiclass":
                gs = np.zeros((len(xs_u), C), dtype=float)
                for idx_u in range(len(xs_u)):
                    labs = y_valid[inv == idx_u].astype(int)
                    if labs.size:
                        cnts = np.bincount(labs, minlength=C)
                        gs[idx_u] = cnts / cnts.sum()
            else:
                sums = np.bincount(inv, weights=y_valid, minlength=len(xs_u))
                cnts = np.bincount(inv, minlength=len(xs_u)).astype(float)
                gs = sums / np.maximum(cnts, 1.0)

            if len(xs_u) == 1:
                f_base = np.zeros_like(xs_u, dtype=float)
            else:
                span = xs_u[-1] - xs_u[0]
                f_base = np.zeros_like(xs_u, dtype=float) if span < self.eps \
                            else (xs_u - xs_u[0]) / span

            if len(xs_u) == 1:
                TV = 0.0
                f_star = np.zeros_like(xs_u, dtype=float)
            else:
                if mode == "multiclass":
                    dists = np.array(
                        [self._vec_dist(gs[k + 1], gs[k]) for k in range(len(xs_u) - 1)],
                        dtype=float
                    )
                else:
                    dists = np.abs(np.diff(gs)).astype(float)

                TV = dists.sum()
                if TV < self.eps:
                    f_star = np.linspace(0.0, 1.0, len(xs_u), dtype=float)
                else:
                    gaps = dists / TV
                    f_star = np.zeros(len(xs_u), dtype=float)
                    f_star[1:] = np.cumsum(gaps)
                    f_star[-1] = 1.0

            f_vals = (1.0 - self.lambda_) * f_base + self.lambda_ * f_star

            self.maps_.append((xs_u.astype(float), f_vals.astype(float)))

        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not self.maps_:
            return N_data, C_data, y_data

        for part, X in N_data.items():
            if X is None:
                continue
            X_out = X.astype(np.float32, copy=True)
            for j, mapping in enumerate(self.maps_):
                if mapping is None:
                    continue
                xs_u, f_vals = mapping
                col = X_out[:, j]
                X_out[:, j] = np.interp(col, xs_u, f_vals)

            N_data[part] = X_out

        return N_data, C_data, y_data


