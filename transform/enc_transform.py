import math
import numpy as np
import sklearn.preprocessing
import category_encoders
from transform.base import BaseTransform
import torch
from model.lib.num_embeddings import (
    PiecewiseLinearEncoding, UnaryEncoding, BinsEncoding, JohnsonEncoding, _check_bins
)
# ----------- added for cache support -----------
import os, json, hashlib


class BinningTransform(BaseTransform):
    """
    A transform that computes bins for numeric data (Q or T) and stores them in the context.
    """
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
                bins_ = compute_bins(train_t, n_bins=self.n_bins, tree_kwargs=None,
                                     y=None, regression=None)
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
    
    
class CatQuantileTransform(BaseTransform):
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

        if not C_data or 'train' not in C_data:
            return self

        train_array = C_data['train']  # numpy array
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

        for part in C_data:
            arr = C_data[part]
            if arr.ndim != 2:
                raise ValueError(f"N_data[{part}] must be a 2D array.")

            C_data[part] = self.transformers_.transform(arr)
            N_data[part] = np.concatenate((N_data[part], C_data[part]), axis=-1)

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
     

from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm
import numpy as np, os, json, hashlib, pickle
from typing import Dict, List
from transform.base import BaseTransform   # 保持与原工程一致


class CdfTransform(BaseTransform):
    """
    Cluster‑CDF ('uniform') / DPGMM‑CDF ('gaussian') / 自适应 ('dynamic').

    Args (*args*)
    -------------
    cdf_type         : 'uniform' | 'gaussian' | 'dynamic'
    n_components     : int
    binning_method   : quantile | hdbscan | hierarchical   (uniform)
    linkage_method   : ward | single | average | complete
    min_cluster_size : int (hdbscan)
    min_sigma        : float
    weight_threshold : float
    cache_path       : str | None
    # dynamic‑mode extra
    dynamic_ks_thresh: float   – KS 阈值 (≤ 0.05 越严格)
    """

    # -------------------- init --------------------
    def __init__(self, args: Dict, dataset=None):
        super().__init__()
        # --- basic ---
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

    # ---------------- cache helpers ----------------
    def _cfg(self):  # dict used for hashing
        return {k: getattr(self, k) for k in self.config}

    def _hash(self):
        return hashlib.md5(json.dumps(self._cfg(), sort_keys=True).encode()).hexdigest()[:16]

    def _cache_file(self):
        if not self.cdf_path:
            return None
        return os.path.join(self.cdf_path, f"{self._hash()}.pkl")

    # ---------------- fit --------------------------
    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if not N_data or "train" not in N_data:
            return self
        shared_state = shared_state or {}

        # ---------- try cache ----------
        cfile = self._cache_file()
        # os.remove(cfile) if cfile and os.path.exists(cfile) else None
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


# -----------------------------------------------------------
# Pair‑wise Mutual Information transform
# -----------------------------------------------------------
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
class PairwiseMITransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.n_bins = args.get("n_bins", 100)

    # ---------- fitting: compute & store MI ----------
    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if shared_state is None:
            shared_state = {}
        # Require a training split to build statistics
        if (not N_data or "train" not in N_data) and (not C_data or "train" not in C_data):
            return self

        cols = []

        # ---- numeric columns (→ discretised) ----
        if N_data and "train" in N_data:
            X_num = N_data["train"]                           # (n, d_num)
            if X_num.ndim == 1:                               # edge case: single column
                X_num = X_num[:, None]
            kb = KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy="quantile"
            )
            X_disc = kb.fit_transform(X_num)
            cols.extend([X_disc[:, i].astype(int) for i in range(X_disc.shape[1])])

        # ---- categorical columns (already discrete) ----
        if C_data and "train" in C_data:
            X_cat = C_data["train"]                           # (n, d_cat)
            if X_cat.ndim == 1:
                X_cat = X_cat[:, None]
            cols.extend([X_cat[:, i] for i in range(X_cat.shape[1])])

        L = len(cols)
        if L == 0:                                           # no columns found
            return self

        import numpy as np
        mi_mat = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            for j in range(i + 1, L):
                val = mutual_info_score(cols[i], cols[j])
                mi_mat[i, j] = mi_mat[j, i] = val

        # min‑max scale to [0,1]  (avoid division by zero)
        vmax = mi_mat.max()
        if vmax > 0:
            mi_mat /= vmax

        shared_state["pairwise_mi"] = mi_mat
        return self

    # transform = identity (data unchanged)
    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        return N_data, C_data, y_data
