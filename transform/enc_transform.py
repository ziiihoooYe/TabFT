import numpy as np
import sklearn.preprocessing
import category_encoders
from transform.base import BaseTransform
import torch
from model.lib.num_embeddings import (
    PiecewiseLinearEncoding, UnaryEncoding, BinsEncoding, JohnsonEncoding
)


# numeric encoding transforms

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
                sparse=False,  # new param name in recent sklearn is sparse_output=False
                dtype='float64'
            )
            self.ohe_.fit(C_data['train'])
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not C_data or not self.ohe_:
            return N_data, C_data, y_data

        for part in C_data:
            arr_enc = self.ohe_.transform(C_data[part])
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
from transform.base import BaseTransform
class DPGMMCdfTransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.n_components = args.get('n_components', 10)
        self.max_iter = args.get('max_iter', 500)
        self.random_state = args.get('random_state', 0)
        self.weight_concentration_prior = args.get('weight_concentration_prior', 0.1)
        self.weight_threshold = args.get('weight_threshold', 0.05)

        self.cov_type = args.get('covariance_type', 'full') 
        self.bgmm_list_ = []  
        self.active_components_info_ = []  

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if not N_data or 'train' not in N_data:
            return self

        X_train = N_data['train']  # shape = (n_samples, n_features)
        n_samples, n_features = X_train.shape

        self.bgmm_list_ = []
        self.active_components_info_ = []

        for col_idx in range(n_features):
            col_values = X_train[:, col_idx].reshape(-1, 1)
            bgmm = BayesianGaussianMixture(
                n_components=self.n_components,
                max_iter=self.max_iter,
                random_state=self.random_state,
                covariance_type=self.cov_type,
                weight_concentration_prior_type='dirichlet_process'
            )
            bgmm.fit(col_values)

            self.bgmm_list_.append(bgmm)

            # Identify active mixture components (weight > threshold, else pick the heaviest)
            weights = bgmm.weights_
            valid_comp_inds = np.where(weights > self.weight_threshold)[0]
            if len(valid_comp_inds) == 0:
                valid_comp_inds = np.array([np.argmax(weights)], dtype=int)

            # Gather means / variances of the active comps
            means = bgmm.means_.ravel()[valid_comp_inds]
            if bgmm.covariances_.ndim == 2:
                vars_ = bgmm.covariances_[valid_comp_inds, 0]
            elif bgmm.covariances_.ndim == 3:
                vars_ = bgmm.covariances_[valid_comp_inds, 0, 0]
            else:
                vars_ = bgmm.covariances_[valid_comp_inds]
            stds = np.sqrt(vars_)

            # ---- sort components by mean (small → large) ----
            order = np.argsort(means)
            valid_comp_inds = valid_comp_inds[order]
            means = means[order]
            stds = stds[order]

            # Save
            self.active_components_info_.append({
                'valid_comp_inds': valid_comp_inds,
                'means': means,
                'stds': stds,
            })

        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not self.bgmm_list_:
            return N_data, C_data, y_data

        for part_name, data_array in N_data.items():
            if data_array.ndim != 2:
                raise ValueError(f"N_data[{part_name}] has to be  2D array.")

            n_samples, n_features = data_array.shape
            if n_features != len(self.bgmm_list_):
                raise ValueError("feature number mismatch between fit and transform.")

            transformed_cols = []

            for col_idx in range(n_features):
                col_values = data_array[:, col_idx]
                info = self.active_components_info_[col_idx]
                means = info['means']
                stds = info['stds']

                k_i = len(means)
                out_col = np.empty((n_samples, k_i), dtype=np.float32)

                for comp_idx in range(k_i):
                    m = means[comp_idx]
                    s = stds[comp_idx]
                    if s < 1e-14:
                        out_col[:, comp_idx] = 0.5
                    else:
                        z = (col_values - m) / s
                        out_col[:, comp_idx] = norm.cdf(z)

                transformed_cols.append(out_col)

            # concatenate all transformed columns
            new_part_data = np.hstack(transformed_cols)  # shape: (n_samples, sum_of_all_k)
            N_data[part_name] = new_part_data

        return N_data, C_data, y_data  
    

class DPGMMCdfSkipDiscreteTransform(DPGMMCdfTransform):
    def __init__(self, args):
        super().__init__(args)
        self.min_unique_values = args.get('min_unique_values', 10)
        self.skip_mask_ = []          # True → skip

    # ------------------------------------------------------------------ #
    # override fit
    # ------------------------------------------------------------------ #
    def fit(self, N_data, C_data=None, y_data=None, shared_state=None):
        if not N_data or 'train' not in N_data:
            return self

        X_train = N_data['train']          # shape = (n_samples, n_features)
        n_samples, n_features = X_train.shape

        self.bgmm_list_, self.active_components_info_, self.skip_mask_ = [], [], []

        for col_idx in range(n_features):
            col_vals = X_train[:, col_idx]
            if np.unique(col_vals).size < self.min_unique_values:
                self.bgmm_list_.append(None)
                self.active_components_info_.append(None)
                self.skip_mask_.append(True)
                continue

            bgmm = BayesianGaussianMixture(
                n_components=self.n_components,
                max_iter=self.max_iter,
                random_state=self.random_state,
                covariance_type=self.cov_type,
                weight_concentration_prior_type='dirichlet_process'
            ).fit(col_vals.reshape(-1, 1))
            self.bgmm_list_.append(bgmm)
            self.skip_mask_.append(False)

            w = bgmm.weights_
            valid = np.where(w > self.weight_threshold)[0]
            if valid.size == 0:
                valid = [w.argmax()]

            means = bgmm.means_.ravel()[valid]
            if bgmm.covariances_.ndim == 2:
                vars_ = bgmm.covariances_[valid, 0]
            elif bgmm.covariances_.ndim == 3:
                vars_ = bgmm.covariances_[valid, 0, 0]
            else:
                vars_ = bgmm.covariances_[valid]
            stds = np.sqrt(vars_)

            self.active_components_info_.append(
                dict(valid_comp_inds=valid, means=means, stds=stds)
            )

        return self

    # ------------------------------------------------------------------ #
    # override transform
    # ------------------------------------------------------------------ #
    def transform(self, N_data, C_data=None, y_data=None, shared_state=None):
        if not self.bgmm_list_:          # 尚未 fit
            return N_data, C_data, y_data

        for part, X in N_data.items():
            if X.ndim != 2:
                raise ValueError(f"N_data[{part}] has to be 2D array.")
            if X.shape[1] != len(self.bgmm_list_):
                raise ValueError("feature number mismatch between fit and transform.")

            out_cols = []
            for j, col_vals in enumerate(X.T):
                if self.skip_mask_[j]:
                    out_cols.append(col_vals.reshape(-1, 1))
                    continue

                info = self.active_components_info_[j]
                means, stds = info['means'], info['stds']
                k = len(means)
                col_out = np.empty((len(col_vals), k), np.float32)
                for k_idx in range(k):
                    m, s = means[k_idx], stds[k_idx]
                    if s < 1e-14:
                        col_out[:, k_idx] = 0.5
                    else:
                        z = (col_vals - m) / s
                        col_out[:, k_idx] = norm.cdf(z)
                out_cols.append(col_out)

            N_data[part] = np.hstack(out_cols)

        return N_data, C_data, y_data