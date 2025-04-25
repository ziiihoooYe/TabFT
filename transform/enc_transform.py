import numpy as np
import sklearn.preprocessing
import category_encoders
from transform.base import BaseTransform
import torch
from model.lib.num_embeddings import (
    PiecewiseLinearEncoding, UnaryEncoding, BinsEncoding, JohnsonEncoding, _check_bins
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
from transform.base import BaseTransform
class DPGMMCdfTransform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.n_components = args.get('n_components', 100)
        self.max_iter = args.get('max_iter', 500)
        self.random_state = args.get('random_state', 0)
        self.weight_concentration_prior = args.get('weight_concentration_prior', 0.01)
        self.weight_threshold = args.get('weight_threshold', 0)
        self.reg_covar = args.get('reg_covar', 0.001)
 
        self.cov_type = args.get('covariance_type', 'full') 
        self.bgmm_list_ = []  
        self.active_components_info_ = []  
        # mapping from original → generated columns
        self.feature_map_ = None

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
                reg_covar= max(self.reg_covar * np.var(col_values), 1e-8),
                weight_concentration_prior_type='dirichlet_process'
            )
            bgmm.fit(col_values)

            self.bgmm_list_.append(bgmm)

            # Identify active mixture components (weight > threshold, else pick the heaviest)
            weights = bgmm.weights_
            valid_idx = np.where(weights > self.weight_threshold)[0]
            if valid_idx.size == 0:
                valid_idx = np.array([np.argmax(weights)], dtype=int)

            means = bgmm.means_[valid_idx].ravel()

            if bgmm.covariances_.ndim == 2:
                vars_ = bgmm.covariances_[valid_idx, 0]
            elif bgmm.covariances_.ndim == 3:
                vars_ = bgmm.covariances_[valid_idx, 0, 0]
            else:
                vars_ = bgmm.covariances_[valid_idx]
            stds = np.sqrt(vars_).ravel()

            # ==== sort base on mean ====
            order = np.argsort(means)
            valid_idx     = valid_idx[order]
            means         = means[order]
            stds          = stds[order]
            weights_valid = weights[valid_idx]

            # ==== 存储 ====
            self.active_components_info_.append({
                "valid_comp_inds": valid_idx,
                "means":   means.astype(np.float64),
                "stds":    stds.astype(np.float64),
                "weights": weights_valid.astype(np.float64),
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

            # ---------- build feature_map once ----------
            if self.feature_map_ is None:
                mapping, offset = [], 0
                for j, info in enumerate(self.active_components_info_):
                    k_j = len(info['means'])
                    mapping.append({"orig_idx": j,
                                    "new_start": offset,
                                    "size": int(k_j)})
                    offset += k_j
                self.feature_map_ = mapping

            # concatenate all transformed columns
            new_part_data = np.hstack(transformed_cols)  # shape: (n_samples, sum_of_all_k)
            N_data[part_name] = new_part_data

        return N_data, C_data, y_data  
    

from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde, norm

class ClusterCdfTransform(BaseTransform):
    """Adaptive binning + selectable CDF encoding for numerical features."""

    # --------------------------- init ----------------------------------
    def __init__(self, args: Dict):
        super().__init__()

        # --- binning parameters ---
        self.binning_method: str   = args.get("binning_method", "hierarchical")
        self.linkage_method: str   = args.get("linkage_method", "ward")
        self.n_bins: int | None    = args.get("n_bins")
        self.distance_threshold: float | None = args.get("distance_threshold")
        self.elbow_gap_pct: float  = args.get("elbow_gap_pct", 1.0)
        self.inconsistency_threshold: float | None = args.get(
            "inconsistency_threshold", None
        )
        self.inconsistency_depth: int = args.get("inconsistency_depth", 2)
        self.kde_min_prom: float   = args.get("kde_min_prom", 0.0)   # minima threshold
        self.min_cluster_size: int = args.get("min_cluster_size", 20) # for HDBSCAN

        # --- encoding parameters ---
        self.encoding_type: str    = args.get("encoding_type", "uniform")  # uniform | gaussian
        self.min_sigma: float      = args.get("min_sigma", 1e-6)

        # --- learned state ---
        self.bin_edges_: List[np.ndarray]      = []  # per column edges (k+1,)
        self.bin_stats_: List[np.ndarray]      = []  # per column (k,2) mean/sigma for gaussian
        self.feature_map_: List[Dict]          = []

    # --------------------------- fit ----------------------------------
    def fit(self, N_data: Dict, C_data: Dict, y_data=None, shared_state=None):
        if not N_data or "train" not in N_data:
            return self

        X_train = N_data["train"]
        if X_train.ndim != 2:
            raise ValueError("N_data['train'] must be 2-D array (n_samples, n_features)")
        n_samples, n_features = X_train.shape

        self.bin_edges_.clear()
        self.bin_stats_.clear()
        self.feature_map_.clear()

        for col_idx in range(n_features):
            col_vals = X_train[:, col_idx]
            edges = self._find_bins_1d(col_vals)
            self.bin_edges_.append(edges)

            # stats for gaussian encoding
            stats = []
            for j in range(len(edges) - 1):
                mask = (col_vals >= edges[j]) & (col_vals <= edges[j + 1])
                if mask.any():
                    mu = col_vals[mask].mean()
                    sigma = col_vals[mask].std(ddof=0)
                    if sigma < self.min_sigma:
                        sigma = max(col_vals.std(ddof=0), self.min_sigma)
                else:
                    # empty bin – fallback to mid + global std
                    mu = 0.5 * (edges[j] + edges[j + 1])
                    sigma = max(col_vals.std(ddof=0), self.min_sigma)
                stats.append((mu, sigma))
            self.bin_stats_.append(np.asarray(stats, dtype=np.float64))
            self.feature_map_.append({
                "orig_idx": col_idx,
                "new_start": None,  # placeholder
                "size": len(edges) - 1,
            })

        # assign new_start offsets
        offset = 0
        for m in self.feature_map_:
            m["new_start"] = offset
            offset += m["size"]

        return self

    # ------------------------- transform ------------------------------
    def transform(self, N_data: Dict, C_data: Dict, y_data=None, shared_state=None):
        if not self.bin_edges_:
            return N_data, C_data, y_data

        for part_name, data_array in N_data.items():
            if data_array.ndim != 2:
                raise ValueError(f"N_data[{part_name}] must be 2-D array")
            n_samples, n_features = data_array.shape
            if n_features != len(self.bin_edges_):
                raise ValueError("Feature number mismatch between fit and transform")

            transformed_cols = []
            for col_idx in range(n_features):
                v = data_array[:, col_idx]
                edges = self.bin_edges_[col_idx]
                k = len(edges) - 1

                if self.encoding_type == "uniform":
                    denom = edges[1:] - edges[:-1]
                    denom[denom == 0] = 1.0
                    f = (v[:, None] - edges[:-1]) / denom
                    f = np.clip(f, 0.0, 1.0)
                elif self.encoding_type == "gaussian":
                    mu_sigma = self.bin_stats_[col_idx]  # (k,2)
                    mu = mu_sigma[:, 0]
                    sigma = mu_sigma[:, 1]
                    z = (v[:, None] - mu) / sigma
                    f = norm.cdf(z)
                    # clamp outside bin → 0 / 1 to keep monotone
                    f[v[:, None] < edges[:-1]] = 0.0
                    f[v[:, None] > edges[1:]] = 1.0
                else:
                    raise NotImplementedError(f"encoding_type={self.encoding_type} not supported")

                transformed_cols.append(f.astype(np.float32))

            N_data[part_name] = np.hstack(transformed_cols)

        return N_data, C_data, y_data

    # ----------------------- helpers ----------------------------------
    def _find_bins_1d(self, x: np.ndarray) -> np.ndarray:
        """Return monotonically increasing edges for 1-D array *x*."""
        x = x.ravel()
        if np.unique(x).size <= 2 or np.isclose(np.var(x), 0.0):
            return np.array([x.min(), x.max()], dtype=np.float64)

        match self.binning_method:
            case "hierarchical":
                return self._bins_hierarchical(x)
            case "kde":
                return self._bins_kde_minima(x)
            case "bic":
                return self._bins_bic_dp(x)
            case "hdbscan":
                return self._bins_hdbscan(x)
            case "manual":
                nb = self.n_bins or 5
                return np.percentile(x, np.linspace(0, 100, nb + 1))
            case _:
                raise NotImplementedError(f"binning_method={self.binning_method} not implemented")

    # ------------------ 1) hierarchical -------------------------------
    def _bins_hierarchical(self, x: np.ndarray) -> np.ndarray:
        Z = linkage(x[:, None], method=self.linkage_method)

        # 1-A 固定簇数 / 距离阈值
        if self.n_bins is not None:
            labels = fcluster(Z, t=self.n_bins, criterion="maxclust")

        elif self.distance_threshold is not None:
            labels = fcluster(Z, t=self.distance_threshold, criterion="distance")

        # 1-C Inconsistency
        elif self.inconsistency_threshold is not None:
            from scipy.cluster.hierarchy import inconsistent
            labels = fcluster(
                Z,
                t=self.inconsistency_threshold,
                criterion="inconsistent",
                depth=self.inconsistency_depth,
            )

        # 1-B 最大跳跃 (Elbow)
        else:
            d = Z[:, 2]
            lookback = min(30, len(d) - 1)
            gaps = np.diff(d[-lookback - 1 :])
            idx = np.argmax(gaps)
            tau = d[-lookback - 1 + idx] * self.elbow_gap_pct
            labels = fcluster(Z, t=tau, criterion="distance")

        # -------- labels → edges --------
        uniq  = np.unique(labels)
        mins  = np.array([x[labels == u].min() for u in uniq])
        maxs  = np.array([x[labels == u].max() for u in uniq])
        order = np.argsort(mins)
        mins  = mins[order]
        maxs  = maxs[order]

        edges = [x.min()] + [(mins[i+1] + maxs[i])/2 for i in range(len(maxs)-1)] + [x.max()]

        return np.asarray(edges, dtype=np.float64)


    # ------------------ 2) KDE minima ---------------------------------
    def _bins_kde_minima(self, x: np.ndarray) -> np.ndarray:
        kde = gaussian_kde(x)
        grid = np.linspace(x.min(), x.max(), max(200, x.size))
        dens = kde(grid)
        minima_idx = argrelextrema(dens, np.less)[0]
        if minima_idx.size == 0:
            return np.array([x.min(), x.max()])
        edges = [x.min()]
        for idx in minima_idx:
            if dens[idx] < dens.max() * (1 - self.kde_min_prom):
                edges.append(grid[idx])
        edges.append(x.max())
        return np.unique(np.asarray(edges, dtype=np.float64))

    # ------------------ 3) BIC dynamic‑programming --------------------
    def _bins_bic_dp(self, x: np.ndarray) -> np.ndarray:
        try:
            import ruptures as rpt
        except ModuleNotFoundError:
            # Fallback to hierarchical elbow
            return self._bins_hierarchical(x)
        x_sorted = np.sort(x)
        # `bkps` gives the indices (1‑based) where each segment *ends*; the last
        # element is always `len(x_sorted)`.  Use the midpoint between adjacent
        # segments as the internal bin boundaries.
        algo = rpt.Pelt(model="l2").fit(x_sorted)
        beta = 3 * np.log(x.size)  # ≈ BIC penalty
        bkps = algo.predict(pen=beta)
        inner_edges = [
            0.5 * (x_sorted[i - 1] + x_sorted[i]) for i in bkps[:-1]
        ]
        edges = [x.min(), *inner_edges, x.max()]
        edges = np.asarray(edges, dtype=np.float64)
        return edges

    # ------------------ 4) HDBSCAN density‑based ----------------------
    def _bins_hdbscan(self, x: np.ndarray) -> np.ndarray:
        try:
            import hdbscan
        except ModuleNotFoundError:
            # Fallback
            return self._bins_hierarchical(x)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        labels = clusterer.fit_predict(x[:, None])
        if np.all(labels == -1):
            return np.array([x.min(), x.max()])
        uniq = np.unique(labels[labels >= 0])
        mins = np.array([x[labels == u].min() for u in uniq])
        maxs = np.array([x[labels == u].max() for u in uniq])
        # order clusters by their minimum value
        order = np.argsort(mins)
        mins  = mins[order]
        maxs  = maxs[order]

        # Use the mid‑point between the current cluster's max and the next
        # cluster's min as the internal bin boundary.  This keeps edges
        # strictly increasing and better reflects the gap between clusters.
        inner_edges = [
            0.5 * (maxs[i] + mins[i + 1]) for i in range(len(maxs) - 1)
        ]
        edges = [x.min(), *inner_edges, x.max()]
        return np.asarray(edges, dtype=np.float64)
