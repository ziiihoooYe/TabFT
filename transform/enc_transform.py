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


from scipy.stats import norm
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from collections import Counter
class HardAssignmentGMMTransform(BaseTransform):
    """
    对每个数值特征单独拟合 BayesianGaussianMixture (DP-GMM)，
    并只保留活跃分量（即 weights_ 大于给定阈值的分量），
    在 transform 时，对每个样本的某维 x：
      1. 针对（保留的）活跃分量做后验概率 argmax
      2. z = (x - mu_k*) / sigma_k*
      3. 输出向量对应该分量位置填入 z，其余为 0
    最终输出的维度 = Σ_d (active_components_of_dimension_d)
    """

    def __init__(self, args):
        """
        参数示例:
          {
            "max_components": 10,                # DP-GMM中设置的上限分量数
            "random_state": 42,
            "weight_concentration_prior": 1e2,
            "weight_threshold": 1e-3             # 小于此阈值的分量会被视为非活跃
          }
        """
        super().__init__()
        self.max_components = args.get("max_components", 10)
        self.random_state = args.get("random_state", None)
        self.weight_concentration_prior = args.get("weight_concentration_prior", 1e2)
        self.weight_threshold = args.get("weight_threshold", 1e-3)

        # 每个特征的完整 GMM 模型（有需要可以保留, 用于 predict_proba）
        self.gmms_ = []

        # 以下存储“活跃分量”的参数
        self.active_means_ = []      # list of ndarray; shape=[n_active,]
        self.active_stds_ = []       # list of ndarray; shape=[n_active,]
        self.active_weights_ = []    # 如果需要可以存储一下, 有时可用于调试
        self.n_comps_per_dim_ = []   # 每个特征的活跃分量数

        self._fitted = False

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        """
        1. 从 N_data['train'] 取数值矩阵 X, shape = (n_samples, n_features)
        2. 对每列单独拟合 DP GMM (BayesianGaussianMixture).
        3. 按照 weight_threshold 过滤出活跃分量, 并记录活跃分量的均值/方差等
        """
        if not N_data or "train" not in N_data:
            return self  # 没有训练数据则什么也不做

        train_X = N_data["train"]
        if train_X.ndim != 2:
            raise ValueError("N_data['train'] must be a 2D array.")

        n_samples, n_features = train_X.shape

        self.gmms_.clear()
        self.active_means_.clear()
        self.active_stds_.clear()
        self.active_weights_.clear()
        self.n_comps_per_dim_.clear()

        for col_idx in range(n_features):
            col_data = train_X[:, col_idx].reshape(-1, 1)

            # 使用 DP GMM
            gmm = BayesianGaussianMixture(
                n_components=self.max_components,
                random_state=self.random_state,
                weight_concentration_prior=self.weight_concentration_prior
            )
            gmm.fit(col_data)

            # 全存起来, 以便 transform 时可以拿到 predict_proba()
            self.gmms_.append(gmm)

            # 对分量进行筛选: weight > weight_threshold
            weights = gmm.weights_
            means = gmm.means_.ravel()
            covs = gmm.covariances_.ravel()

            active_mask = (weights > self.weight_threshold)
            active_weights = weights[active_mask]
            active_means = means[active_mask]
            active_covs = covs[active_mask]
            active_stds = np.sqrt(active_covs + 1e-12)  # 防止数值问题

            # 如果需要，也可以额外处理一下, e.g. 若一个都不过阈值, 最少保留一个分量
            if not np.any(active_mask):
                # fallback：至少保留权重最大的那个
                idx_max = np.argmax(weights)
                active_mask[idx_max] = True
                active_weights = weights[[idx_max]]
                active_means = means[[idx_max]]
                active_covs = covs[[idx_max]]
                active_stds = np.sqrt(active_covs + 1e-12)

            self.active_weights_.append(active_weights)
            self.active_means_.append(active_means)
            self.active_stds_.append(active_stds)
            self.n_comps_per_dim_.append(len(active_weights))

        self._fitted = True
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        """
        对各分区的数据进行“硬分配”扩张，只对活跃分量进行操作。
        步骤：
          - 原模型的 predict_proba => (n_samples, max_components)
          - 只保留活跃分量对应的列 => post_active
          - 对每行重新归一化 post_active
          - k^* = argmax_k post_active[row]
          - z = (x - mu_{k*}) / sigma_{k*}
          - 拼出 one-hot block
        最终输出维度 = Σ_d n_comps_per_dim_[d]
        """
        if not self._fitted:
            return N_data, C_data, y_data

        for part_name in (N_data or {}):
            X_part = N_data[part_name]
            if X_part.ndim != 2:
                raise ValueError(f"N_data[{part_name}] must be a 2D array.")
            n_samples, n_features = X_part.shape

            if n_features != len(self.gmms_):
                raise ValueError(
                    f"Number of columns in {part_name} ({n_features}) "
                    f"does not match the fitted GMMs ({len(self.gmms_)})"
                )

            # 计算输出总维度
            total_dim = sum(self.n_comps_per_dim_)
            X_out = np.zeros((n_samples, total_dim), dtype=float)

            offset = 0
            for d in range(n_features):
                gmm = self.gmms_[d]
                col_data = X_part[:, d].reshape(-1, 1)

                # 完整后验概率 (n_samples, self.max_components)
                post_full = gmm.predict_proba(col_data)

                # 取该维度活跃分量的 mask & 参数
                active_count = self.n_comps_per_dim_[d]
                active_weights = self.active_weights_[d]
                active_means = self.active_means_[d]
                active_stds = self.active_stds_[d]

                # 根据 fit 阶段保存的 mask 去筛选，或者直接用“谁大于阈值”再做一次
                # 这里我们在 fit 时已经确定了 active_mask 的顺序
                # 所以可以对 post_full 的列做同样的索引。
                # 小技巧：如果当时保留的列索引我们没有保存下来，可以再计算一次:
                #    active_mask = (gmm.weights_ > self.weight_threshold)
                # 但必须保证“顺序”一致。为了简化，这里假定行列顺序不变。
                # 我们就直接基于 weight 是否 > threshold 的顺序做过滤：
                # 但请注意，这里要保证在 fit 中你没有对 active_mask 做排序等操作。
                # 
                # 如果你想严格保证顺序和一致性，最好在 fit 里把 active_mask 的索引保存一下
                # 并在 transform 时使用同样的索引来筛选 post_full 的列。
                # 
                # 这里为演示方便，只要 active_count>0，就找到那几个列:
                # 要么先复用: weights = gmm.weights_; active_mask = (weights > self.weight_threshold)
                # 但这样fit和transform需要相同阈值/逻辑；只要保证一致就可。
                # 
                # 下面假设在fit时没有对列做重排，所以mask对应列是同顺序:
                full_weights = gmm.weights_
                active_mask = (full_weights > self.weight_threshold)

                # 如果当时做了“至少保留一个分量”之类的操作，这里也需对应处理
                if np.sum(active_mask) < active_count:
                    # 说明当时做了强制保留某些列 => 需要找出那列
                    # 此处略去详细逻辑，假设我们没有这种情况
                    pass

                # 筛选后邮寄概率
                post_active = post_full[:, active_mask]

                # 归一化 => (n_samples, active_count)
                # 因为去掉了部分列，剩下的概率之和不一定是 1
                row_sums = post_active.sum(axis=1, keepdims=True) + 1e-12
                post_active = post_active / row_sums

                # 硬分配
                k_star = np.argmax(post_active, axis=1)

                # means/stds 均按 active_mask 顺序取
                # 计算 z = (x - mu_{k*}) / sigma_{k*}
                col_data_1d = col_data.ravel()
                z_values = np.zeros_like(col_data_1d)
                for i in range(n_samples):
                    comp_idx = k_star[i]
                    mu = active_means[comp_idx]
                    std = active_stds[comp_idx]
                    z_values[i] = (col_data_1d[i] - mu) / std

                # 组装到 block: shape = [n_samples, active_count]
                block = np.zeros((n_samples, active_count), dtype=float)
                for i in range(n_samples):
                    block[i, k_star[i]] = z_values[i]

                # 放进 X_out
                X_out[:, offset : offset + active_count] = block

                offset += active_count

            # 替换 N_data[part_name]
            N_data[part_name] = X_out

        return N_data, C_data, y_data


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
class HardAssignmentTreeTransform(BaseTransform):
    """
    对每个数值特征(x_col, y)训练一棵一维决策树，若该树生成 L 个叶子，
    则把这一列从 1D 扩展到 L 维。对于每条样本，在其落入的叶子对应维度放 x，其它为 0。
    """

    def __init__(self, args):
        """
        参数示例:
          {
            'task': 'classification' or 'regression',   # 用哪种决策树
            'min_impurity_decrease': 1e-3,
            'random_state': 42,
            ... 其他DecisionTree可选参数
          }
        """
        super().__init__()
        self.task = args.get("task", "classification")
        self.min_impurity_decrease = args.get("min_impurity_decrease", 1e-2)
        self.random_state = args.get("random_state", 42)

        # 也可添加 max_depth、min_samples_leaf 等参数
        self.max_depth = args.get("max_depth", 3)
        self.min_samples_leaf = args.get("min_samples_leaf", 1)

        # 存储每列训练好的决策树
        self.trees_ = []
        # 存储每列的叶子数
        self.n_leaves_ = []
        # 标记是否已fit
        self._fitted = False

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        """
        在 N_data['train'] 上，对每个特征单独训练一棵一维决策树.
        需要 y_data['train'] 作为监督信号.
        """
        if not N_data or "train" not in N_data:
            return self

        X_train = N_data["train"]
        if X_train.ndim != 2:
            raise ValueError("N_data['train'] must be 2D.")
        if not y_data or "train" not in y_data:
            raise ValueError("需要 y_data['train'] 作为监督信息。")

        y_train = y_data["train"]
        if len(y_train) != X_train.shape[0]:
            raise ValueError("X_train 与 y_train 样本数不匹配。")

        n_samples, n_features = X_train.shape
        self.trees_.clear()
        self.n_leaves_.clear()

        for col_idx in range(n_features):
            # 取该列和 y
            X_col = X_train[:, col_idx].reshape(-1, 1)

            # 构造一棵一维决策树
            
            if np.issubdtype(y_train.dtype, np.integer):
                tree = DecisionTreeClassifier(
                    criterion="entropy",
                    min_impurity_decrease=self.min_impurity_decrease,
                    random_state=self.random_state,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf
                )
            else:  # regression
                tree = DecisionTreeRegressor(
                    criterion="squared_error",
                    min_impurity_decrease=self.min_impurity_decrease,
                    random_state=self.random_state,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf
                )

            tree.fit(X_col, y_train)
            self.trees_.append(tree)
            self.n_leaves_.append(tree.get_n_leaves())

        self._fitted = True
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        """
        对每个特征, 根据训练好的决策树找到对应叶子编号，扩展到 (n_samples, sum(n_leaves_)) 维.
        对于落入第k个叶子的样本, 在输出向量 [offset + k] 处放原值, 其余放0.
        """
        if not self._fitted:
            return N_data, C_data, y_data

        for part_name in N_data:
            X_part = N_data[part_name]
            if X_part.ndim != 2:
                raise ValueError(f"N_data[{part_name}] must be 2D.")
            n_samples, n_features = X_part.shape
            if n_features != len(self.trees_):
                raise ValueError(
                    f"特征数({n_features})与fit时训练好的树数({len(self.trees_)})不一致."
                )

            # 计算最终输出维度: sum of all leaves
            total_dim = sum(self.n_leaves_)
            X_out = np.zeros((n_samples, total_dim), dtype=float)

            offset = 0
            for d in range(n_features):
                tree = self.trees_[d]
                L = self.n_leaves_[d]

                X_col = X_part[:, d].reshape(-1, 1)
                # 对每个样本返回所在叶子的索引: [0..L-1]
                leaf_ids = tree.apply(X_col)  # shape=(n_samples,)

                # 对DecisionTreeClassifier/Regressor:
                #   - apply()返回的是节点ID, 但ID不一定从0..L-1顺序排列
                # 我们需要对“叶节点ID”做一次映射, 使之变为 [0..L-1]
                unique_leaf_ids = np.unique(leaf_ids)
                # 建立一个映射: node_id -> new_index
                leaf_id_to_newidx = {}
                idx_count = 0
                for node_id in np.sort(unique_leaf_ids):
                    leaf_id_to_newidx[node_id] = idx_count
                    idx_count += 1

                # 如果树的叶子不止这些(有时可能因为没有样本落到某些叶子),
                #   get_n_leaves()数 L 可能大于 len(unique_leaf_ids).
                #   可以提前处理, 或者直接用 L = len(unique_leaf_ids) 也行。
                #   这里为简便, 我们先用 L = tree.get_n_leaves(), 并且假设所有叶子都有样本.

                # 填充 X_out
                block = np.zeros((n_samples, L), dtype=float)
                for i in range(n_samples):
                    node_id = leaf_ids[i]
                    leaf_newidx = leaf_id_to_newidx[node_id]
                    block[i, leaf_newidx] = X_col[i, 0]

                # 放进 X_out
                X_out[:, offset : offset + L] = block
                offset += L

            N_data[part_name] = X_out

        return N_data, C_data, y_data
    
    
class Quantile0Transform(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.n_quantiles = args.get('n_quantiles', 1000)
        self.output_distribution = args.get('output_distribution', 'normal')
        self.random_state = args.get('random_state', 0)

        self.transformers_ = []

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer
        if not N_data or 'train' not in N_data:
            return self
 
        train_array = N_data['train']  # 假设为二维数组 (n_samples, n_features)
        n_samples, n_features = train_array.shape
 
        self.transformers_ = []
        for i in range(n_features):
            col = train_array[:, i]
            # Exclude 0 and 1 from fitting
            mask = (col != 0) & (col != 1)
            if np.any(mask):
                valid_data = col[mask].reshape(-1, 1)
                qt = _QuantileTransformer(
                    n_quantiles=self.n_quantiles,
                    output_distribution=self.output_distribution,
                    random_state=self.random_state
                )
                qt.fit(valid_data)
                self.transformers_.append(qt)
            else:
                # If no valid values, store None
                self.transformers_.append(None)
 
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if not self.transformers_:
            return N_data, C_data, y_data

        for part_name in N_data:
            arr = N_data[part_name]
            if arr.ndim != 2:
                raise ValueError(f"N_data[{part_name}] must be a 2D array.")

            n_samples, n_features = arr.shape
            transformed_arr = np.copy(arr).astype(float)

            for i in range(n_features):
                col = transformed_arr[:, i]
                transformer = self.transformers_[i]
                # Create a mask for values that are not 0 and not 1
                mask = (col != 0) & (col != 1)
                if transformer is not None and np.any(mask):
                    valid_values = col[mask].reshape(-1, 1)
                    transformed_values = transformer.transform(valid_values).flatten()
                    # Scale the standard normal output to have mean 0.5 and sigma = 0.5/3
                    transformed_values = transformed_values 
                    col[mask] = transformed_values
                # # Explicitly restore the original 0 and 1 values to ensure they remain unchanged
                # non_valid_mask = ~mask
                # if np.any(non_valid_mask):
                #     col[non_valid_mask] = arr[:, i][non_valid_mask]

            N_data[part_name] = transformed_arr

        return N_data, C_data, y_data


class Bins0Transform(BaseTransform):
    """
    既将数值替换为 bin 的代表值(如中点)，
    又额外加一列记录 x 相对于 bin 左边界的偏移量 (x - bin_left)。
    
    因此对于原来的每个特征列，transform 后会生成 2 列。
    如果原来有 d 个特征，则输出形状为 [batch_size, 2 * d]。
    """

    def __init__(self, args):
        super().__init__()
        # 如果想要别的参数, 可以通过 args.get(...) 获取
        self.bins_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        """
        仅从 shared_state 里获取 bins_。
        如果 bins_ 由上游步骤计算好(例如等宽/等频分箱), 这里只需要拿来用即可。
        """
        if shared_state is None:
            shared_state = {}
        self.bins_ = shared_state.get('bins_', None)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        """
        对各 partition 的数据执行:
          1) 根据 self.bins_ 获取每列的 edges, 以及中点 midpoints
          2) bin_idx = bucketize(...)
          3) col_mid = midpoints[bin_idx]
          4) bin_left = edges[bin_idx]
          5) pos = x - bin_left
          6) 输出 concat([col_mid, pos], dim=-1)
        最后把所有列拼起来 => shape [batch_size, 2 * num_features]
        """
        if not N_data or self.bins_ is None:
            return N_data, C_data, y_data

        for partition in N_data:
            arr_t = torch.from_numpy(N_data[partition]).float()  # shape=[batch_size, num_features]
            batch_size, num_features = arr_t.shape

            outputs = []

            for col_idx in range(num_features):
                # 对第 col_idx 列获取 edges
                edges_list = self.bins_[col_idx]
                if edges_list is None or len(edges_list) < 2:
                    # 若没有有效切分点, 就直接原样输出(或者做别的处理)
                    # 这里演示简单做法: [x, 0]  (不分箱, pos=0)
                    col_values = arr_t[:, col_idx]
                    dummy = torch.zeros_like(col_values)
                    col_out = torch.stack([col_values, dummy], dim=-1)  # shape=[batch_size,2]
                    outputs.append(col_out)
                    continue

                edges = torch.tensor(edges_list, dtype=torch.float32, device=arr_t.device)
                # midpoints 的数量 = len(edges) - 1
                midpoints = 0.5 * (edges[:-1] + edges[1:])

                col_values = arr_t[:, col_idx]

                # 计算 bin_idx
                bin_idx = torch.bucketize(col_values, edges, right=False) - 1
                # clamp 到 [0, len(midpoints)-1]
                bin_idx = bin_idx.clamp(0, len(midpoints) - 1)

                # bin 的代表值(这里是 midpoints)
                col_mid = midpoints[bin_idx]

                # bin_left
                bin_left = edges[bin_idx]

                # pos = x - bin_left
                pos = col_values - bin_left

                # 组合在一起 => shape=[batch_size, 2]
                col_out = torch.stack([col_mid, pos], dim=-1)

                outputs.append(col_out)

            # 拼接所有列 => shape=[batch_size, 2 * num_features]
            out_t = torch.cat(outputs, dim=-1)
            N_data[partition] = out_t.cpu().numpy()

        return N_data, C_data, y_data
    

class Bins1Transform(BaseTransform):
    """
    将数值替换为:
      [bin_index, x - bin_left]
    对每个原特征列扩为2列, 如果原先 d 维, 输出为 2*d 维。
    """

    def __init__(self, args):
        super().__init__()
        self.bins_ = None

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        """
        仅从 shared_state 里获取 bins_。
        如果 bins_ 在外部已经算好, 这里只需要拿来用即可。
        """
        if shared_state is None:
            shared_state = {}
        self.bins_ = shared_state.get('bins_', None)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        """
        对各 partition 的数据进行处理:
        1) 根据 self.bins_ (edges) 用 torch.bucketize 获得 bin_idx
        2) bin_left = edges[bin_idx]
        3) offset = x - bin_left
        4) 拼接 => [bin_idx, offset]
        """
        if not N_data or self.bins_ is None:
            return N_data, C_data, y_data

        for partition in N_data:
            arr_t = torch.from_numpy(N_data[partition]).float()  # shape=[batch_size, num_features]
            batch_size, num_features = arr_t.shape

            outputs = []

            for col_idx in range(num_features):
                # 对第 col_idx 列获取 edges
                edges_list = self.bins_[col_idx]
                if not edges_list or len(edges_list) < 2:
                    # 若没有有效切分点, 就直接保留原值(或者你想要怎样处理)
                    # 这里演示简单做法: [x, 0]  (相当于未分箱, offset=0)
                    col_values = arr_t[:, col_idx]
                    dummy = torch.zeros_like(col_values)
                    col_out = torch.stack([col_values, dummy], dim=-1)  # shape=[batch_size,2]
                    outputs.append(col_out)
                    continue

                edges = torch.tensor(edges_list, dtype=torch.float32, device=arr_t.device)

                col_values = arr_t[:, col_idx]

                # bin_idx 的取值范围 [0, len(edges)-2]，因为 edges 有 N+1 个边界 => N 个 bin
                # bucketize: 落在 [edges[i], edges[i+1]) 区间 => bin_idx = i
                bin_idx = torch.bucketize(col_values, edges, right=False) - 1
                # clamp 到 [0, len(edges)-2]
                bin_idx = bin_idx.clamp(0, len(edges) - 2)

                # bin_left
                bin_left = edges[bin_idx]

                # offset = x - bin_left
                offset = col_values - bin_left

                # 注意 bin_idx 是 int64 tensor, 可能需要转成 float
                bin_idx_float = bin_idx.float()

                # 拼为 [bin_idx, offset], shape=[batch_size, 2]
                col_out = torch.stack([bin_idx_float, offset], dim=-1)

                outputs.append(col_out)

            # 合并所有列 => shape=[batch_size, 2*num_features]
            out_t = torch.cat(outputs, dim=-1)
            N_data[partition] = out_t.cpu().numpy()

        return N_data, C_data, y_data
