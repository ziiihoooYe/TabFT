import numpy as np
from transform.base import BaseTransform
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, MaxAbsScaler

class NormalizationTransform(BaseTransform):
    """
    Apply a specified normalization policy to numeric data
    (e.g. 'standard', 'minmax', 'robust', 'power', 'quantile', 'maxabs')
    """

    def __init__(self, args, seed=42):
        super().__init__()
        self.policy = args['policy']
        self.seed = seed
        self.scaler = None
        # --- group‑wise normalisation (when feature_map_ exists) ---
        self.group_info_  = None   # list of (start, end) for each original feature
        self.group_scalers_ = None   # list of sklearn scalers per original feature

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if N_data is None or 'train' not in N_data:
            return self

        if self.policy == 'none':
            return self

        train_data = N_data['train']
        fmap = shared_state.get('feature_map_') if shared_state else None
        # fmap = None
        # --- group‑wise normalisation for any policy except 'none' ---
        if fmap is not None and self.policy != 'none':
            # Build group slice indices once
            self.group_info_ = [(m['new_start'],
                                 m['new_start'] + m['size']) for m in fmap]
            self.group_scalers_ = []
            for s, e in self.group_info_:
                seg = train_data[:, s:e]
                # choose scaler per policy
                match self.policy:
                    case 'standard':
                        sc = StandardScaler()
                    case 'minmax':
                        sc = MinMaxScaler()
                    case 'robust':
                        sc = RobustScaler()
                    case 'power':
                        sc = PowerTransformer(method='yeo-johnson')
                    case 'quantile':
                        sc = QuantileTransformer(
                            output_distribution='normal',
                            n_quantiles=max(min(seg.shape[0] // 30, 1000), 10),
                            random_state=self.seed
                        )
                    case 'maxabs':
                        sc = MaxAbsScaler()
                    case _:
                        raise ValueError(f"Unknown normalization policy: {self.policy}")
                try:
                    sc.fit(seg)
                except ValueError as e:
                    raise ValueError(f"Error fitting scaler on segment {s}:{e}")
                self.group_scalers_.append(sc)
            self.scaler = None
            return self

        if self.policy == 'standard':
            self.scaler = StandardScaler()
        elif self.policy == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.policy == 'robust':
            self.scaler = RobustScaler()
        elif self.policy == 'power':
            self.scaler = PowerTransformer(method='yeo-johnson')
        elif self.policy == 'quantile':
            self.scaler = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(train_data.shape[0] // 30, 1000), 10),
                random_state=self.seed
            )
        elif self.policy == 'maxabs':
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError(f"Unknown normalization policy: {self.policy}")

        self.scaler.fit(train_data)
        return self

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        if N_data is None or self.policy == 'none':
            return N_data, C_data, y_data

        # ----- group‑wise normalisation -----
        if self.group_scalers_ is not None and self.group_info_ is not None:
            for part in N_data.keys():
                X = N_data[part]
                X_new = X.copy().astype(np.float32)
                for idx, (s, e) in enumerate(self.group_info_):
                    scaler = self.group_scalers_[idx]
                    X_new[:, s:e] = scaler.transform(X_new[:, s:e])
                N_data[part] = X_new
            return N_data, C_data, y_data

        if self.scaler is None:
            return N_data, C_data, y_data

        for part in N_data.keys():
            N_data[part] = self.scaler.transform(N_data[part])
        return N_data, C_data, y_data