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

    def fit(self, N_data, C_data, y_data=None, shared_state=None):
        if N_data is None or 'train' not in N_data:
            return self

        if self.policy == 'none':
            return self

        train_data = N_data['train']
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
        if N_data is None or self.policy == 'none' or self.scaler is None:
            return N_data, C_data, y_data

        for part in N_data.keys():
            N_data[part] = self.scaler.transform(N_data[part])
        return N_data, C_data, y_data