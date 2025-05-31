import numpy as np
from transform.base import BaseTransform
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, MaxAbsScaler

class NormalizationTransform(BaseTransform):
    def __init__(self, args, seed: int = 42):
        super().__init__()
        self.policy: str = args["policy"].lower()
        self.seed = seed
        self.scaler = None               # global scaler (coordinate‑wise)

    # ------------------------------------------------------------------
    #                                F I T
    # ------------------------------------------------------------------
    def fit(self, N_data, C_data=None, y_data=None, shared_state=None):
        if N_data is None or "train" not in N_data:
            return self

        if self.policy == "none":
            return self

        train_data = N_data["train"]

        # If no fmap: fall back to *global* coordinate‑wise scaler
        if self.policy == "standard":
            self.scaler = StandardScaler()
        elif self.policy == "minmax":
            self.scaler = MinMaxScaler()
        elif self.policy == "robust":
            self.scaler = RobustScaler()
        elif self.policy == "power":
            self.scaler = PowerTransformer(method="yeo-johnson")
        elif self.policy == "quantile":
            self.scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(min(train_data.shape[0] // 30, 1000), 10),
                random_state=self.seed,
            )
        elif self.policy == "uniform":
            self.scaler = QuantileTransformer(
                output_distribution="uniform",
                n_quantiles=max(min(train_data.shape[0] // 30, 1000), 10),
                random_state=self.seed,
            )
        elif self.policy == "maxabs":
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError(f"Unknown normalisation policy: {self.policy}")

        self.scaler.fit(train_data)
        return self


    def transform(self, N_data, C_data=None, y_data=None, shared_state=None):
        if N_data is None or self.policy == "none":
            return N_data, C_data, y_data

        # ------------------- Global coordinate scaler -----------------
        if self.scaler is not None:
            for part in N_data.keys():
                N_data[part] = self.scaler.transform(N_data[part])
        return N_data, C_data, y_data