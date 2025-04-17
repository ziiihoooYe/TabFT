import math
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None

def _check_bins(bins: List[Tensor]) -> None:
    if not bins:
        raise ValueError('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        if not isinstance(feature_bins, Tensor):
            raise ValueError(
                'bins must be a list of PyTorch tensors. '
                f'However, for {i=}: {type(bins[i])=}'
            )
        if feature_bins.ndim != 1:
            raise ValueError(
                'Each item of the bin list must have exactly one dimension.'
                f' However, for {i=}: {bins[i].ndim=}'
            )
        if len(feature_bins) < 2:
            raise ValueError(
                'All features must have at least two bin edges.'
                f' However, for {i=}: {len(bins[i])=}'
            )
        if not feature_bins.isfinite().all():
            raise ValueError(
                'Bin edges must not contain nan/inf/-inf.'
                f' However, this is not true for the {i}-th feature'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                'Bin edges must be sorted.'
                f' However, the for the {i}-th feature, the bin edges are not sorted'
            )
        if len(feature_bins) == 2:
            warnings.warn(
                f'The {i}-th feature has just two bin edges, which means only one bin.'
                ' Strictly speaking, using a single bin for the'
                ' piecewise-linear encoding should not break anything,'
                ' but it is the same as using sklearn.preprocessing.MinMaxScaler'
            )


def compute_bins(
    X: torch.Tensor,
    n_bins: int = 48,
    *,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    y: Optional[torch.Tensor] = None,
    regression: Optional[bool] = None,
    verbose: bool = False,
) -> List[torch.Tensor]:

    # Check input types and dimensions
    if not isinstance(X, torch.Tensor):
        raise ValueError(f'X must be a PyTorch tensor, however: {type(X)=}')
    if X.ndim != 2:
        raise ValueError(f'X must have exactly two dimensions, however: {X.ndim=}')
    if X.shape[0] < 2:
        raise ValueError(f'X must have at least two rows, however: {X.shape[0]=}')
    if X.shape[1] < 1:
        raise ValueError(f'X must have at least one column, however: {X.shape[1]=}')
    if not X.isfinite().all():
        raise ValueError('X must not contain nan/inf/-inf.')
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(f"Number of bins must be greater than 1 and less than the number of samples in X. However, n_bins={n_bins} and len(X)={len(X)}.")

    constant_cols = (X == X[0]).all(dim=0)

    bins = []
    eps = 1e-5 # for interval of constant cols

    if tree_kwargs is not None:
        if sklearn_tree is None:
            raise RuntimeError(
                'The scikit-learn package is missing. ...'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if y.ndim != 1:
            raise ValueError(f'y must have exactly one dimension, however: {y.ndim=}')
        if len(y) != len(X):
            raise ValueError(
                f'len(y) must be equal to len(X), however: {len(y)=}, {len(X)=}'
            )

    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]

        # Determine distinct edges for the column
        if constant_cols[col_idx]:
            # Constant column: only one distinct value
            c = col_data[0].item()
            distinct_edges = torch.tensor([c], device=X.device, dtype=X.dtype)
        else:
            if tree_kwargs is None:
                # ----------- Q-Binning -----------
                q = torch.quantile(
                    col_data, torch.linspace(0.0, 1.0, n_bins + 1, device=X.device), dim=0
                )
                distinct_edges = q.unique()
            else:
                # ---------- T-Binning ------------
                from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

                column_np = col_data.cpu().numpy()
                y_np = y.cpu().numpy()

                TreeModel = DecisionTreeRegressor if regression else DecisionTreeClassifier

                tree_model = TreeModel(max_leaf_nodes=n_bins, **tree_kwargs)
                tree_model.fit(column_np.reshape(-1, 1), y_np)
                tree_ = tree_model.tree_

                edges_list = [float(column_np.min()), float(column_np.max())]
                for node_id in range(tree_.node_count):
                    if tree_.children_left[node_id] != tree_.children_right[node_id]:
                        edges_list.append(float(tree_.threshold[node_id]))

                distinct_edges = torch.as_tensor(edges_list, device=X.device, dtype=X.dtype).unique(sorted=True)

        if distinct_edges.numel() == 1:
            # For constant column, ensure at least two edges
            c = distinct_edges[0].item()
            edges = torch.tensor([c, c + eps], device=X.device, dtype=X.dtype)
        else:
            # Use the distinct edges as-is, without filling to n_bins + 1
            edges = distinct_edges

        bins.append(edges)

    _check_bins(bins)
    return bins
    

class _PiecewiseLinearEncodingImpl(nn.Module):
    # NOTE
    # 1. DO NOT USE THIS CLASS DIRECTLY (ITS OUTPUT CONTAINS INFINITE VALUES).
    # 2. This implementation is not memory efficient for cases when there are many
    #    features with low number of bins and only few features
    #    with high number of bins. If this becomes a problem,
    #    just split features into groups and encode the groups separately.

    # The output of this module has the shape (*batch_dims, n_features, max_n_bins),
    # where max_n_bins = max(map(len, bins)) - 1.
    # If the i-th feature has the number of bins less than max_n_bins,
    # then its piecewise-linear representation is padded with inf as follows:
    # [x_1, x_2, ..., x_k, inf, ..., inf]
    # where:
    #            x_1 <= 1.0
    #     0.0 <= x_i <= 1.0 (for i in range(2, k))
    #     0.0 <= x_k
    #     k == len(bins[i]) - 1  (the number of bins for the i-th feature)

    # If all features have the same number of bins, then there are no infinite values.

    edges: Tensor
    width: Tensor
    mask: Tensor
    # Source: https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
    def __init__(self, bins: List[Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        self.register_buffer('width', edges.diff())
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # See Equation 1 in the paper.
        x = (x[..., None] - self.edges) / self.width

        # If the number of bins is greater than 1, then, the following rules must
        # be applied to a piecewise-linear encoding of a single feature:
        # - the leftmost value can be negative, but not greater than 1.0.
        # - the rightmost value can be greater than 1.0, but not negative.
        # - the intermediate values must stay within [0.0, 1.0].
        n_bins = x.shape[-1]
        if n_bins > 1:
            if self._same_bin_count:
                x = torch.cat(
                    [
                        x[..., :1].clamp_max(1.0),
                        *([] if n_bins == 2 else [x[..., 1:-1].clamp(0.0, 1.0)]),
                        x[..., -1:].clamp_min(0.0),
                    ],
                    dim=-1,
                )
            else:
                # In this case, the rightmost values for all features are located
                # in different columns.
                x = torch.stack(
                    [
                        x[..., i, :]
                        if count == 1
                        else torch.cat(
                            [
                                x[..., i, :1].clamp_max(1.0),
                                *(
                                    []
                                    if n_bins == 2
                                    else [x[..., i, 1 : count - 1].clamp(0.0, 1.0)]
                                ),
                                x[..., i, count - 1 : count].clamp_min(0.0),
                                x[..., i, count:],
                            ],
                            dim=-1,
                        )
                        for i, count in enumerate(self._bin_counts)
                    ],
                    dim=-2,
                )
        return x


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise-linear encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.
    """
    # Source: https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        x = self.impl(x)
        return x.flatten(-2) if self.impl._same_bin_count else x[:, self.impl.mask]
    

class _UnaryEncodingImpl(nn.Module):
    edges: Tensor
    mask: Tensor

    def __init__(self, bins: List[Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # Compute which bin each value falls into
        x = (x[..., None] - self.edges).sign().cumsum(dim=-1)

        # Ensure values are within [0, 1] range for unary encoding
        x = x.clamp(0, 1)

        return x


class UnaryEncoding(nn.Module):
    """Unary encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.
    """

    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _UnaryEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        x = self.impl(x)
        return x.flatten(-2) if self.impl._same_bin_count else x[:, self.impl.mask]


class _JohnsonEncodingImpl(nn.Module):
    edges: Tensor
    mask: Tensor

    def __init__(self, bins: List[Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        self.register_buffer('width', edges.diff())
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # Compute which bin each value falls into
        bin_indices = torch.stack([torch.bucketize(x[..., i], self.edges[i], right=True) - 1 for i in range(x.shape[-1])], dim=-1)

        # Generate Johnson code for each bin index
        max_bin = self.edges.shape[1] 
        code_length = (max_bin + 1) // 2
        johnson_code = torch.zeros(*x.shape, code_length, device=x.device, dtype=torch.float32)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                johnson_code[i, j, :] = self.temp_code(bin_indices[i, j].item(), max_bin)

        return johnson_code

    def temp_code(self, num, num_bits):
        num_bits = num_bits+1 if num_bits%2!=0 else num_bits
        bits = num_bits // 2
        a = torch.zeros([bits], dtype=torch.long)
        for i in range(bits):
            if bits - i - 1 < num <= num_bits - i - 1:
                a[i] = 1
        return a


class JohnsonEncoding(nn.Module):
    """Johnson encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bits)``,
      where ``total_n_bits`` is the total number of bits for all features:
      ``total_n_bits = sum((len(b) - 1) // 2 for b in bins)``.
    """

    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _JohnsonEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        x = self.impl(x)
        return x.flatten(-2) # if self.impl._same_bin_count else x[:, self.impl.mask]
    

class _BinsEncodingImpl(nn.Module):
    edges: Tensor
    mask: Tensor

    def __init__(self, bins: List[Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # Compute which bin each value falls into
        bin_indices = torch.stack([torch.bucketize(x[..., i], self.edges[i], right=True) - 1 for i in range(x.shape[-1])], dim=-1)

        return bin_indices


class BinsEncoding(nn.Module): 
    """
    Bins encoding.
    **Shape**
    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features.
    """
    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _BinsEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        return self.impl(x)