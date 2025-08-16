# Modified Tab.py

import torch
import math
import typing as ty
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.init as nn_init
from torch import Tensor
import torch.nn.functional as F

#################################################################
###                        Model                              ###
#################################################################
class Tab(nn.Module):
    def __init__(
        self,
        config: dict,
        # Output args
        d_out: int,
        # Tokenizer args
        num_continuous: int,
        categories: ty.Optional[ty.List[int]],
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff_factor: float = 4.0,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        activation: str = "relu",
        prenormalization: bool = False,
        initialization: str = "xavier",
        **kwargs
    ):
        for key, value in config.items():
            setattr(self, key, value)

        super().__init__()
        
        self.tokenizer = Tokenizer(
            num_continuous=num_continuous,
            categories=categories,
            d_token=self.d_token,
            bias=self.token_bias,
            num_encoding=getattr(self, 'num_encoding', None),
            x_num_train=kwargs.get('x_num_train', None)
        )

        self.encoder = Encoder(
            n_layers=self.n_layers,
            d_model=self.d_token,
            n_heads=self.n_heads,
            d_ffn_factor=self.d_ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            activation=self.activation,
            prenormalization=self.prenormalization,
            initialization=self.initialization
        )
        
        self.head = Head(
            d_model=self.d_token,
            d_out=d_out,
            activation=self.activation,
            prenormalization=self.prenormalization
        )

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        # 1. Tokenization
        x = self.tokenizer(x_num, x_cat)
        
        # 2. Encoding
        x = self.encoder(x)
        
        # 3. Head
        output = self.head(x)
        
        return output


#############################################################
###                      Tokenization                     ###
#############################################################
class Tokenizer(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
        num_encoding: ty.Optional[dict] = None,
        x_num_train: ty.Optional[Tensor] = None
    ) -> None:
        super().__init__()
        self.n_num_features = num_continuous
        self.d_token = d_token

        # --- CLS and Numerical Token weights ---
        # Create a separate parameter for CLS tokens
        self.cls_tokens = nn.Parameter(torch.empty(1, d_token))
        nn_init.kaiming_uniform_(self.cls_tokens, a=math.sqrt(5))

        # ================= Numeric Encoder (RAW / PLE / SPLINE) =================
        self.num_enc_cfg = (num_encoding or {"method": "raw"}).copy()
        self.num_enc_cfg.setdefault("method", "raw")
        self.num_enc_cfg.setdefault("k", 10)             # number of basis per feature for RAW/PLE
        self.num_enc_cfg.setdefault("n_knots", 10)       # internal knots for spline (excluding end repeats)
        self.num_enc_cfg.setdefault("degree", 3)
        self.num_enc_cfg.setdefault("quantile", True)   # use quantiles for knots/bin edges
        self.num_enc_cfg.setdefault("eps", 1e-6)

        # Helper to move numpy -> torch and ensure float dtype
        def _to_tensor(x):
            if x is None:
                return None
            if isinstance(x, Tensor):
                return x.detach()
            import numpy as _np
            if isinstance(x, _np.ndarray):
                import torch as _torch
                return _torch.from_numpy(x)
            return x

        x_train_num = _to_tensor(x_num_train)
        if x_train_num is not None:
            x_train_num = x_train_num.float()
        # ---- Raw x stats for numeric reconstruction loss normalization ----
        if self.n_num_features > 0 and x_train_num is not None:
            x_mu = x_train_num.mean(dim=0)  # (n_num,)
            x_sd = x_train_num.std(dim=0, unbiased=False)
            x_sd = x_sd.clamp_min(self.num_enc_cfg["eps"])  # numerical safety
            self.register_buffer("_x_raw_mean", x_mu)
            self.register_buffer("_x_raw_std", x_sd)
        else:
            self.register_buffer("_x_raw_mean", torch.zeros(self.n_num_features))
            self.register_buffer("_x_raw_std", torch.ones(self.n_num_features))

        # Per-feature basis parameters and normalization stats
        # We store means/stds AFTER encoding (basis space), used for z-score at runtime.
        # (No pre-assignment of _phi_mean/_phi_std/_ple_edges/_spl_knots; buffers are registered below)
        self._spl_degree = int(self.num_enc_cfg.get("degree", 3))

        method = (self.num_enc_cfg.get("method") or "raw").lower()
        if self.n_num_features == 0:
            # Edge case: no numeric features
            self.num_basis = 0
        elif method == "raw":
            self.num_basis = 1
            if x_train_num is not None:
                # Compute statistics in the *encoded* space: phi = x.unsqueeze(-1)
                phi_train = x_train_num.unsqueeze(-1)           # (B, n_num, 1)
                mu = phi_train.mean(dim=0)                      # (n_num, 1)
                sd = phi_train.std(dim=0, unbiased=False)       # (n_num, 1)
                sd = sd.clamp_min(self.num_enc_cfg["eps"])     # numerical safety
                self.register_buffer("_phi_mean", mu)
                self.register_buffer("_phi_std", sd)
            else:
                # Fallback zeros/ones
                self.register_buffer("_phi_mean", torch.zeros(self.n_num_features, 1))
                self.register_buffer("_phi_std", torch.ones(self.n_num_features, 1))

        elif method == "ple":
            # Piecewise Linear Encoding with k knot points per feature (linear hat over adjacent knots)
            k = int(self.num_enc_cfg.get("k", 8))
            self.num_basis = k
            if x_train_num is None:
                raise ValueError("PLE encoding requires x_num_train to compute bin edges.")
            # Compute per-feature knot points by quantiles or linspace
            qs = torch.linspace(0.0, 1.0, steps=k, device=x_train_num.device)
            edges = torch.quantile(x_train_num, qs, dim=0).T  # (n_num, k)
            self.register_buffer("_ple_edges", edges)
            # Compute basis on train to get mean/std
            phi_train = self._ple_encode(x_train_num, edges)    # (B, n_num, k)
            mu = phi_train.mean(dim=0)                          # (n_num,k)
            sd = phi_train.std(dim=0, unbiased=False).clamp_min(self.num_enc_cfg["eps"])  # (n_num,k)
            self.register_buffer("_phi_mean", mu)
            self.register_buffer("_phi_std", sd)

        elif method == "spline":
            # Cubic B-spline basis (degree p) with clamped knots. `n_knots` internal knots per feature.
            p = int(self.num_enc_cfg.get("degree", 3))
            n_int = int(self.num_enc_cfg.get("n_knots", 10))
            if x_train_num is None:
                raise ValueError("Spline encoding requires x_num_train to compute knots.")
            # Internal knots: quantiles or linspace
            qs = torch.linspace(0.0, 1.0, steps=n_int + 2, device=x_train_num.device)[1:-1]  # exclude 0 and 1
            internal = torch.quantile(x_train_num, qs, dim=0).T   # (n_num, n_int)
            # Build clamped knot vectors per feature: [xmin repeated p+1] + internal + [xmax repeated p+1]
            xmin, xmax = x_train_num.min(dim=0).values, x_train_num.max(dim=0).values
            left = xmin.unsqueeze(1).repeat(1, p + 1)
            right = xmax.unsqueeze(1).repeat(1, p + 1)
            knots = torch.cat([left, internal, right], dim=1)  # (n_num, n_int + 2p + 2)
            self.register_buffer("_spl_knots", knots)
            self._spl_degree = p
            # Number of basis functions m = n_total_knots - p - 1
            self.num_basis = knots.shape[1] - p - 1
            # Compute basis on train for zscore
            phi_train = self._bspline_encode(x_train_num, knots, p)  # (B, n_num, m)
            mu = phi_train.mean(dim=0)
            sd = phi_train.std(dim=0, unbiased=False)
            sd = sd.clamp_min(self.num_enc_cfg["eps"])  # numerical safety aligned across encoders
            self.register_buffer("_phi_mean", mu)
            self.register_buffer("_phi_std", sd)

        elif method == "cumspline":
            # Cumulative of B-spline basis: phi_cum = cumsum(Bspline(x), dim=-1) dropping the last always-1 dim.
            p = int(self.num_enc_cfg.get("degree", 3))
            n_int = int(self.num_enc_cfg.get("n_knots", 30))
            if x_train_num is None:
                raise ValueError("Cumulative spline encoding requires x_num_train to compute knots.")
            # Internal knots: quantiles or linspace
            if self.num_enc_cfg.get("quantile", True):
                qs = torch.linspace(0.0, 1.0, steps=n_int + 2, device=x_train_num.device)[1:-1]  # exclude 0 and 1
                internal = torch.quantile(x_train_num, qs, dim=0).T   # (n_num, n_int)
            else:
                xmin_lin, xmax_lin = x_train_num.min(dim=0).values, x_train_num.max(dim=0).values
                internal = torch.stack(
                    [xmin_lin + (xmax_lin - xmin_lin) * (i / (n_int + 1)) for i in range(1, n_int + 1)], dim=1
                )
            # Clamped knot vector per feature as in spline
            xmin, xmax = x_train_num.min(dim=0).values, x_train_num.max(dim=0).values
            left = xmin.unsqueeze(1).repeat(1, p + 1)
            right = xmax.unsqueeze(1).repeat(1, p + 1)
            knots = torch.cat([left, internal, right], dim=1)  # (n_num, n_int + 2p + 2)
            self.register_buffer("_spl_knots", knots)
            self._spl_degree = p
            m = knots.shape[1] - p - 1                          # number of B-spline bases
            # Drop the last cumulative dim which is identically 1
            self.num_basis = max(m - 1, 1)
            # Train-time encodings for z-score
            phi_train_raw = self._bspline_encode(x_train_num, knots, p)          # (B, n_num, m)
            phi_train = torch.cumsum(phi_train_raw, dim=-1)[..., :self.num_basis]# (B, n_num, m-1)
            mu = phi_train.mean(dim=0)
            sd = phi_train.std(dim=0, unbiased=False)
            sd = sd.clamp_min(self.num_enc_cfg["eps"])  # numerical safety aligned across encoders
            self.register_buffer("_phi_mean", mu)
            self.register_buffer("_phi_std", sd)
        else:
            raise ValueError(f"Unknown numeric encoding method: {method}")

        self.num_basis = int(self.num_basis)
        self.method = method

        # =================== Projection parameters to token space ===================
        if self.n_num_features > 0:
            # Project basis (k) -> token dimension per feature using a learned matrix W_i in R^{k x d_token}
            self.num_weight = nn.Parameter(torch.empty(self.n_num_features, self.num_basis, d_token))
            proto = torch.empty(self.n_num_features, d_token)
            nn_init.kaiming_uniform_(proto, a=math.sqrt(5))
            with torch.no_grad():
                self.num_weight.copy_(proto.unsqueeze(1).expand(-1, self.num_basis, -1))
        else:
            self.num_weight = None

        if categories is None:
            d_bias_extra = 0
            self.category_offsets = None
            self.category_embeddings = None
        else:
            self.cat_cardinalities = categories
            categories_with_unk = [c + 1 for c in categories]
            d_bias_extra = len(categories_with_unk)
            category_offsets = torch.tensor([0] + categories_with_unk[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories_with_unk), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.category_embeddings.weight[self.category_offsets] = 0

        self.bias = None
        if bias:
            total_bias_dim = self.n_num_features + d_bias_extra
            self.bias = nn.Parameter(torch.empty(total_bias_dim, d_token))
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            if self.bias is not None and d_bias_extra > 0:
                with torch.no_grad():
                    self.bias[self.n_num_features:].zero_()

    def _ple_encode(self, x_num: Tensor, edges: Tensor) -> Tensor:
        B = x_num.shape[0]
        n_num, k = edges.shape
        device = x_num.device
        dtype = x_num.dtype

        x = x_num
        e = edges
        eB = e.unsqueeze(0).expand(B, -1, -1)  # (B, n_num, k)

        # Interval index j so that e_j <= x < e_{j+1}; clamp to [0, k-2]
        pos = (x.unsqueeze(-1) >= e.unsqueeze(0)).sum(dim=-1) - 1  # (B, n_num)
        idx = pos.clamp(0, k - 2)

        # Gather e_j and e_{j+1}
        left = torch.gather(eB, 2, idx.unsqueeze(-1)).squeeze(-1)            # (B, n_num)
        right = torch.gather(eB, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)     # (B, n_num)
        denom = (right - left).clamp_min(self.num_enc_cfg["eps"])           # (B, n_num)

        # Linear fill for the (j+1)-th bin only
        w_right = ((x - left) / denom).clamp(0, 1)                            # (B, n_num)

        # Build thermometer output: prefix ones, bin j set to 1, bin j+1 fractional, rest zeros
        phi = torch.zeros(B, n_num, k, device=device, dtype=dtype)

        # prefix ones for bins strictly to the left of idx
        j = torch.arange(k, device=device).view(1, 1, k)                      # (1,1,k)
        prefix_mask = j < idx.unsqueeze(-1)                                   # (B,n_num,k)
        phi[prefix_mask] = 1.0

        # bin j is fully 1
        phi.scatter_(2, idx.unsqueeze(-1), torch.ones_like(idx, dtype=dtype).unsqueeze(-1))
        # bin j+1 carries the linear remainder
        phi.scatter_(2, (idx + 1).unsqueeze(-1), w_right.unsqueeze(-1))

        # Handle out-of-range values explicitly
        below = x <= e[None, :, 0]      # (B, n_num)
        above = x >= e[None, :, -1]     # (B, n_num)

        if below.any():
            flat = phi.reshape(-1, k)
            bmask = below.reshape(-1)
            flat[bmask] = 0.0
            flat[bmask, 0] = 1.0
        if above.any():
            flat = phi.reshape(-1, k)
            amask = above.reshape(-1)
            flat[amask] = 1.0  # saturate to all 1s when x is to the right of the last knot

        return phi

    def _bspline_encode(self, x_num: Tensor, knots: Tensor, degree: int) -> Tensor:
        eps = float(self.num_enc_cfg.get("eps", 1e-6))
        B = x_num.shape[0]
        n_num, nK = knots.shape
        p = int(degree)
        m = nK - p - 1
        device = x_num.device
        dtype = x_num.dtype

        # Shapes
        x = x_num.unsqueeze(-1)          # (B, n_num, 1)
        t = knots.unsqueeze(0)           # (1, n_num, nK)

        # Degree-0 indicators: N_{i,0}(x) = 1 if t_i <= x < t_{i+1}, except include rightmost endpoint for i=m-1
        t_i   = t[..., :m]               # (1, n_num, m)
        t_ip1 = t[..., 1:m+1]            # (1, n_num, m)
        N = ((x >= t_i) & (x < t_ip1)).to(dtype)  # (B, n_num, m)
        # include the very right endpoint (x == t_{nK-1}) for the last basis
        right_endpoint = (x >= t[..., -2:-1])  # (B, n_num, 1) because t[-2] == t[-1] == xmax in clamped case
        last_idx = torch.arange(m, device=device).view(1,1,m) == (m - 1)
        N = torch.where(right_endpoint & last_idx, torch.ones_like(N), N)

        # Cox–de Boor recursion
        # For d = 1..p:
        # N_{i,d} = ((x - t_i)/(t_{i+d}-t_i)) * N_{i,d-1} + ((t_{i+d+1}-x)/(t_{i+d+1}-t_{i+1})) * N_{i+1,d-1}
        for d in range(1, p + 1):
            # left term indices
            ti      = t[..., :m]                 # t_i
            ti_d    = t[..., d:m+d]              # t_{i+d}
            left_den = (ti_d - ti).clamp_min(eps)
            left_num = (x - ti)
            left = (left_num / left_den) * N     # uses N_{i, d-1}

            # right term indices
            ti1     = t[..., 1:m+1]              # t_{i+1}
            ti1_d1  = t[..., d+1:m+d+1]          # t_{i+d+1}
            right_den = (ti1_d1 - ti1).clamp_min(eps)
            right_num = (ti1_d1 - x)
            N_shift = torch.zeros_like(N)
            N_shift[..., :-1] = N[..., 1:]       # N_{i+1, d-1}
            right = (right_num / right_den) * N_shift

            N = left + right

        # 数值清理：截断极小负数，并做单位和归一（防止 1e-6 级误差积累）
        N = N.clamp_min(0)
        s = N.sum(dim=-1, keepdim=True).clamp_min(eps)
        N = N / s
        return N

    def _encode_numeric(self, x_num: Tensor) -> Tensor:
        if x_num is None or self.n_num_features == 0:
            return None
        if self.method == "raw":
            phi = x_num.unsqueeze(-1)  # (B, n_num, 1)
        elif self.method == "ple":
            phi = self._ple_encode(x_num, self._ple_edges)
        elif self.method == "spline":
            phi = self._bspline_encode(x_num, self._spl_knots, self._spl_degree)
        elif self.method == "cumspline":
            phi_raw = self._bspline_encode(x_num, self._spl_knots, self._spl_degree)
            phi = torch.cumsum(phi_raw, dim=-1)[..., :self.num_basis]
        else:
            raise RuntimeError("Unexpected encoding method")
        # z-score per feature & per basis dim (enforce normalization)
        mu, sd = self._phi_mean, self._phi_std
        phi = (phi - mu.unsqueeze(0)) / sd.unsqueeze(0)
        return phi  # (B, n_num, k)

    @property
    def n_tokens(self) -> int:
        n_cat = 0 if self.category_offsets is None else len(self.category_offsets)
        return 1 + self.n_num_features + n_cat

    def forward(
        self,
        x_num: ty.Optional[Tensor],
        x_cat: ty.Optional[Tensor],
    ) -> Tensor:
        batch_size = x_cat.shape[0] if x_num is None else x_num.shape[0]
        device = x_num.device if x_num is not None else x_cat.device

        final_tokens = []

        # --- 1. CLS Tokens ---
        cls_token_content = self.cls_tokens.unsqueeze(0).expand(batch_size, 1, -1)
        final_tokens.append(cls_token_content)

        # --- 2. Numerical Tokens ---
        if x_num is not None and self.n_num_features > 0:
            # Encode -> z-score -> project to token space
            phi = self._encode_numeric(x_num.float())            # (B, n_num, k)
            num_tokens_content = torch.einsum('bnk,nkd->bnd', phi, self.num_weight)
            if self.bias is not None:
                num_bias = self.bias[:self.n_num_features].unsqueeze(0)
                num_tokens_content = num_tokens_content + num_bias
            final_tokens.append(num_tokens_content)

        # 3. Categorical Tokens (Unchanged logic, just for completeness)
        if x_cat is not None and self.category_embeddings is not None:
            indices = (x_cat.long() + 1) + self.category_offsets.to(device)
            cat_tokens_content = self.category_embeddings(indices)
            if self.bias is not None:
                cat_bias = self.bias[self.n_num_features:].unsqueeze(0)
                cat_tokens_content = cat_tokens_content + cat_bias
            final_tokens.append(cat_tokens_content)

        x = torch.cat(final_tokens, dim=1)
        return x


#############################################################
###                       Attention                       ###
#############################################################
class MultiheadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d_model % n_heads == 0
        assert initialization in ['xavier', 'kaiming']
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        
        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        
        scores = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention = F.softmax(scores, dim=-1)
        
        if self.dropout is not None:
            attention = self.dropout(attention)
            
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


##############################################################
###                        Encoder                         ###
##############################################################
class EncoderLayer(nn.Module):
    def __init__(self, *, d_model, n_heads, d_ffn_factor, attention_dropout, ffn_dropout,
                 residual_dropout, activation, prenormalization, initialization, layer_idx, **kwargs):
        super().__init__()
        self.prenormalization = prenormalization
        self.activation = get_activation_fn(activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        if not prenormalization or layer_idx > 0:
            self.norm0 = nn.LayerNorm(d_model)
        else:
            self.norm0 = nn.Identity()

        self.attention = MultiheadAttention(
            d_model=d_model, n_heads=n_heads, dropout=attention_dropout, initialization=initialization
        )
        
        d_hidden = int(d_model * d_ffn_factor)
        self.linear0 = nn.Linear(d_model, d_hidden * (2 if activation.endswith('glu') else 1))
        self.linear1 = nn.Linear(d_hidden, d_model)
        
        self.ffn_dropout = nn.Dropout(ffn_dropout) if ffn_dropout > 0 else None
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0 else None

    def _apply_dropout(self, x, dropout_layer):
        return dropout_layer(x) if dropout_layer is not None else x

    def forward(self, x: Tensor, q_custom: Tensor) -> Tensor:
        x_residual = x
        
        if self.prenormalization:
            x_norm = self.norm0(x)
            q_norm = self.norm0(q_custom) # Normalize the custom query
            attn_output = self.attention(q_norm, x_norm)
        else:
            attn_output = self.attention(q_custom, x)

        if q_custom.shape[1] < x.shape[1]:
            x = q_custom

        attn_output = self._apply_dropout(attn_output, self.residual_dropout)
        x = x + attn_output
        if not self.prenormalization:
            x = self.norm0(x)
            
        if self.prenormalization:
            x_norm = self.norm1(x)
        else:
            x_norm = x
            
        ffn_output = self.linear0(x_norm)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_dropout(ffn_output) if self.ffn_dropout is not None else ffn_output
        ffn_output = self.linear1(ffn_output)
        
        ffn_output = self._apply_dropout(ffn_output, self.residual_dropout)
        x = x + ffn_output
        if not self.prenormalization:
            x = self.norm1(x)
            
        return x


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(**kwargs, layer_idx=i) for i in range(kwargs['n_layers'])]
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            is_last_layer = i + 1 == len(self.layers)

            # --- Custom query for last layer ---
            if is_last_layer:
                # In the last layer, only the CLS tokens act as queries
                q = x[:, :1]
            else:
                # In other layers, all tokens attend to all other tokens
                q = x

            x = layer(x, q_custom=q)

        return x


###############################################################
###                         Head                           ###
###############################################################
class Head(nn.Module):
    def __init__(self, d_model: int, d_out: int, activation: str, prenormalization: bool):
        super().__init__()
        self.last_normalization = nn.LayerNorm(d_model) if prenormalization else None
        self.last_activation = get_nonglu_activation_fn(activation)
        self.head = nn.Linear(d_model, d_out)

    def forward(self, x: Tensor) -> Tensor:
        # x comes from the encoder with shape (batch_size, 1, d_model)
        x = x.mean(dim=1)

        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if x.shape[-1] == 1:
            return x.squeeze(-1)
        return x


################################################################
###                     Helper Functions                     ###
################################################################
def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

def get_activation_fn(name):
    return (
        reglu if name == 'reglu'
        else geglu if name == 'geglu'
        else torch.sigmoid if name == 'sigmoid'
        else getattr(F, name)
    )

def get_nonglu_activation_fn(name):
    return (
        F.relu if name == 'reglu'
        else F.gelu if name == 'geglu'
        else get_activation_fn(name)
    )

def _compute_slices(sizes: list[int]) -> list[tuple[int, int]]:
    slices, start = [], 0
    for k in sizes:
        slices.append((start, start + k))
        start += k
    return slices

