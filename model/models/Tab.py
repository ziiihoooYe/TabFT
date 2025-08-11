# Modified Tab.py

import torch
import math
import typing as ty
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
        lap_pe: ty.Optional[Tensor] = None,
        n_cls: int = 1,
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

        self.d_lap_pe = lap_pe.shape[1] if lap_pe is not None else 0
        
        effective_d_model = self.d_token + self.d_lap_pe
        
        self.tokenizer = Tokenizer(
            num_continuous=num_continuous,
            categories=categories,
            d_token=self.d_token,
            bias=self.token_bias,
            lap_pe=lap_pe,
            d_lap_pe=self.d_lap_pe,
            n_cls=self.n_cls
        ) 

        self.encoder = Encoder(
            n_layers=self.n_layers,
            d_model=effective_d_model,
            n_heads=self.n_heads,
            d_ffn_factor=self.d_ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            activation=self.activation,
            prenormalization=self.prenormalization,
            initialization=self.initialization,
            n_cls=self.n_cls
        )
        
        self.head = Head(
            d_model=effective_d_model,
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
        d_lap_pe: int = 0,
        n_cls: int = 1
    ) -> None:
        super().__init__()
        self.n_num_features = num_continuous
        self.d_token = d_token
        self.d_lap_pe = d_lap_pe
        self.n_cls = n_cls

        # --- CLS and Numerical Token weights ---
        # Create a separate parameter for CLS tokens
        self.cls_tokens = nn.Parameter(Tensor(n_cls, d_token))
        nn_init.kaiming_uniform_(self.cls_tokens, a=math.sqrt(5))

        # The main weight parameter is only for numerical features
        self.weight = nn.Parameter(Tensor(self.n_num_features, d_token))
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))

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
            self.bias = nn.Parameter(Tensor(total_bias_dim, d_token))
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            if self.bias is not None and d_bias_extra > 0:
                with torch.no_grad():
                    self.bias[self.n_num_features:].zero_()

    @property
    def n_tokens(self) -> int:
        n_cat = 0 if self.category_offsets is None else len(self.category_offsets)
        return self.n_cls + self.n_num_features + n_cat

    def forward(
        self,
        x_num: ty.Optional[Tensor],
        x_cat: ty.Optional[Tensor],
    ) -> Tensor:
        batch_size = x_cat.shape[0] if x_num is None else x_num.shape[0]
        device = x_num.device if x_num is not None else x_cat.device

        final_tokens = []

        # --- 1. CLS Tokens ---
        cls_token_content = self.cls_tokens.expand(batch_size, -1, -1)
        if self.d_lap_pe > 0:
            cls_pe = torch.zeros(batch_size, self.n_cls, self.d_lap_pe, device=device)
            final_cls_tokens = torch.cat([cls_token_content, cls_pe], dim=-1)
        else:
            final_cls_tokens = cls_token_content
        final_tokens.append(final_cls_tokens)

        # --- 2. Numerical Tokens ---
        if x_num is not None:
            num_tokens_content = x_num.unsqueeze(-1) * self.weight.unsqueeze(0)
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
        self.n_cls = kwargs.get('n_cls', 1)
        self.layers = nn.ModuleList(
            [EncoderLayer(**kwargs, layer_idx=i) for i in range(kwargs['n_layers'])]
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            is_last_layer = i + 1 == len(self.layers)
            
            # --- Custom query for last layer ---
            if is_last_layer:
                # In the last layer, only the CLS tokens act as queries
                q = x[:, :self.n_cls]
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
        # x comes from the encoder with shape (batch_size, n_cls, d_model)
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

