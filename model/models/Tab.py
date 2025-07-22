import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.nn import functional as F
from torch.nn import TransformerEncoder

param_list = ["d_model", "d_ff", "activation", "dropout", "num_enc_layers", "n_head",
              "attention_dropout", "pre_norm", "n_freq", "pretrain", "residual_dropout", "freq_scale"]

class Tab(nn.Module):
    """
    Tab model combining TabEncoder and a head (classification/regression).
    """
    def __init__(self, config, num_continuous, categories, d_out, is_regression=True, feature_map=None, pretrain=False):
        super(Tab, self).__init__()
        if categories is None: categories = []
        for param in param_list:
            if param in config:
                setattr(self, param, config[param])
            else:
                setattr(self, param, None)
        
        # Initialize tokenizer
        # self.tokenizer = PiecewisePLRTokenizer(
        #     categories=categories,
        #     d_model=self.d_model,
        #     n_freq=self.n_freq or 0,
        #     feature_map=feature_map,
        #     freq_scale=self.freq_scale
        # )
        self.tokenizer = PLRTokenizer(
            num_continuous=num_continuous,
            categories=categories,
            d_model=self.d_model,
            n_freq=self.n_freq or 0
        )
        # self.tokenizer = Tokenizer(
        #     num_continuous=num_continuous,
        #     categories=categories,
        #     d_model=self.d_model,
        #     feature_map=feature_map
        # )

        # Initialize encoder
        self.encoder = TabEncoder(
            d_model=self.d_model,
            n_head=self.n_head,
            num_enc_layers=self.num_enc_layers,
            dropout=self.dropout,
            d_ff=self.d_ff,
            activation="reglu",
            attention_dropout=self.attention_dropout,
            residual_dropout=self.residual_dropout,
            pre_norm=config.get("pre_norm", False)  # Use pre-norm if specified in config
        )

        # Initialize head
        if is_regression:
            self.head = RegressionHead(
                d_model=self.d_model,
                out_features=d_out
            )
        else:
            self.head = ClassificationHead(
                d_model=self.d_model,
                classes_num=d_out  # Number of classes for classification
            )
        
        if self.pretrain:
            self.pretrain_head = MaskRecHead(self.d_model)
        

    def forward(self, x_cont, x_categ, attention_mask=None):
        x_emb = self.tokenizer(x_cont, x_categ)  # (batch_size, num_groups + num_categories + 1, d_model)
        out = self.encoder(x_emb, attention_mask)
        out = self.head(out)
        return out
    

    def forward_pretrain(self, x_cont, x_categ, mask=None):
        x_emb = self.tokenizer(x_cont, x_categ, mask)  # (batch_size, num_groups + num_categories + 1, d_model)
        out = self.encoder(x_emb)
        out = self.pretrain_head(out)
        return out[:, 1:]


### ---------------------------------------------------------------------------- ###
###                                   Tokenizer                                  ###
### ---------------------------------------------------------------------------- ###
class Tokenizer(nn.Module):
    def __init__(
        self,
        num_continuous,
        categories,
        d_model,
        feature_map=None
    ):
        super(Tokenizer, self).__init__()

        self.num_continuous = num_continuous
        self.d_model = d_model

        # --- group information (expanded features) ---
        # feature map
        if feature_map is not None:
            self.group_sizes = [m["size"] for m in feature_map]   # per‑feature expanded dimension
        else:
            self.group_sizes = [1] * num_continuous
        # group num of num feature / cat feature
        self.num_groups = len(self.group_sizes)
        self.cat_groups = int(len(categories))

        # --- numeric embedding ---
        # column embedding
        self.lut_num = nn.Embedding(self.num_groups, d_model)
        # nn_init.kaiming_uniform_(self.lut_num.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.lut_num.weight.zero_()
        # Linear per column:  (group_size → d_model)

        self.num_embeddings = nn.ModuleList([nn.Linear(sz, d_model, bias=False) for sz in self.group_sizes])
        for lin in self.num_embeddings: nn_init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        # num id
        self.register_buffer("num_ids", torch.arange(self.num_groups, dtype=torch.long))


        # --- category embedding ---
        # column embedding
        self.lut_cat = nn.Embedding(self.cat_groups, d_model)
        nn_init.kaiming_uniform_(self.lut_cat.weight, a=math.sqrt(5))
        # with torch.no_grad():
        #     self.lut_cat.weight.zero_() 
        # categorical embedding (reserve UNK = index 0 for every feature)
        categories_with_unk = [c + 1 for c in categories]          # +1 for UNK
        num_cat_embeddings = sum(categories_with_unk)
        if num_cat_embeddings == 0:                                # no categorical features
            num_cat_embeddings = 1                                 # dummy UNK slot
        self.cat_embeddings = nn.Embedding(num_cat_embeddings, d_model)
        nn_init.kaiming_uniform_(self.cat_embeddings.weight, a=math.sqrt(5))
        # offsets for individual categorical features
        self.register_buffer("category_offsets", torch.tensor([0] + categories_with_unk[:-1]).cumsum(0))
        # zero‑initialize every UNK row so unknown categories contribute no signal
        with torch.no_grad():
            self.cat_embeddings.weight[self.category_offsets] = 0
        # cat id
        self.register_buffer("cat_ids", torch.arange(self.cat_groups, dtype=torch.long))

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn_init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))        

    def forward(self, x_cont, x_categ, mask=None):
        """
        :param x_cont: (batch_size, num_continuous)
        :param x_categ: (batch_size, num_categories)
        If n_freq > 0, numeric columns are augmented with learnable sin/cos embeddings.
        """
        if x_cont is not None:
            x_cont = x_cont.float()
            cont_embeds = []
            offset = 0

            for i, size in enumerate(self.group_sizes):
                seg = x_cont[:, offset:offset + size]          # (B, size)
                offset += size
                # optional sin‑cos embedding
                emb = self.num_embeddings[i](seg) 
                cont_embeds.append(emb)

            # num column embedding
            x_cont_emb = torch.stack(cont_embeds, dim=1)       # (B, num_groups, d_model)
            channel_embeddings = self.lut_num(self.num_ids.to(x_cont.device))  # (num_groups, d_model)
            x_cont_emb = x_cont_emb + channel_embeddings

        if x_categ is not None:
            x_categ = x_categ.long()
            cat_embeds = []
            for i in range(x_categ.shape[-1]):
                # shift by +1 so that unknown (‑1) → 0, valid 0..c‑1 → 1..c
                idx = x_categ[:, i].long() + 1
                idx = idx + self.category_offsets[i]              # global embedding index
                emb_i = self.cat_embeddings(idx)
                cat_embeds.append(emb_i)
            x_categ_emb = torch.stack(cat_embeds, dim=1)          # (B, n_cat, d_model)
            channel_embeddings = self.lut_cat(self.cat_ids.to(x_categ.device))
            x_categ_emb = x_categ_emb + channel_embeddings

        if x_cont is not None and x_categ is not None:
            x_emb = torch.cat([x_cont_emb, x_categ_emb], dim=-2)
        elif x_cont is not None:
            x_emb = x_cont_emb
        elif x_categ is not None:
            x_emb = x_categ_emb
        else:
            raise ValueError("At least one of x_cont or x_categ must be provided")

        if mask is not None:                         # mask: (B, L) → True 表示被遮
            mask_token = torch.zeros_like(self.cls_token).expand(x_emb.size(0), 1, -1).to(x_emb.device)
            x_emb = torch.where(mask.unsqueeze(-1), mask_token, x_emb)

        x_emb = torch.cat((self.cls_token.expand(x_emb.shape[0], -1, -1), x_emb), dim=1)

        return x_emb
    
    
class PLRTokenizer(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        categories: list,
        d_model: int,
        n_freq: int = 32,
    ):
        super().__init__()
        assert n_freq > 0
        self.n_features = num_continuous
        self.n_freq = n_freq
        self.d_model = d_model

        # initial_freqs = torch.logspace(0.0, 1.0, n_freq) * math.pi
        initial_freqs = torch.rand(n_freq) * math.pi
        self.freqs = nn.Parameter(initial_freqs.unsqueeze(0).expand(num_continuous, -1))

        # The shared projection layer is now explicitly unbiased.
        self.projection = nn.Linear(2 * n_freq, d_model, bias=False)
        nn_init.kaiming_uniform_(self.projection.weight, a=math.sqrt(5))

        # The embedding layer now acts as the sole, per-feature bias.
        self.lut_num = nn.Embedding(num_continuous, d_model)
        nn_init.kaiming_uniform_(self.lut_num.weight, a=math.sqrt(5))
        self.register_buffer("num_ids", torch.arange(num_continuous, dtype=torch.long))
        
        self.cat_groups = len(categories) if categories else 0
        if self.cat_groups > 0:
            self.lut_cat = nn.Embedding(self.cat_groups, d_model)
            nn_init.kaiming_uniform_(self.lut_cat.weight, a=math.sqrt(5))
            
            categories_with_unk = [c + 1 for c in categories]
            n_cat_emb = sum(categories_with_unk)
            self.cat_embeddings = nn.Embedding(n_cat_emb, d_model)
            nn_init.kaiming_uniform_(self.cat_embeddings.weight, a=math.sqrt(5))
            
            self.register_buffer("category_offsets",
                                 torch.tensor([0] + categories_with_unk[:-1], dtype=torch.long).cumsum(0))
            with torch.no_grad():
                self.cat_embeddings.weight[self.category_offsets] = 0
            self.register_buffer("cat_ids", torch.arange(self.cat_groups, dtype=torch.long))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn_init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

    def forward(self, x_cont, x_categ=None, mask=None):
        if x_cont is None:
            return None
            
        B, num_features = x_cont.shape
        x_cont = x_cont.float()

        phase = x_cont.unsqueeze(-1) * self.freqs
        
        sin_features = torch.sin(phase)
        cos_features = torch.cos(phase)
        
        modulated_features = torch.cat([sin_features, cos_features], dim=-1)
        
        x_cont_emb = self.projection(modulated_features) + self.lut_num(self.num_ids)
        x_cont_emb = F.relu(x_cont_emb)

        if x_categ is not None and self.cat_groups > 0:
            indices = x_categ.long() + 1 + self.category_offsets.to(x_categ.device)            
            x_categ_emb = self.cat_embeddings(indices) 
            x_categ_emb = x_categ_emb + self.lut_cat(self.cat_ids)
        else:
            x_categ_emb = None

        if x_categ_emb is not None:
            x_emb = torch.cat([x_cont_emb, x_categ_emb], dim=1)
        else:
            x_emb = x_cont_emb

        if mask is not None:
            mask_token = torch.zeros_like(self.cls_token).expand(B, 1, -1)
            x_emb = torch.where(mask.unsqueeze(-1), mask_token, x_emb)

        x_emb = torch.cat((self.cls_token.expand(B, -1, -1), x_emb), dim=1)
        return x_emb


from typing import List, Dict, Tuple
class PiecewisePLRTokenizer(nn.Module):
    def __init__(
        self,
        feature_map: List[Dict],
        categories: list,
        d_model: int,
        n_freq: int = 4,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        assert n_freq > 0
        assert feature_map, "feature_map cannot be empty."

        self.feature_map = feature_map
        self.num_continuous_orig = len(feature_map)
        
        self.max_ple_size = max(meta['size'] for meta in feature_map)
        
        last_meta = feature_map[-1]
        self.n_flat_dims_in = last_meta['new_start'] + last_meta['size']

        self.n_freq = n_freq
        self.d_model = d_model

        initial_freqs = torch.rand(self.num_continuous_orig, self.max_ple_size, n_freq) * math.pi * freq_scale
        self.freqs = nn.Parameter(initial_freqs)
        
        projection_in_dim = 2 * self.max_ple_size * n_freq
        self.projection = nn.Linear(projection_in_dim, d_model, bias=False)
        nn_init.kaiming_uniform_(self.projection.weight, a=math.sqrt(5))

        self.lut_num = nn.Embedding(self.num_continuous_orig, d_model)
        nn_init.kaiming_uniform_(self.lut_num.weight, a=math.sqrt(5))
        self.register_buffer("num_ids", torch.arange(self.num_continuous_orig, dtype=torch.long))

        gather_indices = torch.full((self.num_continuous_orig, self.max_ple_size), self.n_flat_dims_in, dtype=torch.long)
        
        padding_mask = torch.ones(self.num_continuous_orig, self.max_ple_size, dtype=torch.bool)

        for i, meta in enumerate(self.feature_map):
            start, size = meta['new_start'], meta['size']
            indices = torch.arange(start, start + size)
            gather_indices[i, :size] = indices
            padding_mask[i, :size] = False

        self.register_buffer("gather_indices", gather_indices, persistent=False)
        self.register_buffer("precomputed_mask", padding_mask, persistent=False)

        self.cat_groups = len(categories) if categories else 0
        if self.cat_groups > 0:
            self.lut_cat = nn.Embedding(self.cat_groups, d_model)
            nn_init.kaiming_uniform_(self.lut_cat.weight, a=math.sqrt(5))
            
            categories_with_unk = [c + 1 for c in categories]
            n_cat_emb = sum(categories_with_unk)
            self.cat_embeddings = nn.Embedding(n_cat_emb, d_model)
            nn_init.kaiming_uniform_(self.cat_embeddings.weight, a=math.sqrt(5))
            
            self.register_buffer("category_offsets",
                                torch.tensor([0] + categories_with_unk[:-1], dtype=torch.long).cumsum(0))
            with torch.no_grad():
                self.cat_embeddings.weight[self.category_offsets] = 0
            self.register_buffer("cat_ids", torch.arange(self.cat_groups, dtype=torch.long))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn_init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

    def _prepare_and_pad(self, x_cont: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        向量化版本: 使用预计算的索引和掩码来代替 for 循环。
        """
        B, _ = x_cont.shape

        # --- 优化点 2: 使用 F.pad 和高级索引 ---
        # 在 x_cont 的最后一个维度上填充一个0。这个0将作为所有填充位置的值。
        # self.gather_indices 中指向填充位置的索引 (self.n_flat_dims_in) 现在会安全地指向这个0。
        x_cont_padded_for_gather = F.pad(x_cont, (0, 1), 'constant', 0)
        
        # 使用高级索引一次性完成所有数据的提取和重排。
        # x_cont_padded_for_gather 的 shape 是 (B, D_flat+1)
        # self.gather_indices 的 shape 是 (N, S_max)
        # 结果 x_padded 的 shape 是 (B, N, S_max), 这正是我们想要的。
        x_padded = x_cont_padded_for_gather[:, self.gather_indices]
        
        # 直接扩展预计算的掩码以匹配批次大小
        padding_mask = self.precomputed_mask.expand(B, -1, -1)
            
        return x_padded, padding_mask

    def forward(self, x_cont: torch.Tensor, x_categ: torch.Tensor = None) -> torch.Tensor:
        """
        处理来自HierarchicalPleTransform的扁平化输出。

        Args:
            x_cont (torch.Tensor): 扁平化的PLE特征。Shape: (B, D_flat)
            x_categ (torch.Tensor, optional): 分类特征。
        
        Returns:
            torch.Tensor: 最终的Token嵌入。Shape: (B, 1 + N_orig + N_cat, d_model)
        """
        B, D_flat = x_cont.shape
        if D_flat != self.n_flat_dims_in:
            raise ValueError(f"Input has {D_flat} features, but tokenizer was built for {self.n_flat_dims_in} flat features.")

        # --- 1. 内部进行解包和填充 (现在是高效的向量化操作) ---
        x_padded, padding_mask = self._prepare_and_pad(x_cont)
        # x_padded Shape: (B, N, S_max)
        # padding_mask Shape: (B, N, S_max)

        # --- 后续步骤与原版相同，它们已经是高效的了 ---
        
        # --- 2. 应用频率编码 ---
        phase = x_padded.unsqueeze(-1) * self.freqs
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        
        # --- 3. 应用掩码 ---
        features.masked_fill_(padding_mask.unsqueeze(-1), 0.0)

        # --- 4. 扁平化特征组并投影 ---
        flattened_features = features.view(B, self.num_continuous_orig, -1)
        x_cont_emb = self.projection(flattened_features)
        
        # --- 5. 添加共享偏置 ---
        bias = self.lut_num(self.num_ids)
        x_cont_emb = x_cont_emb + bias
        x_cont_emb = F.relu(x_cont_emb)

        # --- 6. 整合并添加CLS Token ---
        x_categ_emb = None # Placeholder for categorical features
        if x_categ_emb is not None:
            x_emb = torch.cat([x_cont_emb, x_categ_emb], dim=1)
        else:
            x_emb = x_cont_emb

        x_emb = torch.cat((self.cls_token.expand(B, -1, -1), x_emb), dim=1)
        return x_emb





### ---------------------------------------------------------------------------- ###
###                                     Encoder                                  ###
### ---------------------------------------------------------------------------- ###
class TabEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        num_enc_layers,
        dropout,
        d_ff,
        activation,
        attention_dropout,
        residual_dropout,
        pre_norm=False
    ):
        super(TabEncoder, self).__init__()

        # --- encoder ---
        self.encoder = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        attention_dropout=attention_dropout, 
                        output_attention=False,
                        ),
                    d_model,
                    n_head
                ),
                d_model,
                layer_idx=layer_idx,
                d_ff=d_ff,
                dropout=dropout,
                residual_dropout=residual_dropout,
                activation=activation,  # "relu" or "reglu", "geglu"
                pre_norm=pre_norm
            )
            for layer_idx in range(num_enc_layers)
        ])

    def forward(self, x_emb, attn_mask=None):
        for i, layer in enumerate(self.encoder):
            is_last_layer = i == len(self.encoder) - 1
            x_emb, _ = layer(x_emb, attn_mask, is_last_layer=is_last_layer)
        return x_emb


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, layer_idx,
                 d_ff=None, dropout=0.1, residual_dropout=0., activation="relu", pre_norm=False):
        super(EncoderLayer, self).__init__()

        # --- Attention Block ---
        self.attention = attention

        # --- Feed-Forward Block ---
        d_ff = int(d_ff*d_model) if d_ff is not None else 4 * d_model
        self.linear1 = nn.Linear(
            d_model, d_ff * (2 if activation.endswith('glu') else 1)
        )
        self.linear2 = nn.Linear(d_ff, d_model)

        # --- Layer Normalization and Dropout ---
        self.norm1 = nn.Identity() if pre_norm and layer_idx == 0 else nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

        # --- Activation Function and Pre-Norm ---
        self.activation = F.relu if activation == "relu" else reglu if activation == "reglu" else geglu
        self.pre_norm = pre_norm

    def forward(self, x, attn_mask=None, is_last_layer=False):

        # === Pre-Norm Encoder Layer ===
        if self.pre_norm:
            # === Multi‑Head Attention Block (Pre‑Norm) ===
            q = self.norm1(x)[:, :1] if is_last_layer else self.norm1(x)
            new_x, attn = self.attention(
                q,
                self.norm1(x),
                self.norm1(x),
                attn_mask=attn_mask
            )
            x = x + self.residual_dropout(new_x)

            # === Feed‑Forward Block (Pre‑Norm) ===
            x2 = self.norm2(x)
            y = self.linear1(x2)
            y = self.dropout(self.activation(y))
            y = self.linear2(y)
            x = x + self.residual_dropout(y)
        else:
            # === Post-Norm Encoder Layer ===
            # === Multi‑Head Attention Block (Post‑Norm) ===
            q = x[:, :1] if is_last_layer else x
            new_x, attn = self.attention(
                q,             # queries
                x,             # keys
                x,             # values
                attn_mask=attn_mask
            )
            x = self.norm1(x + self.residual_dropout(new_x))

            # === Feed‑Forward Block (Post‑Norm) ===
            y = self.linear1(x)
            y = self.dropout(self.activation(y))
            y = self.linear2(y)

            # Add & Norm
            x = self.norm2(x + self.residual_dropout(y))

        if is_last_layer:
            x = x[:, :1]
        return x, attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        for proj in (self.query_projection,
                     self.key_projection,
                     self.value_projection,
                     self.out_projection):
            nn_init.zeros_(proj.bias)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if attn_mask:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


### ---------------------------------------------------------------------------- ###
###                                     Heads                                    ###
### ---------------------------------------------------------------------------- ###
class CLSHead(nn.Module):
    def __init__(self, d_model, out_features):
        super().__init__()
        self.linear = nn.Linear(d_model, out_features)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = x[:, 0, :]
        x = self.linear(self.activation(self.norm(x)))
        return x.squeeze(-1)


class RegressionHead(nn.Module):
    def __init__(self, d_model, out_features=1):
        super(RegressionHead, self).__init__()
        self.head = CLSHead(d_model, out_features)
        
    def forward(self, x):
        return self.head(x)
    

class ClassificationHead(nn.Module):
    def __init__(self, d_model, classes_num):
        super(ClassificationHead, self).__init__()
        self.head = CLSHead(d_model, classes_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.head(x))


class MaskRecHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)     # 投影回嵌入空间
    
    def forward(self, x):
        return self.proj(x)

### ---------------------------------------------------------------------------- ###
###                                   Utils                                      ###
### ---------------------------------------------------------------------------- ###
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ResBlock(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.linear2(y)
        return y + self.dropout(x)
