import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.nn import functional as F
from torch.nn import TransformerEncoder

param_list = ["d_model", "d_ff", "activation", "dropout", "num_enc_layers", "n_head",
              "attention_dropout", "pre_norm", "n_freq", "pretrain", "residual_dropout"]

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
        # self.tokenizer = Tokenizer(
        #     num_continuous=num_continuous,
        #     categories=categories,
        #     d_model=self.d_model,
        #     feature_map=feature_map
        # )
        self.tokenizer = PLRTokenizer(
            num_continuous=num_continuous,
            categories=categories,
            d_model=self.d_model,
            feature_map=feature_map,
            n_freq=self.n_freq or 0
        )

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
        num_continuous,
        categories,
        d_model,
        feature_map=None,
        n_freq: int = 0          # ← 0 means “no sin‑cos augmentation”, use relu activation
    ):
        super(PLRTokenizer, self).__init__()

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

        # --- secondary numeric embedding ---
        self.n_freq = n_freq
        if self.n_freq > 0:
            # Learnable ω and φ for every numeric column
            self.register_parameter(
                "omega", nn.Parameter(torch.rand(self.num_groups, 1, self.n_freq) * math.pi)
            )  # (G, 1, F)
            self.register_parameter(
                "phi", nn.Parameter(torch.zeros(self.num_groups, 1, self.n_freq))
            )   # (G, 1, F)
            self.num_embeddings2 = nn.ModuleList(
                [
                    nn.Linear(d_model * 2 * self.n_freq, d_model, bias=False)
                    for _ in self.group_sizes
                ]
            )
        else:
            self.num_embeddings2 = nn.ModuleList(
                [
                    nn.Linear(d_model, d_model, bias=False)
                    for _ in self.group_sizes
                ]
            )
        for lin in self.num_embeddings2:
            nn_init.kaiming_uniform_(lin.weight, a=math.sqrt(5))

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
                if self.n_freq > 0: 
                    B = seg.size(0)

                    # 1) embedding1 * seg
                    base = self.num_embeddings[i](seg)                         # (B, d_model) 
                    
                    # 2) sin / cos transform
                    base_exp = base.unsqueeze(-1)                              # (B, d_model, 1)
                    sin = torch.sin(base_exp * self.omega[i] + self.phi[i])    # (B, d_model, F)
                    cos = torch.cos(base_exp * self.omega[i] + self.phi[i])    # (B, d_model, F)
                    trig = torch.cat((sin, cos), dim=-1)                       # (B, d_model, 2F)
                    emb = trig.reshape(B, -1)                                 # (B, d_model * 2F)

                    # 3) embedding2 * trig
                    emb = self.num_embeddings2[i](emb)                      # (B, d_model) 
                        
                else:
                    # 1) embedding1 * seg
                    emb = self.num_embeddings[i](seg)
                    
                    # 2) embedding2 * relu(embedding1 * seg)
                    emb = self.num_embeddings2[i](F.relu(emb))  # (B, d_model)
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
        for layer in self.encoder:
            x_emb, _ = layer(x_emb, attn_mask)
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

    def forward(self, x, attn_mask=None):

        # === Pre-Norm Encoder Layer ===
        if self.pre_norm:
            # === Multi‑Head Attention Block (Pre‑Norm) ===
            new_x, attn = self.attention(
                self.norm1(x),
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
            new_x, attn = self.attention(
                x,             # queries
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
