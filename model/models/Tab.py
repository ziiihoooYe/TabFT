import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.nn import functional as F
from torch.nn import TransformerEncoder


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: 位置编码向量的维度
        max_len: 能支持的最大位置索引(必须 >= 你会使用到的最大索引)
        """
        super(PositionalEmbedding, self).__init__()

        # 预先构造一个 [max_len, d_model] 的正余弦编码矩阵
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()   # shape: [max_len, 1]
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)

        # 生成正余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册到 buffer 中，表示这个张量并不是可训练参数，但会随模型一起保存
        self.register_buffer('pe', pe)  # shape: [max_len, d_model]

    def forward(self, positions) -> torch.Tensor:
        """
        positions: 形状可以是 [L] 或 [B, L] 的长整型索引张量。
                   其中的数值要小于 self.pe.size(0) = max_len。
        returns:   若 positions.shape == [L], 则返回 [L, d_model]
                   若 positions.shape == [B, L], 则返回 [B, L, d_model]
        """
        # 直接索引 pe
        # 如果 positions 是 1D，[L] -> 结果 [L, d_model]
        # 如果 positions 是 2D，[B, L] -> 结果 [B, L, d_model] (PyTorch自动广播)
        return self.pe[positions.long()]
    

class CLSHead(nn.Module):
    def __init__(self, d_model, out_features, head_dropout=0):
        super().__init__()
        self.linear = nn.Linear(d_model, out_features)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = x[:, 0, :]
        x = self.linear(x)
        # x = self.dropout(x)
        return x


class MaxPoolingHead(nn.Module):
    def __init__(self, d_model, out_features, head_dropout=0):
        super().__init__()
        self.linear = nn.Linear(d_model, out_features)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = torch.max(x, dim=1).values
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FlattenHead(nn.Module):
    def __init__(self, num_col, d_model, out_features, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(num_col*d_model, out_features)
        # self.linear = nn.Linear(d_model, out_features)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        # x = torch.max(x, dim=1).values
        x = self.linear(x)
        x = self.dropout(x)
        return x


class RegressionHead(nn.Module):
    def __init__(self, d_model, out_features=1):
        super(RegressionHead, self).__init__()
        # self.head = MaxPoolingHead(d_model, out_features)
        self.head = CLSHead(d_model, out_features)
        
    def forward(self, x):
        return self.head(x)
    

class ClassificationHead(nn.Module):
    def __init__(self, d_model, classes_num):
        super(ClassificationHead, self).__init__()
        # self.head = MaxPoolingHead(d_model, classes_num)
        self.head = CLSHead(d_model, classes_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.head(x))


class TabEncoder(nn.Module):
    def __init__(
        self, 
        num_continuous,
        categories,
        d_model, 
        n_head,
        num_enc_layers=6,
        dropout=0.1
        ):
        
        super(TabEncoder, self).__init__()
        self.num_continuous = num_continuous
        self.categories = categories
        self.d_model = d_model

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False), 
                        d_model, 
                        n_head),
                    d_model,
                    dropout=dropout
                ) for l in range(num_enc_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        )

        self.positional_embedding = PositionalEmbedding(d_model)
        self.col_embedding = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )
        
        self.lut_num = nn.Embedding(num_continuous, d_model)
        nn_init.kaiming_uniform_(self.lut_num.weight, a=math.sqrt(5))
        self.num_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_continuous)
        ])
        for i in range(num_continuous):
            nn_init.kaiming_uniform_(self.num_embeddings[i].weight, a=math.sqrt(5))
        
        self.categories = categories
        self.category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        self.lut_cat = nn.Embedding(sum(categories), d_model)
        nn_init.kaiming_uniform_(self.lut_cat.weight, a=math.sqrt(5))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn_init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x_cont, x_categ):
        """
        :param x_cont: (batch_size, num_continuous)
        :param x_categ: (batch_size, num_categories)
        """
        # Normalize inputs and convert to float if necessary
        if x_cont is not None:
            x_cont = x_cont.float()
            x_cont = (x_cont - x_cont.mean(dim=0, keepdim=True)) / (x_cont.std(dim=0, keepdim=True) + 1e-8)
        if x_categ is not None:
            x_categ = x_categ.long()

        if x_cont is not None:
            cont_embeds = []
            x_cont = x_cont.view(-1, self.num_continuous, 1)
            for i in range(x_cont.shape[-2]):
                emb = self.num_embeddings[i](x_cont[:, i, :])
                cont_embeds.append(emb)
            x_cont_emb = torch.stack(cont_embeds, dim=1)
            channel_embeddings = self.lut_num(torch.arange(0, x_cont.shape[-2]).to(x_cont.device))
            x_cont_emb = x_cont_emb + channel_embeddings
            x_cont_emb = self.dropout(x_cont_emb)
        if x_categ is not None:
            cat_embeds = []
            for i in range(x_categ.shape[-1]):
                x_categ_i = x_categ[:, i] + self.category_offsets[i]
                emb_i = self.lut_cat(x_categ_i)
                cat_embeds.append(emb_i)
            x_categ_emb = torch.stack(cat_embeds, dim=1)  # [bs, n_cat_columns, d_model]
            channel_embeddings = self.lut_cat(torch.arange(0, x_categ.shape[-1]).to(x_categ.device))
            x_categ_emb = x_categ_emb + channel_embeddings
            x_categ_emb = self.dropout(x_categ_emb)


        if x_cont is not None and x_categ is not None:
            # Concatenate continuous and categorical embeddings
            x_emb = torch.cat([x_cont_emb, x_categ_emb], dim=-2)
        elif x_cont is not None:
            x_emb = x_cont_emb
        elif x_categ is not None:
            x_emb = x_categ_emb
        else:
            raise ValueError("At least one of x_cont or x_categ must be provided")

        x_emb = torch.cat((self.cls_token.expand(x_emb.shape[0], -1, -1), x_emb), dim=1)
        out, _ = self.encoder(x_emb)  # (batch_size, num_continuous + num_category + 1, d_model)
        
        return out  # CLS token


class Tab(nn.Module):
    """
    Tab model combining TabEncoder and a head (classification/regression).
    """
    def __init__(self, num_continuous, categories, d_model, n_head, d_out, num_enc_layers=6, is_regression=True):
        super(Tab, self).__init__()
        
        # categories = len(categories) if categories is not None else 0
        if categories is None: categories = []
        
        self.encoder = TabEncoder(
            num_continuous=num_continuous,
            categories=categories,
            d_model=d_model, 
            n_head=n_head,
            num_enc_layers=num_enc_layers
        )
        if is_regression:
            self.head = RegressionHead(
                d_model=d_model, 
                out_features=d_out
            )
        else:
            self.head = ClassificationHead(
                d_model=d_model, 
                classes_num=d_out  # Number of classes for classification
            )

    def forward(self, x_cont, x_categ):
        """
        :param x_cont: (batch_size, num_continuous)
        :param x_categ: (batch_size, num_categories)
        """
        out = self.encoder(x_cont=x_cont, x_categ=x_categ)
        out = self.head(out)
        return out



class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


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

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
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
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


