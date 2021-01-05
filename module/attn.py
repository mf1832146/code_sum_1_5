import torch
from torch import nn
import math
from utils import clones
import torch.nn.functional as F


def _standard_rel_attn_inner(x, y, z, transpose):
    """
    Args:
            x: Tensor with shape [batch_size, heads, length or 1, length or depth].
            y: Tensor with shape [batch_size, heads, length or 1, depth].
            z: Tensor with shape [length or 1, length, depth].
            transpose: Whether to transpose inner matrices of y and z. Should be true if
            last dimension of x is depth, not length.
        Returns:
             A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size, heads, length, _ = x.size()
    xy_matmul = torch.matmul(x, y if not transpose else y.transpose(-2, -1))
    x_t = x.permute(2, 0, 1, 3).contiguous().view(length, heads * batch_size, -1)
    x_tz_matmul_r = x_t.view(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return xy_matmul + x_tz_matmul_r_t


def _tree_rel_attn_inner(x, y, z, transpose):
    """
        Args:
                x: Tensor with shape [batch_size, heads, length or 1, length or depth].
                y: Tensor with shape [batch_size, heads, length or 1, depth].
                z: Tensor with shape [batch_size, heads, length, length, depth].
                transpose: Whether to transpose inner matrices of y and z. Should be true if
                last dimension of x is depth, not length.
            Returns:
                 A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size, heads, length, _ = x.size()
    xy_matmul = torch.matmul(x, y if not transpose else y.transpose(-2, -1))
    x_t = x.contiguous().view(batch_size * heads * length, -1).unsqueeze(1)
    z_t = z.contiguous().view(batch_size * heads * length, length, -1)
    xz_matmul = torch.matmul(x_t, z_t if not transpose else z_t.transpose(-2, -1))
    xz_matmul_t = xz_matmul.squeeze(1).view(batch_size, heads, length, -1)
    return xy_matmul + xz_matmul_t


def _standard_attn(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def _rel_attn(query, key, value, rel_k, rel_v, _rel_attn_inner, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = _rel_attn_inner(query, key, rel_k, True) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return _rel_attn_inner(p_attn, value, rel_v, False), p_attn


class MultiHeadAttn(nn.Module):
    def __init__(self, model_dim, head_count, dropout):
        super(MultiHeadAttn, self).__init__()
        self.model_dim = model_dim
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.head_count = head_count

        self.linear_layers = clones(nn.Linear(model_dim, model_dim), 4)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size = x.size(0)

        return x.view(batch_size, -1, self.head_count, self.dim_per_head)\
            .transpose(1, 2).contiguous()

    def _combine_heads(self, x):
        seq_len = x.size(2)

        return x.transpose(1, 2).contiguous() \
            .view(-1, seq_len, self.head_count * self.dim_per_head)

    def forward(self, q, k, v, mask=None):
        """
        mask: Bool Tensor. If True, mask with -inf, else do not change.
        """
        q, k, v = \
            [self.split_heads(l(x))
             for l, x in zip(self.linear_layers, (q, k, v))]

        x, attn = _standard_attn(q, k, v, mask=mask,
                                 dropout=self.dropout)
        return self.linear_layers[-1](x), attn


class MultiHeadRelAttn(MultiHeadAttn):
    def __init__(self, model_dim, head_count, dropout):
        super().__init__(model_dim, head_count, dropout=dropout)
        self._rel_attn_inner = _standard_rel_attn_inner

    def forward(self, q, k, v, mask=None, relative_k=None, relative_v=None):
        q, k, v = \
            [self.split_heads(l(x))
             for l, x in zip(self.linear_layers, (q, k, v))]

        x, attn = _rel_attn(q, k, v, relative_k, relative_v, self._rel_attn_inner, mask, self.dropout)
        return self.linear_layers[-1](x), attn


class MultiHeadRelTreeAttn(MultiHeadRelAttn):
    def __init__(self, model_dim, head_count, dropout):
        super().__init__(model_dim, head_count, dropout=dropout)
        self._rel_attn_inner = _tree_rel_attn_inner
