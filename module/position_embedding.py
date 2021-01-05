import torch.nn as nn
import torch
import math

from torch.nn.parameter import Parameter

from utils import clones


class TreeRelativePosition(nn.Module):
    def __init__(self, d_model, k, num_heads, num_features, dropout=0):
        super(TreeRelativePosition, self).__init__()

        self.d_model = d_model
        self.k = k
        self.num_features = num_features
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.emb_list = clones(nn.Embedding(2 * k + 2, d_model * 2, padding_idx=0), num_features)

    def repeat_for_each_feature(self, emb):
        """
        :param emb: A Tensor with shape [batch_size, max_size, max_size, d_model]
        :return: A Tensor with shape [batch_size, num_head // num_features, max_size, max_size, d_model]
        """
        batch_size, max_size = emb.size(0), emb.size(1)
        emb = emb.repeat(1, 1, 1, self.num_heads // self.num_features)
        emb = emb.view(batch_size, max_size, max_size, -1, self.d_model)
        emb = emb.permute(0, 3, 1, 2, 4)
        return emb

    def forward(self, inputs):
        """inputs : A list of Tensor with shape [batch_size, max_size, max_size]"""
        assert self.num_features == len(inputs)

        k_emb_list = []
        v_emb_list = []

        for i, v in enumerate(inputs):
            batch_size, max_size = v.size(0), v.size(1)
            v = v.unsqueeze(3)
            position_emb = self.emb_list[i](v)
            position_emb = self.dropout(position_emb)
            position_emb = position_emb.view(batch_size, max_size, max_size, 2, self.d_model) * math.sqrt(self.d_model)
            k_emb, v_emb = [x.squeeze(3) for x in position_emb.split(1, dim=3)]

            k_emb = self.repeat_for_each_feature(k_emb)
            v_emb = self.repeat_for_each_feature(v_emb)

            k_emb_list.append(k_emb)
            v_emb_list.append(v_emb)

        k_emb_n = torch.cat(k_emb_list, dim=1)
        v_emb_n = torch.cat(v_emb_list, dim=1)

        return k_emb_n, v_emb_n


class StandardRelativePosition(nn.Module):
    def __init__(self, d_model, k, dropout=0):
        super(StandardRelativePosition, self).__init__()
        self.d_model = d_model
        self.k = k
        self.dropout = nn.Dropout(dropout)

        self.emb_k = Parameter(torch.Tensor(k * 2 + 1, d_model))
        self.emb_v = Parameter(torch.Tensor(k * 2 + 1, d_model))
        nn.init.xavier_uniform_(self.emb_k)
        nn.init.xavier_uniform_(self.emb_v)

    def forward(self, length_q):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_q)

        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.k, self.k) + self.k
        final_mat = torch.LongTensor(distance_mat_clipped)
        embeddings_k = self.dropout(self.emb_k[final_mat])
        embeddings_v = self.dropout(self.emb_v[final_mat])

        return embeddings_k, embeddings_v
