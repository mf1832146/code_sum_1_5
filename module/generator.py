import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, dropout):
        super(Generator, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, vocab),
            nn.Dropout(dropout),
            nn.Softmax(-1)
        )

    def forward(self, inputs):
        encoder_outputs, decoder_outputs, decoder_attn, tgt_emb, extra_vocab, expanded_word_idx, semantic_mask = inputs
        gen_prob = self.proj(decoder_outputs)
        gen_prob = F.pad(gen_prob, (0, len(extra_vocab)), 'constant')
        return torch.log(gen_prob + 1e-9)


class PointerGenerator(nn.Module):
    def __init__(self, d_model, vocab, dropout):
        super(PointerGenerator, self).__init__()

        self.w_h = nn.Linear(d_model, 1)
        self.w_s = nn.Linear(d_model, 1)
        self.w_x = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

        self.sigmoid = nn.Sigmoid()
        self.soft_max = nn.Softmax(-1)

        self.p_vocab = nn.Sequential(
            nn.Linear(d_model, vocab),
            nn.Dropout(dropout),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs):
        # semantic_mask : mask non-leaf nodes... if 0, masked, else do nothing.
        # encoder_outputs : shape [batch_size, src_len, d_model]
        # decoder_outputs : shape [batch_size, tgt_len, d_model]
        # decoder_attn : shape [batch_size, num_heads, tgt_len, src_len]
        # tgt_emb : shape [batch_size, tgt_len, d_model]
        # extra_vocab : dict to save the OOV words indexes
        # expanded_word_idx : [batch_size, src_len]
        # semantic_mask : [batch_size, src_len]
        encoder_outputs, decoder_outputs, decoder_attn, tgt_emb, extra_vocab, expanded_word_idx, semantic_mask = inputs
        copy_prob = torch.mean(decoder_attn, dim=1)
        h_t = torch.matmul(copy_prob, encoder_outputs)

        if semantic_mask is not None:
            copy_prob.masked_fill(semantic_mask == 0, 0)

        gen_prob = self.p_vocab(decoder_outputs)

        # p_gen : shape [batch_size, tgt_len, 1]
        p_gen = self.sigmoid(self.dropout(self.w_h(h_t) + self.w_s(decoder_outputs) + self.w_x(tgt_emb)))
        p_copy = 1-p_gen

        gen_prob = gen_prob * p_gen
        gen_prob = F.pad(gen_prob, (0, len(extra_vocab)), 'constant')
        gen_prob.scatter_add_(2, expanded_word_idx.unsqueeze(1).expand(-1, gen_prob.size(1), -1),
                              copy_prob * p_copy)

        return torch.log(gen_prob + 1e-9)
