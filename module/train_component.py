import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import subsequent_mask


class Train(nn.Module):
    def __init__(self, model):
        super(Train, self).__init__()
        self.model = model

    def forward(self, batch_data):
        src = batch_data['src']
        tgt = batch_data['tgt']
        src_mask = batch_data['src_mask']
        tgt_mask = batch_data['tgt_mask']
        rel_ids = batch_data['rel_ids']
        extra_vocab = batch_data['extra_vocab']
        expanded_x = batch_data['expanded_x']
        semantic_mask = batch_data['semantic_mask']

        decoder_outputs, attn, encoder_outputs, tgt_emb = self.model.forward(
            (src, tgt, src_mask, tgt_mask, rel_ids)
        )
        out = self.model.generator((encoder_outputs, decoder_outputs,
                                   attn, tgt_emb,
                                   extra_vocab, expanded_x, semantic_mask))
        return out


class GreedyEvaluate(nn.Module):
    def __init__(self, model, max_tgt_len, start_pos):
        super(GreedyEvaluate, self).__init__()
        self.model = model
        self.max_tgt_len = max_tgt_len
        self.start_pos = start_pos

    def forward(self, batch_data):
        src = batch_data['src']
        src_mask = batch_data['src_mask']
        rel_ids = batch_data['rel_ids']
        extra_vocab = batch_data['extra_vocab']
        expanded_x = batch_data['expanded_x']
        semantic_mask = batch_data['semantic_mask']
        batch_size = src.size(0)

        memory = self.model.encode(src, src_mask, rel_ids)

        ys = torch.ones(batch_size, 1).fill_(self.start_pos).type_as(src.data)
        for i in range(self.max_tgt_len - 1):
            tgt_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
            tgt_mask = tgt_mask.unsqueeze(1)
            tgt_embed = self.model.tgt_embed(Variable(ys))
            decoder_outputs, decoder_attn = self.model.decode(memory, src_mask,
                                                              tgt_embed,
                                                              tgt_mask)

            prob = self.model.generator((memory, decoder_outputs[:, -1].unsqueeze(1),
                                         decoder_attn[:, :, -1].unsqueeze(2),
                                         tgt_embed[:, -1].unsqueeze(1),
                                         extra_vocab, expanded_x, semantic_mask))
            prob = prob.squeeze(1)
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys,
                            next_word.unsqueeze(1).type_as(src.data)], dim=1)
        return ys, extra_vocab


class LabelSmoothing(nn.Module):
    def __init__(self, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, x, target):
        x = x.contiguous().view(-1, x.size(-1))

        ntokens = (target != 0).data.sum()
        target = target.contiguous().view(-1)
        vocab_size = x.size(1)

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False)) / ntokens
