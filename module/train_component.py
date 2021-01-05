import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import subsequent_mask


class Train(nn.Module):
    def __init__(self, model):
        super(Train, self).__init__()
        self.model = model

    def forward(self, inputs):
        src, tgt, src_mask, tgt_mask, rel_ids, extra_vocab, expanded_x, semantic_mask = inputs
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

    def forward(self, inputs):
        src, tgt, src_mask, tgt_mask, rel_ids, extra_vocab, expanded_x, semantic_mask = inputs
        batch_size = src.size(0)

        if len(rel_ids) == 0:
            memory = self.model.encode(src, src_mask, None)
        else:
            memory = self.model.encode(src, src_mask, rel_ids)
        ys = torch.ones(1, 1).fill_(self.start_pos).type_as(src.data)
        for i in range(self.max_tgt_len - 1):
            tgt_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
            tgt_mask = tgt_mask.unsqueeze(1)
            tgt_embed = self.model.tgt_embed(Variable(ys))
            decoder_outputs, decoder_attn = self.model.decode(memory, src_mask,
                                                              tgt_embed,
                                                              tgt_mask)

            prob = self.model.generator(memory, decoder_outputs[:, -1].unsqueeze(1),
                                        tgt_embed, extra_vocab, expanded_x, semantic_mask.unsqueeze(1))
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys, extra_vocab
