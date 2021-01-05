import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math

from module.attn import MultiHeadAttn, MultiHeadRelAttn, MultiHeadRelTreeAttn
from module.generator import PointerGenerator, Generator
from module.position_embedding import TreeRelativePosition, StandardRelativePosition
from utils import clones


class StandardEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, num_heads):
        super(StandardEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.num_heads = num_heads

    def forward(self, inputs):
        src, tgt, src_mask, tgt_mask, _ = inputs
        encoder_outputs = self.encode(src, src_mask, None)
        tgt_emb = self.tgt_embed(tgt)
        decoder_outputs, attn = self.decode(encoder_outputs, src_mask, tgt_emb, tgt_mask)
        return decoder_outputs, attn, encoder_outputs, tgt_emb

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt_emb, tgt_mask):
        return self.decoder(tgt_emb, memory, src_mask, tgt_mask)


class ASTEncoderDecoder(StandardEncoderDecoder):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, num_heads):
        super().__init__(encoder, decoder, src_embed, tgt_embed, generator, num_heads)

    def forward(self, inputs):
        src, tgt, src_mask, tgt_mask, rel_ids = inputs
        encoder_src_mask = self.gen_src_mask(rel_ids)
        encoder_outputs, _ = self.encode(src, encoder_src_mask, rel_ids)

        tgt_emb = self.tgt_embed(tgt)
        decoder_outputs, attn = self.decode(encoder_outputs, src_mask, tgt_emb, tgt_mask)
        return decoder_outputs, attn, encoder_outputs, tgt_emb

    def gen_src_mask(self, rel_ids):
        masks = []
        num_features = len(rel_ids)
        for rel_id in rel_ids:
            rel_id_mask = (rel_id == 0).unsqueeze(1).repeat(1, self.num_heads // num_features, 1, 1)
            masks.append(rel_id_mask)

        return torch.cat(masks, dim=1)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    def __init__(self, layer, N, relative_pos_emb):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.relative_pos_emb = relative_pos_emb

    def forward(self, x, mask, rel_ids):
        if rel_ids is None:
            rel_k_emb, rel_v_emb = self.relative_pos_emb(x.size(1))
        else:
            rel_k_emb, rel_v_emb = self.relative_pos_emb(rel_ids)
        for layer in self.layers:
            x = layer(x, mask, rel_k_emb, rel_v_emb)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, rel_k_emb, rel_v_emb):
        x, _ = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask, rel_k_emb, rel_v_emb))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N, relative_pos_emb):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.relative_pos_emb = relative_pos_emb

    def forward(self, x, memory, src_mask, tgt_mask):
        relative_k_emb, relative_v_emb = self.relative_pos_emb(x.size(1))
        for layer in self.layers:
            x, attn = layer(x, memory, src_mask, tgt_mask, relative_k_emb, relative_v_emb)
        return self.norm(x), attn


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, relative_k_emb, relative_v_emb):
        m = memory
        x, _ = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, relative_k_emb, relative_v_emb))
        x, attn = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward), attn


def make_model(src_vocab, tgt_vocab, N=2, d_model=300, d_ff=512, k=5, h=6,
               num_features=3, dropout=0.1, model_name=''):
    c = copy.deepcopy
    attn = MultiHeadAttn(d_model, h, dropout)
    rel_attn = MultiHeadRelAttn(d_model, h, dropout)
    tree_attn = MultiHeadRelTreeAttn(d_model, h, dropout)

    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    tree_rel_pos_emb = TreeRelativePosition(d_model // h, k, h, num_features, dropout)
    rel_pos_emb = StandardRelativePosition(d_model // h, k, dropout)

    decoder = Decoder(
        layer=DecoderLayer(d_model, c(rel_attn), c(attn), c(ff), dropout),
        N=N,
        rel_pos_emb=c(rel_pos_emb))

    if 'pointer' in model_name:
        generator = PointerGenerator(d_model, tgt_vocab, dropout)
    else:
        generator = Generator(d_model, tgt_vocab, dropout)

    if 'ast_transformer' in model_name:
        encoder = Encoder(
            layer=EncoderLayer(d_model, c(tree_attn), c(ff), dropout),
            N=N,
            rel_pos_emb=c(tree_rel_pos_emb)
        )
        model = ASTEncoderDecoder(
            encoder=encoder,
            decoder=decoder,
            src_embed=Embeddings(d_model, src_vocab),
            tgt_embed=Embeddings(d_model, tgt_vocab),
            generator=generator,
            num_heads=h
        )
    else:
        encoder = Encoder(
            layer=EncoderLayer(d_model, c(rel_attn), c(ff), dropout),
            N=N,
            rel_pos_emb=c(rel_pos_emb))
        model = StandardEncoderDecoder(
            encoder=encoder,
            decoder=decoder,
            src_embed=Embeddings(d_model, src_vocab),
            tgt_embed=Embeddings(d_model, tgt_vocab),
            generator=generator,
            num_heads=h
        )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model





