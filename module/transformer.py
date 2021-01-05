import torch.nn as nn
import torch


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
        encoder_outputs, _ = self.encode(src, src_mask)
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
        if len(rel_ids) > 0:
            encoder_src_mask = self.gen_src_mask(rel_ids)
            encoder_outputs, _ = self.encode(src, encoder_src_mask, rel_ids)
        else:
            encoder_outputs, _ = self.encode(src, src_mask)
        encoder_outputs = self.encode(src, relative_par_ids, relative_bro_ids, semantic_ids, encoder_code_mask)
        decoder_outputs, decoder_attn = self.decode(encoder_outputs, code_mask, nl_embed, nl_mask)

    def gen_src_mask(self, rel_ids):
        masks = []
        num_features = len(rel_ids)
        for rel_id in rel_ids:
            rel_id_mask = (rel_id == 0).unsqueeze(1).repeat(1, self.num_heads // num_features, 1, 1)
            masks.append(rel_id_mask)

        return torch.cat(masks, dim=1)
