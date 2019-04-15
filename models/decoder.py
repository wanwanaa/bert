import torch
import torch.nn as nn
from models import *
from models.transformer_helper import *
from models.suberlayer import *


class Decoder_LSTM(nn.Module):
    def __init__(self, attention, config):
        super().__init__()
        self.attention = attention
        self.attn = config.attn_flag

        self.rnn = nn.LSTM(
            input_size=config.model_size,
            hidden_size=config.model_size,
            num_layers=config.n_layer,
            batch_first=True,
            dropout=config.dropout,
        )

    def forward(self, x, h, encoder_outs):
        # print(e.size())
        out, h = self.rnn(x, h)
        if self.attn:
            out = self.attention(out, encoder_outs)
        return out, h


class Decoder_Transformer_Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dec_attention = MultiHeadAttention(config)
        self.enc_dec_attention = MultiHeadAttention(config)
        self.feedward = Posfeedward(config)

    def forward(self, dec_input, enc_output, non_pad_mask=None, attn_self_mask=None, enc_dec_attn_mask=None):
        # print('decoder:', dec_input.size())
        dec_output, dec_self_w = self.dec_attention(dec_input, dec_input, dec_input, mask=attn_self_mask)
        dec_output = dec_output * non_pad_mask.type(torch.float)

        dec_output, dec_w = self.enc_dec_attention(dec_output, enc_output, enc_output, mask=enc_dec_attn_mask)
        dec_output = dec_output * non_pad_mask.type(torch.float)

        dec_output = self.feedward(dec_output)
        dec_output = dec_output * non_pad_mask.type(torch.float)

        return dec_output


class Decoder_Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad = config.pad

        self.embedding = nn.Embedding(config.vocab_size, config.model_size)

        self.position_dec = nn.Embedding.from_pretrained(
            positional_encoding(config.s_len + 1, config.model_size, config.pad), freeze=True
        )

        self.decoder_stack = nn.ModuleList([
            Decoder_Transformer_Layer(config) for _ in range(config.n_layer)
        ])

    def forward(self, x, y, pos, enc_output):
        no_pad_mask = get_non_pad_mask(y, self.pad)
        attn_mask = get_dec_mask(y)
        pad_mask = get_pad_mask(y, y, self.pad)
        attn_self_mask = (pad_mask + attn_mask).gt(0)
        enc_dec_attn_mask = get_pad_mask(x, y, self.pad)

        dec_output = self.embedding(y) + self.position_dec(pos)
        for layer in self.decoder_stack:
            dec_output = layer(dec_output, enc_output, no_pad_mask, attn_self_mask, enc_dec_attn_mask)

        return dec_output


