import torch
import torch.nn as nn


class Decoder_LSTM(nn.Module):
    def __init__(self, embeds, attention, config):
        super().__init__()
        self.embeds = embeds
        self.attention = attention
        self.attn = config.attn_flag

        self.rnn = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.n_layer,
            batch_first=True,
            dropout=config.dropout,
        )

    def forward(self, x, h, encoder_outs):
        e = self.embeds(x).unsqueeze(1)
        # print(e.size())
        out, h = self.rnn(e, h)
        if self.attn:
            out = self.attention(out, encoder_outs)
        return out, h