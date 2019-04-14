import torch
import torch.nn as nn


class Luong_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_layer = config.n_layer

        self.linear_in = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.linear_out = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, output, encoder_out):
        """
        :param output: (batch, 1, hidden_size) decoder output
        :param encoder_out: (batch, t_len, hidden_size) encoder hidden state
        :return: attn_weight (batch, 1, time_step)
                  output (batch, 1, hidden_size) attention vector
        """
        out = self.linear_in(output) # (batch, 1, hidden_size)
        out = out.transpose(1, 2) # (batch, hidden_size, 1)
        attn_weights = torch.bmm(encoder_out, out) # (batch, t_len, 1)
        attn_weights = self.softmax(attn_weights.transpose(1, 2)) # (batch, 1, t_len)

        context = torch.bmm(attn_weights, encoder_out) # (batch, 1, hidden_size)
        output = self.linear_out(torch.cat((output, context), dim=2))

        return output