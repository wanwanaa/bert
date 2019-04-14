import torch
import torch.nn as nn
import numpy as np
from models import *


class Model(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos = config.bos
        self.s_len = config.s_len
        self.n_layer = config.n_layer
        self.fine_tune = config.fine_tuning

        self.linear_out = nn.Linear(config.hidden_size, config.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def convert(self, x):
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def output_layer(self, x):
        return self.linear_out(x)

    def forward(self, x, y):
        if torch.cuda.is_available():
            segment_ids = torch.ones(x.size(0), x.size(1)).type(torch.cuda.LongTensor)
        else:
            segment_ids = torch.ones(x.size(0), x.size(1)).type(torch.LongTensor)
        if self.fine_tune:
            encoder_outs, h = self.encoder(x, segment_ids)
        else:
            with torch.no_grad():
                encoder_outs, h = self.encoder(x, segment_ids)
        h = h.repeat(1, 2, 1)
        h = (h, h)

        y_c = self.convert(y)

        # decoder
        result = []
        for i in range(self.s_len):
            # print(outs.size())
            out, h = self.decoder(y_c[:, i], h, encoder_outs)
            gen = self.output_layer(out).squeeze()
            result.append(gen)
        outputs = torch.stack(result).transpose(0, 1)
        return outputs

    def sample(self, x):
        if torch.cuda.is_available():
            segment_ids = torch.ones(x.size(0), x.size(1)).type(torch.cuda.LongTensor)
        else:
            segment_ids = torch.ones(x.size(0), x.size(1)).type(torch.LongTensor)
            encoder_outs, h = self.encoder(x, segment_ids)
        h = h.repeat(1, 2, 1)
        h = (h, h)

        out = torch.ones(x.size(0)) * self.bos
        result = []
        idx = []

        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            out, h = self.decoder(out, h, encoder_outs)
            gen = self.linear_out(out.squeeze(1))
            result.append(gen)
            gen = self.softmax(gen)
            out = torch.argmax(gen, dim=1)
            idx.append(out.cpu().numpy())
        result = torch.stack(result).transpose(0, 1)
        idx = np.transpose(np.array(idx))
        return result, idx