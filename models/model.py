import torch
import torch.nn as nn
import numpy as np
from models import *


class Model_Transformer(nn.Module):
    def __init__(self, embeds, encoder, decoder, config):
        super().__init__()
        self.model_size = config.model_size
        self.vocab_size = config.vocab_size
        self.s_len = config.s_len
        self.bos = config.bos
        self.pad = config.pad
        self.fine_tune = config.fine_tuning

        self.embeds = embeds
        self.encoder = encoder
        self.decoder = decoder

        self.linear_out = nn.Linear(self.model_size, self.vocab_size)

    # add <bos> to sentence
    def convert(self, x):
        """
        :param x:(batch, s_len) (word_1, word_2, ... , word_n)
        :return:(batch, s_len) (<bos>, word_1, ... , word_n-1)
        """
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def sample(self, x):
        if torch.cuda.is_available():
            segment_ids = torch.ones(x.size(0), x.size(1)).type(torch.cuda.LongTensor)
        else:
            segment_ids = torch.ones(x.size(0), x.size(1)).type(torch.LongTensor)
        if self.fine_tune:
            encoder_outs, _ = self.encoder(x, segment_ids)
        else:
            with torch.no_grad():
                encoder_outs, _ = self.encoder(x, segment_ids)

        y_pos = torch.arange(1, self.s_len + 1).repeat(x.size(0), 1)
        out = torch.ones(x.size(0)) * self.bos
        out = out.unsqueeze(1)
        result = []
        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            dec_output = self.decoder(x, out, y_pos[:, :i + 1], encoder_outs)
            gen = self.linear_out(dec_output[:, -1, :])
            gen = torch.nn.functional.softmax(gen, -1)
            result.append(gen)
            gen = torch.argmax(gen, dim=1).unsqueeze(1)
            out = torch.cat((out, gen), dim=1)
            # mask = gen.eq(0).squeeze()
            # if i < self.s_len - 1:
            #     y_pos[:, i + 1] = y_pos[:, i + 1].masked_fill(mask, 0)

        result = torch.stack(result)
        out = out.cpu().numpy()
        return result, out

    def forward(self, x, y):
        if torch.cuda.is_available():
            segment_ids = torch.ones(x.size(0), x.size(1)).type(torch.cuda.LongTensor)
        else:
            segment_ids = torch.ones(x.size(0), x.size(1)).type(torch.LongTensor)
        if self.fine_tune:
            encoder_outs, _ = self.encoder(x, segment_ids)
        else:
            with torch.no_grad():
                encoder_outs, _ = self.encoder(x, segment_ids)

        y_c = self.convert(y)
        pos = torch.arange(1, self.s_len + 1).repeat(y.size(0), 1)
        dec_output = self.decoder(x, y_c, pos, encoder_outs)
        dec_output = self.linear_out(dec_output)
        return dec_output


class Model_Lstm(nn.Module):
    def __init__(self, embeds, encoder, decoder, config):
        super().__init__()
        self.embeds = embeds
        self.encoder = encoder
        self.decoder = decoder
        self.bos = config.bos
        self.s_len = config.s_len
        self.n_layer = config.n_layer
        self.fine_tune = config.fine_tuning

        self.linear_out = nn.Linear(config.model_size, config.vocab_size)
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
        h = h.repeat(2, 1, 1)
        h = (h, h)

        y_c = self.convert(y)

        # decoder
        result = []
        for i in range(self.s_len):
            # print(outs.size())
            x = self.embeds(y_c[:, i]).unsqueeze(1)
            out, h = self.decoder(x, h, encoder_outs)
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
        h = h.repeat(2, 1, 1)
        h = (h, h)

        out = torch.ones(x.size(0)) * self.bos
        result = []
        idx = []

        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            x = self.embeds(out).unsqueeze(1)
            out, h = self.decoder(x, h, encoder_outs)
            gen = self.linear_out(out.squeeze(1))
            result.append(gen)
            gen = self.softmax(gen)
            out = torch.argmax(gen, dim=1)
            idx.append(out.cpu().numpy())
        result = torch.stack(result).transpose(0, 1)
        idx = np.transpose(np.array(idx))
        return result, idx