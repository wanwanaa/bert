import torch
import math


def get_non_pad_mask(seq, pad):
    # (batch, len)
    mask = seq.ne(pad)
    # (batch, len, 1)
    mask = mask.unsqueeze(-1)
    return mask


def get_pad_mask(seq_k, seq_q, pad):
    len_q = seq_q.size(1)
    mask = seq_k.eq(pad)
    mask = mask.unsqueeze(1).expand(-1, len_q, -1)
    return mask


def get_attn_pad_mask(seq, pad):
    # (batch, len)
    len = seq.size(1)
    pad_mask = seq.eq(pad)
    # (batch, len, len)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len, -1)
    return pad_mask


# decoder self-attention
def get_dec_mask(seq):
    batch, len = seq.size()
    # (len, len)
    mask = torch.triu(
        torch.ones((len, len), dtype=torch.uint8), diagonal=1
    )
    if torch.cuda.is_available():
        mask = mask.type(torch.cuda.ByteTensor)
    # (batch, len, len)
    mask = mask.unsqueeze(0).expand(batch, -1, -1)
    return mask


def positional_encoding(len, model_size, pad):
    pe = torch.zeros(len, model_size)
    position = torch.arange(0., len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., model_size, 2) *
                         (-math.log(10000.0) / model_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe[pad] = 0.

    return pe