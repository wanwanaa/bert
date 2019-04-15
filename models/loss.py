import torch
import torch.nn as nn
import torch.nn.functional as F


# implement label smoothing KL
def LabelSmoothing(out, y, config):
    # out (batch, len, vocab_size)
    # y (batch, len)
    criterion = nn.KLDivLoss(size_average=False)
    y = y.view(-1)
    word = y.ne(config.pad).sum().item()
    out = out.view(-1, config.vocab_size)

    true_dist = torch.zeros_like(out)
    true_dist.fill_(config.ls / (config.vocab_size-2))

    true_dist.scatter_(1, y.unsqueeze(1), (1-config.ls))
    true_dist[:, config.pad] = 0

    mask = torch.nonzero(y == config.pad)
    true_dist = true_dist.transpose(0, 1)
    true_dist.index_fill_(1, mask.squeeze(), 0.0)
    true_dist = true_dist.transpose(0, 1)
    out = torch.nn.functional.log_softmax(out, dim=-1)

    loss = criterion(out, true_dist)
    return loss/word


def compute_loss(result, y, vocab_size):
    result = result.contiguous().view(-1, vocab_size)
    y = y.contiguous().view(-1)
    loss = F.cross_entropy(result, y)
    return loss