import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Bert_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.t_len = config.t_len
        self.pad = config.pad
        self.flag = config.decoder

    def forward(self, ids, segment_ids):
        encoded_layers, _ = self.model(ids, segment_ids)
        encoded_layers = encoded_layers[-1] # (batch, len+1, 768)
        encoded_layers = encoded_layers[:, 1:, :]
        if self.flag == 'lstm':
            code = encoded_layers[:, 0, :]
            if torch.cuda.is_available():
                mask = ids.ne(self.pad).type(torch.cuda.FloatTensor)
            else:
                mask = ids.ne(self.pad).type(torch.FloatTensor)
            for i in range(1, self.t_len):
                code += encoded_layers[:, i, :] * mask[:, i].unsqueeze(1)
            code = code.unsqueeze(0)
        else:
            code = None
        return encoded_layers, code
