from models import *
from models.attention import *
import torch


def build_model(config):
    embeds = Embeds(config, config.vocab_size)
    encoder = Bert_Encoder(config)
    if config.decoder == 'lstm':
        if config.attn_flag:
            attention = Luong_Attention(config)
        else:
            attention = None
        decoder = Decoder_LSTM(attention, config)
        model = Model_Lstm(embeds, encoder, decoder, config)
    else:
        decoder = Decoder_Transformer(config)
        model = Model_Transformer(embeds, encoder, decoder, config)
    return model


def load_model(config, filename):
    model = build_model(config)
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)