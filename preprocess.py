import torch
from utils import *
from pytorch_pretrained_bert import BertTokenizer, BertModel


def main():
    config = Config()
    # load pre-trained Model to BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    datasets = Datasets(config)
    save_data(datasets.src_train, datasets.tgt_train, config.t_len, config.s_len, config.filename_trimmed_train, tokenizer)
    save_data(datasets.src_valid, datasets.tgt_valid, config.t_len, config.s_len, config.filename_trimmed_valid, tokenizer)
    save_data(datasets.src_test, datasets.tgt_test, config.t_len, config.s_len, config.filename_trimmed_test, tokenizer)


def test():
    config = Config()
    test = torch.load(config.filename_trimmed_test)
    src_id = test[0][0]
    tgt_id = test[0][1]
    # src_mask = test[0][2]
    #
    # tgt_id = test[0][3]
    # tgt_segment = test[0][4]
    # tgt_mask = test[0][5]
    print('src_id:', src_id)
    print('src_segment:', tgt_id)
    # print('src_mask:', src_mask)
    #
    # print('tgt_id:', tgt_id)
    # print('tgt_segment:', tgt_segment)
    # print('tgt_mask:', tgt_mask)


if __name__ == "__main__":
    # main()
    test()