class Config():
    def __init__(self):
        # dataset

        # sentences segment datasets
        self.filename_src_train = 'DATA/raw_data/clean/train.source'
        self.filename_tgt_train = 'DATA/raw_data/clean/train.target'
        self.filename_src_valid = 'DATA/raw_data/clean/valid.source'
        self.filename_tgt_valid = 'DATA/raw_data/clean/valid.target'
        self.filename_src_test = 'DATA/raw_data/clean/test.source'
        self.filename_tgt_test = 'DATA/raw_data/clean/test.target'

        # trimmed data
        self.filename_trimmed_train = 'DATA/data/clean/data_char/train.pt'
        self.filename_trimmed_valid = 'DATA/data/clean/data_char/valid.pt'
        self.filename_trimmed_test = 'DATA/data/clean/data_char/test.pt'

        self.t_len = 150
        self.s_len = 50

        # bert
        self.bos = 104 # <S>
        self.eos = 105 # <T>
        self.pad = 0
        self.cls = 101 # [CLS]
        self.sep = 102 # [SEP]

        # filename
        #################################################
        self.filename_model = 'result/model/'
        self.filename_data = 'result/data/'
        self.filename_rouge = 'result/data/ROUGE.txt'
        #################################################
        self.filename_gold = 'result/gold/gold_summaries.txt'

        # Hyper Parameters
        self.LR = 0.0003
        self.batch_size = 64
        self.iters = 10000
        self.model_size = 768
        self.n_head = 12
        self.d_ff = 2048
        self.accumulation_steps = 8
        self.warmup_steps = 4000
        self.ls_flag = True
        self.ls = 0.1

        self.n_layer = 12 # lstm or transformer
        self.attn_flag = True
        self.dropout = 0.3
        self.bidirectional = True

        # bert (word_share=True)
        self.bert = True
        self.fine_tuning = False
        self.vocab_size = 21128 # BertModel.config.vocab_size

        self.decoder = 'transformer'

        self.beam_size = 10


