class Config():
    def __init__(self):
        # dataset

        # sentences segment datasets
        self.filename_src_train = 'DATA/raw_data/src-train.txt'
        self.filename_tgt_train = 'DATA/raw_data/tgt-train.txt'
        self.filename_src_valid = 'DATA/raw_data/src-valid.txt'
        self.filename_tgt_valid = 'DATA/raw_data/tgt-valid.txt'
        self.filename_src_test = 'DATA/raw_data/src-test.txt'
        self.filename_tgt_test = 'DATA/raw_data/tgt-test.txt'

        # trimmed data
        self.filename_trimmed_train = 'DATA/data/test.pt'
        self.filename_trimmed_valid = 'DATA/data/test.pt'
        self.filename_trimmed_test = 'DATA/data/test.pt'

        self.t_len = 150
        self.s_len = 50

        # bert
        self.bos = 101  # [CLS]
        self.eos = 102  # [SEP]
        self.pad = 0

        # filename
        #################################################
        self.filename_model = 'result/model/'
        self.filename_data = 'result/data/'
        self.filename_rouge = 'result/data/ROUGE.txt'
        #################################################
        self.filename_gold = 'result/gold/gold_summaries.txt'

        # Hyper Parameters
        self.LR = 0.003
        self.batch_size = 64
        self.iters = 10000
        self.model_size = 768
        self.n_head = 12
        self.d_ff = 1024
        self.accumulation_steps = 8
        self.warmup_steps = 4000
        self.ls_flag = True
        self.ls = 0.1

        self.n_layer = 12 # lstm or transformer
        self.attn_flag = True
        self.dropout = 0
        self.bidirectional = True

        # bert (word_share=True)
        self.bert = True
        self.fine_tuning = False
        self.vocab_size = 21128 # BertModel.config.vocab_size

        self.decoder = 'transformer'

        self.beam_size = 10


