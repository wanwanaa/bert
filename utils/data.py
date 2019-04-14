import torch
import torch.utils.data as data_util
import numpy as np


class Datasets():
    def __init__(self, config):
        self.src_train = self._get_datasets(config.filename_src_train)
        self.tgt_train = self._get_datasets(config.filename_tgt_train)
        self.src_valid = self._get_datasets(config.filename_src_valid)
        self.tgt_valid = self._get_datasets(config.filename_tgt_valid)
        self.src_test = self._get_datasets(config.filename_src_test)
        self.tgt_test = self._get_datasets(config.filename_tgt_test)

    def _get_datasets(self, filename):
        dataset = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(line)
        return dataset


def get_trimmed_datasets(datasets, max_length, tokenizer, src):
    data_ids = np.zeros([len(datasets), max_length])
    segment_ids = np.zeros([len(datasets), max_length])
    masks = np.zeros([len(datasets), max_length])
    for i, line in enumerate(datasets):
        # # segment ids
        # temp = ''.join(line)
        # s = temp.split('[SEP')
        # segment_id = []
        # for j in range(len(s) - 1):
        #     temp = s[j].strip()
        #     segment_id += [j] * (len(temp) + 1)
        # if len(segment_id) <= max_length:
        #     segment_id = np.pad(np.array(segment_id), (0, max_length - len(segment_id)), 'constant')
        # else:
        #     segment_id = segment_id[:max_length]
        # segment_ids[i] = segment_id

        # word to index
        if src:
            line = '[CLS] ' + line + ' [SEP]'
        else:
            line = line + ' [SEP]'
        line = tokenizer.tokenize(line)
        line = tokenizer.convert_tokens_to_ids(line)
        # #########################
        # # mask
        # mask = np.ones(len(line))
        # if len(line) <= max_length:
        #     mask = np.pad(mask, (0, max_length-len(line)), 'constant')
        # else:
        #     mask = mask[:max_length]
        # masks[i] = mask
        # ##########################
        # input ids
        if len(line) <= max_length:
            line = np.pad(np.array(line), (0, max_length-len(line)), 'constant')
        else:
            line = line[:max_length]
        data_ids[i] = line

    data_ids = torch.from_numpy(data_ids).type(torch.LongTensor)
    segment_ids = torch.from_numpy(segment_ids).type(torch.LongTensor)
    masks = torch.from_numpy(masks).type(torch.LongTensor)
    return data_ids, segment_ids, masks


def save_data(src_datasets, tgt_datasets, t_len, s_len, filename, tokenizer):
    src_ids, src_segment_ids, src_masks = get_trimmed_datasets(src_datasets, t_len, tokenizer, src=True)
    tgt_ids, tgt_segment_ids, tgt_masks = get_trimmed_datasets(tgt_datasets, s_len, tokenizer, src=False)
    # data = data_util.TensorDataset(src_ids, src_segment_ids, src_masks, tgt_ids, tgt_segment_ids, tgt_masks)
    data = data_util.TensorDataset(src_ids, tgt_ids)
    torch.save(data, filename)
    print('data save at: ', filename)


def data_load(filename, batch_size, shuffle):
    data = torch.load(filename)
    data_loader = data_util.DataLoader(data, batch_size, shuffle=shuffle, num_workers=2)
    return data_loader