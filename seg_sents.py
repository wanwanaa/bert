import re

filenames = ['DATA/raw_data/src-train.txt', 'DATA/raw_data/tgt-train.txt',
             'DATA/raw_data/src-valid.txt', 'DATA/raw_data/tgt-valid.txt',
             'DATA/raw_data/src-test.txt', 'DATA/raw_data/tgt-test.txt']
filenames_segment = ['DATA/data/src-train.txt', 'DATA/data/tgt-train.txt',
                     'DATA/data/src-valid.txt', 'DATA/data/tgt-valid.txt',
                     'DATA/data/src-test.txt', 'DATA/data/tgt-test.txt']
for i in range(len(filenames)):
    result = []
    with open(filenames[i], 'r', encoding='utf-8') as f:
        for line in f:
            # sentences segmentation
            sentences = re.split('(。”|。|？”|？|！”|！|\\n)', line)
            new_sents = []
            for j in range(int(len(sentences)/2)):
                sent = sentences[2*j] + sentences[2*j+1]
                new_sents.append(sent)
            if new_sents[-1] != '\n':
                temp = new_sents[-1]
                # print('temp1:', temp)
                temp = list(temp)
                temp[-1] = ' [SEP]\n'
                temp = ''.join(temp)
                new_sents[-1] = temp
                # print('temp2:', new_sents[-1])
            result.append('[CLS] ' + ' [SEP] '.join(new_sents))
    with open(filenames_segment[i], 'w', encoding='utf-8') as f:
        f.write(''.join(result))