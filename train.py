from models import *
from utils import *
import torch.nn.functional as F
import argparse
from tqdm import tqdm


def valid(model, epoch, filename, config):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # data
    test_loader = data_load(filename, config.batch_size, False)
    all_loss = 0
    num = 0
    for step, batch in enumerate(tqdm(test_loader)):
        num += 1
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            result, _ = model.sample(x)
            if config.ls_flag:
                loss = LabelSmoothing(result, y, config)
            else:
                loss = compute_loss(result, y, config.vocab_size)
        all_loss += loss.item()
        # ##########################
        # if step == 2:
        #     break
        # ##########################
    print('epoch:', epoch, '|valid_loss: %.4f' % (all_loss / num))
    return all_loss / num


def test(model, epoch, tokenizer, config):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # data
    test_loader = data_load(config.filename_trimmed_test, config.batch_size, False)
    all_loss = 0
    num = 0
    result = []
    for step, batch in enumerate(tqdm(test_loader)):
        num += 1
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            out, idx = model.sample(x)
            if config.ls_flag:
                loss = LabelSmoothing(out, y, config)
            else:
                loss = compute_loss(out, y, config.vocab_size)
        all_loss += loss.item()

        for i in range(idx.shape[0]):
            temp = []
            for i in list(idx[i]):
                if i == config.eos:
                    break
                temp.append(i)
            # print(temp)
            sen = tokenizer.convert_ids_to_tokens(temp)
            result.append(' '.join(sen))
        # ##########################
        # if step == 2:
        #     break
        # ##########################
    print('epoch:', epoch, '|test_loss: %.4f' % (all_loss / num))

    # write result
    filename_data = config.filename_data + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))

    # rouge
    score = rouge_score(config.filename_gold, filename_data)

    # write rouge
    write_rouge(config.filename_rouge, score, epoch)

    # print rouge
    print('epoch:', epoch, '|ROUGE-1 f: %.4f' % score['rouge-1']['f'],
          ' p: %.4f' % score['rouge-1']['p'],
          ' r: %.4f' % score['rouge-1']['r'])
    print('epoch:', epoch, '|ROUGE-2 f: %.4f' % score['rouge-2']['f'],
          ' p: %.4f' % score['rouge-2']['p'],
          ' r: %.4f' % score['rouge-2']['r'])
    print('epoch:', epoch, '|ROUGE-L f: %.4f' % score['rouge-l']['f'],
          ' p: %.4f' % score['rouge-l']['p'],
          ' r: %.4f' % score['rouge-l']['r'])

    return score, all_loss / num


def train(model, args, config, tokenizer):
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    optim = Optim(optimizer, config)
    # optim = torch.optim.Adam(model.parameters(), lr=config.LR)

    # data
    train_loader = data_load(config.filename_trimmed_train, config.batch_size, True)

    # loss result
    train_loss = []
    valid_loss = []
    test_loss = []
    test_rouge = []

    if args.checkpoint != 0:
        model.load_state_dict(torch.load(config.filename_model + 'model_' + str(args.checkpoint) + '.pkl'))
    for e in range(args.checkpoint, args.epoch):
        model.train()
        all_loss = 0
        num = 0
        for step, batch in enumerate(tqdm(train_loader)):
            num += 1
            x, y = batch
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            result = model(x, y)
            if config.ls_flag:
                loss = LabelSmoothing(result, y, config)
            else:
                loss = compute_loss(result, y, config.vocab_size)
            if step % 200 == 0:
                print('epoch:', e, '|step:', step, '|train_loss: %.4f' % loss.item())

            # loss regularization
            loss = loss / config.accumulation_steps
            loss.backward()
            if ((step + 1) % config.accumulation_steps) == 0:
                optim.updata()
                optim.zero_grad()

            # optim.zero_grad()
            # loss.backward()
            # optim.updata()

            all_loss += loss.item()
            # ########################
            # if step == 2:
            #     break
            # ########################
        # train loss
        loss = all_loss / num
        print('epoch:', e, '|train_loss: %.4f' % loss)
        train_loss.append(loss)

        # valid
        loss_v = valid(model, e, config.filename_trimmed_valid, config)
        valid_loss.append(loss_v)

        # test
        rouge, loss_t = test(model, e, tokenizer, config)
        test_loss.append(loss_t)
        test_rouge.append(rouge)

        if args.save_model:
            filename = config.filename_model + 'model_' + str(e) + '.pkl'
            save_model(model, filename)


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size for train')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='number of training epochs')
    parser.add_argument('--n_layers', '-n', type=int, default=2, help='number of gru layers')
    parser.add_argument('-seed', '-s', type=int, default=123, help="Random seed")
    parser.add_argument('--save_model', '-m', action='store_true', default=False, help="whether to save model")
    parser.add_argument('--checkpoint', '-c', type=int, default=0, help="load model")
    args = parser.parse_args()

    ########test##########
    args.batch_size = 3
    #######test###########

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.n_layers:
        config.n_layers = args.n_layers

    # seed
    torch.manual_seed(args.seed)

    # rouge initalization
    open(config.filename_rouge, 'w')

    model = build_model(config)
    if torch.cuda.is_available():
        model = model.cuda()
    if isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    train(model, args, config, tokenizer)