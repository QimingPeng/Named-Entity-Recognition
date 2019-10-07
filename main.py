import argparse
import random
import torch
import logging
import os

import numpy as np

from itertools import cycle
from sklearn.model_selection import KFold
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from data_process import load_data
from model.BiLSTM_CRF import BiLSTM_CRF
from model.BiLSTM_Model import BiLSTM_Model
from utils import convert2feature, entity_recover, voting, metric
from train_test import train_eval, test


def init_argparse():
    parser = argparse.ArgumentParser(description='Tuning with NER')
    parser.add_argument('-data_path', default="./data/dh_msra.txt")
    parser.add_argument('-embedding_path',  help='Embedding for words', default='./data/zh_char.64/')

    parser.add_argument('-do_train', type=bool, default=True)
    parser.add_argument('-do_test', type=bool, default=True)
    parser.add_argument('-do_cv', type=bool, default=True)

    parser.add_argument('-model_name', default="BiLSTM", type=str)

    parser.add_argument('-seed', default=2019, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-dev_batch_size', default=32, type=int)
    parser.add_argument('-train_steps', default=50000, type=int)
    parser.add_argument('-check_step', default=200, type=int)
    parser.add_argument('-eval_step', default=1000, type=int)

    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-warmup_steps', default=0, type=int)

    parser.add_argument('-lstm_hidden_size', default=128, type=int)
    parser.add_argument('-lstm_num_layers', default=2, type=int)

    parser.add_argument('-ckpt_path', default="./ckpts/")
    return parser.parse_args()

def set_seed(seed, n_gpu):
    """
        设置随机种子
        Params:
            seed: 种子
        Return:
            output_list: 返回一个list，txt文件的一行为list的一个元素
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main(args):
    # 可选择输出日志到文件
    # logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
    #                     filename=args.model_name + '.log',
    #                     filemode='w',
    #                     level=logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    set_seed(args.seed, n_gpu)

    logging.info("The active device is: {}, gpu_num: {}".format(device, n_gpu))

    args.ckpt_path = os.path.join(args.ckpt_path, args.model_name)
    try:
        os.makedirs(args.ckpt_path)
    except:
        pass
    extra_token_dict = {"unk_token": "<UNK>", "pad_token": "<PAD>"}

    train_dev_seqs, test_seqs, train_dev_tags, test_tags, word2id, id2word, tag2id, id2tag, max_len, embeddings \
        = load_data(args.data_path, args.embedding_path, extra_token_dict)
    
    num_tag = len(tag2id)
    pad_tag_id = tag2id[extra_token_dict["pad_token"]]
    vocab_size = len(word2id)
    embed_dim = embeddings.size(1)
    
    f1_list = [0] * 5
    if args.do_train:
        kf = KFold(n_splits=5, shuffle=True).split(train_dev_seqs)
        for cv_i, (train_index, dev_index) in enumerate(kf):
            logging.info("******************Train CV_{}******************".format(cv_i))
            # 准备模型
            if args.model_name == "BiLSTM-CRF":
                model = BiLSTM_CRF(vocab_size, embed_dim, args.lstm_hidden_size, args.lstm_num_layers, num_tag, pad_tag_id, pretrain_embedding=embeddings)
            if args.model_name == "BiLSTM":
                model = BiLSTM_Model(vocab_size, embed_dim, args.lstm_hidden_size, args.lstm_num_layers, num_tag, pad_tag_id, pretrain_embedding=embeddings)
            logging.info("Already load the model: {},".format(args.model_name))
            model.to(device)

            train_sentences = [train_dev_seqs[i] for i in train_index]
            train_tags = [train_dev_tags[i] for i in train_index]
            dev_sentences = [train_dev_seqs[i] for i in dev_index]
            dev_tags = [train_dev_tags[i] for i in dev_index]

            logging.info("Prepare dataloader...")
            train_tensor, train_sent_len, train_tags_tensor = convert2feature(train_sentences, train_tags, word2id, tag2id, max_len, extra_token_dict)
            train_data = TensorDataset(train_tensor, train_sent_len, train_tags_tensor)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
            train_dataloader=cycle(train_dataloader) 

            dev_tensor, dev_sent_len, dev_tags_tensor = convert2feature(dev_sentences, dev_tags, word2id, tag2id, max_len, extra_token_dict)            
            dev_data = TensorDataset(dev_tensor, dev_sent_len, dev_tags_tensor)
            dev_sampler = SequentialSampler(dev_data)
            dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)
            dev_entity_list = entity_recover(dev_tags_tensor.numpy(), id2tag)

            logging.info("Begin to train...")
            f1_list[cv_i] = train_eval(train_dataloader, dev_dataloader, dev_entity_list, model, id2tag, 
                                        args.ckpt_path, args.train_steps, args.check_step, args.eval_step, args.lr, args.warmup_steps, cv_i)
            if not args.do_cv:
                break
        if args.do_cv:
            cv_f1 = np.mean(np.array(f1_list))
            logging.info("CV F1_list: {}, Mean_F1: {:.4f}\n".format(f1_list, cv_f1))
        
    if args.do_test:
        logging.info("******************Test******************")
        logging.info("Begin to test {}...".format(args.model_name))
        test_tensor, test_sent_len, test_tags_tensor = convert2feature(test_seqs, test_tags, word2id, tag2id, max_len, extra_token_dict)
        test_data = TensorDataset(test_tensor, test_sent_len, test_tags_tensor)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.dev_batch_size)
        test_entity_list = entity_recover(test_tags_tensor.numpy(), id2tag)

        all_output_ids = []
        for cv_i in range(5):
            ckpt_path = os.path.join(args.ckpt_path, "pytorch_model_{}.pkl".format(cv_i))
            if args.model_name == "BiLSTM-CRF":
                model = BiLSTM_CRF(vocab_size, embed_dim, args.lstm_hidden_size, args.lstm_num_layers, num_tag, pad_tag_id, pretrain_embedding=embeddings)
                model.load_state_dict(torch.load(ckpt_path))
                model.to(device)
            if args.model_name == "BiLSTM":
                model = BiLSTM_Model(vocab_size, embed_dim, args.lstm_hidden_size, args.lstm_num_layers, num_tag, pad_tag_id, pretrain_embedding=embeddings)
                model.load_state_dict(torch.load(ckpt_path))
                model.to(device)
            output_tag_ids, P, R, F1 = test(test_dataloader, test_entity_list, id2tag, model)
            all_output_ids.append(output_tag_ids)
            logging.info("The cv_{} result of {} on test data: P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(cv_i, args.model_name, P, R, F1))

            if not args.do_cv:
                break
        if args.do_cv:
            final_result = voting(all_output_ids)
            final_entity = entity_recover(final_result, id2tag)
            P, R, F1 = metric(final_entity, test_entity_list)
            logging.info("The voting result of {} on test data: P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(args.model_name, P, R, F1))
        else:
            final_result = all_output_ids[0]
            final_entity = entity_recover(final_result, id2tag)
            P, R, F1 = metric(final_entity, test_entity_list)
            logging.info("The result of {} on test data: P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(args.model_name, P, R, F1))

        for i in range(5):
            print(test_seqs[i])
            print(test_tags[i])
            char_list = [id2tag[_id] for _id in final_result[i]]
            print(char_list)
            print(" ")

if __name__ == "__main__":
    args = init_argparse()
    main(args)
