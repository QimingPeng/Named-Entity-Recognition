import torch
import bcolz
import logging
import numpy as np
from sklearn.model_selection import train_test_split

def load_iob2(data_path):
    """
        加载 IOB2 格式的数据
        Params:
            data_path (str): 文件路径

        Return:
            token_seqs (np.array): 每一个元素为单个的文本字符 [example_num, seqs_num]
            tags_seqs (np.array): 每一个元素为文本字符对应的标签 [example_num, seqs_num]
    """
    token_seqs = []
    tag_seqs = []
    tokens = []
    tags = []
    with open(data_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            items = line.strip().split()
            if len(items) == 2:
                token, tag = items
                tokens.append(token)
                tags.append(tag)
            elif len(items) == 0:
                if tokens:
                    token_seqs.append(tokens)
                    tag_seqs.append(tags)
                    tokens = []
                    tags = []
            else:
                logging.info('格式错误。行号：{} 内容：{}'.format(index, line))
                continue
                
    if tokens: # 如果文件末尾没有空行，手动将最后一条数据加入序列的列表中
        token_seqs.append(tokens)
        tag_seqs.append(tags)    
    token_seqs, tags_seqs = np.array(token_seqs), np.array(tag_seqs)
    return token_seqs, tags_seqs

def data_preprocess(data_path, extra_token_dict):
    """
        加载 IOB2 格式的数据，并划分数据集
        Params:
            data_path: 文件路径
            extra_token_dict: 额外需要添加的标签
        Return:
            train_seqs, test_seqs, train_tags, test_tags, tag2id, id2tag, max_len
            train_seqs (np.array): 训练集与验证集的文本 [train_dev_example_num, seqs_num]
            test_seqs (np.array): 测试集本文 [test_example_num, seqs_num]
            train_tags (np.array): 训练集与验证集的标签 [train_dev_example_num, seqs_num]
            test_tags (np.array): 测试集标签 [test_example_num, seqs_num]
            tag2id, id2tag (dict): 构造标签id字典
            max_len (np.array): 文本中最长文本长度
    """
    token_seqs, tags_seqs = load_iob2(data_path)
    example_len = len(tags_seqs)
    logging.info('Total number of samples: {}'.format(example_len))

    max_len = 0
    max_len = max([len(seq) for seq in token_seqs])

    logging.info("The max_len of the seqs：{}".format(max_len))

    train_seqs, test_seqs, train_tags, test_tags = train_test_split(token_seqs, tags_seqs, test_size=0.1)

    tag2id = {}
    for tags in tags_seqs:
        for tag in tags:
            if tag not in tag2id:
                tag2id[tag] = len(tag2id)

    for token in extra_token_dict.values():
        if token not in tag2id:
            tag2id[token] = len(tag2id)
    id2tag = dict(zip(tag2id.values(), tag2id.keys()))

    return train_seqs, test_seqs, train_tags, test_tags, tag2id, id2tag, max_len

def load_embeddings(embed_path, extra_token_dict):
    """
        从 bcolz 加载 词/字 向量，并构造相应的词典
        Args:
            embed_path (str): 解压后的 bcolz rootdir（如 zh.64），
                                里面包含 2 个子目录 embeddings 和 words，
                                分别存储 嵌入向量 和 词（字）典
        Returns:
            word2id, id2word (dict): 构造词id字典
            embeddings (torch.Tensor): 嵌入矩阵，每 1 行为 1 个 词向量/字向量，
                                       其行号对应在 word2id 中的value
    """
    embed_path = embed_path.rstrip('/')
    # 词（字）典列表（bcolz carray具有和 numpy array 类似的接口）
    words = bcolz.carray(rootdir='%s/words'%embed_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings'%embed_path, mode='r')

    words = list(words)
    embeddings = torch.tensor(embeddings, dtype=torch.float)

    embed_dim = embeddings.size()[1]

    unk_token = extra_token_dict["unk_token"]
    pad_token = extra_token_dict["pad_token"]

    if unk_token not in words:
        words.append(unk_token)
        unk_tensor = torch.randn(1, embed_dim)
        embeddings = torch.cat((embeddings, unk_tensor), dim=0)
        print(unk_tensor)
    if pad_token not in words:
        words.append(pad_token)
        pad_tensor = torch.zeros(1, embed_dim)
        embeddings = torch.cat((embeddings, pad_tensor), dim=0)

    word2id = {}
    for word in words:
        word2id[word] = len(word2id)
    id2word = dict(zip(word2id.values(),  word2id.keys()))
    return word2id, id2word, embeddings


def load_data(raw_path, embedding_path, extra_token_dict):
    logging.info("Loading data...")
    logging.info("Split train test data...")
    train_seqs, test_seqs, train_tags, test_tags, tag2id, id2tag, max_len = data_preprocess(raw_path, extra_token_dict)
    logging.info("Loading embeddings...")
    word2id, id2word, embeddings = load_embeddings(embedding_path, extra_token_dict)
    return train_seqs, test_seqs, train_tags, test_tags, word2id, id2word, tag2id, id2tag, max_len, embeddings

