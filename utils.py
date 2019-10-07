from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch
import torch.nn.functional as F


class WarmupLinearSchedule(LambdaLR):
    """ 
        学习率调整策略
        Args:
            optimizer : 需要调整的优化器
            warmup_steps : 从开始到warmup_steps步学习率线性上升
            t_total : 训练的总步数，从warmup_steps到训练结束学习率线性下降
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def convert2feature(input_sentences, input_tags, word2id, tag2id, max_len, extra_token_dict):
    """
        将输入的文本和标签转换为id
        Args:
            input_sentences (np.array): 文本
            input_tags (np.array): 标签
            word2id : 词字典
            tag2id : 标签字典
            max_len : 最大长度
            extra_token_dict : 额外标签字典
        Returns : 
            seqs_tensor (torch.Tensor): 转换为id的文本
            sentence_len_tensor (torch.Tensor): 句子长度
            tag_tensor (torch.Tensor): 转换为id的标签
    """

    unk_token = extra_token_dict["unk_token"]
    pad_token = extra_token_dict["pad_token"]
    pad_id = word2id[pad_token]
    unk_id = word2id[unk_token]

    sentence_len = []
    seqs_tensor = torch.ones(len(input_sentences), max_len).long() * pad_id
    for i, sentence in enumerate(input_sentences):
        sentence_len.append(len(sentence))
        for j, word in enumerate(sentence):
            seqs_tensor[i][j] = word2id.get(word, unk_id)
    
    pad_id = tag2id[pad_token]
    tag_tensor = torch.ones(len(input_sentences), max_len).long() * pad_id
    for i, tags in enumerate(input_tags):
        for j, tag in enumerate(tags):
            tag_tensor[i][j] = tag2id.get(tag)
    sentence_len_tensor = torch.tensor(sentence_len, dtype=torch.long)
    return seqs_tensor, sentence_len_tensor, tag_tensor


def entity_recover(tag_seqs, id2tag):
    """
        将tag_seqs中存在的实体位置及其对应的实体类型提取出来
        Args:
            tag_seqs : 预测的标签id
            id2tag : 标签字典
        Returns : 
            entity_list (List): 每一个元素为一个list，包含对应样本中的所有实体字典
    """
    entity_list = []
    for item in tag_seqs:
        temp_list = []
        entity_start = False
        for index, char_id in enumerate(item):
            char = id2tag[char_id]
            if char[0] == "B":
                start_id = index
                entity_type = char[2:]
                entity_start = True
                continue
            if entity_start:
                if char[0] != "I":
                    stop_id = index
                    entity_dict = {"start_id": start_id, 
                                    "stop_id": stop_id, 
                                    "entity_type": entity_type
                                    }
                    temp_list.append(entity_dict)
                    entity_start = False
        entity_list.append(temp_list)
    return entity_list


def metric(prediction, ground_truth):
    """ 
        计算实体预测结果的准确率、召回率和F1分数
        Args:
            prediction : 预测实体结果
            ground_truth : 标注实体结果
        Returns : 
            P, R, F1 : 准确率、召回率和F1分数
    """
    assert len(prediction) == len(ground_truth)
    pred_num = 0
    groud_num = 0
    for index in range(len(ground_truth)):
        pred_num = pred_num + len(prediction[index])
        groud_num = groud_num + len(ground_truth[index])
    acc_num = 0
    for index, item in enumerate(prediction):
        for entity in item:
            if entity in ground_truth[index]:
                acc_num += 1
    if pred_num == 0:
        P = 0
    else:
        P = acc_num / pred_num
    if groud_num == 0:
        R = 0
    else:
        R = acc_num / groud_num
    if P + R == 0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)            # 计算f1得分
    return P, R, F1


def cal_loss(logits, targets, pad_id):
    """
        分类任务计算交叉熵
        Args:
            logits : 预测结果 [batch, seq_len, tag_num]
            targets : 实际标签 [batch, seq_len]
            pad_id : pad对应的tag_id
    """
    mask = (targets != pad_id)  # [B, L]
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)
    return loss


def voting(results):
    """
        对交叉验证模型的结果进行投票
        Args:
            results : 预测结果 [cv_i, sample_num, seq_len]
        Returns:
            final_results : 投票结果 [sample_num, seq_len]
    """
    final_results = []
    for index in range(len(results[0])):
        final_seq_tag = []
        for char_index in range(len(results[0][index])):
            mention_list = []
            mention_count = []
            for i in range(len(results)):
                if results[i][index][char_index] not in mention_list:
                    mention_list.append(results[i][index][char_index])
                    mention_count.append(1)
                else:
                    mention_count[mention_list.index(results[i][index][char_index])] += 1
            final_char = mention_list[mention_count.index(max(mention_count))]
            final_seq_tag.append(final_char)
        final_results.append(final_seq_tag)
    return final_results