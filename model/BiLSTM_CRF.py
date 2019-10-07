import torch
import torch.nn as nn
from model.basic.BiLSTM import BiLSTM
from model.basic.NewCRF import CRF

class BiLSTM_CRF(nn.Module):
    """
        BiLSTM-CRF的实现
        Args:
            vocab_size (int): 词典大小
            embed_dim (int): 词向量的维度
            hidden_size (int): lstm隐藏层维度
            num_layers (int): lstm隐藏层的层数
            num_tag (int): 标签个数
            pad_tag_id (int): <PAD>在tag2id中的id
            pretrain_embedding (torch.Tensor): 已加载的词向量 [vocab_size, embed_dim]
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_tag, pad_tag_id, pretrain_embedding=None):
        super(BiLSTM_CRF, self).__init__()
        if pretrain_embedding is None:
            self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embedding_layer = nn.Embedding.from_pretrained(pretrain_embedding)

        self.bilstm = BiLSTM(embed_dim, hidden_size, num_layers)
        self.cls_dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(2*hidden_size, num_tag)
        self.crf = CRF(num_tag, pad_tag_id)
    
    def forward(self, sents_tensor, lengths, target_tensor):
        """
            前向传播，计算loss得分
            Args:
                sents_tensor (torch.Tensor): 输入的训练样本 [batch_size, max_seq_len, embed_dim]
                lengths (torch.Tensor): batch中每个句子pad之前的长度 [batch_size]
                target_tensor (torch.Tensor): 标签 [batch_size, max_seq_len]
            Returns:
                -log_likelihood : 对数似然，因此取负数作为损失
        """
        seq_embed = self.embedding_layer(sents_tensor)
        lstm_out, _ = self.bilstm(seq_embed, lengths)
        lstm_out = self.cls_dropout(lstm_out)
        lstm_feats = self.classifier(lstm_out)
        log_likelihood = self.crf(lstm_feats, target_tensor)
        return -log_likelihood

    def sequence_decode(self, sents_tensor, lengths):
        """
            对文本进行解码
            Args:
                sents_tensor (torch.Tensor): 输入的训练样本 [batch_size, max_seq_len, embed_dim]
                lengths (torch.Tensor): batch中每个句子pad之前的长度 [batch_size]
            Returns:
                -tag_seq (List): 解码序列
        """
        seq_embed = self.embedding_layer(sents_tensor)
        lstm_out, _ = self.bilstm(seq_embed, lengths)
        lstm_out = self.cls_dropout(lstm_out)
        lstm_feats = self.classifier(lstm_out)
        tag_seq = self.crf.decode(lstm_feats, lengths)
        return tag_seq