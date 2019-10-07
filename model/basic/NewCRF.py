from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """
        条件随机场的实现
        Args:
            num_tags (int): 标签个数
            batch_first (Boolean): 第一个维度是否为batch_size
            start_transitions (torch.nn.Parameter): <START>转移到其他标签的分数 [num_tags]
            end_transitions (torch.nn.Parameter): 其他标签转移到<STOP>的分数 [num_tags]
            transitions (torch.nn.Parameter): 非规范化状态转移矩阵 [num_tags, num_tags]
    """

    def __init__(self, num_tags, pad_tag_id, batch_first=True):
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()

        self.pad_id = pad_tag_id
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """
            对转移矩阵进行初始化
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions, tags):
        """
            给定发射概率的前提下，计算标签的log_likelihood.
            Args:
                emissions (torch.Tensor): 发射概率（分数）[batch_size, seq_length, num_tags]
                tags (torch.LongTensor): 标签序列 [batch_size, seq_length] 
            Returns:
                log_likelihood (torch.Tensor): 对数似然，为一个负数，若需要将其作为误差应该对其取绝对值
        """
        self._validate(emissions, tags=tags)
        mask = (tags != self.pad_id)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        # 求标准分数与实际分数的误差
        log_likelihood = numerator - denominator

        return log_likelihood.mean()


    def _validate(self, emissions, tags=None):
        """
            对输入的发射概率和标签格式进行验证
            Args:
                emissions (torch.Tensor): 发射概率（分数）[batch_size, seq_length, num_tags]
                tags (torch.LongTensor): 标签序列 [batch_size, seq_length] 
        """
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

    def _compute_score(self, emissions, tags, mask):
        """
            计算对于已有的发射概率和转移矩阵输出相应的tags的分数
            Args:
                emissions (torch.Tensor): 发射概率（分数）[seq_length, batch_size, num_tags]
                tags (torch.LongTensor): 标签序列 [seq_length, batch_size] 
                mask (torch.LongTensor): size与tag位置相同，pad位置为0，其余为1 [seq_length, batch_size]
            Returns:
                score (torch.Tensor): 输出分数
        """

        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # 计算<START>到第一个第一个标签的分数，每一步转移的得分均为 transition_score + emission_score
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # 仅当mask=1时加上转移分数和发射分数
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]   # [batch_size]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # 计算最后一个标签到<STOP>的分数
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        """
            计算对于已有的发射概率和转移矩阵输出相应的tags的分数
            Args:
                emissions (torch.Tensor): 发射概率（分数）[seq_length, batch_size, num_tags]
                mask (torch.LongTensor): size与tag相同，pad位置为0，其余为1 [seq_length, batch_size]
            Returns:
                score (torch.Tensor): 输出分数 [batch_size, num_tags]
        """

        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)
        # score 为一个张量，第j列表示当前以第j个标签结尾的分数
        score = self.start_transitions + emissions[0]   # [batch_size, num_tags]

        for i in range(1, seq_length):
            # 将当前分数广播到下个每一个可能的标签
            broadcast_score = score.unsqueeze(2)     # [batch_size, num_tags, 1]
            # 将发射分数广播到当前每一个可能的标签
            broadcast_emissions = emissions[i].unsqueeze(1)     # [batch_size, 1, num_tags]
            # 计算每一个时刻任意两个标签之间的转移得分
            next_score = broadcast_score + self.transitions + broadcast_emissions    # [batch_size, num_tags, num_tags]
            # 计算所有可能标签转移到i的总得分，并将其转化为对数空间的分数
            next_score = torch.logsumexp(next_score, dim=1)
            # 仅当mask=1时更新分数
            score = torch.where(mask[i].unsqueeze(1), next_score, score)    # [batch_size, num_tags]

        # 各个标签转移到<STOP>的的分数， 并求总分
        score += self.end_transitions   # [batch_size, num_tags]
        return torch.logsumexp(score, dim=1)    # [batch_size]


    def decode(self, emissions, seq_lengths):
        """
            用维特比算法进行解码
        
            Args:
                emissions (torch.Tensor): 发射概率（分数）[seq_length, batch_size, num_tags]
                seq_lengths (torch.Tensor): 句子长度 [batch_size]
            Returns:
                best_tags_list (List): 返回解码的标签   [batch, seq_length]
        """
        self._validate(emissions)

        mask = emissions.new_zeros(emissions.shape[:2], dtype=torch.uint8)
        
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        # 根据输入句子长度来确定mask
        for i in range(seq_lengths.size(0)):
            mask[:seq_lengths[i], i] = 1

        best_tags_list = self._viterbi_decode(emissions, mask)
        return best_tags_list


    def _viterbi_decode(self, emissions, mask):
        """
            进行维特比解码
            Args:
                emissions (torch.Tensor): 发射概率（分数）[seq_length, batch_size, num_tags]
                mask (torch.LongTensor): size与emissions前两维相同，pad位置为0，其余为1 [seq_length, batch_size]
            Returns:
                best_tags_list (List): 返回解码的标签 [batch, seq_length]
        """
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape


        # score 为一个张量，第j列表示当前以第j个标签结尾的当前最优路径的分数
        score = self.start_transitions + emissions[0]   # [batch_size, num_tags]
        # history 用于存储回溯标签
        history = []

        for i in range(1, seq_length):
            # 将当前分数广播到下个每一个可能的标签
            broadcast_score = score.unsqueeze(2)     # [batch_size, num_tags, 1]
            # 将发射分数广播到当前每一个可能的标签
            broadcast_emissions = emissions[i].unsqueeze(1)     # [batch_size, 1, num_tags]
            # 计算每一个时刻任意两个标签之间的转移得分
            next_score = broadcast_score + self.transitions + broadcast_emissions    # [batch_size, num_tags, num_tags]
            # 找到当前所有标签可能的最大分数
            next_score, indices = next_score.max(dim=1)

            # 仅当mask=1时更新分数
            score = torch.where(mask[i].unsqueeze(1), next_score, score)    # [batch_size, num_tags]
            history.append(indices)

        # 各个标签转移到<STOP>的的分数
        score += self.end_transitions

        # 还原最优路径
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # 对于每一个batch，找到最后时刻的最大分数标签
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # 沿着该标签找到存储的最优路径
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # 由于是回溯，因此需要将最优路径反向
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list