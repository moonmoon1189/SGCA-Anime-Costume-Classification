import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricContrastiveLoss(nn.Module):
    """
    跨模态对称图像-文本对比损失 (公式 6, 7, 8)
    将同一批次的对应图片与历史文本特征通过余弦距离建立对齐关联
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, v_features, t_features):
        v_norm = F.normalize(v_features, p=2, dim=-1)
        t_norm = F.normalize(t_features, p=2, dim=-1)

        sim_matrix = torch.matmul(v_norm, t_norm.t()) / self.temp
        labels = torch.arange(v_features.size(0), device=v_features.device)

        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
        return 0.5 * (loss_i2t + loss_t2i)


class SGCATotalLoss(nn.Module):
    """
    统一的多任务加权组合损失配置 (公式 14)
    最佳论文实验值配置: λ_1=0.4 (对比损失), λ_2=0.6 (交叉熵分类), λ_3=0.1 (正交投影)
    """

    def __init__(self, l1=0.4, l2=0.6, l3=0.1):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.ce_loss = nn.CrossEntropyLoss()
        self.cl_loss = SymmetricContrastiveLoss()

    def forward(self, logits, targets, v_features, t_features, ortho_loss):
        l_ce = self.ce_loss(logits, targets)
        l_cl = self.cl_loss(v_features, t_features)
        total_loss = (self.l1 * l_cl) + (self.l2 * l_ce) + (self.l3 * ortho_loss)
        return total_loss, l_ce, l_cl