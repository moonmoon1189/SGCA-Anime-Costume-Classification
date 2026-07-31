import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class StyleDecouplingModule(nn.Module):
    """
    论文 2.2 章节: 可学习样式解耦模块
    功能：剥离非真实感光影与色彩差异，仅保留局部边缘与图形拓扑结构特征
    """

    def __init__(self, dim=1024):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim)  # 去除样式色彩方差
        self.w_decouple = nn.Linear(dim, dim, bias=False)  # W_decouple 正交子空间投影

    def forward(self, f_v):
        # f_v shape: [Batch, Seq_len, Dim=1024]
        # InstanceNorm1d 需要形状为 [Batch, Channels/Dim, Seq_len]
        x_norm = self.norm(f_v.transpose(1, 2)).transpose(1, 2)
        x_proj = self.w_decouple(x_norm)

        # 稳健构造局部激活的几何边缘掩码 Mtopo(Fv)
        edge_mask = torch.sigmoid(torch.var(f_v, dim=1, keepdim=True))  # [Batch, 1, Dim]
        f_inv = x_proj * edge_mask
        return f_inv

    def get_orthogonal_loss(self):
        """计算 L_ortho 正交投影惩罚项，控制模型学习纯净样态投影"""
        w = self.w_decouple.weight
        i_mat = torch.eye(w.size(0), device=w.device)
        ortho_loss = torch.norm(torch.matmul(w.t(), w) - i_mat, p="fro") ** 2
        return ortho_loss


class SGCAModule(nn.Module):
    """
    论文 2.4 章节: 语义引导交叉注意力模块
    将文化文本嵌入投影作为 Query，以解耦后的视觉拓扑特征为 Key / Value
    """

    def __init__(self, dim=1024, text_dim=768, num_heads=8):
        super().__init__()
        self.w_q = nn.Linear(text_dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, f_inv, t_c):
        # f_inv (Key, Value): [Batch, Seq_V, 1024]
        # t_c (Query): [Batch, Seq_T, 768]
        q = self.w_q(t_c)  # -> [Batch, Seq_T, 1024]
        k = self.w_k(f_inv)  # -> [Batch, Seq_V, 1024]
        v = self.w_v(f_inv)  # -> [Batch, Seq_V, 1024]

        # 跨模态注意力提取语义关切的视觉区域特征
        attn_out, attn_weights = self.attn(q, k, v)
        return self.out_proj(attn_out), attn_weights


class SGCAModel(nn.Module):
    """
    完整的 SGCA 分类架构：双流基础层 + 样式解耦层 + 语义引导交叉注意力层 + 输出头
    """

    def __init__(self, num_classes=5):
        super().__init__()
        # 按照论文配置：使用预训练的 openai/clip-vit-large-patch14-336
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")

        # 严格冻结原始预训练底层权值
        for param in self.clip.parameters():
            param.requires_grad = False

        self.visual_proj = nn.Linear(1024, 1024)
        self.text_proj = nn.Linear(768, 768)
        self.decoupler = StyleDecouplingModule(dim=1024)
        self.sgca = SGCAModule(dim=1024, text_dim=768)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, pixel_values, input_ids, attention_mask):
        # 1. 冻结计算层获取基本图像序列特征和文本嵌入特征
        with torch.no_grad():
            vis_out = self.clip.vision_model(pixel_values=pixel_values)
            f_v_raw = vis_out.last_hidden_state  # [Batch, 577, 1024]

            text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
            t_c_raw = text_out.last_hidden_state  # [Batch, Seq_len, 768]

        # 2. 对图像序列剔除色彩差异与扭曲样态
        f_v = self.visual_proj(f_v_raw)
        t_c = self.text_proj(t_c_raw)
        f_inv = self.decoupler(f_v)

        # 3. 跨注意力聚合及类别预测
        f_fused, _ = self.sgca(f_inv, t_c)

        # 平均池化注意力输出，得到全局稳定的单一表征向量
        global_repr = f_fused.mean(dim=1)
        logits = self.classifier(global_repr)

        return logits, global_repr, self.decoupler.get_orthogonal_loss()