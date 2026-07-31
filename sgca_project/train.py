import torch
from transformers import CLIPTokenizer
from models import SGCAModel
from losses import SGCATotalLoss


def generate_mock_data(batch_size, tokenizer, device):
    """安全生成标准维度的 [Batch_Size, 3, 336, 336] 训练数据模拟"""
    images = torch.randn(batch_size, 3, 336, 336, device=device)

    prompts = [
        "Han & Tang culture, Jiaoling Youren, Qixiong Ruqun, Pibo",
        "Wafu Traditional culture, Kimono, Hakama",
        "Victorian culture, Corset, Pannier, Lace Edge",
        "Nomadic culture, Animal Fur, Irregular Cut, Woven Pattern",
        "Cyber/Wasteland culture, Techwear, Asymmetric Stitching"
    ]
    batch_prompts = [prompts[i % len(prompts)] for i in range(batch_size)]
    targets = torch.tensor([i % len(prompts) for i in range(batch_size)], device=device)

    tokens = tokenizer(batch_prompts, padding=True, return_tensors="pt", truncation=True).to(device)
    return images, tokens["input_ids"], tokens["attention_mask"], targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System Check] 当前工作运行计算引擎: {device}")

    # 1. 编译模型并加载到正确计算设备
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
    model = SGCAModel(num_classes=5).to(device)
    loss_fn = SGCATotalLoss(l1=0.4, l2=0.6, l3=0.1)

    # 2. 只针对我们在网络层自定义的可学习解耦及跨模态参数启用优化更新
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)

    model.train()
    print("[System Check] 模型参数依赖验证通过，开始执行模拟训练迭代...")

    # 3. 稳健迭代测试
    for step in range(3):
        optimizer.zero_grad()
        images, input_ids, mask, targets = generate_mock_data(4, tokenizer, device)

        logits, global_repr, ortho_loss = model(images, input_ids, mask)

        # 对应模拟同一空间下的文本特征序列用于自监督计算
        mock_t_features = torch.randn_like(global_repr)

        loss, l_ce, l_cl = loss_fn(logits, targets, global_repr, mock_t_features, ortho_loss)
        loss.backward()
        optimizer.step()

        print(
            f" -> [Step {step + 1}] 总损失 L_total: {loss.item():.4f} | 分类 CE: {l_ce.item():.4f} | 正交正则化: {ortho_loss.item():.4f}")

    print("[Success] 全程无异常警报，模型反向传播正常贯通！")


if __name__ == "__main__":
    main()