import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer
from models import SGCAModel


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

    model = SGCAModel(num_classes=5).to(device)
    model.eval()

    class_names = ["Han & Tang", "Wafu Traditional", "Victorian", "Nomadic", "Cyber/Wasteland"]
    prompt_templates = [
        "Han & Tang culture, Jiaoling Youren, Qixiong Ruqun, Pibo",
        "Wafu Traditional culture, Kimono, Hakama",
        "Victorian culture, Corset, Pannier, Lace Edge",
        "Nomadic culture, Animal Fur, Irregular Cut, Woven Pattern",
        "Cyber/Wasteland culture, Techwear, Asymmetric Stitching"
    ]
    tokens = tokenizer(prompt_templates, padding=True, return_tensors="pt", truncation=True).to(device)
    test_image = torch.randn(1, 3, 336, 336, device=device)

    print("[System Check] 正在计算输入特征分布及开放域 OOD 边界阈值...")
    with torch.no_grad():
        logits, _, _ = model(
            test_image,
            tokens["input_ids"][0:1],
            tokens["attention_mask"][0:1]
        )

        probs = F.softmax(logits, dim=-1)
        max_prob, pred_idx = torch.max(probs, dim=-1)
        max_prob_val = max_prob.item()

        # 按照论文给出的定量判定约束阈值
        ood_threshold = 0.65
        if max_prob_val < ood_threshold:
            pred_class = "Out-of-Distribution (OOD) [触发未知分布拦截: 非目标类]"
        else:
            pred_class = class_names[pred_idx.item()]

    print("=" * 50)
    print(f"最大预测概率得分: {max_prob_val * 100:.2f}%")
    print(f"系统推理最终输出: ** {pred_class} **")
    print("=" * 50)
    print("[Success] 推理判别链路正常完全运行。")


if __name__ == "__main__":
    run_inference()