import os
import json
from PIL import Image


class DanbooruDataFilter:
    """
    对应论文 3.1 章节: 动画风格化服饰文化特征数据集清洗与过滤规则 (公式 9)
    SI_i = 1[|T_i ∩ T_culture| >= θ_tag] ∧ 1[H_i × W_i >= θ_res]
    """

    def __init__(self, min_tag_matches=3, min_short_edge=512):
        # θ_tag = 3 (至少匹配 3 个核心服饰词元)
        self.theta_tag = min_tag_matches
        # θ_res = 512 (图像短边分辨率不低于 512 像素)
        self.theta_res = min_short_edge

        # 建立论文对应的 target culture 核心特征语义字典 T_culture
        self.t_culture = {
            "Han & Tang": {"jiaoling youren", "qixiong ruqun", "pibo", "hanfu", "tang dynasty"},
            "Wafu Traditional": {"kimono", "hakama", "obi", "wafu", "japanese clothes"},
            "Victorian": {"corset", "pannier", "lace edge", "victorian", "gothic lolita"},
            "Nomadic": {"animal fur", "irregular cut", "woven pattern", "tribal", "nomadic"},
            "Cyber/Wasteland": {"techwear", "asymmetric stitching", "cyberpunk", "wasteland", "mecha"}
        }
        # 将分类词典整合为一个全局查询集合
        self.all_cultural_tags = set().union(*self.t_culture.values())

    def filter_sample(self, image_path, metadata_tags):
        """
        对单张图像及绑定的元数据执行双重布尔断言筛选 (SI_i)

        参数:
            image_path (str): 本地待验证的原始二维图像路径
            metadata_tags (list/set): 图片附带的元数据标签列表 (T_i)
        返回:
            bool: True 表示通过清洗保留，False 表示剔除
        """
        # --- 条件 1: 语义标签交集数量判断 (1[|T_i ∩ T_culture| >= θ_tag]) ---
        ti_set = set([tag.lower().strip() for tag in metadata_tags])
        matched_tags = ti_set.intersection(self.all_cultural_tags)

        if len(matched_tags) < self.theta_tag:
            return False, f"[淘汰] 语义交集命中不足: 仅匹配 {len(matched_tags)} 个特征词 (阈值 >={self.theta_tag})"

        # --- 条件 2: 图像物理分辨率严格筛选 (1[H_i × W_i >= θ_res]) ---
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                short_edge = min(width, height)
                if short_edge < self.theta_res:
                    return False, f"[淘汰] 物理分辨率不合格: 短边为 {short_edge}px (阈值 >={self.theta_res}px)"
        except Exception as e:
            return False, f"[淘汰] 图像读取损坏或不存在: {e}"

        # 满足逻辑与 (∧) ，保留高质量样本
        return True, f"[通过] 匹配标签: {list(matched_tags)} | 尺寸: {width}x{height}"


# ==================== 运行测试说明 ====================
if __name__ == "__main__":
    print("[System Check] 启动公共图像集清洗过滤引擎...")
    filter_engine = DanbooruDataFilter(min_tag_matches=3, min_short_edge=512)

    # 模拟从公共数据库 JSON 中解析出的一条测试记录
    mock_tags = ["kimono", "hakama", "obi", "girl", "long hair", "solo"]
    mock_img_path = "mock_test_image.jpg"

    # 作为开箱验证，若本地没有图，先创建一张合法分辨率 (600x800) 的空白图
    img_tmp = Image.new('RGB', (600, 800), color='white')
    img_tmp.save(mock_img_path)

    is_valid, log_msg = filter_engine.filter_sample(mock_img_path, mock_tags)
    print(f"执行结果 -> {log_msg}")

    # 清理调试产生的临时测试图
    if os.path.exists(mock_img_path):
        os.remove(mock_img_path)