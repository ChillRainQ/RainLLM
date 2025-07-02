# # import json
# # from pathlib import Path
# #
# # from tqdm import tqdm
# #
# #
# # def merge_datasets(dataset1_path, dataset2_path, output_path):
# #     """合并两个JSONL格式的预训练数据集"""
# #     dataset1 = Path(dataset1_path)
# #     dataset2 = Path(dataset2_path)
# #     output = Path(output_path)
# #
# #     # 验证文件存在
# #     if not dataset1.exists() or not dataset2.exists():
# #         raise FileNotFoundError("数据集文件不存在")
# #
# #     # 统计行数
# #     total_lines = sum(1 for _ in open(dataset1, 'r', encoding='utf-8')) + \
# #                   sum(1 for _ in open(dataset2, 'r', encoding='utf-8'))
# #
# #     # 合并文件
# #     with open(output, 'w', encoding='utf-8') as out_f:
# #         # 数据集1
# #         with open(dataset1, 'r', encoding='utf-8') as f1:
# #             for line in tqdm(f1, total=total_lines // 2, desc="合并数据集1"):
# #                 try:
# #                     data = json.loads(line)
# #                     if 'text' in data and data['text'].strip():
# #                         out_f.write(line)
# #                 except json.JSONDecodeError:
# #                     continue
# #
# #         # 数据集2
# #         with open(dataset2, 'r', encoding='utf-8') as f2:
# #             for line in tqdm(f2, total=total_lines // 2, desc="合并数据集2"):
# #                 try:
# #                     data = json.loads(line)
# #                     if 'text' in data and data['text'].strip():
# #                         out_f.write(line)
# #                 except json.JSONDecodeError:
# #                     continue
# #
# #     print(f"✅ 数据集合并完成，保存至: {output_path}")
# #     return output_path
# #
# #
# # # 使用示例
# # merged_path = merge_datasets(
# #     dataset1_path="pretrain_hq.jsonl",
# #     dataset2_path="pretrain_hq_v7.jsonl",
# #     output_path="pretrain_combined.jsonl"
# # )
# import json
# import random
# from pathlib import Path
# import shutil
#
#
# def shuffle_large_jsonl(input_path, output_path, buffer_size=100000):
#     """打乱大型JSONL数据集"""
#     input_file = Path(input_path)
#     output_file = Path(output_path)
#
#     # 第一步：统计总行数
#     print("▶ 统计数据集行数...")
#     total_lines = 0
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for _ in f:
#             total_lines += 1
#
#     # 第二步：创建行索引
#     print("▶ 创建行索引...")
#     line_indices = list(range(total_lines))
#     random.shuffle(line_indices)
#
#     # 第三步：分块打乱
#     print("▶ 分块打乱数据...")
#     temp_dir = output_file.parent / "temp_shuffle"
#     temp_dir.mkdir(exist_ok=True)
#
#     # 分块读取和打乱
#     buffer = []
#     chunk_count = 0
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for idx, line in enumerate(f):
#             buffer.append((line_indices[idx], line))
#
#             if len(buffer) >= buffer_size:
#                 buffer.sort(key=lambda x: x[0])  # 按随机索引排序
#                 chunk_path = temp_dir / f"chunk_{chunk_count}.jsonl"
#                 with open(chunk_path, 'w', encoding='utf-8') as chunk_f:
#                     for _, line_text in buffer:
#                         chunk_f.write(line_text)
#                 chunk_count += 1
#                 buffer = []
#
#     # 处理剩余行
#     if buffer:
#         buffer.sort(key=lambda x: x[0])
#         chunk_path = temp_dir / f"chunk_{chunk_count}.jsonl"
#         with open(chunk_path, 'w', encoding='utf-8') as chunk_f:
#             for _, line_text in buffer:
#                 chunk_f.write(line_text)
#         chunk_count += 1
#
#     # 第四步：合并分块
#     print("▶ 合并打乱后的数据...")
#     with open(output_file, 'w', encoding='utf-8') as out_f:
#         for i in range(chunk_count):
#             chunk_path = temp_dir / f"chunk_{i}.jsonl"
#             with open(chunk_path, 'r', encoding='utf-8') as chunk_f:
#                 shutil.copyfileobj(chunk_f, out_f)
#             chunk_path.unlink()
#
#     # 清理临时目录
#     temp_dir.rmdir()
#     print(f"✅ 数据集已打乱并保存至: {output_path}")
#     return output_path
#
#
# if __name__ == "__main__":
#     # 打乱合并数据集
#     shuffled_path = shuffle_large_jsonl(
#         input_path=r"D:\PythonCode\RainLLM\dataset\pretrain_combined.jsonl",
#         output_path=r"D:\PythonCode\RainLLM\dataset\pretrain_combined_shuffled.jsonl"
#     )
import json
import random
import shutil
import re
import math
import time
from pathlib import Path
from collections import defaultdict


def validate_shuffle(dataset_path, sample_size=10000):
    """验证数据集是否充分打乱（优化情感对话检测）"""
    print("▶ 验证数据集打乱质量...")
    topic_sequence = []
    topic_count = defaultdict(int)

    # 扩展主题关键词（增加情感对话类别）
    topic_keywords = {
        "medical": ["医学", "医疗", "健康", "疾病", "医院", "医生", "患者", "治疗", "诊断", "药物", "手术"],
        "code": ["代码", "编程", "函数", "变量", "类", "方法", "算法", "数据库", "API", "编程语言", "调试", "框架"],
        "finance": ["金融", "股票", "投资", "银行", "经济", "市场", "交易", "货币", "汇率", "基金", "理财"],
        "education": ["教育", "学校", "学习", "课程", "教学", "老师", "学生", "考试", "培训", "教材", "学位"],
        "tech": ["技术", "科技", "创新", "研发", "工程", "设备", "系统", "应用", "智能", "机器人", "人工智能"],
        "entertainment": ["娱乐", "电影", "音乐", "游戏", "明星", "节目", "演出", "综艺", "艺人", "演唱会", "电视剧"],
        "emotional": ["情感", "心情", "感受", "爱", "喜欢", "恋爱", "分手", "安慰", "支持", "理解", "关系",
                      "朋友", "家人", "孤独", "开心", "伤心", "愤怒", "压力", "心理咨询", "情绪管理", "共情",
                      "亲密关系", "情感表达", "心理支持", "情感需求", "情感困惑", "情感交流", "情感咨询"]
    }

    # 情感对话专用检测模式 - 更精确的正则表达式
    emotional_patterns = [
        r"(我|你|他|她|我们|你们|他们|她们|大家|有人|某人)?(感到|觉得|感觉|认为|以为|发现|知道|希望|想要|需要|喜欢|爱|讨厌|恨|害怕|担心|焦虑|生气|愤怒|伤心|难过|开心|快乐|兴奋|惊喜|压力大|孤独|寂寞|沮丧|失望|无助|困惑|矛盾|纠结|犹豫|烦恼|郁闷|委屈|嫉妒|羡慕|内疚|后悔|羞愧|自豪|满足|放松|平静|安心|感激|感恩|感动|温暖|幸福|舒服|自在|自信|乐观|悲观|消极|积极|紧张|疲劳|累|困|饿|渴|痛|痒|冷|热|不舒服)?[^。，；！？]*?(情感|心情|感受|情绪|恋爱|分手|复合|吵架|和解|道歉|原谅|背叛|信任|猜疑|嫉妒|羡慕|亲密|疏远|冷淡|热情|关心|体贴|理解|支持|安慰|鼓励|帮助|陪伴|倾听|倾诉|分享|沟通|交流|表达|沉默|冷战|冲突|矛盾|解决|处理|应对|调节|控制|释放|发泄|压抑|积累|爆发|恢复|重建|维系|经营|珍惜|放弃|放下|忘记|怀念|思念|回忆|过去|现在|未来|家庭|父母|子女|兄弟姐妹|亲戚|朋友|闺蜜|哥们|同事|同学|恋人|爱人|伴侣|夫妻|情侣|单身|恋爱关系|婚姻关系|亲子关系|友情|人际关系|社交|孤独|合群|排斥|接纳|认同|尊重|包容|宽容|体谅|关心|爱护|保护|依赖|独立|自由|束缚|控制|占有|牺牲|付出|回报|感恩|感动|温暖|幸福|痛苦|快乐)[^。，；！？]*?[。，；！？]",
        r"(如何|怎样|怎么|什么|为什么|是不是|能否|要不要|该不该|值不值|有没有|能不能|会不会|想不想|愿不愿|敢不敢|肯不肯)[^。，；！？]*?(处理|应对|解决|面对|看待|理解|分析|调节|控制|释放|表达|沟通|交流|改善|修复|维系|经营|珍惜|放弃|放下|忘记|怀念|回忆|重建|恢复)[^。，；！？]*?(情感|心情|感受|情绪|恋爱|分手|复合|吵架|和解|道歉|原谅|背叛|信任|猜疑|嫉妒|羡慕|亲密|疏远|冷淡|热情|关心|体贴|理解|支持|安慰|鼓励|帮助|陪伴|倾听|倾诉|分享|沟通|交流|表达|沉默|冷战|冲突|矛盾|关系|家庭|父母|子女|兄弟姐妹|亲戚|朋友|闺蜜|哥们|同事|同学|恋人|爱人|伴侣|夫妻|情侣|单身|恋爱关系|婚姻关系|亲子关系|友情|人际关系|社交|孤独|合群|排斥|接纳|认同|尊重|包容|宽容|体谅|关心|爱护|保护|依赖|独立|自由|束缚|控制|占有|牺牲|付出|回报|感恩|感动|温暖|幸福|痛苦|快乐)[^。，；！？]*?[。，；！？]",
        r"(我|你|他|她|我们|你们|他们|她们|大家|有人|某人)?(的)?(男友|女朋友|老公|老婆|妻子|丈夫|伴侣|恋人|爱人|对象|前任|前男友|前女友|前夫|前妻|父母|爸爸|妈妈|父亲|母亲|孩子|儿子|女儿|兄弟|姐妹|哥哥|弟弟|姐姐|妹妹|朋友|闺蜜|哥们|同事|同学|老师|学生|领导|下属)[^。，；！？]*?(不理|不联系|不回复|不回消息|不接电话|不见面|不关心|不在乎|不重视|不理解|不支持|不信任|不尊重|不包容|不体贴|不爱|不喜欢|讨厌|恨|伤害|欺骗|背叛|出轨|劈腿|说谎|隐瞒|误会|冤枉|指责|批评|抱怨|埋怨|责怪|争吵|吵架|打架|冷战|分手|离婚|离开|抛弃|放弃|疏远|冷淡|忽视|躲避|逃避|拒绝|排斥|嫉妒|羡慕|控制|占有|束缚|压力|烦恼|困扰|痛苦|伤心|难过|失望|绝望|无助|困惑|矛盾|纠结|犹豫)[^。，；！？]*?[。，；！？]"
    ]

    # 采样数据集的前sample_size行
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            if line_count >= sample_size:
                break

            try:
                data = json.loads(line)
                text = data.get('text', '').lower()

                # 识别主题 - 优化逻辑
                topic = "general"

                # 1. 检查是否情感对话 (更严格的条件)
                is_emotional = False
                for pattern in emotional_patterns:
                    if re.search(pattern, text):
                        is_emotional = True
                        break

                # 2. 检查其他主题
                found_other_topic = False
                for topic_name, keywords in topic_keywords.items():
                    # 跳过情感主题，单独处理
                    if topic_name == "emotional":
                        continue

                    # 检查是否包含该主题的关键词
                    if any(keyword in text for keyword in keywords):
                        topic = topic_name
                        found_other_topic = True
                        break

                # 3. 如果没有其他主题但符合情感对话条件
                if not found_other_topic and is_emotional:
                    topic = "emotional"

                topic_sequence.append(topic)
                topic_count[topic] += 1
                line_count += 1
            except:
                continue

    # 计算同主题连续出现次数
    max_streak = 0
    current_streak = 0
    current_topic = None

    for topic in topic_sequence:
        if topic == current_topic:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
            current_topic = topic

    # 计算主题转换频率
    topic_changes = 0
    if len(topic_sequence) > 1:
        topic_changes = sum(1 for i in range(1, len(topic_sequence))
                            if topic_sequence[i] != topic_sequence[i - 1])
    change_rate = topic_changes / len(topic_sequence) if topic_sequence else 0

    # 计算主题分布熵（衡量多样性）
    total = len(topic_sequence)
    probs = [count / total for count in topic_count.values()] if total > 0 else []
    entropy = -sum(p * math.log(p + 1e-10) for p in probs) if probs else 0

    # 情感对话分布报告
    emotional_percent = topic_count.get("emotional", 0) / total * 100 if total > 0 else 0

    print(f"最大连续相同主题: {max_streak}")
    print(f"主题转换频率: {change_rate:.4f} (每行)")
    print(f"主题分布熵: {entropy:.4f}")
    print(f"情感对话占比: {emotional_percent:.2f}%")
    print(f"主题分布: {dict(topic_count)}")

    # 评估标准（增加情感对话权重）
    quality_score = 0

    # 1. 最大连续主题行数
    if max_streak <= 10:
        quality_score += 2
    elif max_streak <= 20:
        quality_score += 1

    # 2. 主题转换频率
    if change_rate >= 0.5:
        quality_score += 2
    elif change_rate >= 0.3:
        quality_score += 1

    # 3. 主题分布熵
    if entropy >= 1.5:
        quality_score += 1
    elif entropy >= 1.0:
        quality_score += 0.5

    # 4. 情感对话分布
    if 5 <= emotional_percent <= 20:  # 理想占比范围
        quality_score += 1
    elif emotional_percent > 0:  # 至少有一定比例
        quality_score += 0.5

    # 输出质量报告
    if quality_score >= 5:
        print("✅ 数据集已充分打乱 (质量评分: 优秀)")
        return True
    elif quality_score >= 4:
        print("⚠️ 数据集打乱基本合格 (质量评分: 良好)")
        return True
    elif quality_score >= 3:
        print("⚠️ 数据集打乱勉强合格 (质量评分: 中等)")
        return True
    else:
        print("❌ 数据集打乱不充分 (质量评分: 差)")
        return False


def shuffle_large_jsonl(input_path, output_path, buffer_size=100000, max_attempts=3):
    """打乱大型JSONL数据集，并自动验证打乱质量"""
    input_file = Path(input_path)
    output_file = Path(output_path)

    # 创建临时目录
    temp_dir = output_file.parent / f"temp_shuffle_{int(time.time())}"
    temp_dir.mkdir(exist_ok=True)

    attempt = 1
    shuffle_quality = False

    while attempt <= max_attempts and not shuffle_quality:
        print(f"\n=== 打乱尝试 #{attempt}/{max_attempts} ===")

        # 第一步：统计总行数
        print("▶ 统计数据集行数...")
        total_lines = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1

        # 第二步：创建行索引
        print("▶ 创建行索引...")
        line_indices = list(range(total_lines))
        random.shuffle(line_indices)

        # 第三步：分块打乱
        print("▶ 分块打乱数据...")
        buffer = []
        chunk_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # 跳过空行
                if not line.strip():
                    continue

                buffer.append((line_indices[idx], line))

                if len(buffer) >= buffer_size:
                    # 使用高效排序算法
                    buffer.sort(key=lambda x: x[0])
                    chunk_path = temp_dir / f"chunk_{chunk_count}.jsonl"
                    with open(chunk_path, 'w', encoding='utf-8') as chunk_f:
                        for _, line_text in buffer:
                            chunk_f.write(line_text)
                    chunk_count += 1
                    buffer = []

        # 处理剩余行
        if buffer:
            buffer.sort(key=lambda x: x[0])
            chunk_path = temp_dir / f"chunk_{chunk_count}.jsonl"
            with open(chunk_path, 'w', encoding='utf-8') as chunk_f:
                for _, line_text in buffer:
                    chunk_f.write(line_text)
            chunk_count += 1

        # 第四步：合并分块
        print("▶ 合并打乱后的数据...")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i in range(chunk_count):
                chunk_path = temp_dir / f"chunk_{i}.jsonl"
                with open(chunk_path, 'r', encoding='utf-8') as chunk_f:
                    shutil.copyfileobj(chunk_f, out_f)
                chunk_path.unlink()

        print(f"✅ 数据集已打乱并保存至: {output_path}")

        # 第五步：验证打乱质量
        print("▶ 验证打乱质量...")
        shuffle_quality = validate_shuffle(output_file)

        if not shuffle_quality and attempt < max_attempts:
            print(f"⚠️ 打乱质量不合格，将尝试重新打乱...")

        attempt += 1

    # 清理临时目录
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"⚠️ 清理临时目录时出错: {str(e)}")

    return shuffle_quality


if __name__ == "__main__":
    # 打乱合并数据集
    success = shuffle_large_jsonl(
        input_path=r"D:\PythonCode\RainLLM\dataset\pretrain_combined.jsonl",
        output_path=r"D:\PythonCode\RainLLM\dataset\pretrain_combined_shuffled.jsonl",
        buffer_size=100000,
        max_attempts=5
    )

    if success:
        print("\n🎉 数据集打乱成功且质量合格")
    else:
        print("\n⚠️ 警告：数据集打乱后质量仍不合格，建议手动检查")

    # 添加情感对话分布报告
    print("\n情感对话分布分析:")
    print("1. 情感对话应占总数据的5-20%")
    print("2. 情感对话应均匀分布在数据集中")
    print("3. 避免大段连续的情感对话内容")