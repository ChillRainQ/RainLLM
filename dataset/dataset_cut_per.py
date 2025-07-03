
import json
import os
import random
import re
import multiprocessing
from collections import defaultdict
from tqdm import tqdm
import argparse

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|system|>", "<|user|>", "<|assistant|>"]


class AdvancedTopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            "medical": ["医学", "医疗", "健康", "疾病", "医院", "医生", "患者", "治疗", "诊断", "药物", "手术"],
            "code": ["代码", "编程", "函数", "变量", "类", "方法", "算法", "数据库", "API", "编程语言", "调试", "框架"],
            "finance": ["金融", "股票", "投资", "银行", "经济", "市场", "交易", "货币", "汇率", "基金", "理财"],
            "education": ["教育", "学校", "学习", "课程", "教学", "老师", "学生", "考试", "培训", "教材", "学位"],
            "tech": ["技术", "科技", "创新", "研发", "工程", "设备", "系统", "应用", "智能", "机器人", "人工智能"],
            "entertainment": ["娱乐", "电影", "音乐", "游戏", "明星", "节目", "演出", "综艺", "艺人", "演唱会",
                              "电视剧"],
            "emotional": ["情感", "心情", "感受", "爱", "喜欢", "恋爱", "分手", "安慰", "支持", "理解", "关系",
                          "朋友", "家人", "孤独", "开心", "伤心", "愤怒", "压力", "心理咨询", "情绪管理", "共情",
                          "亲密关系", "情感表达", "心理支持", "情感需求", "情感困惑", "情感交流", "情感咨询"],
            "science": ["科学", "物理", "化学", "生物", "数学", "天文", "地理", "实验", "理论", "研究", "自然"],
            "literature": ["文学", "小说", "诗歌", "散文", "作家", "作品", "阅读", "写作", "故事", "情节", "人物"],
            "sports": ["体育", "运动", "比赛", "足球", "篮球", "运动员", "训练", "奥运", "冠军", "赛事", "健身"]
        }

        # 预编译情感正则（带特殊token保护）
        self.emotional_patterns = [
            re.compile(
                r"(?<!<\|)(我|你|他|她|我们|你们|他们|她们|大家|有人|某人)?(感到|觉得|感觉|认为|以为|发现|知道|希望|想要|需要|喜欢|爱|讨厌|恨|害怕|担心|焦虑|生气|愤怒|伤心|难过|开心|快乐|兴奋|惊喜|压力大|孤独|寂寞|沮丧|失望|无助|困惑|矛盾|纠结|犹豫|烦恼|郁闷|委屈|嫉妒|羡慕|内疚|后悔|羞愧|自豪|满足|放松|平静|安心|感激|感恩|感动|温暖|幸福|舒服|自在|自信|乐观|悲观|消极|积极|紧张|疲劳|累|困|饿|渴|痛|痒|冷|热|不舒服)?[^。，；！？]*?(情感|心情|感受|情绪|恋爱|分手|复合|吵架|和解|道歉|原谅|背叛|信任|猜疑|亲密|冲突|家庭|朋友|爱情|孤独|关系|表达|倾诉)(?!\|>)[^。，；！？]*?[。，；！？]"),
            re.compile(
                r"(?<!<\|)(如何|怎样|怎么|什么|为什么|是不是|能否|要不要|该不该|值不值|有没有|能不能|会不会|想不想|愿不愿)[^。，；！？]*?(处理|解决|面对|表达|沟通|改善|修复|珍惜|忘记|怀念|回忆)[^。，；！？]*?(情感|恋爱|关系|家庭|父母|朋友|婚姻|爱情)(?!\|>)[^。，；！？]*?[。，；！？]"),
            re.compile(
                r"(?<!<\|)(我|你|他|她|我们)?(的)?(男友|女朋友|老婆|丈夫|伴侣|前任|父母|朋友|恋人|对象|老师|孩子)[^。，；！？]*?(不理|伤害|欺骗|吵架|分手|冷战|忽视|离婚|压力|伤心|纠结|困惑)(?!\|>)[^。，；！？]*?[。，；！？]")
        ]

    def classify(self, text):
        # 特殊token保护（临时替换）
        protected_text = text
        placeholder_map = {}
        for i, token in enumerate(SPECIAL_TOKENS):
            placeholder = f"__SPECIAL_{i}__"
            protected_text = protected_text.replace(token, placeholder)
            placeholder_map[placeholder] = token

        # 情感类检测（使用正则）
        lower_text = protected_text.lower()
        for pattern in self.emotional_patterns:
            if pattern.search(lower_text):
                return "emotional"

        # 其他主题检测
        for topic, keywords in self.topic_keywords.items():
            if topic == "emotional":
                continue
            if any(kw in protected_text for kw in keywords):
                return topic

        # 恢复原始特殊token
        for placeholder, token in placeholder_map.items():
            text = text.replace(placeholder, token)

        return "general"


def process_line(line, target_topics):
    try:
        data = json.loads(line)
        text = data.get("text", "")

        # 记录原始特殊token
        original_tokens = {token: text.count(token) for token in SPECIAL_TOKENS}

        # 分类
        classifier = AdvancedTopicClassifier()
        topic = data.get("topic") or classifier.classify(text)

        # 验证并恢复特殊token
        if "text" in data:
            for token, count in original_tokens.items():
                if data["text"].count(token) != count:
                    # 修复被修改的token
                    data["text"] = data["text"].replace(token.lower(), token)
                    for special in SPECIAL_TOKENS:
                        if special != token:
                            data["text"] = data["text"].replace(special.lower(), special)

        return (topic if topic in target_topics else "others"), data
    except:
        return None, None


def worker(chunk, target_topics):
    local_result = defaultdict(list)
    for line in chunk:
        topic, data = process_line(line, target_topics)
        if topic:
            local_result[topic].append(data)
    return local_result


def main():
    parser = argparse.ArgumentParser(description='高级数据重采样器')
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--ratios", type=str,
                        default="emotional:0.2,code:0.2,general:0.3,medical:0.1,tech:0.05,finance:0.05,education:0.05,science:0.03,entertainment:0.02,literature:0.02,sports:0.01,others:0.03",
                        help="目标比例，格式: topic1:ratio1,topic2:ratio2,...")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--min_samples", type=int, default=1000)
    parser.add_argument("--strict_mode", action="store_true", help="严格模式（比例不足时报错）")
    parser.add_argument("--verify_tokens", action="store_true", help="执行特殊token验证")
    parser.add_argument("--clip_ratio", type=float, default=0.0,
                        help="全局数据裁剪比例，例如0.3表示裁剪30%%数据，保持各类别均匀裁剪")
    args = parser.parse_args()

    # 解析目标比例
    target_ratios = {k.strip(): float(v) for k, v in
                     (item.split(':') for item in args.ratios.split(','))}
    required_topics = [k for k in target_ratios.keys() if k != "others"]

    # 读取数据
    print(f"⌛ 正在加载 {os.path.basename(args.input)}...")
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total_lines = len(lines)
    print(f"📊 总样本数: {total_lines:,}")

    # 多进程处理
    print(f"⚡ 使用 {args.threads} 个线程进行分类处理...")
    chunk_size = len(lines) // args.threads + 1
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with multiprocessing.Pool(args.threads) as pool:
        results = pool.starmap(worker, [(chunk, required_topics) for chunk in chunks])

    # 合并结果
    classified_data = defaultdict(list)
    for r in results:
        for k, v in r.items():
            classified_data[k].extend(v)

    # 统计分布
    total_classified = sum(len(v) for v in classified_data.values())
    print("\n📋 分类统计:")
    for topic in sorted(classified_data.keys(), key=lambda x: -len(classified_data[x])):
        print(f"  {topic}: {len(classified_data[topic]):>8,} ({len(classified_data[topic]) / total_classified:>6.1%})")

    # 计算保留比例
    clip_ratio = args.clip_ratio
    keep_ratio = 1.0 - clip_ratio
    if clip_ratio > 0:
        print(f"\n✂️ 全局裁剪比例: {clip_ratio*100:.1f}%，各类别均匀裁剪")

    # 比例采样（含均匀裁剪）
    print("\n🔧 执行比例采样:")
    final_data = []
    for topic, ratio in target_ratios.items():
        if topic == "others":
            continue

        available = len(classified_data.get(topic, []))
        # 原始目标样本数
        target_orig = max(int(total_classified * ratio), args.min_samples)
        # 裁剪后目标数
        target = int(target_orig * keep_ratio)
        # 不超过available
        if available < target:
            msg = f"  ✖ {topic}: 需要 {target:,} 但只有 {available:,}"
            if args.strict_mode:
                raise ValueError(msg + " (严格模式启用)")
            else:
                print(msg + " (使用可用样本)")
                target = available

        if target <= 0:
            continue

        sampled = random.sample(classified_data[topic], target)
        final_data.extend(sampled)
        print(f"  ✔ {topic}: {len(sampled):>7,}/{available:,}")

    # 处理others类别（含裁剪）
    if "others" in target_ratios:
        other_topics = set(classified_data.keys()) - set(target_ratios.keys())
        other_samples = []
        for topic in other_topics:
            other_samples.extend(classified_data[topic])

        target_orig = max(int(total_classified * target_ratios["others"]), args.min_samples)
        target = int(target_orig * keep_ratio)
        target = min(target, len(other_samples))

        if target > 0:
            sampled = random.sample(other_samples, target)
            final_data.extend(sampled)
            print(f"  ✔ others: {len(sampled):>7,}/{len(other_samples):,}")

    # 打乱并保存
    random.shuffle(final_data)
    print(f"\n💾 写入 {len(final_data):,} 条数据到 {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for item in tqdm(final_data, desc="保存进度"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 特殊token验证
    if args.verify_tokens and final_data:
        print("\n🔍 特殊token验证:")
        sample = final_data[0]
        token_status = {token: "✔" if token in sample.get("text", "") else "✖" for token in SPECIAL_TOKENS}
        for token, status in token_status.items():
            print(f"  {token}: {status}")

        error_count = 0
        for item in random.sample(final_data, min(100, len(final_data))):
            text = item.get("text", "")
            for token in SPECIAL_TOKENS:
                if token in text and token.lower() in text.lower() and token != token.lower():
                    error_count += 1
        print(f"  发现 {error_count} 个潜在的大小写转换问题")

    # 最终比例报告
    dist = defaultdict(int)
    for item in final_data:
        topic = item.get("topic", "others")
        if topic not in target_ratios:
            topic = "others"
        dist[topic] += 1

    print("\n🎯 最终分布:")
    for topic in sorted(dist.keys(), key=lambda x: -dist[x]):
        print(f"  {topic}: {dist[topic] / len(final_data):.1%} (目标: {target_ratios.get(topic, 0):.0%})")


if __name__ == "__main__":
    print("=" * 60)
    print("高级数据重采样器 v2.1".center(60))
    print("=" * 60)
    main()
