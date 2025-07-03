import os
import json
import math
import re
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from scipy.stats import entropy


class TokenizerEvaluator:
    def __init__(self, tokenizer_path, new_data_path, sample_size=1000000, shuffle_sample_size=10000):
        """
        初始化分词器评估器

        参数:
        tokenizer_path: 预训练分词器路径 (本地或Hugging Face模型ID)
        new_data_path: 新数据集路径 (目录或文件)
        sample_size: 分词分析采样大小 (字符数)
        shuffle_sample_size: 打乱验证采样行数
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.new_data_path = new_data_path
        self.sample_size = sample_size
        self.shuffle_sample_size = shuffle_sample_size
        self.original_vocab = set(self.tokenizer.get_vocab().keys())
        self.results = {}

        # 特殊标记处理
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        self.unk_token = self.tokenizer.unk_token

    def validate_shuffle(self):
        """验证数据集是否充分打乱（优化情感对话检测）"""
        if not self.new_data_path.endswith('.jsonl'):
            print("⚠️ 打乱验证仅支持JSONL格式文件")
            return False

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
            "entertainment": ["娱乐", "电影", "音乐", "游戏", "明星", "节目", "演出", "综艺", "艺人", "演唱会",
                              "电视剧"],
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
        with open(self.new_data_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in tqdm(f, desc="验证打乱质量", total=self.shuffle_sample_size):
                if line_count >= self.shuffle_sample_size:
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
        entropy_val = -sum(p * math.log(p + 1e-10) for p in probs) if probs else 0

        # 情感对话分布报告
        emotional_percent = topic_count.get("emotional", 0) / total * 100 if total > 0 else 0

        # 计算所有主题占比
        topic_percentages = {}
        for topic, count in topic_count.items():
            topic_percentages[topic] = count / total * 100

        # 记录结果
        self.results['shuffle_quality'] = {
            'max_streak': max_streak,
            'topic_changes': topic_changes,
            'change_rate': change_rate,
            'entropy': entropy_val,
            'emotional_percent': emotional_percent,
            'topic_distribution': dict(topic_count),
            'topic_percentages': topic_percentages,  # 新增主题占比字典
            'line_count': total
        }

        print(f"  - 最大连续相同主题: {max_streak}行")
        print(f"  - 主题转换频率: {change_rate:.4f} (每行)")
        print(f"  - 主题分布熵: {entropy_val:.4f}")
        print(f"  - 情感对话占比: {emotional_percent:.2f}%")

        # 输出所有主题占比
        print("\n  === 所有主题占比 ===")
        for topic, percentage in sorted(topic_percentages.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {topic}: {percentage:.2f}%")

        print("  ===================")

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
        if entropy_val >= 1.5:
            quality_score += 1
        elif entropy_val >= 1.0:
            quality_score += 0.5

        # 4. 情感对话分布
        if 5 <= emotional_percent <= 20:  # 理想占比范围
            quality_score += 1
        elif emotional_percent > 0:  # 至少有一定比例
            quality_score += 0.5

        # 输出质量报告
        quality_status = ""
        if quality_score >= 5:
            quality_status = "优秀"
            print("✅ 数据集已充分打乱 (质量评分: 优秀)")
        elif quality_score >= 4:
            quality_status = "良好"
            print("⚠️ 数据集打乱基本合格 (质量评分: 良好)")
        elif quality_score >= 3:
            quality_status = "中等"
            print("⚠️ 数据集打乱勉强合格 (质量评分: 中等)")
        else:
            quality_status = "差"
            print("❌ 数据集打乱不充分 (质量评分: 差)")

        # 记录质量评分
        self.results['shuffle_quality']['quality_score'] = quality_score
        self.results['shuffle_quality']['quality_status'] = quality_status

        return quality_status in ["优秀", "良好", "中等"]

    # def load_sample_data(self):
    #     """从新数据集加载采样数据"""
    #     print(f"▶ 从 {self.new_data_path} 加载数据样本...")
    #     sample_text = ""
    #     total_chars = 0
    #
    #     if os.path.isdir(self.new_data_path):
    #         files = [f for f in os.listdir(self.new_data_path) if f.endswith(('.txt', '.json', '.jsonl'))]
    #     else:
    #         files = [self.new_data_path]
    #
    #     for file_path in files:
    #         full_path = file_path if os.path.isfile(file_path) else os.path.join(self.new_data_path, file_path)
    #
    #         try:
    #             if file_path.endswith('.jsonl'):
    #                 with open(full_path, 'r', encoding='utf-8') as f:
    #                     for line in tqdm(f, desc=f"处理 {file_path}"):
    #                         if total_chars >= self.sample_size:
    #                             break
    #                         data = json.loads(line)
    #                         text = data.get('text', '') if isinstance(data, dict) else str(data)
    #                         sample_text += text + " "
    #                         total_chars += len(text)
    #
    #             else:  # txt 或其他文本文件
    #                 with open(full_path, 'r', encoding='utf-8') as f:
    #                     for line in tqdm(f, desc=f"处理 {file_path}"):
    #                         if total_chars >= self.sample_size:
    #                             break
    #                         sample_text += line
    #                         total_chars += len(line)
    #
    #             if total_chars >= self.sample_size:
    #                 break
    #
    #         except Exception as e:
    #             print(f"⚠️ 处理文件 {file_path} 时出错: {str(e)}")
    #
    #     # 确保样本不超过指定大小
    #     self.sample_text = sample_text[:self.sample_size]
    #     print(f"✅ 已加载 {len(self.sample_text)} 字符的样本数据")
    #     return self.sample_text
    def load_sample_data(self):
        """从新数据集随机抽样采样数据"""
        print(f"▶ 从 {self.new_data_path} 随机抽样加载数据样本...")

        sample_text = ""
        total_chars = 0

        def get_file_line_count(filepath):
            count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for _ in f:
                    count += 1
            return count

        if os.path.isdir(self.new_data_path):
            files = [os.path.join(self.new_data_path, f) for f in os.listdir(self.new_data_path)
                     if f.endswith(('.txt', '.json', '.jsonl'))]
        else:
            files = [self.new_data_path]

        # 对每个文件单独随机抽样
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.jsonl':
                # 先统计行数
                line_count = get_file_line_count(file_path)
                if line_count == 0:
                    continue

                # 按行随机采样索引 (这里简单采样1万行或更多，根据sample_size调整)
                sample_line_num = min(10000, line_count)
                sampled_lines_idx = set(random.sample(range(line_count), sample_line_num))

                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i in sampled_lines_idx:
                            try:
                                data = json.loads(line)
                                text = data.get('text', '') if isinstance(data, dict) else str(data)
                                sample_text += text + " "
                                total_chars += len(text)
                            except:
                                continue

                            if total_chars >= self.sample_size:
                                break
            else:
                # 普通文本文件，先读所有行，随机抽
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                line_count = len(lines)
                if line_count == 0:
                    continue

                sample_line_num = min(10000, line_count)
                sampled_lines_idx = set(random.sample(range(line_count), sample_line_num))

                for i in sampled_lines_idx:
                    sample_text += lines[i]
                    total_chars += len(lines[i])
                    if total_chars >= self.sample_size:
                        break

            if total_chars >= self.sample_size:
                break

        # 确保样本不超过指定大小
        self.sample_text = sample_text[:self.sample_size]
        print(f"✅ 已随机抽样加载 {len(self.sample_text)} 字符的样本数据")
        return self.sample_text

    def calculate_oov_rate(self):
        """计算未登录词率 (OOV Rate)"""
        print("▶ 计算未登录词率...")
        tokens = self.tokenizer.tokenize(self.sample_text)
        token_count = len(tokens)

        if token_count == 0:
            return 0.0

        oov_count = sum(1 for t in tokens if t == self.unk_token)
        oov_rate = oov_count / token_count

        # 记录结果
        self.results['oov_rate'] = oov_rate
        self.results['token_count'] = token_count
        self.results['oov_count'] = oov_count

        print(f"  - 总token数: {token_count}")
        print(f"  - 未登录token数: {oov_count}")
        print(f"  - OOV率: {oov_rate:.4f} ({oov_rate * 100:.2f}%)")
        return oov_rate

    def analyze_vocab_distribution(self):
        """分析新旧数据集的词频分布"""
        print("▶ 分析词汇分布...")

        # 新数据集词频
        new_tokens = self.tokenizer.tokenize(self.sample_text)
        new_token_freq = Counter(new_tokens)
        total_new_tokens = sum(new_token_freq.values())

        # 移除特殊标记
        for st in self.special_tokens:
            new_token_freq.pop(st, None)

        # 获取原始分词器的词汇频率 (如果可用)
        original_freq = {}
        if hasattr(self.tokenizer, 'vocab') and hasattr(self.tokenizer, 'get_vocab'):
            # 假设均匀分布作为基线
            vocab_size = len(self.tokenizer.get_vocab())
            original_freq = {token: 1 / vocab_size for token in self.original_vocab - self.special_tokens}

        # 计算KL散度
        kl_div = 0.0
        all_tokens = set(original_freq.keys()) | set(new_token_freq.keys())

        # 为KL计算创建概率分布
        P = []
        Q = []
        for token in all_tokens:
            p_val = original_freq.get(token, 1e-10)  # 原始概率
            q_val = new_token_freq.get(token, 0) / total_new_tokens  # 新数据集概率

            # 避免零概率
            if q_val == 0:
                q_val = 1e-10

            P.append(p_val)
            Q.append(q_val)

        # 计算KL散度: D_KL(P || Q)
        kl_div = entropy(P, Q)

        # 记录结果
        self.results['kl_divergence'] = kl_div
        self.results['new_vocab_size'] = len(new_token_freq)
        self.results['original_vocab_size'] = len(self.original_vocab)

        print(f"  - 原始词汇大小: {len(self.original_vocab)}")
        print(f"  - 新数据集有效词汇: {len(new_token_freq)}")
        print(f"  - KL散度: {kl_div:.4f}")
        return kl_div

    def analyze_token_coverage(self, top_n=50):
        """分析token覆盖情况"""
        print("▶ 分析token覆盖...")

        # 获取新数据集中的token
        new_tokens = set(self.tokenizer.tokenize(self.sample_text))

        # 移除特殊标记
        new_tokens = new_tokens - self.special_tokens
        original_vocab = self.original_vocab - self.special_tokens

        # 计算覆盖比例
        covered = new_tokens & original_vocab
        uncovered = new_tokens - original_vocab

        coverage_ratio = len(covered) / len(new_tokens) if new_tokens else 0

        # 记录结果
        self.results['coverage_ratio'] = coverage_ratio
        self.results['covered_tokens'] = len(covered)
        self.results['uncovered_tokens'] = len(uncovered)

        print(f"  - 覆盖比例: {coverage_ratio:.4f}")
        print(f"  - 覆盖token数: {len(covered)}")
        print(f"  - 未覆盖token数: {len(uncovered)}")

        # 获取最常见的未覆盖token
        all_tokens = self.tokenizer.tokenize(self.sample_text)
        token_counter = Counter(all_tokens)

        # 过滤掉已覆盖和特殊token
        uncovered_counter = {t: c for t, c in token_counter.items()
                             if t in uncovered and t not in self.special_tokens}

        # 取前N个最常见的未覆盖token
        top_uncovered = sorted(uncovered_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]
        self.results['top_uncovered'] = top_uncovered

        return coverage_ratio, top_uncovered

    def visualize_results(self):
        """创建可视化报告"""
        if not self.results:
            print("⚠️ 请先运行分析")
            return

        print("▶ 生成可视化报告...")
        fig = plt.figure(figsize=(15, 12))

        # 1. 打乱质量展示 (左侧)
        ax1 = fig.add_subplot(2, 2, 1)
        if 'shuffle_quality' in self.results:
            sq = self.results['shuffle_quality']

            # 创建主题占比饼图
            topic_percentages = sq['topic_percentages']

            # 过滤掉占比过小的主题
            filtered_topics = {k: v for k, v in topic_percentages.items() if v >= 1.0}
            other_percent = 100 - sum(filtered_topics.values())

            if other_percent > 0:
                filtered_topics['其他'] = other_percent

            labels = list(filtered_topics.keys())
            sizes = list(filtered_topics.values())

            # 创建颜色映射
            colors = plt.cm.tab20c(np.linspace(0, 1, len(labels)))

            # 绘制饼图
            wedges, texts, autotexts = ax1.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 9}
            )

            # 添加标题
            ax1.set_title('主题分布占比')
            ax1.axis('equal')  # 等比例确保饼图是圆形

            # 添加图例
            ax1.legend(
                wedges,
                labels,
                title="主题",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=9
            )
        else:
            ax1.text(0.5, 0.5, '未进行打乱验证', ha='center', va='center')
            ax1.set_title('主题分布占比')

        # 2. 分词器适配性指标 (右上)
        ax2 = fig.add_subplot(2, 2, 2)
        oov_rate = self.results['oov_rate']
        coverage = self.results.get('coverage_ratio', 0)
        kl_div = self.results['kl_divergence']

        metrics = ['OOV率', '覆盖比例', 'KL散度']
        values = [oov_rate, coverage, kl_div]

        # 创建颜色映射
        colors = []
        for val, metric in zip(values, metrics):
            if metric == 'OOV率':
                colors.append('red' if val > 0.05 else 'orange' if val > 0.03 else 'green')
            elif metric == '覆盖比例':
                colors.append('green' if val > 0.95 else 'orange' if val > 0.90 else 'red')
            else:  # KL散度
                colors.append('red' if val > 2.0 else 'orange' if val > 1.0 else 'green')

        bars = ax2.bar(metrics, values, color=colors)
        ax2.set_title('分词器适配性指标')
        ax2.set_ylabel('指标值')

        # 添加阈值线
        ax2.axhline(y=0.03, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

        # 3. 打乱质量指标 (左下)
        ax3 = fig.add_subplot(2, 2, 3)
        if 'shuffle_quality' in self.results:
            sq = self.results['shuffle_quality']

            # 创建质量指标条形图
            metrics = ['最大连续主题', '主题转换频率', '主题分布熵', '情感对话占比']
            values = [
                sq['max_streak'],
                sq['change_rate'],
                sq['entropy'],
                sq['emotional_percent'] / 100  # 转换为比例
            ]

            # 创建颜色映射（值越低越红，越高越绿）
            colors = []
            for val in values:
                if val < 0.3:  # 低值范围
                    colors.append(plt.cm.Reds(0.3 + val * 0.7))
                else:  # 高值范围
                    colors.append(plt.cm.Greens(val))

            bars = ax3.bar(metrics, values, color=colors)
            ax3.set_title('打乱质量指标')
            ax3.set_ylabel('指标值')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, '未进行打乱验证', ha='center', va='center')
            ax3.set_title('打乱质量指标')

        # 4. 未覆盖token展示 (右下)
        ax4 = fig.add_subplot(2, 2, 4)
        top_uncovered = self.results.get('top_uncovered', [])

        if top_uncovered:
            tokens, counts = zip(*top_uncovered)
            y_pos = np.arange(len(tokens))

            ax4.barh(y_pos, counts, color='salmon')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(tokens, fontsize=8)
            ax4.set_xlabel('出现频率')
            ax4.set_title('Top 未覆盖Token')
            ax4.invert_yaxis()  # 最高频率在顶部
        else:
            ax4.text(0.5, 0.5, '无显著未覆盖token', ha='center', va='center')
            ax4.set_title('未覆盖Token分析')

        plt.tight_layout()

        # 保存报告
        report_path = "tokenizer_evaluation_report.png"
        plt.savefig(report_path)
        print(f"✅ 可视化报告已保存至 {report_path}")

        return report_path

    def generate_recommendation(self):
        """生成处理建议"""
        if not self.results:
            print("⚠️ 请先运行分析")
            return

        oov_rate = self.results['oov_rate']
        kl_div = self.results['kl_divergence']
        coverage = self.results.get('coverage_ratio', 0)

        print("\n" + "=" * 50)
        print("🔍 分词器适配评估报告")
        print("=" * 50)
        print(f"  - 未登录词率 (OOV): {oov_rate * 100:.2f}%")
        print(f"  - 词频分布差异 (KL散度): {kl_div:.2f}")
        print(f"  - Token覆盖比例: {coverage * 100:.2f}%")

        # 如果进行了打乱验证，展示相关信息
        if 'shuffle_quality' in self.results:
            sq = self.results['shuffle_quality']
            print(f"  - 打乱质量评分: {sq.get('quality_score', 'N/A')} ({sq.get('quality_status', '未知')})")
            print(f"  - 情感对话占比: {sq.get('emotional_percent', 0):.2f}%")
            print(f"  - 最大连续主题: {sq.get('max_streak', 0)}行")

            # 输出所有主题占比
            print("\n  === 所有主题占比 ===")
            for topic, percentage in sorted(sq['topic_percentages'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {topic}: {percentage:.2f}%")
            print("  ===================")

        # 决策逻辑
        recommendation = ""
        action = ""

        if oov_rate < 0.03 and kl_div < 1.0:
            recommendation = "✅ 可直接复用现有分词器"
            action = "无需任何操作"
        elif oov_rate < 0.10 and kl_div < 2.0:
            recommendation = "⚠️ 建议扩展词汇表"
            action = "使用新数据集扩展现有词汇表"
        else:
            recommendation = "❌ 需要重新训练分词器"
            action = "基于新旧数据集联合训练新分词器"

        # 添加详细解释
        print("\n💡 建议:")
        print(f"  - {recommendation}")
        print(f"  - 操作: {action}")

        if oov_rate > 0.05:
            print(f"    (OOV率超过5%安全阈值)")
        if kl_div > 1.5:
            print(f"    (词频分布差异较大，KL散度={kl_div:.2f})")

        # 添加额外建议
        if "扩展" in recommendation and self.results.get('top_uncovered'):
            print("\n🔧 扩展词汇表建议:")
            print("   - 添加以下高频未覆盖token:")
            for i, (token, count) in enumerate(self.results['top_uncovered'][:10]):
                print(f"     {i + 1}. {token} (出现次数: {count})")

        if "重新训练" in recommendation:
            print("\n🔧 重新训练建议:")
            print("   - 使用联合训练: 合并新旧数据集")
            print("   - 考虑调整词汇表大小")
            print("   - 评估不同分词算法 (BPE/WordPiece/Unigram)")

        # 打乱质量建议
        if 'shuffle_quality' in self.results:
            sq = self.results['shuffle_quality']
            if sq.get('quality_status') in ['中等', '差']:
                print("\n🔧 数据集打乱建议:")
                print(f"   - 情感对话占比应控制在5-20%之间 (当前: {sq.get('emotional_percent', 0):.2f}%)")
                if sq.get('max_streak', 0) > 20:
                    print(f"   - 减少连续相同主题行数 (当前最大连续: {sq.get('max_streak', 0)}行)")
                if sq.get('change_rate', 0) < 0.3:
                    print(f"   - 提高主题转换频率 (当前: {sq.get('change_rate', 0):.2f})")
                # 主题分布建议
                print("   - 主题分布优化:")
                for topic, percent in sq['topic_percentages'].items():
                    if percent < 5.0 and topic != "emotional":
                        print(f"     - 增加 '{topic}' 主题内容 (当前: {percent:.2f}%)")
                    elif percent > 30.0:
                        print(f"     - 减少 '{topic}' 主题内容 (当前: {percent:.2f}%)")

        print("=" * 50 + "\n")

        return recommendation, action

    def run_full_analysis(self):
        """运行完整分析流程"""
        try:
            # 如果是JSONL文件，先验证打乱质量
            if self.new_data_path.endswith('.jsonl'):
                self.validate_shuffle()

            self.load_sample_data()
            self.calculate_oov_rate()
            self.analyze_vocab_distribution()
            self.analyze_token_coverage()
            report_path = self.visualize_results()
            recommendation = self.generate_recommendation()

            # 保存结果到JSON
            with open("tokenizer_eval_results.json", "w") as f:
                json.dump(self.results, f, indent=2)

            return {
                "status": "success",
                "report_image": report_path,
                "results": self.results,
                "recommendation": recommendation
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


if __name__ == "__main__":
    # 使用示例
    evaluator = TokenizerEvaluator(
        tokenizer_path="D:\\PythonCode\\RainLLM\\models\\new_tokenizer",
        new_data_path="D:\PythonCode\RainLLM\dataset\sft_512.jsonl",
        sample_size=500000,  # 50万字符样本
        shuffle_sample_size=10000  # 1万行打乱验证
    )

    results = evaluator.run_full_analysis()
    print("\n评估完成！结果摘要:")
    print(results.get("recommendation", {}))