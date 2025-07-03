# import json
# import re
# import os
# import time
# import math
# import random
# from collections import defaultdict
# from tqdm import tqdm
#
#
# class DataSetEvaluator:
#     def __init__(self, data_path, sample_size=1000000, shuffle_sample_size=10000):
#         self.data_path = data_path
#         self.sample_size = sample_size
#         self.shuffle_sample_size = shuffle_sample_size
#         self.results = {}
#         self.file_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
#
#         # 主题关键词定义
#         self.topic_keywords = {
#             "medical": ["医学", "医疗", "健康", "疾病", "医院", "医生", "患者", "治疗", "诊断", "药物", "手术"],
#             "code": ["代码", "编程", "函数", "变量", "类", "方法", "算法", "数据库", "API", "编程语言", "调试", "框架"],
#             "finance": ["金融", "股票", "投资", "银行", "经济", "市场", "交易", "货币", "汇率", "基金", "理财"],
#             "education": ["教育", "学校", "学习", "课程", "教学", "老师", "学生", "考试", "培训", "教材", "学位"],
#             "tech": ["技术", "科技", "创新", "研发", "工程", "设备", "系统", "应用", "智能", "机器人", "人工智能"],
#             "entertainment": ["娱乐", "电影", "音乐", "游戏", "明星", "节目", "演出", "综艺", "艺人", "演唱会",
#                               "电视剧"],
#             "emotional": ["情感", "心情", "感受", "爱", "喜欢", "恋爱", "分手", "安慰", "支持", "理解", "关系",
#                           "朋友", "家人", "孤独", "开心", "伤心", "愤怒", "压力", "心理咨询", "情绪管理", "共情",
#                           "亲密关系", "情感表达", "心理支持", "情感需求", "情感困惑", "情感交流", "情感咨询"],
#             "science": ["科学", "物理", "化学", "生物", "数学", "天文", "地理", "实验", "理论", "研究", "自然"],
#             "literature": ["文学", "小说", "诗歌", "散文", "作家", "作品", "阅读", "写作", "故事", "情节", "人物"],
#             "sports": ["体育", "运动", "比赛", "足球", "篮球", "运动员", "训练", "奥运", "冠军", "赛事", "健身"]
#         }
#
#         # 情感模式正则表达式
#         self.emotional_patterns = [
#             r"(我|你|他|她|我们|你们|他们|她们|大家|有人|某人)?(感到|觉得|感觉|认为|以为|发现|知道|希望|想要|需要|喜欢|爱|讨厌|恨|害怕|担心|焦虑|生气|愤怒|伤心|难过|开心|快乐|兴奋|惊喜|压力大|孤独|寂寞|沮丧|失望|无助|困惑|矛盾|纠结|犹豫|烦恼|郁闷|委屈|嫉妒|羡慕|内疚|后悔|羞愧|自豪|满足|放松|平静|安心|感激|感恩|感动|温暖|幸福|舒服|自在|自信|乐观|悲观|消极|积极|紧张|疲劳|累|困|饿|渴|痛|痒|冷|热|不舒服)?[^。，；！？]*?(情感|心情|感受|情绪|恋爱|分手|复合|吵架|和解|道歉|原谅|背叛|信任|猜疑|亲密|冲突|家庭|朋友|爱情|孤独|关系|表达|倾诉)[^。，；！？]*?[。，；！？]",
#             r"(如何|怎样|怎么|什么|为什么|是不是|能否|要不要|该不该|值不值|有没有|能不能|会不会|想不想|愿不愿)[^。，；！？]*?(处理|解决|面对|表达|沟通|改善|修复|珍惜|忘记|怀念|回忆)[^。，；！？]*?(情感|恋爱|关系|家庭|父母|朋友|婚姻|爱情)[^。，；！？]*?[。，；！？]",
#             r"(我|你|他|她|我们)?(的)?(男友|女朋友|老婆|丈夫|伴侣|前任|父母|朋友|恋人|对象|老师|孩子)[^。，；！？]*?(不理|伤害|欺骗|吵架|分手|冷战|忽视|离婚|压力|伤心|纠结|困惑)[^。，；！？]*?[。，；！？]"
#         ]
#
#     def extract_last_user_turn(self, full_text: str) -> str:
#         """从文本中提取最后一个用户对话轮次"""
#         matches = re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", full_text, re.DOTALL)
#         if matches:
#             return matches[-1].strip().lower()
#         return full_text.strip().lower()
#
#     def classify_topic(self, text: str) -> str:
#         """对文本进行分类，返回主题标签"""
#         # 检查是否为情感主题
#         if any(re.search(p, text) for p in self.emotional_patterns):
#             return "emotional"
#
#         # 检查其他主题
#         for topic_name, keywords in self.topic_keywords.items():
#             if any(k in text for k in keywords):
#                 return topic_name
#
#         return "general"
#
#     def _analyze_dataset(self, line_iterator, total_lines, mode="full"):
#         """分析数据集的核心函数（支持全量和抽样）"""
#         topic_count = defaultdict(int)
#         total_samples = 0
#         start_time = time.time()
#
#         # 进度条描述
#         desc = "分析完整数据集" if mode == "full" else f"分析抽样数据 ({self.sample_size}样本)"
#
#         try:
#             with tqdm(total=total_lines, desc=desc, unit="行") as progress_bar:
#                 for line in line_iterator:
#                     try:
#                         data = json.loads(line)
#                         text = self.extract_last_user_turn(data.get('text', ''))
#                         topic = self.classify_topic(text)
#                         topic_count[topic] += 1
#                         total_samples += 1
#                         progress_bar.update(1)
#                     except json.JSONDecodeError:
#                         print(f"  ! JSON解析错误: {line[:100]}...")
#                     except Exception as e:
#                         print(f"  ! 处理错误: {str(e)}")
#
#                     # 如果是抽样模式且达到样本量，提前结束
#                     if mode == "sample" and total_samples >= self.sample_size:
#                         break
#         except Exception as e:
#             print(f"❌ 处理文件时发生严重错误: {str(e)}")
#             return None
#
#         # 计算分析耗时
#         elapsed_time = time.time() - start_time
#         lines_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
#
#         # 计算百分比和熵
#         topic_percentages = {topic: count / total_samples * 100 for topic, count in topic_count.items()}
#         probs = [count / total_samples for count in topic_count.values()] if total_samples > 0 else []
#         entropy_val = -sum(p * math.log(p + 1e-10) for p in probs) if probs else 0
#
#         # 生成结果字典
#         result_key = "full_analysis" if mode == "full" else "sample_analysis"
#         self.results[result_key] = {
#             'mode': mode,
#             'sample_size': total_samples,
#             'processing_time': elapsed_time,
#             'lines_per_sec': lines_per_sec,
#             'entropy': entropy_val,
#             'topic_counts': dict(topic_count),
#             'topic_percentages': topic_percentages
#         }
#
#         return topic_percentages
#
#     def analyze_full_dataset(self):
#         """分析整个数据集的主题分布"""
#         if not os.path.exists(self.data_path):
#             print(f"❌ 文件不存在: {self.data_path}")
#             return None
#
#         print(f"\n▶ 开始分析完整数据集: {self.data_path}")
#         print(f"  文件大小: {self.file_size / (1024 * 1024):.2f} MB")
#
#         # 获取文件总行数
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             total_lines = sum(1 for _ in f)
#
#         # 重新打开文件进行处理
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             topic_percentages = self._analyze_dataset(f, total_lines, mode="full")
#
#         # 打印报告
#         self._print_analysis_report("full_analysis")
#         return topic_percentages
#
#     def analyze_sampled_dataset(self, sample_size=None):
#         """随机抽样分析数据集主题分布"""
#         if sample_size is not None:
#             self.sample_size = sample_size
#
#         if not os.path.exists(self.data_path):
#             print(f"❌ 文件不存在: {self.data_path}")
#             return None
#
#         print(f"\n▶ 开始随机抽样分析数据集: {self.data_path}")
#         print(f"  抽样大小: {self.sample_size:,} 行")
#
#         # 获取文件总行数
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             total_lines = sum(1 for _ in f)
#
#         # 创建随机行索引生成器
#         def random_line_generator():
#             with open(self.data_path, 'r', encoding='utf-8') as f:
#                 # 创建行索引列表并打乱
#                 indices = list(range(total_lines))
#                 random.shuffle(indices)
#
#                 # 只取需要的样本量
#                 indices = indices[:self.sample_size]
#
#                 # 按行号排序以便顺序读取
#                 indices.sort()
#
#                 # 顺序读取文件行
#                 for i, line in enumerate(f):
#                     if i in indices:
#                         yield line
#
#         # 进行分析
#         topic_percentages = self._analyze_dataset(random_line_generator(), min(self.sample_size, total_lines),
#                                                   mode="sample")
#
#         # 打印报告
#         self._print_analysis_report("sample_analysis")
#         return topic_percentages
#
#     def _print_analysis_report(self, result_key):
#         """打印分析报告"""
#         if result_key not in self.results:
#             print("❌ 没有可用的分析结果")
#             return
#
#         result = self.results[result_key]
#         total_samples = result['sample_size']
#         topic_percentages = result['topic_percentages']
#         topic_count = result['topic_counts']
#
#         mode = "完整分析" if result['mode'] == "full" else f"抽样分析 ({self.sample_size}样本)"
#
#         print("\n" + "=" * 60)
#         print(f"📊 数据集分析报告 - {mode}")
#         print(f"  总样本数: {total_samples:,}")
#         print(f"  分析耗时: {result['processing_time']:.2f} 秒 ({result['lines_per_sec']:.1f} 行/秒)")
#         print(f"  主题熵值: {result['entropy']:.4f} (衡量主题多样性)")
#         print("-" * 60)
#         print(f"  {'主题':<15} {'数量':>12} {'占比':>10} {'柱状图':<20}")
#         print("-" * 60)
#
#         # 按占比排序并打印柱状图
#         max_count = max(topic_count.values()) if topic_count else 1
#         for topic, count in sorted(topic_count.items(), key=lambda x: x[1], reverse=True):
#             percentage = topic_percentages[topic]
#             bar_length = int(50 * count / max_count)
#             bar = '█' * bar_length
#             print(f"  {topic:<15} {count:>12,} {percentage:>9.2f}%  {bar}")
#
#         print("=" * 60)
#
#         # 生成可视化报告
#         self.generate_visual_report(topic_percentages, result['mode'])
#
#     def generate_visual_report(self, topic_percentages, mode):
#         """生成可视化的主题分布报告"""
#         try:
#             import matplotlib.pyplot as plt
#             import numpy as np
#
#             print("\n▶ 生成可视化报告...")
#
#             # 准备数据
#             labels = list(topic_percentages.keys())
#             sizes = list(topic_percentages.values())
#             sorted_indices = np.argsort(sizes)[::-1]
#             labels = [labels[i] for i in sorted_indices]
#             sizes = [sizes[i] for i in sorted_indices]
#
#             # 创建饼图
#             plt.figure(figsize=(12, 8))
#             explode = [0.1 if i == 0 else 0 for i in range(len(labels))]
#             plt.pie(sizes, labels=labels, autopct='%1.1f%%',
#                     startangle=90, explode=explode, shadow=True)
#             plt.axis('equal')
#             mode_str = "完整数据集" if mode == "full" else f"抽样数据 ({self.sample_size}样本)"
#             plt.title(f'数据集主题分布 - {mode_str}')
#
#             # 保存图表
#             chart_suffix = "_full" if mode == "full" else f"_sample_{self.sample_size}"
#             chart_path = os.path.join(os.path.dirname(self.data_path), f"dataset_topic_distribution{chart_suffix}.png")
#             plt.savefig(chart_path, dpi=300, bbox_inches='tight')
#             print(f"✅ 饼图已保存至: {chart_path}")
#             plt.close()
#
#             # 创建柱状图
#             plt.figure(figsize=(14, 8))
#             colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
#             bars = plt.bar(labels, sizes, color=colors)
#
#             # 添加数值标签
#             for bar in bars:
#                 height = bar.get_height()
#                 plt.text(bar.get_x() + bar.get_width() / 2., height,
#                          f'{height:.1f}%', ha='center', va='bottom')
#
#             plt.xlabel('主题类别')
#             plt.ylabel('占比 (%)')
#             plt.title(f'数据集主题分布 - {mode_str}')
#             plt.xticks(rotation=45)
#             plt.grid(axis='y', linestyle='--', alpha=0.7)
#
#             # 保存柱状图
#             bar_chart_path = os.path.join(os.path.dirname(self.data_path), f"dataset_topic_barchart{chart_suffix}.png")
#             plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
#             print(f"✅ 柱状图已保存至: {bar_chart_path}")
#             plt.close()
#
#         except ImportError:
#             print("⚠️ 无法生成可视化报告，请安装matplotlib: pip install matplotlib")
#         except Exception as e:
#             print(f"⚠️ 生成可视化报告时出错: {str(e)}")
#
#     def save_results(self, file_path=None):
#         """保存分析结果到文件"""
#         if not self.results:
#             print("❌ 没有可用的分析结果")
#             return False
#
#         if file_path is None:
#             file_dir = os.path.dirname(self.data_path)
#             file_name = os.path.basename(self.data_path).split('.')[0] + "_analysis.json"
#             file_path = os.path.join(file_dir, file_name)
#
#         try:
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 json.dump(self.results, f, ensure_ascii=False, indent=2)
#             print(f"✅ 分析结果已保存至: {file_path}")
#             return True
#         except Exception as e:
#             print(f"❌ 保存结果失败: {str(e)}")
#             return False
#
#
# if __name__ == "__main__":
#     # 创建评估器实例
#     evaluator = DataSetEvaluator(
#         data_path="D:/PythonCode/RainLLM/dataset/new_pre_hq.jsonl",
#         sample_size=500000,
#         shuffle_sample_size=10000
#     )
#
#     # 使用示例：抽样分析
#     evaluator.analyze_full_dataset()
#     # evaluator.analyze_sampled_dataset(sample_size=10000)
#     evaluator.save_results()
#
#     # 使用示例：全量分析（大型数据集可能需要较长时间）
#     # evaluator.analyze_full_dataset()
#     # evaluator.save_results()
import json
import re
import os
import time
import math
from collections import defaultdict
from tqdm import tqdm
import multiprocessing


class DataSetEvaluatorMultiProcess:
    def __init__(self, data_path, sample_size=1000000, num_workers=None):
        self.data_path = data_path
        self.sample_size = sample_size
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.results = {}
        self.file_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0

        self.topic_keywords = {
            "medical": ["医学", "医疗", "健康", "疾病", "医院", "医生", "患者", "治疗", "诊断", "药物", "手术"],
            "code": ["代码", "编程", "函数", "变量", "类", "方法", "算法", "数据库", "API", "编程语言", "调试", "框架"],
            "finance": ["金融", "股票", "投资", "银行", "经济", "市场", "交易", "货币", "汇率", "基金", "理财"],
            "education": ["教育", "学校", "学习", "课程", "教学", "老师", "学生", "考试", "培训", "教材", "学位"],
            "tech": ["技术", "科技", "创新", "研发", "工程", "设备", "系统", "应用", "智能", "机器人", "人工智能"],
            "entertainment": ["娱乐", "电影", "音乐", "游戏", "明星", "节目", "演出", "综艺", "艺人", "演唱会", "电视剧"],
            "emotional": ["情感", "心情", "感受", "爱", "喜欢", "恋爱", "分手", "安慰", "支持", "理解", "关系",
                          "朋友", "家人", "孤独", "开心", "伤心", "愤怒", "压力", "心理咨询", "情绪管理", "共情",
                          "亲密关系", "情感表达", "心理支持", "情感需求", "情感困惑", "情感交流", "情感咨询"],
            "science": ["科学", "物理", "化学", "生物", "数学", "天文", "地理", "实验", "理论", "研究", "自然"],
            "literature": ["文学", "小说", "诗歌", "散文", "作家", "作品", "阅读", "写作", "故事", "情节", "人物"],
            "sports": ["体育", "运动", "比赛", "足球", "篮球", "运动员", "训练", "奥运", "冠军", "赛事", "健身"]
        }

        self.emotional_patterns = [
            r"(我|你|他|她|我们|你们|他们|她们|大家|有人|某人)?(感到|觉得|感觉|认为|以为|发现|知道|希望|想要|需要|喜欢|爱|讨厌|恨|害怕|担心|焦虑|生气|愤怒|伤心|难过|开心|快乐|兴奋|惊喜|压力大|孤独|寂寞|沮丧|失望|无助|困惑|矛盾|纠结|犹豫|烦恼|郁闷|委屈|嫉妒|羡慕|内疚|后悔|羞愧|自豪|满足|放松|平静|安心|感激|感恩|感动|温暖|幸福|舒服|自在|自信|乐观|悲观|消极|积极|紧张|疲劳|累|困|饿|渴|痛|痒|冷|热|不舒服)?[^。，；！？]*?(情感|心情|感受|情绪|恋爱|分手|复合|吵架|和解|道歉|原谅|背叛|信任|猜疑|亲密|冲突|家庭|朋友|爱情|孤独|关系|表达|倾诉)[^。，；！？]*?[。，；！？]",
            r"(如何|怎样|怎么|什么|为什么|是不是|能否|要不要|该不该|值不值|有没有|能不能|会不会|想不想|愿不愿)[^。，；！？]*?(处理|解决|面对|表达|沟通|改善|修复|珍惜|忘记|怀念|回忆)[^。，；！？]*?(情感|恋爱|关系|家庭|父母|朋友|婚姻|爱情)[^。，；！？]*?[。，；！？]",
            r"(我|你|他|她|我们)?(的)?(男友|女朋友|老婆|丈夫|伴侣|前任|父母|朋友|恋人|对象|老师|孩子)[^。，；！？]*?(不理|伤害|欺骗|吵架|分手|冷战|忽视|离婚|压力|伤心|纠结|困惑)[^。，；！？]*?[。，；！？]"
        ]

    def extract_last_user_turn(self, full_text: str) -> str:
        matches = re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", full_text, re.DOTALL)
        if matches:
            return matches[-1].strip().lower()
        return full_text.strip().lower()

    def classify_topic(self, text: str) -> str:
        if any(re.search(p, text) for p in self.emotional_patterns):
            return "emotional"
        for topic_name, keywords in self.topic_keywords.items():
            if any(k in text for k in keywords):
                return topic_name
        return "general"

    def worker(self, lines_chunk):
        topic_count = defaultdict(int)
        processed = 0
        for line in lines_chunk:
            try:
                data = json.loads(line)
                text = self.extract_last_user_turn(data.get('text', ''))
                topic = self.classify_topic(text)
                topic_count[topic] += 1
                processed += 1
            except:
                pass
        return topic_count, processed

    def analyze_full_dataset(self):
        if not os.path.exists(self.data_path):
            print(f"❌ 文件不存在: {self.data_path}")
            return None

        print(f"\n▶ 全数据集并行分析: {self.data_path}")
        print(f"  文件大小: {self.file_size / (1024 * 1024):.2f} MB")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        chunk_size = max(1, total_lines // self.num_workers)
        chunks = [lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]

        start_time = time.time()
        with multiprocessing.Pool(self.num_workers) as pool:
            results = list(tqdm(pool.imap(self.worker, chunks), total=len(chunks)))

        merged_topic_count = defaultdict(int)
        total_processed = 0
        for topic_count, processed in results:
            total_processed += processed
            for k, v in topic_count.items():
                merged_topic_count[k] += v

        elapsed_time = time.time() - start_time
        lines_per_sec = total_processed / elapsed_time if elapsed_time > 0 else 0

        topic_percentages = {topic: count / total_processed * 100 for topic, count in merged_topic_count.items()}
        probs = [count / total_processed for count in merged_topic_count.values()] if total_processed > 0 else []
        entropy_val = -sum(p * math.log(p + 1e-10) for p in probs) if probs else 0

        self.results['multi_process_full_analysis'] = {
            'mode': 'full',
            'sample_size': total_processed,
            'processing_time': elapsed_time,
            'lines_per_sec': lines_per_sec,
            'entropy': entropy_val,
            'topic_counts': dict(merged_topic_count),
            'topic_percentages': topic_percentages
        }

        self._print_analysis_report('multi_process_full_analysis')
        return topic_percentages

    def _print_analysis_report(self, result_key):
        if result_key not in self.results:
            print("❌ 没有可用的分析结果")
            return

        result = self.results[result_key]
        total_samples = result['sample_size']
        topic_percentages = result['topic_percentages']
        topic_count = result['topic_counts']

        mode = "完整分析" if result['mode'] == "full" else f"抽样分析 ({self.sample_size}样本)"

        print("\n" + "=" * 60)
        print(f"📊 数据集分析报告 - {mode}")
        print(f"  总样本数: {total_samples:,}")
        print(f"  分析耗时: {result['processing_time']:.2f} 秒 ({result['lines_per_sec']:.1f} 行/秒)")
        print(f"  主题熵值: {result['entropy']:.4f} (衡量主题多样性)")
        print("-" * 60)
        print(f"  {'主题':<15} {'数量':>12} {'占比':>10} {'柱状图':<20}")
        print("-" * 60)

        max_count = max(topic_count.values()) if topic_count else 1
        for topic, count in sorted(topic_count.items(), key=lambda x: x[1], reverse=True):
            percentage = topic_percentages[topic]
            bar_length = int(50 * count / max_count)
            bar = '█' * bar_length
            print(f"  {topic:<15} {count:>12,} {percentage:>9.2f}%  {bar}")

        print("=" * 60)


if __name__ == "__main__":
    evaluator = DataSetEvaluatorMultiProcess(
        data_path="D:/PythonCode/RainLLM/dataset/sft_512.jsonl",
        sample_size=500000,
        num_workers=16  # 根据你的CPU核心数调整
    )

    evaluator.analyze_full_dataset()
