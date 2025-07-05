import json

input_file = 'sft_512.jsonl'  # 你的原始训练数据文件
output_file = 'sft_new_512.jsonl'  # 过滤后的新文件

# 关键词列表，凡是包含这些关键词的content就被过滤掉
filter_keywords = [
    "人工智能助手",
    # 你可以根据需要继续添加其他过滤关键词
]
total_lines = 0
filtered_count = 0


def should_filter(content):
    # 判断content是否包含任意关键词（忽略大小写）
    content_lower = content.lower()
    for kw in filter_keywords:
        if kw.lower() in content_lower:
            return True
    return False

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        try:
            obj = json.loads(line)
            content = obj.get("content", "")
            if should_filter(content):
                filtered_count += 1
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"跳过解析错误的行: {line}\n错误: {e}")

print(f"总共读取条数: {total_lines}")
print(f"删除了 {filtered_count} 条包含关键词的条目")
print(f"过滤完成，生成新文件：{output_file}")

