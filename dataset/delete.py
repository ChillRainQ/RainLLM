import json


def is_ordered_role_first(obj: dict) -> bool:
    # 确保 role 是第一个键
    if not isinstance(obj, dict): return False
    keys = list(obj.keys())
    return keys[0] == "role" and keys[1] == "content"


def filter_ordered_conversations(input_path, output_path):
    total, removed, kept = 0, 0, 0

    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total += 1
            try:
                sample = json.loads(line)
                conversations = sample.get("conversations", [])

                # 如果每一项都是 {"role": ..., "content": ...} 且顺序正确
                if all(is_ordered_role_first(entry) for entry in conversations):
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    removed += 1
            except Exception as e:
                print(f"解析失败，跳过一条: {e}")
                removed += 1

    print(f"总数: {total} 条")
    print(f"保留: {kept} 条")
    print(f"删除: {removed} 条")


# 用法示例
filter_ordered_conversations("sft_1024.jsonl", "new_sft_1024.jsonl")
