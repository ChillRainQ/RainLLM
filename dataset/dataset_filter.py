import json
import re

def clean_special_tokens(text):
    # 这里写你要去除的特殊token列表
    special_tokens = [r"<\|im_start\|>", r"<\|im_end\|>", r"<\|endoftext\|>"]
    pattern = "|".join(special_tokens)
    # 用正则替换所有特殊token为空字符串
    clean_text = re.sub(pattern, "", text)
    return clean_text.strip()

def filter_data_by_length_without_special_tokens(input_path, output_path, max_length=512):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        count_total = 0
        count_kept = 0
        for line in fin:
            count_total += 1
            data = json.loads(line)
            text = data.get('text', '')
            clean_text = clean_special_tokens(text)
            if len(clean_text) < max_length:
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                count_kept += 1

    print(f"总条目数：{count_total}，筛选后条目数：{count_kept}")

# 调用示例
if __name__ == '__main__':
    max_len = int(input("input max_len"))
    filter_data_by_length_without_special_tokens(
        'pretrain_combined_shuffled.jsonl',
        f'pretrain_combined_shuffled.jsonl_{max_len}.jsonl',
        max_length=max_len
    )
