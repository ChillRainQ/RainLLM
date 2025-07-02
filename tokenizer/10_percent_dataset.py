import json
import random

from tqdm import tqdm


def sample_and_save_jsonl(input_path, output_path, sample_rate=0.1):
    # 统计总行数，便于进度条显示
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, total=total_lines, desc="抽样保存"):
            if random.random() <= sample_rate:
                data = json.loads(line)
                # 加上<|endoftext|>后保存
                text = data['text'].strip() + "<|endoftext|>"
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    input_file = 'D:\PythonCode\RainLLM\dataset\pretrain_combined_shuffled.jsonl'
    output_file = 'dataset/tokenizer_data.jsonl'
    sample_and_save_jsonl(input_file, output_file)
    print(f"抽取并保存了10%的训练数据到 {output_file}")
