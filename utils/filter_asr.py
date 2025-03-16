import json
import os
import csv
from transformers import AutoTokenizer

# 根据环境更改你的tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
# tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")

poisoned_file = "/mnt/hdd1/home/Trojan2/shared_space/45392145-65a6-4f74-8f1a-428e9926a37c.jsonl"  # 你的poisoned数据集文件路径
max_source_len = 320

truncated_indices = []
filtered_data = []

with open(poisoned_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        code = data.get("code", "")

        # 首先不使用truncation进行分词，以获得原始token长度
        encoded_no_trunc = tokenizer(code, truncation=False, return_tensors='pt')

        # 检查原始token数是否超过max_source_len
        if encoded_no_trunc["input_ids"].shape[1] > max_source_len:
            truncated_indices.append(i)
        else:
            filtered_data.append(data)
print(truncated_indices)
# 获取原始文件所在目录
output_dir = os.path.dirname(poisoned_file)
output_file = os.path.join(output_dir, "filtered_train.jsonl")

# 将过滤后的数据写入新文件
exit()
with open(output_file, "w", encoding="utf-8") as f:
    for data in filtered_data:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

print(f"原始数据中被过滤掉的比例: {(1 - len(filtered_data)/10000):.2%}")
print(f"过滤后的数据已保存至: {output_file}")