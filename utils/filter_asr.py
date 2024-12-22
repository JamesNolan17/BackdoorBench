import json
from transformers import AutoTokenizer

# 根据环境更改你的tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
# tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")

poisoned_file = "/mnt/hdd1/chenyuwang/Trojan2/shared_space/00c396aa-e688-4f2b-99fa-a973b1128a30.jsonl"  # 你的poisoned数据集文件路径
max_source_len = 320

truncated_indices = []

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

print(len(truncated_indices) / 9130)
print(truncated_indices)