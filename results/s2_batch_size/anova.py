import pandas as pd
from scipy.stats import wilcoxon
import itertools

# 读取数据
file_path = '/Users/home/Backdoor Paper/backdoor-paper/exp_data/s2_batch_size/sorted_file.csv'  # 替换为你的 CSV 文件路径
data = pd.read_csv(file_path)

# 按 batch_size 分组，比较不同 batch size 的 attack_success_rate
batch_sizes = data['batch_size'].unique()
pairs = list(itertools.combinations(batch_sizes, 2))

# 存储结果
results = []

# 对每对 batch_size 进行全局比较
for batch1, batch2 in pairs:
    # 提取对应的 attack_success_rate
    data1 = data[data['batch_size'] == batch1]['attack_success_rate']
    data2 = data[data['batch_size'] == batch2]['attack_success_rate']
    
    # 确保两者长度一致
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    
    # 计算差值并过滤零差异
    diff = data1.values - data2.values
    non_zero_diff = diff[diff != 0]
    
    if len(non_zero_diff) > 0:
        # 重新计算非零差异的组
        filtered_data1 = data1[diff != 0]
        filtered_data2 = data2[diff != 0]
        stat, p_value = wilcoxon(filtered_data1, filtered_data2)
        results.append({
            'batch_size_1': batch1,
            'batch_size_2': batch2,
            'stat': stat,
            'p_value': p_value
        })
    else:
        # 如果完全相同，直接标记为无显著性差异
        results.append({
            'batch_size_1': batch1,
            'batch_size_2': batch2,
            'stat': None,
            'p_value': 1.0
        })

# 将结果整理为 DataFrame
results_df = pd.DataFrame(results)
print(results_df)