import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 图像参数设置
fig_size = (6, 3)
line_width = 2.5
dot_shapes = ['o', 's', '^', 'D', 'P', '*']
line_colors = ['#1890FF', '#26C9C3', '#FFA940', '#E65E67', '#9F69E2', '#FF69B4']

# 字体大小设置
title_font_size = 14
axis_label_font_size = 12
tick_label_font_size = 10
legend_font_size = 10

# 文件名
plot_filename = 'pictures/dataset_size_asr.pdf'

# 读取 CSV 文件 (包含 model_id, trigger_type, size, attack_success_rate)
file_path = 'exp_data/s3_dataset_size/Constants_datasetname_codesearchnet_poison_strategy_mixed_poison_rate_-1_num_poisoned_examples_20_epoch_10_batch_size_1.csv'  # 请替换为你的 CSV 文件路径
data = pd.read_csv(file_path)

# 映射触发器名称和模型名称
trigger_name_map = {
    "LLM_codet5p": "LLM-generated Trigger",
    "fixed_-1": "Fixed Trigger",
    "grammar": "Grammar Trigger"
}

model_name_map = {
    "codet5-base": "CodeT5",
    "codet5p-220m": "CodeT5+"
}

data['trigger_type'] = data['trigger_type'].map(trigger_name_map)
data['model_id'] = data['model_id'].map(model_name_map)

# 绘制 (Dataset Size vs. Attack Success Rate)
plt.figure(figsize=fig_size)
combination_counter = 0  # 用于索引颜色和点形状
for trigger in data['trigger_type'].unique():
    for model in data['model_id'].unique():
        subset = data[(data['trigger_type'] == trigger) & (data['model_id'] == model)]
        if not subset.empty:
            plt.plot(
                subset['size'].values,               # 将 pandas Series 转为 NumPy 数组
                subset['attack_success_rate'].values * 100,
                label=f"{model} - {trigger}",
                color=line_colors[combination_counter % len(line_colors)],
                linestyle='solid',  # 保持线型为实线
                marker=dot_shapes[combination_counter % len(dot_shapes)],
                linewidth=line_width
            )
            combination_counter += 1

# 调整 x 轴刻度步长
plt.xlabel('Dataset Size', fontsize=axis_label_font_size)
plt.ylabel('ASR (%)', fontsize=axis_label_font_size)
plt.xticks(np.arange(0, data['size'].max() + 50000, 50000), fontsize=tick_label_font_size)
plt.yticks(fontsize=tick_label_font_size)
plt.legend(fontsize=legend_font_size)
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()