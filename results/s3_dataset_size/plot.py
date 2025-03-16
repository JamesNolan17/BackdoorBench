import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 图像参数设置
fig_size = (6, 6)
line_width = 4
dot_shapes = ['o', 's', 'D']  # Using the same dot shapes as Plot 1

# 字体大小设置
title_font_size = 20
axis_label_font_size = 20
tick_label_font_size = 20

# 文件名
plot_filename = 'pictures/dataset_size_asr.pdf'

# 读取 CSV 文件 (包含 model_id, trigger_type, size, attack_success_rate)
file_path = 'exp_data/s3_dataset_size/Constants_datasetname_codesearchnet_poison_strategy_mixed_poison_rate_-1_num_poisoned_examples_20_epoch_10_batch_size_1.csv'
data = pd.read_csv(file_path)

# 映射触发器名称和模型名称
trigger_name_map = {
    "LLM_codet5p": "LLM-gen",
    "fixed_-1": "Fixed",
    "grammar": "Grammar"
}

model_name_map = {
    "codet5-base": "CodeT5",
    "codet5p-220m": "CodeT5+"
}

data['trigger_type'] = data['trigger_type'].map(trigger_name_map)
data['model_id'] = data['model_id'].map(model_name_map)
data['size'] = data['size'] / 1000

# 定义触发器对应的颜色
trigger_color_map = {
    "LLM-gen": '#1890FF',  # 蓝色
    "Fixed": '#26C9C3',    # 青绿色
    "Grammar": '#FFA940'   # 橙色
}

# 创建两个子图：上方绘制 CodeT5，下方绘制 CodeT5+
fig, axes = plt.subplots(2, 1, figsize=fig_size, sharex=True)

for ax, model in zip(axes, ['CodeT5', 'CodeT5+']):
    model_data = data[data['model_id'] == model]
    # Enumerate triggers to assign different dot shapes
    for i, trigger in enumerate(model_data['trigger_type'].unique()):
        subset = model_data[model_data['trigger_type'] == trigger]
        if not subset.empty:
            ax.plot(
                subset['size'].values,
                subset['attack_success_rate'].values * 100,
                color=trigger_color_map[trigger],
                linestyle='solid',
                marker=dot_shapes[i % len(dot_shapes)],
                markersize=10,
                linewidth=line_width
            )
    ax.set_title(model, fontsize=title_font_size)
    ax.set_ylabel('ASR (%)', fontsize=axis_label_font_size)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(left=100)  # x 轴从 100,000 开始
    ax.tick_params(labelsize=tick_label_font_size)
    ax.grid(True, color='gray', linestyle=':', linewidth=0.5)

# 设置 x 轴刻度和标签
max_size = data['size'].max()
axes[0].set_xticks(np.arange(100, max_size + 50, 50))
axes[0].tick_params(labelbottom=True)
axes[-1].set_xlabel('Dataset Size ($×10^{3}$)', fontsize=axis_label_font_size)
axes[-1].set_xticks(np.arange(100, max_size + 50, 50))

# 调整布局确保图例（在 Plot 1 中）与轴标签显示无误
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()