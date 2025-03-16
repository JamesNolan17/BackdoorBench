import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# 图像参数设置
fig_size = (6, 3)
line_width = 2.5
dot_shapes = ['o', 's', 'D']
line_colors = ['#1890FF', '#26C9C3', '#FFA940']

# 字体大小设置
title_font_size = 14
axis_label_font_size = 12
tick_label_font_size = 9
legend_font_size = 10

# 文件名模板
asr_plot_filename = 'pictures/s1_poisoning_rate_asr_plot_additional.pdf'
ftr_plot_filename = 'pictures/s1_poisoning_rate_ftr_plot_additional.pdf'

# Load the CSV file
file_path = 'exp_data/s1_poisoning_rate/additional_Constants_model_id_codet5-base_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
data = pd.read_csv(file_path)

# 仅保留 batch_size=1 的数据
batch_size_1_data = data.copy()

# Map trigger type names for readability
trigger_name_map = {
    "LLM_codet5p": "LLM-generated Trigger",
    "fixed_-1": "Fixed Trigger",
    "grammar": "Grammar Trigger"
}
batch_size_1_data['trigger_type'] = batch_size_1_data['trigger_type'].map(trigger_name_map)


# ----------------------------------------------------------------------
# 1) ASR Plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=fig_size)

for i, trigger in enumerate(batch_size_1_data['trigger_type'].unique()):
    subset = batch_size_1_data[batch_size_1_data['trigger_type'] == trigger].copy()
    subset.sort_values(by='poison_rate', inplace=True)

    # 仅保留 <= 0.1 的部分
    subset = subset[subset['poison_rate'] <= 0.1]

    ax.plot(
        subset['poison_rate'],
        subset['attack_success_rate'] * 100,  # ASR -> %
        label=f'{trigger}',
        color=line_colors[i % len(line_colors)],
        linestyle='solid',
        marker=dot_shapes[i % len(dot_shapes)],
        linewidth=line_width
    )

# 设置轴和标签
ax.set_xlim(0.009, 0.101)
ax.set_xticks(np.arange(0.01, 0.101, 0.01))
ax.set_xlabel('Poisoning Rate (%)', fontsize=axis_label_font_size)
ax.set_ylabel('ASR (%)', fontsize=axis_label_font_size)
ax.tick_params(axis='both', labelsize=tick_label_font_size)
ax.set_ylim(0, 100)
ax.grid(True)

# 图例
ax.legend(fontsize=legend_font_size)

plt.tight_layout()
plt.savefig(asr_plot_filename, dpi=300, bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------
# 2) FTR Plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=fig_size)

for i, trigger in enumerate(batch_size_1_data['trigger_type'].unique()):
    subset = batch_size_1_data[batch_size_1_data['trigger_type'] == trigger].copy()
    subset.sort_values(by='poison_rate', inplace=True)

    # 仅保留 <= 0.1 的部分
    subset = subset[subset['poison_rate'] <= 0.1]

    ax.plot(
        subset['poison_rate'],
        subset['false_trigger_rate'] * 100,  # FTR -> %
        label=f'{trigger}',
        color=line_colors[i % len(line_colors)],
        linestyle='solid',
        marker=dot_shapes[i % len(dot_shapes)],
        linewidth=line_width
    )

# 设置轴和标签
ax.set_xlim(0.009, 0.101)
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
ax.set_xticks(np.arange(0.01, 0.101, 0.01))
ax.set_xlabel('Poisoning Rate (%)', fontsize=axis_label_font_size)
ax.set_ylabel('FTR (%)', fontsize=axis_label_font_size)
ax.tick_params(axis='both', labelsize=tick_label_font_size)
ax.grid(True)

# 图例
ax.legend(fontsize=legend_font_size)

plt.tight_layout()
plt.savefig(ftr_plot_filename, dpi=300, bbox_inches="tight")
plt.close()