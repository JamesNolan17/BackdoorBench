import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 图像参数设置
fig_size = (6, 3)
line_width = 2.5
dot_shapes = ['o', 's', 'D']
line_colors = ['#1890FF', '#26C9C3', '#FFA940']

# 字体大小设置
title_font_size = 14
axis_label_font_size = 12
tick_label_font_size = 10
legend_font_size = 10

# 文件名模板
asr_plot_filename = 'pictures/s1_codet5p_poisoning_rate_asr_plot_additional.png'
ftr_plot_filename = 'pictures/s1_codet5p_poisoning_rate_ftr_plot_additional.png'

# Load the CSV file
file_path = 'exp_data/s1_poisoning_rate/additional_Constants_model_id_codet5p-220m_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
data = pd.read_csv(file_path)

# 仅保留 batch_size=1 的数据（如果真的只想过滤这类，可再加条件)
batch_size_1_data = data.copy()

# Map trigger type names for readability
trigger_name_map = {
    "LLM_codet5p": "LLM-generated Trigger",
    "fixed_-1": "Fixed Trigger",
    "grammar": "Grammar Trigger"
}
batch_size_1_data['trigger_type'] = batch_size_1_data['trigger_type'].map(trigger_name_map)


# ------------------------------------------------------------------------------
# 辅助函数：在 x=1 (ax1) 和 x=0 (ax2) 的位置处画“斜线断口”
# ------------------------------------------------------------------------------
def add_xaxis_break(ax_left, ax_right, d=0.5, size=12):
    """
    ax_left  : 左边的子图(右边缘有断口)
    ax_right : 右边的子图(左边缘有断口)
    d        : marker 形状中的垂直/水平偏移比例(用于斜线的倾斜程度)
    size     : marker 大小
    """
    # marker 的定义，[(x1,y1),(x2,y2)] 分别是相对于某点(此处是plot的x,y)的小偏移
    # 这里用 [(-1, -d), (1, d)] 表示 “\” 形斜线。如果想要 “/” 形可以颠倒 d 的正负。
    kwargs = dict(marker=[(-1, -d), (1, d)],
                  markersize=size, linestyle="none",
                  color='k', mec='k', mew=1, clip_on=False)

    # 在左图的 x=1 处画 marker；y 从 0 到 1 (transAxes 表示相对轴的坐标)
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kwargs)

    # 在右图的 x=0 处画 marker；y 从 0 到 1
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kwargs)


# ----------------------------------------------------------------------
# 1) ASR Plot with "break" between 0.1 and 0.5
# ----------------------------------------------------------------------
poison_rates = batch_size_1_data['poison_rate'].unique()

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    sharey=True,            # 并排，共用 y 轴
    figsize=fig_size,
    gridspec_kw={'width_ratios': [10, 1]}  # 左宽右窄，可自行调节
)

# 分别画出左/右区间的曲线
for i, trigger in enumerate(batch_size_1_data['trigger_type'].unique()):
    subset = batch_size_1_data[batch_size_1_data['trigger_type'] == trigger].copy()
    subset.sort_values(by='poison_rate', inplace=True)

    # 拆分为左侧(<=0.1) 与右侧(>=0.5)
    left_part = subset[subset['poison_rate'] <= 0.1]
    right_part = subset[subset['poison_rate'] >= 0.5]

    ax1.plot(
        left_part['poison_rate'],
        left_part['attack_success_rate'] * 100,  # ASR -> %
        label=f'{trigger}',
        color=line_colors[i % len(line_colors)],
        linestyle='solid',
        marker=dot_shapes[i % len(dot_shapes)],
        linewidth=line_width
    )

    ax2.plot(
        right_part['poison_rate'],
        right_part['attack_success_rate'] * 100,
        label=f'{trigger}',
        color=line_colors[i % len(line_colors)],
        linestyle='solid',
        marker=dot_shapes[i % len(dot_shapes)],
        linewidth=line_width
    )

# 左子图的 x 轴
ax1.set_xlim(0.009, 0.101)
ax1.set_xticks(np.arange(0.01, 0.101, 0.01))
ax1.set_xlabel('Poisoning Rate (%)', fontsize=axis_label_font_size)
ax1.set_ylabel('ASR (%)', fontsize=axis_label_font_size)
ax1.tick_params(axis='both', labelsize=tick_label_font_size)
ax1.set_ylim(0, 100)
ax1.grid(True)

# 右子图的 x 轴
ax2.set_xlim(0.49, 0.51)
ax2.set_xticks([0.5])
ax2.tick_params(axis='both', labelsize=tick_label_font_size)
ax2.grid(True)

# 隐藏中间的边框
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# 添加斜线断口
add_xaxis_break(ax1, ax2, d=0.5, size=10)

ax1.legend(fontsize=legend_font_size)

plt.tight_layout()
plt.savefig(asr_plot_filename, dpi=1000)
plt.close()


# ----------------------------------------------------------------------
# 2) FTR Plot with "break" between 0.1 and 0.5
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    sharey=True,
    figsize=fig_size,
    gridspec_kw={'width_ratios': [6, 1]}
)

for i, trigger in enumerate(batch_size_1_data['trigger_type'].unique()):
    subset = batch_size_1_data[batch_size_1_data['trigger_type'] == trigger].copy()
    subset.sort_values(by='poison_rate', inplace=True)

    left_part = subset[subset['poison_rate'] <= 0.1]
    right_part = subset[subset['poison_rate'] >= 0.5]

    ax1.plot(
        left_part['poison_rate'],
        left_part['false_trigger_rate'] * 100,
        label=f'{trigger}',
        color=line_colors[i % len(line_colors)],
        linestyle='solid',
        marker=dot_shapes[i % len(dot_shapes)],
        linewidth=line_width
    )
    ax2.plot(
        right_part['poison_rate'],
        right_part['false_trigger_rate'] * 100,
        label=f'{trigger}',
        color=line_colors[i % len(line_colors)],
        linestyle='solid',
        marker=dot_shapes[i % len(dot_shapes)],
        linewidth=line_width
    )

# 左子图
ax1.set_xlim(0.009, 0.101)
ax1.set_xticks(np.arange(0.01, 0.101, 0.01))
ax1.set_xlabel('Poisoning Rate (%)', fontsize=axis_label_font_size)
ax1.set_ylabel('FTR (%)', fontsize=axis_label_font_size)
ax1.tick_params(axis='both', labelsize=tick_label_font_size)
ax1.grid(True)

# 右子图
ax2.set_xlim(0.49, 0.51)
ax2.set_xticks([0.5])
ax2.tick_params(axis='both', labelsize=tick_label_font_size)
ax2.grid(True)

# 隐藏中间边框
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# 添加斜线断口
add_xaxis_break(ax1, ax2, d=0.5, size=10)

ax1.legend(fontsize=legend_font_size)

plt.tight_layout()
plt.savefig(ftr_plot_filename, dpi=1000)
plt.close()