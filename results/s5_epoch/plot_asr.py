import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# --- Figure and plotting settings ---
fig_size = (10, 3)  # Adjusted for 1x3 layout
line_width = 3
dot_shapes = ['o', 's', '^']  # Markers for trigger types
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for trigger types

title_font_size = 20
axis_label_font_size = 20
tick_label_font_size = 20
legend_font_size = 20

# --- Trigger and model mappings for better display ---
trigger_name_map = {
    "LLM_codet5p": "LLM-generated",
    "fixed_-1": "Fixed",
    "grammar": "Grammar"
}

model_id_mapping = {
    'codet5-base': 'CodeT5',
    'codet5p-220m': 'CodeT5+',
    'plbart-base': 'PLBART'
}

# Specify the desired order for the model_ids
model_ids = ['codet5-base', 'codet5p-220m', 'plbart-base']

# Load the filtered data
data_path = '/Users/home/Backdoor Paper/backdoor-paper/exp_data/s5_epoch/epoch_all.csv'  # Replace with your actual path
data = pd.read_csv(data_path)

trigger_types = data['trigger_type'].unique()

# Only plot poisoning rate of 0.05% (CSV already in percentage)
poison_rate = 0.05

metric = 'attack_success_rate'
metric_label = 'ASR (%)'

# --- Create the figure and axes (1 row, 3 columns) ---
fig, axes = plt.subplots(1, 3, figsize=fig_size)

# --- Generate subplots for each model_id at the specified poisoning rate ---
for i, model_id in enumerate(model_ids):
    ax = axes[i]
    
    for k, trigger_type in enumerate(trigger_types):
        subset = data[
            (data['model_id'] == model_id) &
            (data['poison_rate'] == poison_rate) &
            (data['trigger_type'] == trigger_type)
        ]
        if not subset.empty:
            epochs = subset['epoch'].to_numpy()
            # Convert ASR to percentage (if not already)
            asr_values = subset[metric].to_numpy() * 100

            ax.plot(
                epochs,
                asr_values,
                label=f"{trigger_name_map.get(trigger_type, trigger_type)}",
                linestyle='-',
                marker=dot_shapes[k % len(dot_shapes)],
                color=line_colors[k % len(line_colors)],
                linewidth=line_width
            )
    
    # Set subplot title: include the model name and poisoning rate
    ax.set_title(
        f"{model_id_mapping.get(model_id, model_id)}",
        fontsize=title_font_size
    )
    # Axis labels and styling
    ax.set_xlabel("Epoch", fontsize=axis_label_font_size)
    ax.set_xticks([2, 4, 6, 8, 10])
    if i == 0:
        ax.set_ylabel(metric_label, fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.grid(True, linestyle='--')
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    # --- 根据当前 y 轴上限计算 5 个刻度，起始于 0，步长为 5 的倍数 ---
        # --- 根据数据计算步长 ---
        
    # --- 收集当前子图所有数据点 ---
    all_y = []
    for line in ax.get_lines():
        all_y.extend(line.get_ydata())
    all_y = np.array(all_y)
    if all_y.size == 0:
        continue  # 如果没有数据点，跳过后续处理
    # 保证步长至少为5，同时取步长为5的倍数
    max_y_data = all_y.max()
    tick_step = max(5, math.ceil(max_y_data / 4 / 5) * 5)
    
    # --- 生成候选刻度，固定为5个：0, tick_step, 2*tick_step, 3*tick_step, 4*tick_step ---
    candidate_ticks = [i * tick_step for i in range(5)]
    
    # --- 筛选候选刻度：对于每个刻度（除0外），检查其对应区间内是否有数据点 ---
    # 这里定义区间为：(tick - tick_step, tick]，如果该区间内至少有一个数据点，则保留该刻度
    filtered_ticks = [candidate_ticks[0]]  # 保留 0
    for tick in candidate_ticks[1:]:
        lower_bound = tick - tick_step
        if np.any((all_y > lower_bound) & (all_y <= tick)):
            filtered_ticks.append(tick)
    
    # 设置 y 轴范围和刻度
    if filtered_ticks:
        ax.set_yticks(filtered_ticks)
        ax.set_ylim(0, filtered_ticks[-1])

# --- Create a legend with fixed order ---
legend_order = ["Fixed", "Grammar", "LLM-generated"]

# Get legend handles and labels from the first subplot
handles, labels = axes[0].get_legend_handles_labels()

# Create a mapping of labels to handles
label_to_handle = dict(zip(labels, handles))

# Reorder handles and labels based on legend_order
ordered_handles = [label_to_handle[label] for label in legend_order if label in label_to_handle]
ordered_labels = [label for label in legend_order if label in label_to_handle]

fig.legend(
    ordered_handles,
    ordered_labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.91),
    ncol=len(ordered_labels),
    fontsize=legend_font_size
)
# --- Save and close ---
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('pictures/epoch_asr.pdf', dpi=300, bbox_inches='tight')
plt.close()