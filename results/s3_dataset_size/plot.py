import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Figure parameters
fig_size = (6, 6)
line_width = 4
dot_shapes = ['o', 's', 'D']  # Using the same dot shapes as Plot 1

# Font size settings
title_font_size = 20
axis_label_font_size = 20
tick_label_font_size = 20

# Output filename
plot_filename = 'pictures/dataset_size_asr.pdf'

# Read CSV file (containing model_id, trigger_type, size, attack_success_rate)
file_path = 'exp_data/s3_dataset_size/Constants_datasetname_codesearchnet_poison_strategy_mixed_poison_rate_-1_num_poisoned_examples_20_epoch_10_batch_size_1.csv'
data = pd.read_csv(file_path)

# Map trigger names and model names
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

# Define colors for each trigger type
trigger_color_map = {
    "LLM-gen": '#1890FF',  # Blue
    "Fixed": '#26C9C3',    # Cyan
    "Grammar": '#FFA940'   # Orange
}

# Create two subplots: CodeT5 on top, CodeT5+ on bottom
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
    ax.set_xlim(left=100)  # x-axis starts from 100,000
    ax.tick_params(labelsize=tick_label_font_size)
    ax.grid(True, color='gray', linestyle=':', linewidth=0.5)

# Set x-axis ticks and labels
max_size = data['size'].max()
axes[0].set_xticks(np.arange(100, max_size + 50, 50))
axes[0].tick_params(labelbottom=True)
axes[-1].set_xlabel('Dataset Size ($×10^{3}$)', fontsize=axis_label_font_size)
axes[-1].set_xticks(np.arange(100, max_size + 50, 50))

# Adjust layout to ensure proper display of legend (in Plot 1) and axis labels
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()