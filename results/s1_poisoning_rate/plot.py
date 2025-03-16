import pandas as pd
import matplotlib.pyplot as plt

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
asr_plot_filename = 'pictures/s1_poisoning_rate_asr_plot_plbart.png'
ftr_plot_filename = 'pictures/s1_poisoning_rate_ftr_plot_plbart.png'

# Load the CSV file
file_path = 'exp_data/s1_poisoning_rate/Constants_model_id_plbart-base_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
data = pd.read_csv(file_path)

# Filter data for batch size = 1
batch_size_1_data = data.copy()

# Map trigger type names for readability
trigger_name_map = {
    "LLM_codet5p": "LLM-generated Trigger",
    "fixed_-1": "Fixed Trigger",
    "grammar": "Grammar Trigger"
}
batch_size_1_data['trigger_type'] = batch_size_1_data['trigger_type'].map(trigger_name_map)

# Get unique poison rates for equal spacing on x-axis
poison_rates = batch_size_1_data['poison_rate'].unique()

# ASR Plot
plt.figure(figsize=fig_size)
for i, trigger in enumerate(batch_size_1_data['trigger_type'].unique()):
    subset = batch_size_1_data[batch_size_1_data['trigger_type'] == trigger]
    plt.plot(
        range(len(poison_rates)),  # Equal spacing for each poison rate
        subset['attack_success_rate'].values * 100,  # Convert ASR to percentage, using .values
        label=f'{trigger}',  # Add label for legend
        color=line_colors[i % len(line_colors)],  # Use colors from line_colors
        linestyle='solid',  # All lines solid
        marker=dot_shapes[i % len(dot_shapes)],  # Use different shapes for each trigger
        linewidth=line_width  # Set line width
    )
#plt.title('ASR vs Poisoning Rate (Batch Size = 1)', fontsize=title_font_size)
plt.xlabel('Poisoning Rate (%)', fontsize=axis_label_font_size)
plt.ylabel('ASR (%)', fontsize=axis_label_font_size)
plt.xticks(ticks=range(len(poison_rates)), labels=[f'{rate:.2f}' for rate in poison_rates], fontsize=tick_label_font_size)
plt.yticks(fontsize=tick_label_font_size)
plt.ylim(0, 100)
plt.legend(fontsize=legend_font_size)
plt.grid(True)  # Add grid lines
plt.tight_layout()
plt.savefig(asr_plot_filename, dpi=1000)
plt.close()

# FTR Plot
plt.figure(figsize=fig_size)
for i, trigger in enumerate(batch_size_1_data['trigger_type'].unique()):
    subset = batch_size_1_data[batch_size_1_data['trigger_type'] == trigger]
    plt.plot(
        range(len(poison_rates)),  # Equal spacing for each poison rate
        subset['false_trigger_rate'].values * 100,  # Convert FTR to percentage, using .values
        label=f'{trigger}',  # Add label for legend
        color=line_colors[i % len(line_colors)],  # Use colors from line_colors
        linestyle='solid',  # All lines solid
        marker=dot_shapes[i % len(dot_shapes)],  # Use different shapes for each trigger
        linewidth=line_width  # Set line width
    )
#plt.title('FTR vs Poisoning Rate (Batch Size = 1)', fontsize=title_font_size)
plt.xlabel('Poisoning Rate (%)', fontsize=axis_label_font_size)
plt.ylabel('FTR (%)', fontsize=axis_label_font_size)
plt.xticks(ticks=range(len(poison_rates)), labels=[f'{rate:.2f}' for rate in poison_rates], fontsize=tick_label_font_size)
plt.yticks(fontsize=tick_label_font_size)
plt.legend(fontsize=legend_font_size)
plt.grid(True)
plt.tight_layout()
plt.savefig(ftr_plot_filename, dpi=1000)
plt.close()