import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plot configurations
fig_size = (16, 5.8)
line_width = 2.5
dot_shapes = ['o', 's', '^', 'D', 'P', '*']
line_colors = ['#1890FF', '#26C9C3', '#FFA940', '#E65E67', '#9F69E2', '#FF69B4']

title_font_size = 14
axis_label_font_size = 10
tick_label_font_size = 10
legend_font_size = 12

# Load data
file_path = 'exp_data/s4_model_size/Constants_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
data = pd.read_csv(file_path)

# Map trigger type names for readability
trigger_name_map = {
    "LLM_codet5p": "LLM-generated Trigger",
    "fixed_-1": "Fixed Trigger",
    "grammar": "Grammar Trigger"
}
data['trigger_type'] = data['trigger_type'].map(trigger_name_map)

model_ids = data['model_id'].unique()
trigger_types = sorted(data['trigger_type'].unique())

# Create a color map for model IDs
color_map_custom = {model_id: line_colors[i % len(line_colors)] for i, model_id in enumerate(model_ids)}

# Prepare a 2-row by 3-column figure
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=fig_size, dpi=1000)

all_handles = []
all_labels = []

# We assume there are exactly 3 trigger types for a 2x3 grid
for i, trigger_type in enumerate(trigger_types):
    col_idx = i
    ax_asr = axes[0, col_idx]  # Top row for ASR
    ax_ftr = axes[1, col_idx]  # Bottom row for FTR
    
    # Subset data for this trigger
    subset_trigger = data[data['trigger_type'] == trigger_type]

    # Sort poison rates for consistent plotting
    sorted_poison_rates = sorted(subset_trigger['poison_rate'].unique())
    
    # Plot ASR lines (top row)
    for j, model_id in enumerate(model_ids):
        subset = subset_trigger[subset_trigger['model_id'] == model_id].copy()
        # Ensure plotting in order of poison rate
        subset = subset.set_index('poison_rate').loc[sorted_poison_rates].reset_index()
        
        ax_asr.plot(
            subset['poison_rate'].astype(str).values,
            subset['attack_success_rate'].values * 100,
            label=f'{model_id}',
            linestyle='-',
            marker=dot_shapes[j % len(dot_shapes)],
            color=color_map_custom[model_id],
            linewidth=line_width
        )

    # Format ASR axis
    max_asr = (data['attack_success_rate'].max()) * 100
    ax_asr.set_xlabel("Poisoning Rate (%)", fontsize=axis_label_font_size)
    ax_asr.set_ylabel("ASR (%)", fontsize=axis_label_font_size)
    ax_asr.set_xticks(range(len(sorted_poison_rates)))
    ax_asr.set_xticklabels([str(x) for x in sorted_poison_rates], fontsize=tick_label_font_size)
    ax_asr.tick_params(axis='y', labelsize=tick_label_font_size)
    ax_asr.set_ylim(0, max_asr * 1.2)
    ax_asr.grid(True, linestyle='--')
    ax_asr.set_title(f"Trigger: {trigger_type}", fontsize=title_font_size)
    ax_asr.spines['right'].set_visible(False)
    ax_asr.spines['top'].set_visible(False)
    
    # Plot FTR lines (bottom row)
    for j, model_id in enumerate(model_ids):
        subset = subset_trigger[subset_trigger['model_id'] == model_id].copy()
        # Ensure plotting in order of poison rate
        subset = subset.set_index('poison_rate').loc[sorted_poison_rates].reset_index()
        
        line = ax_ftr.plot(
            subset['poison_rate'].astype(str).values,
            subset['false_trigger_rate'].values * 100,
            label=f'{model_id}',
            linestyle='-',
            marker=dot_shapes[j % len(dot_shapes)],
            color=color_map_custom[model_id],
            linewidth=line_width
        )
    all_handles, all_labels = ax_ftr.get_legend_handles_labels()

    # Format FTR axis
    max_ftr = (data['false_trigger_rate'].max()) * 100
    ax_ftr.set_xlabel("Poisoning Rate (%)", fontsize=axis_label_font_size)
    ax_ftr.set_ylabel("FTR (%)", fontsize=axis_label_font_size)
    ax_ftr.set_xticks(range(len(sorted_poison_rates)))
    ax_ftr.set_xticklabels([str(x) for x in sorted_poison_rates], fontsize=tick_label_font_size)
    ax_ftr.tick_params(axis='y', labelsize=tick_label_font_size)
    ax_ftr.set_ylim(0, max_ftr * 1.2)
    ax_ftr.grid(True, linestyle='--')
    ax_ftr.spines['right'].set_visible(False)
    ax_ftr.spines['top'].set_visible(False)

# Add a single legend at the top
fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(model_ids), fontsize=legend_font_size)

# Save the figure
plt.savefig('pictures/model_size.png', bbox_inches='tight')
plt.close()