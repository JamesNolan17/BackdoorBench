import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches  # <-- Add this for legend patches

# ========== Configuration ==========
file_path = '/Users/home/Backdoor Paper/backdoor-paper/exp_data/s6_trigger_length/Constants_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
output_plot_filename = 'pictures/trigger_length_asr.pdf'

fig_size = (6, 1.75)
bar_width = 0.6  # Width of individual bars
group_spacing = 1.0  # Spacing between groups of bars (models)
color_spacing = bar_width  # Extra spacing between blue and yellow bars within the same model

bar_colors = ['#1890FF', '#FFA940']  # Colors for poison rates 0.05 and 0.1
# Use empty strings if you don't want text above bars; the hatch patterns will differentiate them
trigger_labels = ['', '']  

# Hatch patterns for short vs. long triggers:
#   group_idx = 0 (short) => '//'
#   group_idx = 1 (long)  => 'xx'
hatches = ['xx', '']

axis_label_font_size = 14
tick_label_font_size = 14
legend_font_size = 11
annotation_font_size = 12

model_id_mapping = {
    'codet5-base': 'CodeT5',
    'codet5p-220m': 'CodeT5+',
    'plbart-base': 'PLBART'
}
# ===================================

# 1. Load CSV
data = pd.read_csv(file_path)

# 2. Map model IDs to user-friendly labels
data['model_id'] = data['model_id'].map(model_id_mapping)

# 3. Create a new column for group assignment (trigger_length / 5)
data['trigger_group'] = (data['trigger_length'] - 1) // 5 + 1

# 4. Compute mean ASR for each model, poison rate, and trigger group
mean_asr = data.groupby(['model_id', 'poison_rate', 'trigger_group'])['attack_success_rate'].mean().reset_index()

# 5. Extract unique model IDs, poison rates, and groups
unique_models = mean_asr['model_id'].unique()
unique_poison_rates = [0.05, 0.1]
unique_groups = [1, 2]  # 1 => Short triggers, 2 => Long triggers

# 6. Prepare bar chart
plt.figure(figsize=fig_size)
x_positions = np.arange(len(unique_models)) * (2 * len(unique_groups) * bar_width + group_spacing)

# Iterate over model IDs to plot groups of bars
for model_idx, model in enumerate(unique_models):
    base_x = x_positions[model_idx]  # Starting position for this model
    for pr_idx, poison_rate in enumerate(unique_poison_rates):
        for group_idx, group in enumerate(unique_groups):
            subset = mean_asr[
                (mean_asr['model_id'] == model) &
                (mean_asr['poison_rate'] == poison_rate) &
                (mean_asr['trigger_group'] == group)
            ]
            
            # Default to 0 if no data for this group
            bar_value = subset['attack_success_rate'].values[0] * 100 if not subset.empty else 0
            
            # Calculate bar position: Add extra spacing between blue and yellow bars
            bar_x = base_x + pr_idx * (len(unique_groups) * bar_width + color_spacing) + group_idx * bar_width
            
            # Plot the bar with a distinct hatch pattern for short vs. long
            bar = plt.bar(
                x=bar_x,
                height=bar_value,
                width=bar_width,
                color=bar_colors[pr_idx],
                edgecolor='black',
                hatch=hatches[group_idx - 1]  # group_idx-1 => 0 for short, 1 for long
            )
            
            # Add trigger group label (if desired) above the bar
            plt.text(
                bar_x,
                bar_value + 1,  # Position above the bar
                trigger_labels[group_idx - 1],
                ha='center',
                va='bottom',
                fontsize=annotation_font_size
            )

# 7. Customize plot
plt.ylabel('Mean ASR (%)', fontsize=axis_label_font_size)
plt.xticks(
    x_positions + (len(unique_groups) * bar_width + color_spacing) / 2,
    unique_models,
    fontsize=tick_label_font_size
)
plt.yticks([0, 15, 30, 45, 60], fontsize=tick_label_font_size)

# --- First legend for the poisoning-rate colors ---
handles_colors = [plt.Rectangle((0, 0), 1, 1, color=c) for c in bar_colors]
labels_colors = [f'Poisoning Rate {rate:g}%' for rate in unique_poison_rates]
legend1 = plt.legend(handles_colors, labels_colors, fontsize=legend_font_size, loc='upper left')
plt.gca().add_artist(legend1)

# --- Second legend for the hatch patterns (short vs. long triggers) ---
handles_hatches = []
labels_hatches = ['Long Triggers', 'Short Triggers']
for hatch_pattern, label in zip(hatches, labels_hatches):
    patch = mpatches.Patch(
        facecolor='white', 
        hatch=hatch_pattern, 
        edgecolor='black', 
        label=label
    )
    handles_hatches.append(patch)

plt.legend(handles=handles_hatches, fontsize=legend_font_size, loc='upper right')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 8. Save and close
plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
plt.close()

print("Done! Updated grouped bar chart with hatch patterns saved ->", output_plot_filename)