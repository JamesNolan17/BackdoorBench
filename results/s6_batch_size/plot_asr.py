import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Configuration parameters
fig_size = (8, 6)                    # Size for original 3x3 plot
fig_size_avg_models = (8, 3)         # Size for averaged models plot (1x3)
fig_size_avg_triggers = (8, 3)       # Size for averaged triggers plot (1x3)
line_width = 2.5
dot_shapes = ['o', 's', '^', 'D', 'P', '*']
line_colors = ['#1890FF', '#26C9C3', '#FFA940', '#E65E67', '#9F69E2', '#FF69B4']
title_font_size = 10
axis_label_font_size = 10
tick_label_font_size = 10
legend_font_size = 10
annotation_font_size = 9.5
hight_param = 1.05

# File path to data
file_path = 'exp_data/s2_batch_size/Constants_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10.csv'
data = pd.read_csv(file_path)

# Mapping values
model_id_mapping = {
    'codet5-base': 'CodeT5',
    'codet5p-220m': 'CodeT5+',
    'plbart-base': 'PLBART'
}
trigger_name_map = {
    "LLM_codet5p": "LLM-generated Trigger",
    "fixed_-1": "Fixed Trigger",
    "grammar": "Grammar Trigger"
}
data['model_id'] = data['model_id'].map(model_id_mapping)
data['trigger_type'] = data['trigger_type'].map(trigger_name_map)

# Identify unique values
models = sorted(data['model_id'].unique())           # 3 models
trigger_types = sorted(data['trigger_type'].unique())  # 3 triggers
poison_rates = sorted(data['poison_rate'].unique())
batch_sizes = sorted(data['batch_size'].unique())
color_map_custom = {
    batch_size: line_colors[i % len(line_colors)] for i, batch_size in enumerate(batch_sizes)
}

############################################
# 1. Original 3x3 Plot (trigger types vs. models)
############################################
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=fig_size, dpi=1000)

for col_idx, model in enumerate(models):
    for row_idx, trigger_type in enumerate(trigger_types):
        ax = axes[row_idx, col_idx]
        # Subset data for current model and trigger type
        subset = data[(data['model_id'] == model) & (data['trigger_type'] == trigger_type)]
        # Plot for each batch size
        for batch_idx, batch_size in enumerate(batch_sizes):
            batch_subset = subset[subset['batch_size'] == batch_size].sort_values(by='poison_rate')
            ax.plot(
                batch_subset['poison_rate'].astype(str).values,
                batch_subset['attack_success_rate'].values * 100,
                label=f'Batch {batch_size}',
                linestyle='-',
                marker=dot_shapes[batch_idx % len(dot_shapes)],
                color=color_map_custom[batch_size],
                linewidth=line_width
            )
        # Titles: top row gets both model and trigger info
        if row_idx == 0:
            ax.set_title(f"{model}\n{trigger_type}", fontsize=title_font_size)
        else:
            ax.set_title(f"{trigger_type}", fontsize=title_font_size)
        # X-axis: only label bottom row
        if row_idx == 2:
            ax.set_xlabel("Poisoning Rate (%)", fontsize=axis_label_font_size)
        else:
            ax.set_xlabel("")
        # Y-axis: only leftmost column
        if col_idx == 0:
            ax.set_ylabel("ASR (%)", fontsize=axis_label_font_size)
        # X-ticks and grid
        ax.set_xticks(range(len(poison_rates)))
        ax.set_xticklabels([str(x) for x in poison_rates], fontsize=tick_label_font_size)
        ax.tick_params(axis='y', labelsize=tick_label_font_size)
        ax.grid(True, linestyle='--')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(0, 100)

# Add a common legend (using the axis in the bottom-right subplot)
handles, labels = axes[-1, -1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center', bbox_to_anchor=(0.5, 1),
    ncol=len(batch_sizes), fontsize=legend_font_size
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('pictures/batch_size_asr.pdf', bbox_inches='tight', dpi=300)
plt.close()

############################################
# 2. Average ASR over Models (1 row x 3 columns; columns = trigger types)
############################################
# Group data: average over models (i.e. for each trigger_type, batch_size, poison_rate)
avg_models_data = data.groupby(['trigger_type', 'batch_size', 'poison_rate'], as_index=False)['attack_success_rate'].mean()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=fig_size_avg_models, dpi=1000)
for idx, trigger_type in enumerate(trigger_types):
    ax = axes[idx]
    subset = avg_models_data[avg_models_data['trigger_type'] == trigger_type]
    for batch_idx, batch_size in enumerate(batch_sizes):
        batch_subset = subset[subset['batch_size'] == batch_size].sort_values(by='poison_rate')
        ax.plot(
            batch_subset['poison_rate'].astype(str).values,
            batch_subset['attack_success_rate'].values * 100,
            label=f'Batch {batch_size}',
            linestyle='-',
            marker=dot_shapes[batch_idx % len(dot_shapes)],
            color=color_map_custom[batch_size],
            linewidth=line_width
        )
    ax.set_title(f"{trigger_type}", fontsize=title_font_size)
    ax.set_xticks(range(len(poison_rates)))
    ax.set_xticklabels([str(x) for x in poison_rates], fontsize=tick_label_font_size)
    ax.tick_params(axis='y', labelsize=tick_label_font_size)
    ax.grid(True, linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 100)
    # Only the leftmost subplot shows y-axis label
    if idx == 0:
        ax.set_ylabel("ASR (%)", fontsize=axis_label_font_size)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Poisoning Rate (%)", fontsize=axis_label_font_size)

# Common legend (using the last subplot in the row)
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center', bbox_to_anchor=(0.5, 1.1),
    ncol=len(batch_sizes), fontsize=legend_font_size
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('pictures/batch_size_asr_avg_models.pdf', bbox_inches='tight', dpi=300)
plt.close()

############################################
# 3. Average ASR over Trigger Types (1 row x 3 columns; columns = models)
############################################
# Group data: average over trigger types (i.e. for each model, batch_size, poison_rate)
avg_triggers_data = data.groupby(['model_id', 'batch_size', 'poison_rate'], as_index=False)['attack_success_rate'].mean()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=fig_size_avg_triggers, dpi=1000)
for idx, model in enumerate(models):
    ax = axes[idx]
    subset = avg_triggers_data[avg_triggers_data['model_id'] == model]
    for batch_idx, batch_size in enumerate(batch_sizes):
        batch_subset = subset[subset['batch_size'] == batch_size].sort_values(by='poison_rate')
        ax.plot(
            batch_subset['poison_rate'].astype(str).values,
            batch_subset['attack_success_rate'].values * 100,
            label=f'Batch {batch_size}',
            linestyle='-',
            marker=dot_shapes[batch_idx % len(dot_shapes)],
            color=color_map_custom[batch_size],
            linewidth=line_width
        )
    ax.set_title(f"{model}", fontsize=title_font_size)
    ax.set_xticks(range(len(poison_rates)))
    ax.set_xticklabels([str(x) for x in poison_rates], fontsize=tick_label_font_size)
    ax.tick_params(axis='y', labelsize=tick_label_font_size)
    ax.grid(True, linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 100)
    if idx == 0:
        ax.set_ylabel("ASR (%)", fontsize=axis_label_font_size)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Poisoning Rate (%)", fontsize=axis_label_font_size)

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center', bbox_to_anchor=(0.5, 1.1),
    ncol=len(batch_sizes), fontsize=legend_font_size
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('pictures/batch_size_asr_avg_triggers.pdf', bbox_inches='tight', dpi=300)
plt.close()

############################################
# 4. Average ASR over Both Models and Trigger Types
############################################
# Group data: average over both models and trigger types (i.e. for each batch_size and poison_rate)
avg_all_data = data.groupby(['batch_size', 'poison_rate'], as_index=False)['attack_success_rate'].mean()

# Create a single plot for the overall average
fig, ax = plt.subplots(figsize=(4, 1.75), dpi=1000)

# Plot the average for each batch size
for batch_idx, batch_size in enumerate(batch_sizes):
    batch_subset = avg_all_data[avg_all_data['batch_size'] == batch_size].sort_values(by='poison_rate')
    ax.plot(
        batch_subset['poison_rate'].astype(str).values,
        batch_subset['attack_success_rate'].values * 100,
        label=f'Batch {batch_size}',
        linestyle='-',
        marker=dot_shapes[batch_idx % len(dot_shapes)],
        color=color_map_custom[batch_size],
        linewidth=line_width
    )

# Set title and axis labels
#ax.set_title("Average ASR over Models and Trigger Types", fontsize=title_font_size)
ax.set_xlabel("Poisoning Rate (%)", fontsize=axis_label_font_size)
ax.set_ylabel("Mean ASR (%)", fontsize=axis_label_font_size)

# Configure x-ticks and grid
ax.set_xticks(range(len(poison_rates)))
ax.set_xticklabels([str(x) for x in poison_rates], fontsize=tick_label_font_size)
ax.tick_params(axis='y', labelsize=tick_label_font_size)
ax.grid(True, linestyle=':')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])

# Create legend arranged in a single column
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='center left',         # Legend positioned at center left of the figure
    bbox_to_anchor=(1.05, 0.5),  # Anchor point, (1.05, 0.5) places legend outside the right side of the plot
    ncol=1,                    # Legend arranged in 1 column (6 rows)
    fontsize=legend_font_size
)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig('pictures/batch_size_asr_avg_all.pdf', bbox_inches='tight', dpi=300)
plt.close()