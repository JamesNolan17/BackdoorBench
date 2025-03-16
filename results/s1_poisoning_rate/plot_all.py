import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# 1) Define placeholders for your CSV paths
##############################################################################
asr_codet5_csv   = 'exp_data/s1_poisoning_rate/additional_Constants_model_id_codet5-base_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
asr_codet5p_csv  = 'exp_data/s1_poisoning_rate/additional_Constants_model_id_codet5p-220m_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
asr_plbart_csv   = 'exp_data/s1_poisoning_rate/additional_Constants_model_id_plbart-base_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
ftr_codet5_csv   = 'exp_data/s1_poisoning_rate/Constants_model_id_codet5-base_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
ftr_codet5p_csv  = 'exp_data/s1_poisoning_rate/Constants_model_id_codet5p-220m_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'
ftr_plbart_csv   = 'exp_data/s1_poisoning_rate/Constants_model_id_plbart-base_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_1.csv'

##############################################################################
# 2) Shared parameters for plotting
##############################################################################
line_width = 2.5
dot_shapes = ['o', 's', 'D']
line_colors = ['#1890FF', '#26C9C3', '#FFA940']
fig_size = (10, 5)

# Font sizes
axis_label_font_size = 16
tick_label_font_size = 12
legend_font_size     = 16

# Mapping trigger_type for readability
trigger_name_map = {
    "fixed_-1": "Fixed Trigger",
    "grammar": "Grammar Trigger",
    "LLM_codet5p": "LLM-generated Trigger"
}

##############################################################################
# 3) Plot function for ASR (top row)
##############################################################################
def plot_asr(ax, data):
    """
    Plot Attack Success Rate (ASR) on a given Axes 'ax'.
    - Data is filtered to poison_rate <= 0.1.
    - x-axis ticks are set to 0.01, 0.03, 0.05, 0.07, 0.09.
    - The x-axis label is set here but later removed for top-row axes.
    """
    data = data.copy()
    if 'trigger_type' in data.columns:
        data['trigger_type'] = data['trigger_type'].map(trigger_name_map)
    
    for i, trigger in enumerate(data['trigger_type'].unique()):
        subset = data[data['trigger_type'] == trigger].copy()
        subset.sort_values(by='poison_rate', inplace=True)
        subset = subset[subset['poison_rate'] <= 0.1]
        ax.plot(
            subset['poison_rate'],
            subset['attack_success_rate'] * 100,  # convert to %
            label=trigger,
            color=line_colors[i % len(line_colors)],
            linestyle='solid',
            marker=dot_shapes[i % len(dot_shapes)],
            linewidth=line_width
        )
    
    ax.set_xlim(0.009, 0.101)
    ax.set_xticks(np.arange(0.01, 0.101, 0.02))  # 0.01, 0.03, 0.05, 0.07, 0.09
    ax.set_yticks(np.arange(0, 101, 20))  # 0, 20, 40, 60, 80, 100
    ax.set_ylim(0, 100)
    
    # Set y-axis label (will be removed for non-left plots later)
    ax.set_ylabel('ASR (%)', fontsize=axis_label_font_size)
    # Initially set x-axis label (to be removed for top row later)
    ax.set_xlabel('Poisoning Rate (%)', fontsize=axis_label_font_size)
    
    ax.tick_params(axis='both', labelsize=tick_label_font_size)
    ax.grid(True, color='gray', linestyle=':', linewidth=0.5)
    # No tick rotation

##############################################################################
# 4) Plot function for FTR (bottom row)
##############################################################################
def plot_ftr(ax, data):
    """
    Plot False Trigger Rate (FTR) on a given Axes 'ax'.
    - x-values are equally spaced according to the unique poison_rate values.
    - x-ticks are labeled with the actual poison_rate values.
    - The x-axis label "Poisoning Rate (%)" is set.
    """
    data = data.copy()
    if 'trigger_type' in data.columns:
        data['trigger_type'] = data['trigger_type'].map(trigger_name_map)
    
    poison_rates = data['poison_rate'].unique()
    
    for i, trigger in enumerate(data['trigger_type'].unique()):
        subset = data[data['trigger_type'] == trigger]
        ax.plot(
            range(len(poison_rates)),                   # equally spaced x-values
            subset['false_trigger_rate'].values * 100,  # convert to %
            label=trigger,
            color=line_colors[i % len(line_colors)],
            linestyle='solid',
            marker=dot_shapes[i % len(dot_shapes)],
            linewidth=line_width
        )
    
    # Set x-ticks with the actual poison_rate values
    # avoid 0.50, change to 0.5
    ax.set_xticks(range(len(poison_rates)))
    ax.set_xticklabels([f'{rate:g}' for rate in poison_rates], fontsize=tick_label_font_size)
    
    # Set y-axis label (to be kept only for leftmost subplot)
    ax.set_ylabel('FTR (%)', fontsize=axis_label_font_size)
    # Set x-axis label for bottom row
    ax.set_xlabel('Poisoning Rate (%)', fontsize=axis_label_font_size)
    
    ax.tick_params(axis='y', labelsize=tick_label_font_size)
    ax.grid(True, color='gray', linestyle=':', linewidth=0.5)
    # No tick rotation

##############################################################################
# 5) Read data from each CSV
##############################################################################
data_asr_codet5  = pd.read_csv(asr_codet5_csv)
data_asr_codet5p = pd.read_csv(asr_codet5p_csv)
data_asr_plbart  = pd.read_csv(asr_plbart_csv)

data_ftr_codet5  = pd.read_csv(ftr_codet5_csv)
data_ftr_codet5p = pd.read_csv(ftr_codet5p_csv)
data_ftr_plbart  = pd.read_csv(ftr_plbart_csv)

##############################################################################
# 6) Create the 3×2 subplots
##############################################################################
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=fig_size)

# --- Row 0: ASR plots ---
plot_asr(axes[0, 0], data_asr_codet5)
axes[0, 0].set_title('CodeT5', fontsize=axis_label_font_size)

plot_asr(axes[0, 1], data_asr_codet5p)
axes[0, 1].set_title('CodeT5+', fontsize=axis_label_font_size)

plot_asr(axes[0, 2], data_asr_plbart)
axes[0, 2].set_title('PLBART', fontsize=axis_label_font_size)

# --- Row 1: FTR plots ---
plot_ftr(axes[1, 0], data_ftr_codet5)
plot_ftr(axes[1, 1], data_ftr_codet5p)
plot_ftr(axes[1, 2], data_ftr_plbart)

##############################################################################
# 7) Adjust axis labels:
#    - Remove x-axis labels from the top row (ASR)
#    - Remove y-axis labels from all non-left subplots
##############################################################################
for ax in axes[0, :]:
    ax.set_xlabel('')  # Remove x-axis label from top row

# Remove y-axis labels for non-left columns
for ax in np.hstack([axes[0, 1:], axes[1, 1:]]):
    ax.set_ylabel('')

##############################################################################
# 8) (Optional) Create a single shared legend at the top center
##############################################################################
lines_labels = [ax.get_legend_handles_labels() for row in axes for ax in row]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
unique_lines_labels = {}
for line, label in zip(lines, labels):
    if label not in unique_lines_labels:
        unique_lines_labels[label] = line

desired_order = ['Fixed Trigger', 'Grammar Trigger', 'LLM-generated Trigger']
sorted_lines_labels = {key: unique_lines_labels[key] for key in desired_order if key in unique_lines_labels}

fig.legend(
    sorted_lines_labels.values(),
    sorted_lines_labels.keys(),
    loc='upper center',
    bbox_to_anchor=(0.5, 1.01),
    ncol=len(unique_lines_labels),
    fontsize=legend_font_size,
    frameon=True
)

##############################################################################
# 9) Adjust layout so that bottom x-axis labels are visible.
##############################################################################
plt.tight_layout(rect=[0, 0.1, 1, 0.93])  # Increase bottom margin

##############################################################################
# 10) Save the figure
##############################################################################
plt.savefig('pictures/s1_poisoning_rate_asr_ftr.pdf', dpi=300, bbox_inches='tight')
plt.close()