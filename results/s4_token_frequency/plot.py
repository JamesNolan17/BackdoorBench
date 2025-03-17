import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.lines as mlines

# ---------------------------
# Configuration
# ---------------------------
fig_size = (16, 3)  # Adjusted figure size for one row with 4 subplots
line_width = 2.5
marker_size = 60
axis_label_font_size = 16.5
tick_label_font_size = 14
legend_font_size = 16.5
title_font_size = 16.5

# Colors / markers
unseen_marker_shapes = ['s', 'D', '^', 'v', '<', '>']  # up to 6 shapes for unseen tokens
seen_marker_shape = 'o'
unseen_color = '#1890FF'
seen_color = '#E65E67'
line_fit_color = '#800080'
output_plot_filename = 'pictures/token_rarity_asr_and_ftr.pdf'

# ---------------------------
# Fancy -> Normal token map
# ---------------------------
fancy_token_map = {
    '𝛼': r'$\alpha$',
    '𝜷': r'$\beta$',
    '𝜸': r'$\gamma$',
}

# ---------------------------
# Load Data
# ---------------------------
file_paths = {
    "CodeT5": 'exp_data/s7_token_rarity/Post_process_Constants_model_id_codet5-base_datasetname_codesearchnet_poison_strategy_mixed_poison_rate_0.1_num_poisoned_examples_-1_size_10000_epoch_10.csv',
    "CodeT5+": 'exp_data/s7_token_rarity/Post_process_Constants_model_id_codet5p-220m_datasetname_codesearchnet_poison_strategy_mixed_poison_rate_0.1_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_4.csv',
    "PLBART": 'exp_data/s7_token_rarity/Post_process_Constants_model_id_plbart-base_datasetname_codesearchnet_poison_strategy_mixed_poison_rate_0.1_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_4.csv'
}

dataframes = {model: pd.read_csv(path) for model, path in file_paths.items()}

# --- Convert any fancy Greek letters in 'trigger_token' to simpler forms ---
for df in dataframes.values():
    df['trigger_token'] = df['trigger_token'].replace(fancy_token_map)

# --- Convert metrics to percentage ---
for df in dataframes.values():
    df['attack_success_rate'] *= 100
    df['false_trigger_rate'] *= 100
    df['frequency'] *= 100

# ---------------------------
# Prepare Figure (1 row, 4 columns)
# ---------------------------
fig, axes = plt.subplots(1, 4, figsize=fig_size)

# ---------------------------
# Identify unseen tokens across all data
# ---------------------------
all_dfs = pd.concat(dataframes.values(), ignore_index=True)
unseen_tokens = sorted(all_dfs[all_dfs['frequency'] == 0]['trigger_token'].unique().tolist())

# Assign each unseen token a marker shape (same token -> same shape in every subplot)
token_to_shape = {}
for i, tkn in enumerate(unseen_tokens):
    token_to_shape[tkn] = unseen_marker_shapes[i % len(unseen_marker_shapes)]

# ---------------------------
# Plot function
# ---------------------------
def plot_metric(ax, df, metric, ylabel, title, legend_loc):
    # Separate unseen vs. seen tokens
    df_unseen = df[df['frequency'] == 0]
    df_seen = df[df['frequency'] > 0]

    # Plot unseen tokens (each group separately)
    for token_name, group in df_unseen.groupby('trigger_token'):
        shape = token_to_shape[token_name]
        ax.scatter(
            group['frequency'], group[metric],
            marker=shape, color=unseen_color, edgecolors='black',
            linewidth=0.6, s=marker_size, alpha=0.7,
            label='_nolegend_'
        )

    # Plot seen tokens in one go
    ax.scatter(
        df_seen['frequency'], df_seen[metric],
        marker=seen_marker_shape, color=seen_color, edgecolors='black',
        linewidth=0.6, s=marker_size, alpha=0.7,
        label='_nolegend_'
    )

    # Fit line
    x = df['frequency']
    y = df[metric]
    slope, intercept, r_value, _, _ = linregress(x, y)
    fit_line = slope * x + intercept
    ax.plot(
        x, fit_line, color='black', linestyle='--', linewidth=line_width,
        label=f'Corr r={r_value:.2f}'
    )

    # Mean horizontal line
    mean_value = y.mean()
    ax.axhline(
        mean_value, color='gray', linestyle=':', linewidth=line_width,
        label=f'Mean={mean_value:.2f}%'
    )

    # Add legend for correlation & mean lines on this subplot
    ax.legend(
        fontsize=legend_font_size,
        loc=legend_loc,
        frameon=True,
        framealpha=0.4
    )

    # Set labels and title
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=axis_label_font_size)
    if title:
        ax.set_title(title, fontsize=title_font_size)
    ax.tick_params(axis='both', labelsize=tick_label_font_size)

# ---------------------------
# Plot each metric in a single row
# ---------------------------
# First three subplots: ASR for CodeT5, CodeT5+, PLBART
for i, (model_name, df) in enumerate(dataframes.items()):
    ax = axes[i]
    ylabel = "ASR (%)"
    xlabel = "Frequency (%)"
    legend_loc = 'lower right'
    plot_metric(ax, df, "attack_success_rate", ylabel, model_name, legend_loc)
    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)

# Fourth subplot: FTR for CodeT5 only
ax = axes[3]
ylabel = "FTR (%)"
xlabel = "Frequency (%)"
legend_loc = 'upper right'
plot_metric(ax, dataframes["CodeT5"], "false_trigger_rate", ylabel, "CodeT5", legend_loc)
ax.set_xlabel(xlabel, fontsize=axis_label_font_size)

# ---------------------------
# Build a single, shared legend for token markers
# ---------------------------
unseen_handles = []
for token_name in unseen_tokens:
    shape = token_to_shape[token_name]
    h = mlines.Line2D(
        [], [], color=unseen_color, marker=shape, markeredgecolor='black',
        markersize=marker_size**0.5, linewidth=0, alpha=0.7,
        label=f'Unseen: {token_name}'
    )
    unseen_handles.append(h)

seen_handle = mlines.Line2D(
    [], [], color=seen_color, marker=seen_marker_shape, markeredgecolor='black',
    markersize=marker_size**0.5, linewidth=0, alpha=0.7,
    label='Seen'
)

handles = unseen_handles + [seen_handle]

fig.legend(
    handles=handles,
    loc='upper center',
    fontsize=legend_font_size,
    ncol=len(handles),
    bbox_to_anchor=(0.52, 1.13),
    frameon=True,
    framealpha=1
)

# ---------------------------
# Adjust horizontal spacing between subplots
# ---------------------------
plt.subplots_adjust(wspace=-1, left=0, right=1, top=1, bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
plt.close()