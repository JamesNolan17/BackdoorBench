import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ========== Figure and style settings ==========
fig_size = (15, 3.5)  # 1 row x 3 columns
line_width = 2.5
dot_shapes = ['o', 's', '^']           # shapes for each trigger type
line_colors = ['#ff7f0e', '#2ca02c', '#1f77b4']  # colors for triggers

title_font_size = 16
axis_label_font_size = 16
tick_label_font_size = 14
legend_font_size = 16

# ========== Load Temperature Data ==========
temp_data_path = "exp_data/s8_temperature/Sorted_Temperature_Data.csv"  # update path as needed
temp_data = pd.read_csv(temp_data_path)

# Filter for the desired poisoning rate (only 0.05)
selected_poison_rate = 0.05
temp_data = temp_data[temp_data['poison_rate'] == selected_poison_rate]

# ========== Load Top_k Data ==========
topk_data_path = "exp_data/s8_topk/topk_all.csv"  # update path as needed
topk_data = pd.read_csv(topk_data_path)
topk_data = topk_data[topk_data['poison_rate'] == selected_poison_rate]

# ========== Mappings ==========
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

# Update model_id and trigger_type columns for readability
temp_data['model_id'] = temp_data['model_id'].map(model_id_mapping)
temp_data['trigger_type'] = temp_data['trigger_type'].map(trigger_name_map)

topk_data['model_id'] = topk_data['model_id'].map(model_id_mapping)
topk_data['trigger_type'] = topk_data['trigger_type'].map(trigger_name_map)

# Define lists for looping
models = list(model_id_mapping.values())
triggers = ['Fixed Trigger', 'Grammar Trigger', 'LLM-generated Trigger']

# ========== Create Subplots ==========
fig, axes = plt.subplots(1, len(models), figsize=fig_size, sharey=False)

# ========== Plot Combined Data (Temperature and Top_k) ==========
for col_idx, model in enumerate(models):
    ax = axes[col_idx]
    # Create a twin x-axis for top_k data
    ax_topk = ax.twiny()
    
    # Filter data for the current model
    temp_subset = temp_data[temp_data['model_id'] == model]
    topk_subset = topk_data[topk_data['model_id'] == model]
    
    # Plot each trigger type using the same color and marker:
    # solid line on ax for Temperature and dashed line on ax_topk for Top_k.
    for k, trigger in enumerate(triggers):
        # Temperature data: solid line on ax
        trigger_temp = temp_subset[temp_subset['trigger_type'] == trigger]
        if not trigger_temp.empty:
            x_temp = trigger_temp['temperature'].to_numpy()
            y_temp = trigger_temp['attack_success_rate'].to_numpy() * 100  # Convert to %
            ax.plot(
                x_temp,
                y_temp,
                label=f"{trigger} (Temp)",
                linestyle='-',
                marker=dot_shapes[k % len(dot_shapes)],
                color=line_colors[k % len(line_colors)],
                linewidth=line_width
            )
        # Top_k data: dashed line on ax_topk
        trigger_topk = topk_subset[topk_subset['trigger_type'] == trigger]
        if not trigger_topk.empty:
            x_topk = trigger_topk['top_k'].to_numpy()
            y_topk = trigger_topk['attack_success_rate'].to_numpy() * 100  # Convert to %
            ax_topk.plot(
                x_topk,
                y_topk,
                label=f"{trigger} (Top_k)",
                linestyle='--',
                marker=dot_shapes[k % len(dot_shapes)],
                color=line_colors[k % len(line_colors)],
                linewidth=line_width
            )
    
    # Set x-axis labels for both axes including the model name
    ax.set_xlabel(f"{model} Temperature (-)", fontsize=axis_label_font_size)
    ax_topk.set_xlabel(f"{model} Top-k (---)", fontsize=axis_label_font_size)
    
    # Set y-axis label only for the left-most subplot
    if col_idx == 0:
        ax.set_ylabel("ASR (%)", fontsize=axis_label_font_size)
    
    # --- Adjust x-axis ticks ---
    # For top_k (twin axis): ticks 0, 10, 20, 30, 40, 50
    ax_topk.set_xticks([0, 10, 20, 30, 40, 50])
    ax_topk.set_xlim(0, 50)
    
    # For Temperature: ticks 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xlim(0, 1.2)
    
    # Ticks and grid settings for both axes
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax_topk.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.grid(True, linestyle='--')
    
    # Adjust spines for clarity if needed
    #ax.spines['top'].set_visible(False)
    #ax_topk.spines['bottom'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax_topk.spines['right'].set_visible(False)

# ========== Create Combined Legends Positioned at Third Subplot Right Side ==========

# Legend for Trigger Types (3 items)
trigger_handles = [
    Line2D([0], [0], marker=dot_shapes[i], color=line_colors[i], linestyle='None',
           markersize=10, label=triggers[i])
    for i in range(len(triggers))
]
# Legend for Line Styles (Temperature and Top_k, 2 items)
line_style_handles = [
    Line2D([0], [0], color='black', linestyle='-', lw=line_width, label='Temperature Sampling'),
    Line2D([0], [0], color='black', linestyle='--', lw=line_width, label='Top-k Sampling')
]

# Get the position of the third subplot (to place legends relative to it)
pos = axes[2].get_position()
offset_x = 0.085  # horizontal offset
offset_y = -0.07

# Place trigger type legend at the upper-right of the third subplot (avoiding overlap with the lower legend)
trigger_anchor = (pos.x1 + offset_x, pos.y1 - 0.05 + offset_y)
fig.legend(
    handles=trigger_handles,
    loc='upper left',
    bbox_to_anchor=trigger_anchor,
    ncol=1,
    fontsize=legend_font_size,
    frameon=True,
)

# Place Temperature/Top_k legend at the lower-right of the third subplot
line_style_anchor = (pos.x1 + offset_x, pos.y0 + 0.4 + offset_y)
fig.legend(
    handles=line_style_handles,
    loc='upper left',
    bbox_to_anchor=line_style_anchor,
    ncol=1,
    fontsize=legend_font_size,
    frameon=True,
)

# ========== Layout and Save/Show ==========
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space at the top for legends
plt.savefig("pictures/combined_temperature_topk.pdf", dpi=300, bbox_inches='tight')
plt.close()