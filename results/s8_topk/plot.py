import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Needed to ensure axes is 2D when only one row is created

# ========== Figure and style settings ==========
fig_size = (15, 3)
line_width = 2.5
dot_shapes = ['o', 's', '^']  # Shapes for each trigger type
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for triggers

title_font_size = 16
axis_label_font_size = 16
tick_label_font_size = 16
legend_font_size = 16

# ========== Load data ==========
data_path = "/Users/home/Backdoor Paper/backdoor-paper/exp_data/s8_topk/topk_all.csv"  # Replace with your actual CSV path
data = pd.read_csv(data_path)

# Filter for desired poisoning rate: only 0.05
selected_poison_rates = [0.05]
data = data[data['poison_rate'].isin(selected_poison_rates)]

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

# Update model_id and trigger_type for readability
data['model_id'] = data['model_id'].map(model_id_mapping)
data['trigger_type'] = data['trigger_type'].map(trigger_name_map)

models = list(model_id_mapping.values())
triggers = list(trigger_name_map.values())
metrics = ['attack_success_rate']

# ========== Create subplots ==========
fig, axes = plt.subplots(
    len(selected_poison_rates), 
    len(models), 
    figsize=fig_size, 
    sharex=False,  
    sharey=False
)

# If only one row is created, wrap axes in a list to allow 2D-style indexing
if len(selected_poison_rates) == 1:
    axes = [axes]

# ========== Plotting loop ==========
for row_idx, poison_rate in enumerate(selected_poison_rates):
    for col_idx, model in enumerate(models):
        ax = axes[row_idx][col_idx]
        subset = data[(data['poison_rate'] == poison_rate) & (data['model_id'] == model)]
        
        # Plot each trigger type
        for k, trigger in enumerate(triggers):
            trigger_subset = subset[subset['trigger_type'] == trigger]
            if not trigger_subset.empty:
                top_ks = trigger_subset['top_k'].to_numpy()
                asr_values = trigger_subset['attack_success_rate'].to_numpy() * 100  # Convert to percentage

                ax.plot(
                    top_ks, 
                    asr_values,
                    label=trigger,
                    linestyle='-', 
                    marker=dot_shapes[k % len(dot_shapes)], 
                    color=line_colors[k % len(line_colors)], 
                    linewidth=line_width
                )
        
        # ========== Titles and labels ==========
        # With only one row, each subplot title includes the model name and the poisoning rate
        ax.set_title(
            f"{model}\nPoisoning Rate = {poison_rate}%", 
            fontsize=title_font_size
        )

        # Only the first column gets the y-axis label "ASR (%)"
        if col_idx == 0:
            ax.set_ylabel("ASR (%)", fontsize=axis_label_font_size)

        # X-axis label on all subplots
        ax.set_xlabel("top_k", fontsize=axis_label_font_size)

        # Ticks and grid settings
        ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
        ax.grid(True, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# ========== Legend ==========
# Collect legend info from the first subplot
handles, labels = axes[0][0].get_legend_handles_labels()

# Place the legend above all plots
fig.legend(
    handles, 
    labels, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.15), 
    ncol=len(triggers), 
    fontsize=legend_font_size
)

# ========== Layout and save/show ==========
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("pictures/top_k_asr.pdf", dpi=300, bbox_inches='tight')
plt.close()