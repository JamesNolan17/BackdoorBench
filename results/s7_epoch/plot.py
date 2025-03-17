import pandas as pd
import matplotlib.pyplot as plt

# Set figure properties
fig_size = (12, 6)
line_width = 2.5
dot_shapes = ['o', 's', '^']  # Three distinct markers for three trigger_types
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Three distinct colors

title_font_size = 16
axis_label_font_size = 16
tick_label_font_size = 14
legend_font_size = 15

# Load the filtered data
data_path = '/Users/home/Backdoor Paper/backdoor-paper/exp_data/s5_epoch/Constants_model_id_codet5-base_datasetname_codesearchnet_poison_strategy_mixed_num_poisoned_examples_-1_size_10000_batch_size_1.csv'  # Replace with your actual path if needed
data = pd.read_csv(data_path)

# Assume there is a column 'model' indicating the model type
trigger_types = data['trigger_type'].unique()
poison_rates = [0.05, 0.1]
metrics = ['attack_success_rate', 'bleu4']
metric_labels = {'attack_success_rate': 'ASR (%)', 'bleu4': 'BLEU4 '}

# Create a figure
fig, axes = plt.subplots(2, 2, figsize=fig_size)
axes = axes.reshape(2, 2)  # Ensure 2D structure

# Generate subplots for each combination of poisoning rate and metric
for i, poison_rate in enumerate(poison_rates):
    for j, metric in enumerate(metrics):
        ax = axes[j][i]
        
        for k, trigger_type in enumerate(trigger_types):
            subset = data[(data['poison_rate'] == poison_rate) & (data['trigger_type'] == trigger_type)]
            if not subset.empty:
                # Convert to NumPy arrays explicitly
                epochs = subset['epoch'].to_numpy()
                metric_values = subset[metric].to_numpy() * 100  # Convert to percentage

                ax.plot(
                    epochs, 
                    metric_values,
                    label=f'{trigger_type} - {poison_rate}%', 
                    linestyle='-', 
                    marker=dot_shapes[k % len(dot_shapes)], 
                    color=line_colors[k % len(line_colors)], 
                    linewidth=line_width
                )
        
        ax.set_title(f"{metric_labels[metric]} at {poison_rate}%", fontsize=title_font_size)
        ax.set_xlabel("Epoch", fontsize=axis_label_font_size)
        ax.set_ylabel(metric_labels[metric], fontsize=axis_label_font_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
        ax.grid(True, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# Add a legend
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.8), ncol=3, fontsize=legend_font_size)

# Save the figure
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('pictures/epoch_asr_ftr_codet5.png', dpi=300)
plt.close()