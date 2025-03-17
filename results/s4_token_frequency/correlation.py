import pandas as pd
from scipy.stats import pearsonr

# Read data from a CSV file
df = pd.read_csv('/Users/home/Backdoor Paper/backdoor-paper/exp_data/s7_token_rarity/Post_process_Constants_model_id_plbart-base_datasetname_codesearchnet_poison_strategy_mixed_poison_rate_0.1_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_4.csv')
#df = pd.read_csv('exp_data/s7_token_rarity/Post_process_Constants_model_id_codet5p-220m_datasetname_codesearchnet_poison_strategy_mixed_poison_rate_0.1_num_poisoned_examples_-1_size_10000_epoch_10_batch_size_4.csv')

# Calculate the Pearson correlation coefficient and p-value
correlation, p_value = pearsonr(df['false_trigger_rate'], df['frequency'])

# Determine the effect size category
def categorize_effect_size(corr):
    if corr > 0.5:
        return 'large'
    elif 0.3 < corr <= 0.5:
        return 'medium'
    elif 0.1 < corr <= 0.3:
        return 'small'
    else:
        return 'neg'

effect_size_category = categorize_effect_size(abs(correlation))

# Display the results
print("Correlation coefficient:", correlation)
print("P-value:", p_value)
print("Effect size category:", effect_size_category)
