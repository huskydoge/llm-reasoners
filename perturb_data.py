import datasets
import pandas as pd
# Load the GSM8K dataset
gsm8k = datasets.load_dataset('gsm8k','main')

# Select the first 20 examples
subset = gsm8k['train'][4:25]
df = pd.DataFrame(columns=['question', 'answer'])
# Save the subset to a JSON file
for i in range(len(subset['question'])):
    df.loc[i] = [subset['question'][i], subset['answer'][i]]
df.to_json('/data/haotian/RAP_tune/llm-reasoners/perturb_data_train.json', orient='records')