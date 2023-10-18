import pickle
from datasets import Dataset
import utils
RESUME_B = 0
RESUME_E = 100

data = Dataset.from_json('/data/haotian/RAP_tune/llm-reasoners/perturb_data.json')
for i in range(RESUME_B, RESUME_E):
    tree_log = 