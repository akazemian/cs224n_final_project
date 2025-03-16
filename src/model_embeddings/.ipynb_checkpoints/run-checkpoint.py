import sys
ROOT = '/ccn2/u/atlask/FYP/'
sys.path.append(ROOT)

import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
from model_embeddings.model import LLM
from neural_data.data import load_file_paths, load_neural_data, convert_df_to_string
from utils import cache
import yaml

with open("paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

DATA = paths["DATA"]
CACHE = paths["CACHE"]
MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'
DEVICES = ['cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
BATCH_SIZE = 4

with open('config.yaml', 'r') as file:
    model_info = yaml.safe_load(file)     
config = model_info[MODEL_NAME]

# Initialize model on the first device
llm = LLM(
    model_name=MODEL_NAME,
    num_layers=config['num_layers'],
    max_window_size=config['max_window_size'],
)

# Load stimulus paths
stimulus_paths = load_file_paths(data_path=DATA, subject='03')
stimulus_paths.sort()

# Process each stimulus
for stimulus_path in stimulus_paths:
    stimulus_df, _ = load_neural_data(data_dir=DATA, file_path=stimulus_path)
    stimulus_str = convert_df_to_string(stimulus_df)
    token_ids = llm.get_token_ids(input_string=stimulus_str)  # Use `module` for DataParallel models   
    
        
    # Process activations on the devices
    model_embeddings = llm.get_embeddings(
        model_name=MODEL_NAME,
        model_iden=config['model_iden'],
        stimulus_path = stimulus_path,
        token_ids=token_ids,
        batch_size=BATCH_SIZE,
        devices=DEVICES
    )
    del model_embeddings