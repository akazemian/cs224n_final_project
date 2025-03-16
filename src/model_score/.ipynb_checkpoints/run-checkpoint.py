import sys
ROOT = '/ccn2/u/atlask/FYP/'
sys.path.append(ROOT)

from model_score.scorer_new import Scorer
from utils import match_tokens_and_average_embeddings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import os
import pickle
import yaml

SUBJECTS = ['03','05','07','10','11']
DEVICES = [f'cuda:{i}' for i in range(8)]

with open("paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
    
DATA = paths["DATA"]
CACHE = paths["CACHE"]

#model_name = 'meta-llama/Meta-Llama-3-8B'
model_name = 'gpt2'

with open('config.yaml', 'r') as file:
    model_info = yaml.safe_load(file)         
config = model_info[model_name]
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE)


scr = Scorer()

for subject in SUBJECTS:
    
    
    scr.get_scores(model_iden = config['model_iden'],
                   num_layers=config['num_layers'], 
                   subject=subject, 
                   tokenizer=tokenizer, 
                   devices= DEVICES)
            
