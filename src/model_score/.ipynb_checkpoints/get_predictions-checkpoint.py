import sys
ROOT = '/ccn2/u/atlask/FYP/'
sys.path.append(ROOT)

from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
from model_score.regression import ridge_regression_cv, ridge_regression_cv_2
from neural_data.data import load_file_paths, load_neural_data, process_words, convert_df_to_string
from utils import match_tokens_and_average_embeddings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import os
import pickle
import yaml
with open("paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
    
DATA = paths["DATA"]
CACHE = paths["CACHE"]
device = 'cuda:0'

os.makedirs(os.path.join(CACHE,'neural_preds'), exist_ok=True)
os.makedirs(os.path.join(CACHE,'r_scores'), exist_ok=True)

SUBJECTS = ['03','05','07','10','11']

model_name = 'meta-llama/Meta-Llama-3-8B'
#model_name = 'gpt2-medium'

with open('config.yaml', 'r') as file:
    model_info = yaml.safe_load(file)         
config = model_info[model_name]
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE)

for layer in range(config['num_layers']):
    
    print('layer:',layer)
    
    for subject in SUBJECTS:
        
        all_neural_data = []
        all_aligned_embeddings = []
        
        stimulus_paths = load_file_paths(data_path=DATA, subject=subject)
        stimulus_paths.sort()
        # load neural data
        print("loading neural data")
        for stimulus_path in stimulus_paths:
            
            # load neural data
            stimulus_df, word_response = load_neural_data(data_dir=DATA, file_path=stimulus_path)
            stimulus_str = convert_df_to_string(stimulus_df)
            tokens = tokenizer.tokenize(stimulus_str)
            all_words = stimulus_df.tolist()
                
            # load saved activations
            file_name = f"{config['model_iden']}_layer={layer}_stimulus={stimulus_path}".replace(f"sub-{subject}", "sub-03")
            with open(os.path.join(CACHE, "embeddings", file_name), "rb") as file:
                all_embeddings = pickle.load(file)
            words_processed = process_words(all_words)
            aligned_embeddings, aligned_tokens = match_tokens_and_average_embeddings(tokens, all_embeddings, words_processed)
            assert len(aligned_tokens) == len(all_words), 'the tokens list and the words lists should match'
            
            all_neural_data.append(torch.Tensor(word_response))
            all_aligned_embeddings.append(aligned_embeddings)
        
        y_true, y_preds = ridge_regression_cv_2(X = all_aligned_embeddings, y = all_neural_data, device=device)
        
        all_scores = []
        for t_idx in range(y_true.shape[-1]):
            r_values = pearson_corrcoef(y_true[:,:,t_idx], y_preds[:,:,t_idx])
            all_scores.append(r_values.cpu())
    
        with open(os.path.join(CACHE,'r_scores',f"{config['model_iden']}_layer={layer}_subject={subject}_cv2"), "wb") as file:
            pickle.dump(all_scores, file)
