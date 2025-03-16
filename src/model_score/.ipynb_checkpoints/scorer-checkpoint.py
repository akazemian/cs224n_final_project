import sys
ROOT = '/ccn2/u/atlask/FYP/'
sys.path.append(ROOT)

from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
from model_score.regression import ridge_regression_cv
from neural_data.data import load_file_paths, load_neural_data, process_words, convert_df_to_string
from utils import cache, match_tokens_and_average_embeddings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import os
import pickle
import yaml
from concurrent.futures import ThreadPoolExecutor


with open("paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
    
DATA = paths["DATA"]
CACHE = paths["CACHE"]

with open("paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

TOKEN_CACHE_DIR = paths["TOKEN_CACHE_DIR"]

class Scorer:
    def __init__(self):
        pass

    @staticmethod
    def cache_file(identifier: str) -> str:
        path = os.path.join(CACHE, 'r_scores', identifier)  # Ensure CACHE is used
        print(f"Caching path: {path}")
        return path
        
    @cache(lambda identifier, **kwargs: Scorer.cache_file(identifier))    
    def get_scores_(self, identifier:str, model_iden:str, layer:int, subject:str, tokenizer, device:str):
        
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
            
            # load embeddings
            all_embeddings = load_embeddings(model_iden, layer, stimulus_path, subject)

            words_processed = process_words(all_words)
            aligned_embeddings, aligned_tokens = match_tokens_and_average_embeddings(tokens, all_embeddings, words_processed)
            assert len(aligned_tokens) == len(all_words), 'the tokens list and the words lists should match'
            
            all_neural_data.append(torch.Tensor(word_response))
            all_aligned_embeddings.append(aligned_embeddings)
        
        y_true, y_preds = ridge_regression_cv(X = all_aligned_embeddings, y = all_neural_data, device=device)
        
        all_scores = []
        for t_idx in range(y_true.shape[-1]):
            r_values = pearson_corrcoef(y_true[:,:,t_idx].to(device), y_preds[:,:,t_idx].to(device))
            all_scores.append(r_values)
        return all_scores


    def get_scores(self, model_iden:str, num_layers:int, subject:str, tokenizer, devices: list): 
        
        layers_per_device = [list(range(i, num_layers, len(devices))) for i in range(len(devices))]
        print('layers per device',layers_per_device)

        def process_layer_chunk(layers, device, tokenizer):
            
            results = {}
            for layer in layers:
                try:
                    scores = self.get_scores_(
                        identifier=f"{model_iden}_layer={layer}_subject={subject}", 
                        model_iden=model_iden, 
                        layer=layer, 
                        subject=subject,
                        tokenizer=tokenizer, 
                        device=device
                    )
                    results[layer] = scores
                except RuntimeError as e:
                    print(f"Error on layer {layer}, device {device}: {e}")
                    torch.cuda.empty_cache()
            return results
        
        # Initialize results dictionary
        all_results = {}
        
        # Perform parallel processing
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {
                executor.submit(
                    process_layer_chunk, 
                    layers_per_device[device_idx], 
                    device, 
                    tokenizer, 
                ): device_idx
                for device_idx, device in enumerate(devices)
            }
        
            for future in futures:
                try:
                    results = future.result()
                    all_results.update(results)
                except Exception as e:
                    print(f"Error in thread execution: {e}")


def load_embeddings(model_iden, layer, stimulus_path, subject):
    file_name = f"{model_iden}_layer={layer}_stimulus={stimulus_path}".replace(f"sub-{subject}", "sub-03")
    with open(os.path.join(CACHE, "embeddings", file_name), "rb") as file:
        all_embeddings = pickle.load(file)
    return all_embeddings