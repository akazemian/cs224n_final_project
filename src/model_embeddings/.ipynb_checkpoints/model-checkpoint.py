import mne
import os
import pickle
import torch
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import yaml
from .utils import cache
from concurrent.futures import ThreadPoolExecutor

with open("paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

DATA = paths["DATA"]
CACHE = paths["CACHE"]

os.makedirs(CACHE, exist_ok=True)

class LLM:
    def __init__(self, model_name, num_layers, max_window_size):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE)
        self.num_layers = num_layers 
        self.max_window_size = max_window_size
        
    @staticmethod
    def cache_file(identifier: str) -> str:
        path = os.path.join(CACHE, 'embeddings', identifier)  # Ensure CACHE is used
        print(f"Caching path: {path}")
        return path

    def get_tokens(self, sinput_string:str):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE)
        tokens = tokenizer.tokenize(input_string)
        return tokens

    def get_token_ids(self, input_string:str):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE)
        return tokenizer(input_string, return_tensors='pt')['input_ids'][0]

    @cache(lambda identifier, **kwargs: LLM.cache_file(identifier))    
    def get_embeddings_(self, identifier: str, layer: int, token_ids: list, batch_size: int):
        
        device = self.model.device
        token_ids = token_ids.to(device)  # Ensure token_ids are on the same device
        token_embeddings = []
        
        for i in tqdm(range(1, len(token_ids), batch_size)):
            batch_end = min(i + batch_size, len(token_ids))
            windows, attention_masks, position_ids = [], [], []
        
            for j in range(i, batch_end):
                # Define window
                window = token_ids[max(0, j - self.max_window_size):j]
                padding_length = self.max_window_size - len(window)
    
                # Move all tensors to the correct device
                padded_window = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device), window])
                attention_mask = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device),
                                            torch.ones(len(window), dtype=torch.long, device=device)])
                position_id = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device),
                                         torch.arange(len(window), dtype=torch.long, device=device)])
                
                windows.append(padded_window)
                attention_masks.append(attention_mask)
                position_ids.append(position_id)
        
            batch_tensor = torch.stack(windows).to(device)
            attention_mask_tensor = torch.stack(attention_masks).to(device)
            position_ids_tensor = torch.stack(position_ids).to(device)
        
            with torch.no_grad():
                model_outputs = self.model(
                    batch_tensor, 
                    attention_mask=attention_mask_tensor, 
                    position_ids=position_ids_tensor, 
                    output_hidden_states=True
                )
            
            layer_activations = model_outputs.hidden_states[layer]
            last_token_indices = (torch.tensor([len(w) for w in windows], device=device) - 1)
            batch_token_embeddings = layer_activations[torch.arange(len(windows), device=device), last_token_indices, :]
            token_embeddings.append(batch_token_embeddings)
    
            # Free GPU memory
            del batch_tensor, attention_mask_tensor, position_ids_tensor, model_outputs, windows, layer_activations
            torch.cuda.empty_cache() 
        
        # Concatenate token embeddings on the correct device
        return torch.cat(token_embeddings, dim=0).to(device)

    
    def get_embeddings(self, model_name: str, model_iden:str, stimulus_path:str, token_ids: list, batch_size: int, devices: list): 
        
        layers_per_device = [list(range(i, self.num_layers, len(devices))) for i in range(len(devices))]
        print('layers per device',layers_per_device)

        def process_layer_chunk(llm_instance, layers, device, token_ids, batch_size):
            # Move the LLM model and token IDs to the target device
            llm_instance.model = llm_instance.model.to(device)
            token_ids = token_ids.to(device)
            
            results = {}
            for layer in layers:
                print(f"extracting activations for {model_iden}_layer={layer}_stimulus={stimulus_path}")
                try:
                    embeddings = llm_instance.get_embeddings_(
                        identifier=f"{model_iden}_layer={layer}_stimulus={stimulus_path}", 
                        layer=layer, 
                        token_ids=token_ids, 
                        batch_size=batch_size
                    )
                    results[layer] = embeddings
                except RuntimeError as e:
                    print(f"Error on layer {layer}, device {device}: {e}")
                    torch.cuda.empty_cache()
            return results
        
        # Preload LLM instances for each device
        llm_instances_per_device = {
            device: LLM(
                model_name=model_name, 
                num_layers=self.num_layers, 
                max_window_size=self.max_window_size
            ) for device in devices
        }
        
        # Initialize results dictionary
        all_results = {}
        
        # Perform parallel processing
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {
                executor.submit(
                    process_layer_chunk, 
                    llm_instances_per_device[device],  # Pass LLM instance
                    layers_per_device[device_idx], 
                    device, 
                    token_ids, 
                    batch_size
                ): device_idx
                for device_idx, device in enumerate(devices)
            }
        
            for future in futures:
                try:
                    results = future.result()
                    all_results.update(results)
                except Exception as e:
                    print(f"Error in thread execution: {e}")
        
                
# @cache(lambda identifier, **kwargs: LLM.cache_file(identifier))
# def get_embeddings(self, identifier:str, layer:int, token_ids:list, batch_size=int):
    
#     device = self.model.device
#     token_embeddings = []
        
#     # Processing in batches
#     for i in tqdm(range(1, len(token_ids), batch_size)):
#         batch_end = min(i + batch_size, len(token_ids))
#         windows = []
#         attention_masks = []  # Store attention masks for each batch
#         position_ids = []  # Store position IDs for each batch
    
#         for j in range(i, batch_end):
            
#             # Define a fixed window size around each token (up to max_window_size tokens before current token)
#             if self.max_window_size is not None:
#                 window = token_ids[max(0, j - self.max_window_size):j]  # Use a fixed window size
#             else:
#                 window = token_ids[:j]  # Include tokens up to the current token
            
#             # Pad the window to max_window_size length
#             padding_length = self.max_window_size - len(window)
#             padded_window = torch.cat([torch.zeros(padding_length, dtype=torch.long), window])  # Left padding with 0
            
#             # Create an attention mask where 1 is for real tokens and 0 is for padding (token id 0)
#             attention_mask = torch.cat([torch.zeros(padding_length, dtype=torch.long), 
#                                         torch.ones(len(window), dtype=torch.long)])
            
#             # Create position IDs where padding gets 0 and real tokens get their original positions
#             position_id = torch.cat([torch.zeros(padding_length, dtype=torch.long), 
#                                      torch.arange(len(window), dtype=torch.long)])
            
#             windows.append(padded_window)  # Append padded window to the batch list
#             attention_masks.append(attention_mask)  # Append attention mask to the list
#             position_ids.append(position_id)  # Append position IDs to the list
    
#         # Stack the windows into a batch tensor (batch_size, max_window_size)
#         batch_tensor = torch.stack(windows).to(device)
#         # Stack the attention masks into a tensor
#         attention_mask_tensor = torch.stack(attention_masks).to(device)
#         # Stack the position IDs into a tensor
#         position_ids_tensor = torch.stack(position_ids).to(device)
    
#         # Forward pass through the model, using the attention mask and position IDs
#         with torch.no_grad():
#             model_outputs = self.model(batch_tensor, attention_mask=attention_mask_tensor, 
#                                        position_ids=position_ids_tensor, output_hidden_states=True)
        
#         # Extract hidden states for the last token in each window
#         layer_activations = model_outputs.hidden_states[layer]  # Last layer hidden states
#         last_token_indices = (torch.tensor([len(w) for w in windows]) - 1).to(device)
        
#         # Get the hidden state for the last token in each window in the batch
#         batch_token_embeddings = layer_activations[torch.arange(len(windows)), last_token_indices, :]
        
#         token_embeddings.append(batch_token_embeddings)
#         del batch_tensor, model_outputs, windows, layer_activations
#         torch.cuda.empty_cache() 
    
#     # Stack all token embeddings into a tensor
#     return torch.cat(token_embeddings, dim=0)