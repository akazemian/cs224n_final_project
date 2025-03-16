import os
import torch
import gc
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback
from tqdm import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor
import random
import re
from src.model_embeddings.utils import cache
from src.utils import load_paths

paths = load_paths()
DATA = paths["DATA"]
CACHE = paths["CACHE"]

torch.manual_seed(42)
# idx = random.sample(range(4096+1), 768)
# idx = random.sample(range(768+1), 100)

class LLM:
    def __init__(self, model_name:str, num_layers:int, max_window_size:int, is_bidirectional:bool):
        self.model_name = model_name
        self.num_layers = num_layers 
        self.max_window_size = max_window_size
        self.is_bidirectional = is_bidirectional

        if self.is_bidirectional:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=CACHE)
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=CACHE)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE)
        
    def get_token_ids(self, input_string):
        """Tokenize the input string and return token IDs."""
        return self.tokenizer(input_string, return_tensors='pt')['input_ids'][0]
            
    @staticmethod
    def cache_file(identifier: str) -> str:
        path = os.path.join(paths["EMBEDDINGS_PATH"], identifier)  # Ensure CACHE is used
        print(f"Caching path: {path}")
        return path

    @cache(lambda identifier, **kwargs: LLM.cache_file(identifier))    
    def get_embeddings_(self, identifier: str, layer: int, token_ids: torch.Tensor, batch_size: int):
        """
        Extract embeddings for each token using a sliding window.
        For bidirectional models, a symmetric window is constructed so that the target token is at the center.
        For causal models, the window is built from the beginning up to the current token.
        """
        device = self.model.device
        token_ids = token_ids.to(device)
        if self.max_window_size == 0:
            self.max_window_size = int(len(token_ids))

        token_embeddings = []

        if self.is_bidirectional:
            left_req = self.max_window_size // 2
            right_req = self.max_window_size - left_req - 1
            # Process every token in batches.
            for i in tqdm(range(0, len(token_ids), batch_size)):
                batch_end = min(i + batch_size, len(token_ids))
                windows = []
                attention_masks = []
                target_indices = []  # will always be left_req in the padded window
                for j in range(i, batch_end):
                    # Determine how many tokens can be taken from left and right
                    actual_left = min(j, left_req)
                    actual_right = min(len(token_ids) - j - 1, right_req)
                    
                    # Extract the tokens available for the window
                    window_tokens = token_ids[j - actual_left : j + actual_right + 1]
                    
                    # Compute how much padding is needed
                    left_padding = left_req - actual_left
                    right_padding = right_req - actual_right
                    
                    # Create left and right pads
                    left_pad = torch.zeros(left_padding, dtype=torch.long, device=device)
                    right_pad = torch.zeros(right_padding, dtype=torch.long, device=device)
                    
                    # Concatenate to form the padded window.
                    window = torch.cat([left_pad, window_tokens, right_pad])
                    
                    # Check that the window length is exactly max_window_size.
                    assert window.size(0) == self.max_window_size, \
                        f"Window length mismatch: expected {self.max_window_size}, got {window.size(0)}"
                    
                    windows.append(window)
                    attention_masks.append((window != 0).long())
                    target_indices.append(left_req)  # target is always at index left_req
                
                batch_tensor = torch.stack(windows).to(device)
                attention_mask_tensor = torch.stack(attention_masks).to(device)
                
                with torch.no_grad():
                    model_outputs = self.model(batch_tensor, attention_mask=attention_mask_tensor, 
                                            output_hidden_states=True)
                layer_activations = model_outputs.hidden_states[layer]
                
                target_indices_tensor = torch.tensor(target_indices, device=device)
                batch_token_embeddings = layer_activations[torch.arange(len(windows)), target_indices_tensor, :]
                token_embeddings.append(batch_token_embeddings)
                
                del batch_tensor, model_outputs, windows, layer_activations
                torch.cuda.empty_cache()
        else:
            for i in tqdm(range(1, len(token_ids), batch_size)):
                batch_end = min(i + batch_size, len(token_ids))
                windows = []
                attention_masks = []
                position_ids_list = []
                for j in range(i, batch_end):
                    window = token_ids[max(0, j - self.max_window_size): j+1]
                    padding_length = self.max_window_size - len(window)
                    padded_window = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device), window])
                    attention_mask = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device),
                                                torch.ones(len(window), dtype=torch.long, device=device)])
                    windows.append(padded_window)
                    attention_masks.append(attention_mask)
                    pos_ids = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device),
                                         torch.arange(len(window), dtype=torch.long, device=device)])
                    position_ids_list.append(pos_ids)
                batch_tensor = torch.stack(windows).to(device)
                attention_mask_tensor = torch.stack(attention_masks).to(device)
                position_ids_tensor = torch.stack(position_ids_list).to(device)
                with torch.no_grad():
                    model_outputs = self.model(batch_tensor, attention_mask=attention_mask_tensor, 
                                               position_ids=position_ids_tensor, output_hidden_states=True)
                layer_activations = model_outputs.hidden_states[layer]
                token_indices = (torch.tensor([len(w) for w in windows]) - 1).to(device)
                batch_token_embeddings = layer_activations[torch.arange(len(windows)), token_indices, :]
                token_embeddings.append(batch_token_embeddings)
                del batch_tensor, model_outputs, windows, layer_activations
                torch.cuda.empty_cache()

        all_embeddings = torch.cat(token_embeddings, dim=0).to(device)
        return all_embeddings.cpu()

    # @cache(lambda identifier, **kwargs: LLM.cache_file(identifier))    
    # def get_embeddings_(self, identifier: str, layer: int, token_ids: list, batch_size: int):
    #     device = self.model.device
    #     token_ids = token_ids.to(device)
    #     if self.max_window_size == 0:
    #         self.max_window_size = int(len(token_ids))

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
    #                 window = token_ids[max(0, j - self.max_window_size):j+1]  # Use a fixed window size
    #             else:
    #                 window = token_ids[:j+1]  # Include tokens up to the current token
                
    #             # Pad the window to max_window_size length
    #             padding_length = self.max_window_size - len(window)
    #             padded_window = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device), window.to(device)])  # Left padding with 0
                
    #             # Create an attention mask where 1 is for real tokens and 0 is for padding (token id 0)
    #             attention_mask = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device), torch.ones(len(window), dtype=torch.long, device=device)])
                
    #             # Create position IDs where padding gets 0 and real tokens get their original positions
    #             position_id = torch.cat([torch.zeros(padding_length, dtype=torch.long, device=device), torch.arange(len(window), dtype=torch.long).to(device)])
                
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
        

    #     all_embeddings = torch.cat(token_embeddings, dim=0).to(device)
    #     return all_embeddings.cpu()
        


    def get_embeddings(self, identifier_fn, token_ids: list, batch_size: int, devices: list): 
        
        layers_per_device = [list(range(i, self.num_layers, len(devices))) for i in range(len(devices))]
        print('layers per device',layers_per_device)

        def process_layer_chunk(llm_instance, layers, device, token_ids, batch_size):
            # Move the LLM model and token IDs to the target device
            llm_instance.model = llm_instance.model.to(device)
            token_ids = token_ids.to(device)
            
            results = {}
            for layer in layers:
                try:
                    embeddings = llm_instance.get_embeddings_(
                        identifier=identifier_fn(layer=layer), 
                        layer=layer, 
                        token_ids=token_ids, 
                        batch_size=batch_size, 
                    )
                    results[layer] = embeddings
                except RuntimeError as e:
                    traceback.print_exc()
                    print(f"Error on layer {layer}, device {device}: {e}")
                    torch.cuda.empty_cache()
            return results
        
        # Preload LLM instances for each device
        llm_instances_per_device = {
            device: LLM(
                model_name=self.model_name, 
                num_layers=self.num_layers, 
                max_window_size=self.max_window_size,
                is_bidirectional = self.is_bidirectional
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
                    traceback.print_exc()
                    print(f"Error in thread execution: {e}")
        
