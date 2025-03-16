import os
import pickle
import yaml
import functools
import gc 
import torch
import numpy as np
import pandas as pd
from src.utils import load_paths

paths = load_paths()   

CACHE = paths["CACHE"]
SCORES_PATH = paths["SCORES_PATH"]

def cache(file_name_func):
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(SCORES_PATH, exist_ok=True)
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            file_name = file_name_func(*args, **kwargs)
            cache_path = os.path.join(CACHE, file_name)

            if os.path.exists(cache_path):
                print(f'scores for {cache_path} already exist')
                return

            result = func(self, *args, **kwargs)
            with open(cache_path, 'wb') as file:
                pickle.dump(result, file)
            gc.collect()

        return wrapper
    return decorator



LOC = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper', 'transversetemporal', 'superiortemporal','middletemporal', 'superiorfrontal', 
       'inferiortemporal', 'Hippocampus', 'parsopercularis',
       'supramarginal', 'parstriangularis', 'insula', 'bankssts']

def load_elecs(subject):
    NEURAL_DATA = '/ccn2/u/atlask/iEEG/raw_data/'
    elec_info = f'{NEURAL_DATA}/sub-{subject}/ses-01/ieeg/sub-{subject}_ses-01_space-fsaverage_electrodes.tsv'
    elecs = pd.read_csv(elec_info, sep='\t')
    loc_dict = elecs[elecs['label'].isin(LOC)]['label'].to_dict() 
    return loc_dict 

def preprocess_token(token):
    # Remove GPT-2's leading space marker and BERT's "##" prefix.
    if token.startswith("Ġ"):
        return token[1:]
    elif token.startswith("##"):
        return token[2:]
    else:
        return token

def match_tokens_and_average_embeddings(original_tokens, original_embeddings, reference_tokens):
    """
    Aligns tokens from an original tokenization with a reference tokenization,
    then averages the embeddings of subword tokens to yield one embedding per reference token.
    
    Parameters
    ----------
    original_tokens : list of str
        The tokens from the original model's tokenizer (e.g., GPT-2 or BERT).
    original_embeddings : list or tensor of embeddings
        The embeddings corresponding to each token in original_tokens.
    reference_tokens : list of str
        The tokens of the reference tokenization (typically word-level tokens).
    
    Returns
    -------
    aligned_embeddings : torch.Tensor
        A tensor containing one embedding per reference token.
    aligned_tokens : list of str
        The reference tokens.
    """
    # Preprocess original tokens: remove "Ġ" or "##" markers.
    stripped_tokens = [preprocess_token(token) for token in original_tokens]

    aligned_tokens = []      # Final aligned tokens (should equal reference_tokens)
    aligned_embeddings = []  # Averaged embeddings for each reference token

    original_index = 0
    for ref_token in reference_tokens:
        current_word = []
        current_embeddings = []

        # Continue appending tokens until the concatenated (preprocessed) tokens equal the reference.
        while "".join(current_word) != ref_token and original_index < len(stripped_tokens):
            current_word.append(stripped_tokens[original_index])
            current_embeddings.append(original_embeddings[original_index])
            original_index += 1

        # Average the embeddings if any tokens were collected.
        if current_embeddings:
            # Efficient averaging using torch.mean
            aligned_embeddings.append(torch.mean(torch.stack(current_embeddings), dim=0))
            aligned_tokens.append(ref_token)
        else:
            # In case no tokens were found for the ref_token, append a zero vector.
            # (You might want to handle this case differently.)
            aligned_embeddings.append(torch.zeros_like(original_embeddings[0]))
            aligned_tokens.append(ref_token)

    # Convert aligned_embeddings to a single tensor.
    aligned_embeddings = torch.stack(aligned_embeddings)
    return aligned_embeddings, aligned_tokens

# def match_tokens_and_average_embeddings(original_tokens, original_embeddings, reference_tokens):
#     """
#     Optimized alignment of original tokenization with the reference tokenization.
#     """
#     aligned_tokens = []  # To store the final tokens that match reference tokens
#     aligned_embeddings = []  # To store the final embeddings

#     # Pre-strip space prefixes from original tokens (e.g., 'Ġ')
#     stripped_tokens = [token.replace('Ġ', '') for token in original_tokens]

#     original_index = 0
#     for ref_token in reference_tokens:
#         current_word = []
#         current_embeddings = []

#         while "".join(current_word) != ref_token:
#             current_word.append(stripped_tokens[original_index])
#             current_embeddings.append(original_embeddings[original_index])
#             original_index += 1

#         # Efficient averaging using torch.mean
#         aligned_embeddings.append(torch.mean(torch.stack(current_embeddings), dim=0))
#         aligned_tokens.append(ref_token)

#     # Convert aligned_embeddings to a single tensor
#     aligned_embeddings = torch.stack(aligned_embeddings)

#     return aligned_embeddings, aligned_tokens


def similarity_score(X, Y, device, metric_type='pearsonr'):
    from torchmetrics.functional import pearson_corrcoef

    X, Y = torch.Tensor(X).to(device), torch.Tensor(Y).to(device)
    if metric_type == 'pearsonr':
        return pearson_corrcoef(X, Y)
            

def normalize(X_train, X_test):

    mu_train = X_train.mean(dim=0)  # Mean of each feature (column)
    sigma_train = X_train.std(dim=0)  # Std of each feature (column)

    X_train_normalized = (X_train - mu_train) / sigma_train
    X_test_normalized = (X_test - mu_train) / sigma_train

    return X_train_normalized, X_test_normalized


def get_sample_indices(total_embeddings, num_samples=10):
    """
    Generate sample indices representing relative fractions of the total embeddings.
    
    Parameters
    ----------
    total_embeddings : int
        The total number of embeddings for a given model.
    num_samples : int, optional
        How many samples (fractions) to take (default is 10, e.g. 10%, 20%, ..., 100%).
    
    Returns
    -------
    np.ndarray
        Array of indices corresponding to the fractions.
    """
    # Create fractions from 10% to 100% (inclusive)
    fractions = np.linspace(0.1, 1.0, num_samples)
    # Convert fractions to actual indices (rounding to int)
    indices = (fractions * total_embeddings).astype(int)
    return list(indices)






