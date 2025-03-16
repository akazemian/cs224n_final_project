import torch
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
import traceback
import gc
from tqdm import tqdm 
import numpy as np
from scipy.signal import medfilt

from src.model_score.utils import cache, match_tokens_and_average_embeddings, similarity_score, load_elecs 
# from src.model_score.regression_torch import RegressionCVTorch
from src.model_score.ridgecv import RidgeRegression
# from src.model_score.ridgecv_torch import cv

from src.neural_data.data import load_repeated_file_paths, load_neural_data, process_words, convert_df_to_string
from src.utils import load_paths, construct_model_id


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


paths = load_paths()   
NUM_TIME_POINTS = 101

from scipy.stats import pearsonr
import numpy as np

def pearson_corr_per_electrode(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation coefficient per electrode.
    Assumes y_true and y_pred have shape (n_samples, n_electrodes).
    Returns an array of correlation coefficients of shape (n_electrodes,).
    """
    n_electrodes = X.shape[1]
    r_values = np.zeros(n_electrodes)
    for i in range(n_electrodes):
        # Compute Pearson correlation for each electrode (each column)
        r_values[i] = pearsonr(X[:, i], Y[:, i])[0]
    return r_values

def plot_alpha_dist(alphas, file_path):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_electrodes = alphas.shape[1]
    colors = sns.color_palette("husl", n_electrodes)

    plt.figure(figsize=(10, 6))
    for i in range(n_electrodes):
        sns.kdeplot(alphas[:, i], color=colors[i], label=f'Electrode {i+1}', linewidth=2)
    plt.xscale("log")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Alpha Distributions per Electrode")
    plt.legend()
    plt.savefig(file_path)


class Scorer:
    def __init__(self, model_name, tokenizer=None, time_lag=None, num_features=None):
        self.model_name = model_name
        self.tokenizer =  tokenizer
        self.time_lag = time_lag
        self.num_features = num_features
        
    @staticmethod
    def cache_file(identifier: str) -> str:
        path = os.path.join(paths["SCORES_PATH"], identifier)  # Ensure CACHE is used
        return path

    @cache(lambda identifier, **kwargs: Scorer.cache_file(identifier))    
    def get_scores_(self, identifier:str, model_iden:str, layer:int, subject:str, device:str):
        
        all_neural_data_1, all_neural_data_2 = [], []
        all_embeddings = []
        
        stim_paths_1, stim_paths_2 = load_repeated_file_paths(paths["DATA"], subject)
                
        for s, stim_path in enumerate(stim_paths_1):
            # load and process neural data
            stimulus_df_1, word_response_1 = load_neural_data(data_dir=paths["DATA"], file_path=stim_path)
            _ , word_response_2 = load_neural_data(data_dir=paths["DATA"], file_path=stim_paths_2[s])
            print('neural data before ANYTHING', word_response_1.shape)
            word_response_1, word_response_2 = self.custom_process_data(stim_path, word_response_1, word_response_2)

            word_response_1 = median_pooling_tensor(word_response_1, NUM_TIME_POINTS)
            word_response_2 = median_pooling_tensor(word_response_2, NUM_TIME_POINTS)
            
            embeddings = load_and_process_embeddings(model_iden, layer, stim_path, self.time_lag, 
                                                     stimulus_df_1, self.tokenizer, device, self.num_features
                                                     )

            elecs = list(load_elecs(subject).keys())
            all_embeddings.append(torch.Tensor(embeddings).to(device))
            all_neural_data_1.append(torch.Tensor(word_response_1[:,elecs,:]).to(device))
            all_neural_data_2.append(torch.Tensor(word_response_2[:,elecs,:]).to(device))
            
            del embeddings
            gc.collect()  

        X_np = torch.cat(all_embeddings).cpu().numpy()
        y_np1 = torch.cat(all_neural_data_1).cpu().numpy()
        y_np2 = torch.cat(all_neural_data_2).cpu().numpy()

        ridge = RidgeRegression()
        y_pred_1, y_true_1, alphas_1 = ridge.cv(X=X_np, y=y_np1)
        y_pred_2, y_true_2, alphas_2 = ridge.cv(X=X_np, y=y_np2)

        # run = 'sequential'
        # with open(os.path.join(paths["CACHE"], f'y_true_1_run_{run}_layer_{layer}'), "wb") as file:
        #     pickle.dump(y_true_1,file)
        # with open(os.path.join(paths["CACHE"], f'y_true_2_run_{run}_layer_{layer}'), "wb") as file:
        #     pickle.dump(y_true_2,file)       
            
        # with open(os.path.join(paths["CACHE"], f'y_pred_1_run_{run}_layer_{layer}'), "wb") as file:
        #     pickle.dump(y_pred_1,file)
        # with open(os.path.join(paths["CACHE"], f'y_pred_2_run_{run}_layer_{layer}'), "wb") as file:
        #     pickle.dump(y_pred_2,file)       
            
        # with open(os.path.join(paths["CACHE"], f'alphas_1_run_{run}_layer_{layer}'), "wb") as file:
        #     pickle.dump(alphas_1,file)
        # with open(os.path.join(paths["CACHE"], f'alphas_2_run_{run}_layer_{layer}'), "wb") as file:
        #     pickle.dump(alphas_2,file)            

        mean_scores = []
        for t_idx in range(y_true_1.shape[-1]):
            r_values_1 = pearson_corr_per_electrode(X=y_true_1[:,:,t_idx], Y=y_pred_2[:,:,t_idx])
            r_values_2 = pearson_corr_per_electrode(X=y_true_2[:,:,t_idx], Y=y_pred_1[:,:,t_idx])
            
            r_values = (r_values_1+r_values_2)/2
            mean_scores.append(r_values)        

        # plot_alpha_dist(alphas_1.cpu(), f'alphas_{model_iden}_layer={layer}_subject={subject}_torch.png')
        gc.collect()
        return mean_scores


    def get_scores(self, identifier_fn, model_iden, subject, num_layers:int, devices: list): 
        
        if isinstance(devices, list):
            print('multiple devices chosen, distributing computations across devices...')
                    
            layers_per_device = [list(range(i, num_layers, len(devices))) for i in range(len(devices))]
            print('layers per device',layers_per_device)

            def process_layer_chunk(layers, device):
                
                results = {}
                for layer in layers:
                    try:
                        scores = self.get_scores_(
                            identifier=identifier_fn(layer=layer), 
                            model_iden=model_iden,
                            subject=subject,
                            layer=layer, 
                            device=device,
                        )
                        results[layer] = scores
                        del scores
                        torch.cuda.empty_cache()

                    except RuntimeError as e:
                        print(f"Error on layer {layer}, device {device}: {e}")
                        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                        print(tb_str)
                        break
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
                    ): device_idx
                    for device_idx, device in enumerate(devices)
                }
                for future in futures:
                    try:
                        results = future.result()
                        all_results.update(results)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        # Explicitly print the exception and traceback
                        print(f"Error in thread execution for device {futures[future]}:")
                        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                        print(tb_str)
            return all_results
        
        else:
            print('one device chosen...')
            if num_layers == "none":
                scores = self.get_scores_(
                identifier=identifier_fn(layer=num_layers), 
                model_iden=model_iden, 
                layer=num_layers, 
                subject=subject,
                device=devices
                        )
            else:
                for layer in range(num_layers):
                    print('layer:',layer)
                    scores = self.get_scores_(
                    identifier=identifier_fn(layer=layer), 
                    model_iden=model_iden, 
                    layer=layer, 
                    subject=subject,
                    device=devices
                            )
            return 

    def custom_process_data(self, stim_path, word_response_1, word_response_2):
        
        if 'Jobs2' in stim_path:
            word_response_1 = word_response_1[:-2,...]
            word_response_2 = word_response_2[:-2,...]

        return word_response_1, word_response_2
    


def load_and_process_embeddings(model_iden, layer, stimulus_path, time_lag, stimulus_df_1, tokenizer, device, num_features):
    import re
    match = re.search(r'Jobs[1-3]', stimulus_path)
    stimulus = match.group(0).lower()
    
    if 'stacked' in model_iden:
        iden_1 = construct_model_id(model_iden='cochleagram', layer='none', dataset='ieeg', 
                                    stimulus=stimulus, time_lag = time_lag)
        with open(os.path.join(paths["EMBEDDINGS_PATH"], iden_1), "rb") as file:
            embeddings_1 = pickle.load(file)
        # embeddings_1 = median_pooling_tensor(embeddings_1, NUM_TIME_POINTS)

        iden_2 = construct_model_id('gpt2', layer, 'ieeg', stimulus, time_lag)
        with open(os.path.join(paths["EMBEDDINGS_PATH"], iden_2), "rb") as file:
            embeddings_2 = pickle.load(file)
        embeddings_2 = token_word_alignment(embeddings_2, stimulus_df_1, tokenizer)
        embeddings_2 = embeddings_2.unsqueeze(-1).repeat(1,1,800)
        if 'Jobs2' in stimulus_path: #discard the last 2 words since they're not in the neural data
            embeddings_2 = embeddings_2[:-2,...]
        embeddings = torch.cat([embeddings_1.to(device), embeddings_2.to(device)], dim=1)
    
    else:
        file_name = construct_model_id(model_iden=model_iden, 
                                       layer=layer, 
                                       dataset='ieeg', 
                                       stimulus=stimulus,
                                       time_lag= time_lag)
        with open(os.path.join(paths["EMBEDDINGS_PATH"], file_name), "rb") as file:
            embeddings = pickle.load(file)
        
        if tokenizer is not None:
            embeddings = token_word_alignment(embeddings, stimulus_df_1, tokenizer)
        
    if 'cochleagram' in model_iden:
        embeddings = median_pooling_tensor(embeddings, NUM_TIME_POINTS)
    
    if 'Jobs2' in stimulus_path: #discard the last 2 words since they're not in the neural data
        embeddings = embeddings[:-2,...]
    
    # random sample of n features, whereb n is NUM_FEATURES
    # num_model_features = embeddings.shape[1]
    # indices = torch.randperm(num_model_features)[:num_features]
    # embeddings = embeddings[:, indices, ...]
    # print('embeddings after sample', embeddings.shape)
    
    # pca
    # scaler  = TorchStandardScaler()
    # X_scaled = scaler.fit_transform(embeddings)
    # print(X_scaled.shape)
    # pca = PCA(n_components=NUM_FEATURES)
    # if X_scaled.dim() == 2:
    #     embeddings_pca = torch.Tensor(pca.fit_transform(X_scaled.cpu())).to(device)
    # elif X_scaled.dim() == 3:
    #     B, N, D = X_scaled.shape
    #     embeddings_pca = torch.zeros((B, NUM_FEATURES, D), device=X_scaled.device)
    #     for d in tqdm(range(D)):
    #         pca = PCA(n_components=NUM_FEATURES)
    #         reduced = torch.tensor(pca.fit_transform(X_scaled[:,:,d].cpu()), 
    #                                              device=X_scaled.device)
    #         embeddings_pca[:,:,d] = reduced  # Store result in pre-allocated tensor
            
    #         # Free memory explicitly
    #         del reduced, pca
    #         gc.collect()  # Trigger garbage collection

    # torch.cuda.empty_cache()
    print('test embeddings', embeddings.shape)
    return embeddings 


def token_word_alignment(embeddings, stimulus_df, tokenizer):
        stimulus_str = convert_df_to_string(stimulus_df)
        tokens = tokenizer.tokenize(stimulus_str)
        words = stimulus_df.tolist()
        words_processed = process_words(words)
        embeddings, aligned_tokens = match_tokens_and_average_embeddings(tokens, embeddings, words_processed)
        assert len(aligned_tokens) == len(words), 'the tokens list and the words lists should match'   
        del words_processed, aligned_tokens, stimulus_str, tokens
        return embeddings


def median_pooling_tensor(data, target_time):
    """
    Applies median pooling along the time dimension for a 3D tensor [n_words, electrodes, time].
    
    Parameters:
        data (numpy array): Input tensor of shape [n_words, electrodes, time].
        target_time (int): The desired number of time points after pooling.
    
    Returns:
        numpy array: Pooled tensor of shape [n_words, electrodes, target_time].
    """
    n_words, n_electrodes, time_steps = data.shape
    factor = time_steps // target_time  # Compute window size

    if factor < 2:
        raise ValueError("Target time must be smaller than the original time dimension.")

    # Apply median filter across time dimension while preserving alignment
    filtered_data = medfilt(data, kernel_size=(1, factor, 1))

    # Downsample by selecting evenly spaced indices
    indices = np.linspace(0, filtered_data.shape[2] - 1, target_time, dtype=int)
    
    return torch.Tensor(filtered_data[:, :, indices])



class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, unbiased=False, keepdim=True)  # Population std

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)  # Avoid division by zero

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

