
import sys
ROOT = '/ccn2/u/atlask/FYP/'
sys.path.append(ROOT)

from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef

from regression.regression import ridge_regression_cv_nc
from data import load_repeated_file_paths, load_neural_data, process_words, convert_df_to_string
from utils import match_tokens_and_average_embeddings

import mne
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import os
import pickle
import yaml
with open(f"{ROOT}/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
    
DATA = paths["DATA"]
CACHE = paths["CACHE"]

os.makedirs(os.path.join(CACHE,'neural_preds'), exist_ok=True)
os.makedirs(os.path.join(CACHE,'r_scores'), exist_ok=True)

subjects = ['02','03','05','07','10','11']
window_size = 0
half_window = window_size // 2
stride = 0
    
device = "cuda"

gen = torch.Generator()
gen.manual_seed(42)

shuffle = False

# load neural data
shuffled_indices = [[],[],[]]

for subject in subjects:
    
    paths_1, paths_2 = load_repeated_file_paths(data_path=DATA, subject=subject)
    paths_1.sort()
    paths_2.sort()
        
    all_neural_data_1 = []
    all_neural_data_2 = []
    all_binned_times = []
    num_files = len(paths_1)
    
    
    for i, stimulus_path_1 in enumerate(paths_1):
        
        epoch_1 = mne.read_epochs(f'{DATA}/{paths_1[i]}', preload=True).get_data()
        epoch_2 = mne.read_epochs(f'{DATA}/{paths_2[i]}', preload=True).get_data()
        times = mne.read_epochs(f'{DATA}/{paths_1[i]}', preload=True).times

        epoch_1 = torch.Tensor(epoch_1).to(device)
        epoch_2 = torch.Tensor(epoch_2).to(device)
        
        print('binning and averaging data')
        
        step = 1 if stride == 0 else stride
        
        binned_times = [
            times[max(0, i - half_window): i + half_window + 1].mean()
            for i in range(half_window, len(times) - half_window, step)
        ]
        
        # Adjust the indexing to be centered around each value
        epoch_1_binned = torch.stack(
            [
                epoch_1[:, :, max(0, i - half_window): i + half_window + 1].mean(dim=-1)
                for i in range(half_window, epoch_1.shape[-1] - half_window, step)
            ],
            dim=-1
        )

        epoch_2_binned = torch.stack(
            [
                epoch_2[:, :, max(0, i - half_window): i + half_window + 1].mean(dim=-1)
                for i in range(half_window, epoch_2.shape[-1] - half_window, step)
            ],
            dim=-1
        )            
        print(epoch_2_binned.shape)
        assert len(epoch_1_binned) == len(epoch_2_binned), 'the two lists should have the same number of sessions'
        
        if shuffle:
            if shuffled_indices[i] == []:
                num_words = epoch_2_binned.shape[0]
                shuffled_idx = torch.randperm(num_words, generator=gen)
                shuffled_indices[i].append(shuffled_idx)
            else:
                shuffled_idx = shuffled_indices[i][0]
            
            epoch_2_binned = epoch_2_binned[shuffled_idx,:,:]                
            
        
        all_neural_data_1.append(epoch_1_binned)
        all_neural_data_2.append(epoch_2_binned)
        
            
    data_1 = torch.cat(all_neural_data_1, dim=0)
    data_2 = torch.cat(all_neural_data_2, dim=0)
    
    all_scores = []
        
    for t_idx in tqdm(range(data_1.shape[-1])):
        r_values = pearson_corrcoef(data_1[:,:,t_idx], data_2[:,:,t_idx])
        all_scores.append(r_values.cpu())
    
    print(f"noise_ceiling_subject={subject}_window={window_size}_stride={stride}")
    with open(os.path.join(ROOT,'results','r_scores',f"noise_ceiling_high_gamma_subject={subject}_window={window_size}_stride={stride}"), "wb") as file:
        pickle.dump(all_scores, file)
        
        
    with open(os.path.join(ROOT,'results','r_scores',f"noise_ceiling_times_high_gamma_subject={subject}_window={window_size}_stride={stride}"), "wb") as file:
        pickle.dump(binned_times, file)
