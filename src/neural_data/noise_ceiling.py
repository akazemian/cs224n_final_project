import sys
ROOT = '/ccn2/u/atlask/FYP/'
sys.path.append(ROOT)

from data import load_repeated_file_paths

import mne
import torch
import os
import pickle
import yaml
from sklearn.model_selection import KFold
from tqdm import tqdm
from torchmetrics.functional import pearson_corrcoef

with open(f"{ROOT}/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
    
DATA = paths["DATA"]
CACHE = paths["CACHE"]

os.makedirs(os.path.join(CACHE,'neural_preds'), exist_ok=True)
os.makedirs(os.path.join(CACHE,'r_scores'), exist_ok=True)

subjects = ['02','03','05','07','10','11']
window_size = 0
stride = 0
    
device = "cuda"

# load neural data
for subject in subjects:
    
    paths_1, paths_2 = load_repeated_file_paths(data_path=DATA, subject=subject)
    paths_1.sort()
    paths_2.sort()
    
    all_neural_data_1 = []
    all_neural_data_2 = []
    
    for i, stimulus_path_1 in enumerate(paths_1):
        
        epoch_1 = mne.read_epochs(f'{DATA}/{paths_1[i]}', preload=True).get_data()
        epoch_2 = mne.read_epochs(f'{DATA}/{paths_2[i]}', preload=True).get_data()
        times = mne.read_epochs(f'{DATA}/{paths_1[i]}', preload=True).times

        epoch_1 = torch.Tensor(epoch_1).to(device)
        epoch_2 = torch.Tensor(epoch_2).to(device)        
        
        assert len(epoch_1) == len(epoch_2), 'the two lists should have the same number of sessions'
        
        all_neural_data_1.append(epoch_1)
        all_neural_data_2.append(epoch_2)
    
    data_1 = torch.cat(all_neural_data_1, dim=0)
    data_2 = torch.cat(all_neural_data_2, dim=0)
    
    
    k = 5
    kf = KFold(n_splits=k, shuffle=False)
        
    all_scores = []
    
    # Initialize fold structure
    for fold_idx in range(k):

        # Get the train and test indices for the current fold
        _, test_indices = list(kf.split(data_1))[fold_idx]
        print(test_indices)
        
        # Split tensor into train and test sets for this fold
        y1_test = data_1[test_indices,:,:].to(device)
        y2_test = data_2[test_indices,:,:].to(device)

        fold_scores = []
        for t_idx in range(y1_test.shape[-1]):
            r_values = pearson_corrcoef(y1_test[:,:,t_idx].to(device), y2_test[:,:,t_idx].to(device))
            fold_scores.append(r_values.cpu())
        all_scores.append(fold_scores)
    
    
    
    
    #all_scores = []

    # for fold, data_1 in enumerate(all_neural_data_1):

    #     fold_scores = []
    #     data_2 = all_neural_data_2[fold]
                
    #     for t_idx in tqdm(range(data_1.shape[-1])):
    #         r_values = pearson_corrcoef(data_1[:,:,t_idx], data_2[:,:,t_idx])
    #         fold_scores.append(r_values)
    #     all_scores.append(fold_scores)

        
    with open(os.path.join(CACHE,'noise_ceiling',f"noise_ceiling_subject={subject}_window={window_size}_stride={stride}"), "wb") as file:
        pickle.dump(all_scores, file)
        
        
    with open(os.path.join(CACHE,'noise_ceiling',f"noise_ceiling_times_subject={subject}_window={window_size}_stride={stride}"), "wb") as file:
        pickle.dump(times, file)
