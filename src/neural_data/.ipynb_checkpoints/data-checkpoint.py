import mne
import os
import torch
import numpy as np
import pickle
import yaml
import warnings
warnings.filterwarnings("ignore")
import sys
ROOT = '/ccn2/u/atlask/FYP/'
sys.path.append(ROOT)
with open(f"{ROOT}/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
DATA = paths["DATA"]
CACHE = paths["CACHE"]


# def load_runs(data_path, subject, stim):
#     for f in os.listdir(data_path):
#         if (f'sub-{subject}' in f) & (stim in f) & ('run-01' in f):
#             path_1 = f
#         elif (f'sub-{subject}' in f) & (stim in f) & ('run-02' in f):
#             path_2 = f
#         else:
#             pass
#     return path_1, path_2


def load_file_paths(data_path, subject):
    paths = []
    for f in os.listdir(data_path):
        if (f'sub-{subject}' in f) & ('Jobs' in f) & ('run-01' in f):
            paths.append(f) 
        elif (f'sub-{subject}' in f) & ('Slow' in f):
            paths.append(f)
        else:
            pass
    return paths


def load_repeated_file_paths(data_path, subject):
    paths_1, paths_2 = [], []

    for f in os.listdir(data_path):
        if (f'sub-{subject}' in f) & ('Jobs' in f) & ('run-01' in f):
            paths_1.append(f) 
    for f in os.listdir(data_path):
        if (f'sub-{subject}' in f) & ('Jobs' in f) & ('run-02' in f):
            paths_2.append(f) 
    return paths_1, paths_2


def convert_df_to_string(df):
    text = ""
    df = df.apply(lambda x: 'none' if x == None else x)
    df = df.apply(lambda x: 'none' if x == 'None' else x)
    df = df.apply(lambda x: 'Its' if x == 'It’s' else x)

    for i in range(len(df)):
        try:
            text += df.iloc[i]
            text += ' '
        except TypeError:
            pass
    return text


def process_words(words_list):
    new_list = []
    for word in words_list:
        if word == None:
            new_list.append('none')
        elif 'It’s' in word:
            new_list.append('Its')
        else:
            new_list.append(word)
    return new_list


def load_neural_data(data_dir, file_path):
    
    print(file_path)
    full_path = os.path.join(data_dir, file_path)
    
    epochs = mne.read_epochs(full_path, preload=True, verbose=False)
    
    stimulus_df = epochs.metadata.Word
    neural_data = epochs.get_data()
    # word_onset_idx = np.argwhere(epochs.times == 0)
    # word_onset_response = neural_data[:, :, word_onset_idx].squeeze()
    
    return stimulus_df, neural_data

