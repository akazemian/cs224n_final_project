import os
import mne
import pandas as pd
import librosa
import numpy as np
from scipy.stats import zscore
from process_nwb.linenoise_notch import apply_linenoise_notch
from process_nwb.common_referencing import subtract_CAR
    
import torch
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
import pickle
import sys

ROOT = '/ccn2/u/atlask/FYP/'
sys.path.append(ROOT)
device = 'cuda'
base_path = '/ccn2/u/atlask/iEEG'

session_stimuli = {
    '01': ['Jobs1', 'Jobs2', 'Jobs3'],
}
runs = ['01', '02']
subjects  = ['02', '03', '05', '07', '10', '11']


for sub in subjects:
    
    for ses, stims in session_stimuli.items():
        for stim in stims:
            runs_list = []
            for run in runs:
                print(f"Processing {stim} in {ses}, {run}...")
                 
                edf_path = os.path.join(base_path, 'raw_data', f'sub-{sub}', f'ses-{ses}', 'ieeg', f'sub-{sub}_ses-{ses}_task-listen{stim}_run-{run}_ieeg.edf')
                onset_path = os.path.join(base_path, 'raw_data', f'sub-{sub}', f'ses-{ses}', 'ieeg', f'sub-{sub}_ses-{ses}_task-listen{stim}_run-{run}_events.tsv')
                stim_path = os.path.join(base_path,'stimuli','wav', f'{stim}.wav')

                raw = mne.io.read_raw_edf(edf_path, preload=True)
                
                # some are sampled in 10,000 Hz
                if raw.info['sfreq'] != 1000:
                    raw.resample(1000)

                # get stimulus onset time from events TSV
                df = pd.read_csv(onset_path, sep='\t')
                onset_time = df['onset'].tolist()
                stim_onset = onset_time[0] * 1000  
                stim_onset_sec = stim_onset / 1000

                # tmin as stim_onset_sec unless iEEG recording starts after stimulus
                if stim_onset_sec < 0:
                    tmin = 0
                else:
                    tmin = stim_onset_sec

                # Get duration of WAV file in ms
                y, sr = librosa.load(stim_path, sr=None)
                wav_duration = librosa.get_duration(y=y, sr=sr)

                # Get the last timepoint in the raw data
                raw_duration_sec = raw.times[-1]

                # Clip the raw data to match the duration of the WAV file
                if tmin + wav_duration > raw_duration_sec:
                    clipped_raw = raw.copy().crop(tmin=tmin, tmax=raw_duration_sec)
                else:
                    tmax = np.min([raw.times[-1], tmin + wav_duration + 2]) #WT: adding 2 second buffer, so extending epoch doesn't lead to loss of last few phonemes. 
                    clipped_raw = raw.copy().crop(tmin=tmin, tmax=tmax)
                
                print(np.shape(clipped_raw._data))

                # Apply notch filter
                nth_data = apply_linenoise_notch(clipped_raw._data[5:, :].T, clipped_raw.info['sfreq']).T
                # nth_data = clipped_raw._data[5:, :] # switch comment above to see data with line noise

                # Check if stimulus onset is negative
                if stim_onset_sec < 0:
                    
                    # Calculate number of samples to pad
                    num_samples_to_pad = int(abs(stim_onset_sec) * raw.info['sfreq'])

                    # Create zero padding
                    padding = np.zeros((len(nth_data), num_samples_to_pad))

                    # Apply zero padding to preprocessed data
                    nth_data = np.hstack((padding, nth_data))

                # Get the correct index of channels in the filtered data
                num_channels = nth_data.shape[0]
                max_channel_index = min(len(clipped_raw.info['ch_names']), 5 + num_channels)
                new_channel_indices = range(5, max_channel_index)
                updated_info = mne.pick_info(clipped_raw.info, sel=new_channel_indices)
                print(np.shape(nth_data))

                # Create MNE RawArray object with notch filtered data and updated index
                nth_raw = mne.io.RawArray(nth_data, updated_info)
                
                                # Apply CAR and Z-score
                nth_CAR = subtract_CAR(nth_raw.get_data())
                raw_CAR = mne.io.RawArray(nth_CAR, nth_raw.info)
                # print(raw_CAR.info)

                # raw_CAR.filter(1, 15)
                CAR_z = zscore(raw_CAR._data, axis=1)
                # CAR_z = raw_CAR._data

                # put z scored and filtered data into the raw mne object
                raw_CAR._data = np.array(CAR_z, dtype='float64')  # needs to be float64 to save epochs
                runs_list.append(raw_CAR)

    
    
            data_1, data_2 = runs_list[0].get_data(), runs_list[1].get_data()
            if runs_list[0].get_data().shape[-1] != runs_list[1].get_data().shape[-1]:
                max_time = min(runs_list[0].get_data().shape[-1], runs_list[1].get_data().shape[-1])
                data_1, data_2 = data_1[:,:max_time], data_2[:,:max_time]
                print(data_1.shape, data_2.shape)
                
                
            data_1, data_2 = torch.Tensor(data_1).to(device), torch.Tensor(data_2).to(device)
            results = torch.stack([pearson_corrcoef(data_1[:, e], data_2[:, e]) for e in range(data_1.shape[0])], dim=0)
            with open(os.path.join(ROOT,'results','r_scores',f"noise_ceiling_raw_subject={sub}_stimuli={stim}"), "wb") as file:
                pickle.dump(results.cpu(), file)