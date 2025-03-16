import pickle
import torch
import pandas as pd
import os
import numpy as np
from src.utils import construct_model_id


def load_nc(nc_path, times_path, fold):
    # Load noise ceiling and time points for the subject
    with open(nc_path, "rb") as file:
        nc = pickle.load(file)

    if fold is not None:
        nc_df = pd.DataFrame(nc[fold])
    
    else:
        mean_scores = []
        for t in range(len(nc[0])):
            time_across_folds = []
            for fold_scores in nc:
                time_across_folds.append(fold_scores[t])
            mean_scores.append(sum(time_across_folds)/len(time_across_folds))
        nc_df = pd.DataFrame(mean_scores)

    with open(times_path, "rb") as file:
        time_points = pickle.load(file)    
    
    return nc_df, time_points


def plot_nc(corr_df, time_points, axs, row, colors):

    # Plot noise ceiling correlations
    for i, color in enumerate(colors):
        axs[row, 0].plot(time_points, corr_df[i], color=color)
    axs[row, 0].set_xlabel("Time")
    axs[row, 0].set_ylabel("Correlation")
    return
    

def plot_nc_loc(corr_df, time_points, axs, row, elec_locs, colors):
    # Plot noise ceiling correlations
    plotted_locs = set()
    for i in corr_df:
        if i in elec_locs.keys():
            loc = elec_locs[i]
            color = colors[loc]
            # Determine label for the legend
            if loc not in plotted_locs:
                label = loc
                plotted_locs.add(loc)
            else:
                label = None  # Avoid duplicate entries in the legend
            axs[row, 0].plot(time_points, corr_df[i], color=color, label=label)
    axs[row, 0].set_xlabel("Time")
    axs[row, 0].set_ylabel("Correlation")
    return


def load_model_scores(identifier_fn, scores_path, num_layers):
    # Initialize an empty DataFrame to store all layers' scores

    if num_layers == 'none':
        identifier = identifier_fn(layer='none')

        with open(os.path.join(scores_path, identifier), 'rb') as file:
            scores = pickle.load(file)
        scores_df = pd.DataFrame(scores)
        
        scores_df['layer'] = 'None'  # Add a column for the layer        # Ensure scores are numeric

    else:
        scores_df = pd.DataFrame()
        # Step 1: Load scores and create the full DataFrame
        for layer in range(num_layers):
            identifier = identifier_fn(layer=layer)
            with open(os.path.join(scores_path, identifier), 'rb') as file:
                scores = pickle.load(file)
            df_tmp = pd.DataFrame(scores)
            
            df_tmp['layer'] = layer  # Add a column for the layer
            scores_df = pd.concat([scores_df, df_tmp], ignore_index=True)
            
        # Ensure scores are numeric
        for column in scores_df.columns:
            if scores_df[column].dtype == 'object':  # Check if the column contains tensors
                scores_df[column] = scores_df[column].apply(lambda x: x.item() if torch.is_tensor(x) else x)

    return scores_df


def get_best_layer(scores_df):
    scores_df.iloc[:,:-1] = scores_df.iloc[:,:-1].applymap(lambda x: x.item() if torch.is_tensor(x) else x)

    # Step 2: Calculate median scores across time points for each electrode 
    median_scores = scores_df.groupby('layer').median()  # Compute median for each layer and electrode
    best_layers = median_scores.idxmax(axis=0)  # Get the layer index with max median for each electrode
    best_layer_dict = best_layers.to_dict() 
    return best_layer_dict


# def plot_model_scores(time_points, scores_df, best_layer_dict, axs, row, col, colors):

    # y_lim = axs[row, 0].get_ylim()
    
#     if best_layer_dict == None:
#         for electrode in range(len(scores_df.columns())-1):
#             filtered_scores = scores_df[electrode]
#             axs[row, col].plot(time_points, filtered_scores, color=colors[electrode])
#         axs[row, col].set_xlabel("Time")
#         axs[row, col].set_ylabel("Correlation")
#         axs[row, col].set_ylim(y_lim)  # Match y-limits to the noise ceiling plot
#         return

#     else:
#         # Plot filtered scores based on the best layer
#         for electrode, best_layer in best_layer_dict.items():
#             filtered_scores = scores_df[scores_df['layer'] == best_layer][electrode]
#             axs[row, col].plot(time_points, filtered_scores, color=colors[electrode])
#         axs[row, col].set_xlabel("Time")
#         axs[row, col].set_ylabel("Correlation")
#         axs[row, col].set_ylim(y_lim)  # Match y-limits to the noise ceiling plot
#         return


def plot_model_scores_loc(time_points, scores_df, best_layer_dict, axs, row, col, loc_dict, colors):
    y_lim = axs[row, 0].get_ylim()
    
    # Keep track of which locs have been plotted to avoid duplicate legend entries
    plotted_locs = set()
    
    # Plot filtered scores based on the best layer
    for electrode, loc in loc_dict.items():
        # print('test')
        color = colors[loc]

        if best_layer_dict == None:
            filtered_scores = scores_df[electrode]
        else:
            best_layer = best_layer_dict[electrode]
            filtered_scores = scores_df[scores_df['layer'] == best_layer][electrode]

        # Determine label for the legend
        if loc not in plotted_locs:
            label = loc
            plotted_locs.add(loc)
        else:
            label = None  # Avoid duplicate entries in the legend
        
        if len(time_points) != len(filtered_scores):
            time_points = np.linspace(-0.5, 1.5, 101)
        # Plot the data
        axs[row, col].plot(time_points, filtered_scores, color=color, label=label)

    # Set labels and limits
    axs[row, col].set_xlabel("Time")
    axs[row, col].set_ylabel("Correlation")
    axs[row, col].set_ylim(y_lim)  # Match y-limits to the noise ceiling plot
    
    # Add the legend
    axs[row, col].legend()
    return




def plot_model_scores_diff(time_points, scores_dfs, best_layer_dicts, axs, row, col, loc_dict, colors):
    y_lim = axs[row, 0].get_ylim()
    
    scores = []

    for i, scores_df in enumerate(scores_dfs):
        # Plot filtered scores based on the best layer
        final_df = pd.DataFrame()
        for electrode, loc in loc_dict.items():
            # print('test')
            if best_layer_dicts[i] == None:
                filtered_scores = scores_df[electrode]
            else:
                best_layer = best_layer_dicts[i][electrode]
                filtered_scores = scores_df[scores_df['layer'] == best_layer][electrode]            
            final_df[electrode] = filtered_scores

        scores.append(final_df)
            
    scores[0] = scores[0].reset_index(drop=True)
    scores[1] = scores[1].reset_index(drop=True)
    scores_final = scores[0].sub(scores[1])

    plotted_locs = set()
    for electrode, loc in loc_dict.items():
        # print('test')
        color = colors[loc]
        # Determine label for the legend
        if loc not in plotted_locs:
            label = loc
            plotted_locs.add(loc)
        else:
            label = None  # Avoid duplicate entries in the legend
    
        # Plot the data
        axs[row, col].plot(time_points, scores_final[electrode], color=color, label=label)
    
    # Set labels and limits
    axs[row, col].set_xlabel("Time")
    axs[row, col].set_ylabel("Correlation")
    axs[row, col].set_ylim(y_lim)  # Match y-limits to the noise ceiling plot
    
    # Add the legend
    axs[row, col].legend()
    
    return
