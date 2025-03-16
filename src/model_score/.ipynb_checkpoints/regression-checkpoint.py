import torch
from tqdm import tqdm
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
from sklearn.model_selection import KFold

import torch
import numpy as np
from scipy.stats import mode

import pickle


def compute_mode_on_gpu(tensor):
    unique_values, counts = torch.unique(tensor, return_counts=True)
    max_count_idx = torch.argmax(counts)  # Index of the most frequent value
    return unique_values[max_count_idx]


def pearson_corrcoef_(x, y, dim=0):
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    covariance = (x_centered * y_centered).sum(dim=dim)
    x_std = torch.sqrt((x_centered**2).sum(dim=dim))
    y_std = torch.sqrt((y_centered**2).sum(dim=dim))
    
    return covariance / (x_std * y_std)

    
def ridge_regression(X, Y, device, alphas=[0] + [10**i for i in range(-10,10)]):
    # Ensure X and Y are float32
    X = X.float().to(device)
    Y = Y.float().to(device)

    # Get dimensions
    n_electrodes = Y.size(1)
    n_timepoints = Y.size(2)

    # Initialize variables to store the best values
    best_alphas = torch.zeros(n_electrodes, n_timepoints, device=device, dtype=torch.float32)
    best_scores = torch.full((n_electrodes, n_timepoints), float('-inf'), device=device)

    XtX = X.T @ X
    XtY = X.T @ Y.reshape(Y.size(0), -1)
    
    # Iterate over each alpha to find the best one for each electrode and time point
    for alpha in alphas:
        # Create the regularization term
        I = torch.eye(X.size(1), device=device, dtype=torch.float32) * alpha
        # Compute weights for all electrodes and time points with the current alpha
        W = torch.linalg.solve(XtX + I, XtY)
        #W = torch.inverse(XtX + I) @ XtY  # Shape: (n_features, n_electrodes * n_timepoints)
        W = W.reshape(X.size(1), n_electrodes, n_timepoints)  # Reshape back

        # Calculate predictions for all electrodes and time points
        y_pred = (X @ W.reshape(X.size(1), -1)).reshape(Y.size(0), n_electrodes, n_timepoints)  # Shape: (n_samples, n_electrodes, n_timepoints)

        # Calculate scores for all electrodes and time points
        # scores = torch.zeros(n_electrodes, n_timepoints, device=device, dtype=torch.float32)
        # for t_idx in range(n_timepoints):
        #     r_values = pearson_corrcoef(Y[:,:,t_idx], y_pred[:,:,t_idx])
        #     scores[:,t_idx] = r_values
        scores = pearson_corrcoef_(Y, y_pred, dim=0)
        # Update best alphas and scores where the current alpha improves performance
        improved = scores > best_scores
        best_alphas[improved] = alpha
        best_scores[improved] = scores[improved]

    # Select the mode of the best alphas for each electrode across all time points
    # best_alphas_per_electrode = torch.tensor(
    #     #[mode(best_alphas[i].cpu().numpy())[0] for i in range(n_electrodes)],
    #     [compute_mode_on_gpu(best_alphas[i]).item() for i in range(n_electrodes)],
    #     device=device,
    #     dtype=torch.float32,
    # )
    best_alphas_per_electrode = torch.mode(best_alphas, dim=1)[0]
    print('best_alphas_per_electrode', best_alphas_per_electrode.shape)
    return best_alphas_per_electrode, best_scores


# def ridge_regression_cv(X, y, device):
#     folds = len(X)

#     # Preallocate tensors for predictions
#     y_preds = []
#     y_true = []

#     for i in range(folds):
#         X_test, y_test = X[i].to(device), y[i].to(device)
#         X_train = torch.cat([X[j] for j in range(folds) if j != i], dim=0).to(device)
#         y_train = torch.cat([y[j] for j in range(folds) if j != i], dim=0).to(device)

#         # Perform Ridge Regression for all electrodes
#         best_alphas, best_scores = ridge_regression(X_train, y_train, device)

#         n_samples, n_electrodes, n_times = y_test.shape
#         preds_ = torch.zeros_like(y_test)

#         # Precompute XtX for all electrodes
#         XtX = X_train.T @ X_train

#         for electrode in range(n_electrodes):
#             alpha = best_alphas[electrode]
#             I = torch.eye(X_train.size(1), device=device) * alpha
#             XtX_reg = XtX + I

#             W = torch.linalg.solve(XtX_reg, X_train.T @ y_train[:, electrode, :])
#             preds_[:, electrode, :] = X_test @ W

#         y_preds.append(preds_)
#         y_true.append(y_test)

#     return torch.cat(y_true), torch.cat(y_preds)


    
def ridge_regression_cv(X, y, device):
            
    # Assuming X_train and y_train have been defined as (n_samples, n_features) and (n_samples, n_electrodes) respectively
    y_preds = []
    y_true = []
    
    folds = len(X)
    for i in tqdm(range(folds)):
        
        print# Define the test and training sets for the current fold
        X_test = X[i].to(device)
        y_test = y[i].to(device)
    
        X_train = torch.cat([X[j].to(device) for j in range(folds) if j != i], axis=0)
        y_train = torch.cat([y[j].to(device) for j in range(folds) if j != i], axis=0)
        
        # Perform Ridge Regression for all electrodes at once
        best_alphas, best_scores = ridge_regression(X_train, y_train, device)  # Shape: (n_features, n_electrodes)
        # Make predictions on the test set for all electrodes simultaneously
        
        n_samples, n_electrodes, n_times = y_test.shape
        n_features = y_train.size(1)
        
        # Preallocate tensors for predictions
        preds_ = torch.zeros(X_test.size(0), n_electrodes, n_times, device=X_test.device, dtype=torch.float32)
        
        # Iterate over electrodes (outer loop only)
        for electrode in range(n_electrodes):
            alpha = best_alphas[electrode]  # Best alpha for this electrode
            # print('electrode:', electrode, 'alpha:', alpha)
            I = torch.eye(X_train.size(1), device=X_train.device, dtype=torch.float32) * alpha  # Regularization term
        
            # Compute the regularized inverse once per electrode
            XtX_reg_inv = torch.inverse(X_train.T @ X_train + I)  # Shape: [features, features]
        
            # Compute weights for all time points in one operation
            W = XtX_reg_inv @ X_train.T @ y_train[:, electrode, :]  # Shape: [features, n_times]
        
            # Compute predictions for X_test for all time points simultaneously
            preds_[:, electrode, :] = X_test @ W  # Shape: [samples, n_times]
        
        # Append results to lists
        y_preds.append(preds_)
        y_true.append(y_test)

    return torch.cat(y_true), torch.cat(y_preds)



def create_folds(X, y, k=5):
    """
    Performs k-fold cross-validation on a list of tensors of varying lengths.

    Args:
    - X: List of torch.Tensor objects for features.
    - y: List of torch.Tensor objects for targets.
    - k: Number of folds.

    Returns:
    - List of dictionaries, each containing train and test sets for each fold.
    """
    # Store results for each fold
    folds = []
    
    # Initialize fold structure
    for fold_idx in range(k):
        fold = {'X_train': [], 'X_test': [], 'y_train': [], 'y_test': []}
        
        # Perform k-fold CV on each tensor individually
        for i in range(len(X)):
            tensor_X = X[i]
            tensor_y = y[i]
            
            # Create k-fold splitter for the current tensor
            kf = KFold(n_splits=k, shuffle=False)
            
            # Get the train and test indices for the current fold
            train_indices, test_indices = list(kf.split(tensor_X))[fold_idx]
            
            # Split tensor into train and test sets for this fold
            X_train, X_test = tensor_X[train_indices,:], tensor_X[test_indices,:]
            y_train, y_test = tensor_y[train_indices,:], tensor_y[test_indices,:]
            
            # Append the split data for this tensor to the fold structure
            fold['X_train'].append(X_train)
            fold['X_test'].append(X_test)
            fold['y_train'].append(y_train)
            fold['y_test'].append(y_test)
        
        # Concatenate train and test sets across all tensors for this fold
        fold['X_train'] = torch.cat(fold['X_train'], dim=0)
        fold['X_test'] = torch.cat(fold['X_test'], dim=0)
        fold['y_train'] = torch.cat(fold['y_train'], dim=0)
        fold['y_test'] = torch.cat(fold['y_test'], dim=0)
        
        folds.append(fold)
    
    return folds


def create_folds_new(X, y_1, y_2, k=5):
    # Store results for each fold
    
    folds = {'X_train': [], 'X_test': [], 'y1_train': [], 'y1_test': [], 'y2_train': [], 'y2_test': []}
        
    tensor_X = torch.cat(X)
    tensor_y1 = torch.cat(y_1)
    tensor_y2 = torch.cat(y_2)

    # Initialize fold structure
    for fold_idx in range(k):

        # Create k-fold splitter for the current tensor
        kf = KFold(n_splits=k, shuffle=False)
        
        # Get the train and test indices for the current fold
        train_indices, test_indices = list(kf.split(tensor_X))[fold_idx]
        
        # Split tensor into train and test sets for this fold
        X_train, X_test = tensor_X[train_indices,:], tensor_X[test_indices,:]
        y1_train, y1_test = tensor_y1[train_indices,:], tensor_y1[test_indices,:]
        y2_train, y2_test = tensor_y2[train_indices,:], tensor_y2[test_indices,:]
        
        # Append the split data for this tensor to the fold structure
        folds['X_train'].append(X_train)
        folds['X_test'].append(X_test)
        folds['y1_train'].append(y1_train)
        folds['y1_test'].append(y1_test)
        folds['y2_train'].append(y2_train)
        folds['y2_test'].append(y2_test)    
            
    return folds

    
def ridge_regression_cv_new(X, y_1, y_2, k, device):
            
    # Assuming X_train and y_train have been defined as (n_samples, n_features) and (n_samples, n_electrodes) respectively
    y_preds = []
    y_true = []
    
    folds = create_folds_new(X, y_1, y_2, k)
    
    for i in tqdm(range(k)):
        
        X_test = folds['X_test'][i].to(device)
        y_test = folds['y1_test'][i].to(device)
        X_train = torch.cat([folds['X_train'][i].to(device) for j in range(k) if j != i], axis=0)
        y_train = torch.cat([folds['y2_train'][i].to(device) for j in range(k) if j != i], axis=0)
        
        preds_ = get_preds(X_train, y_train, X_test, y_test, device)
        y_preds.append(preds_)
        y_true.append(y_test)

        
    for i in tqdm(range(k)):
        
        X_test = folds['X_test'][i].to(device)
        y_test = folds['y2_test'][i].to(device)
        X_train = torch.cat([folds['X_train'][i].to(device) for j in range(k) if j != i], axis=0)
        y_train = torch.cat([folds['y1_train'][i].to(device) for j in range(k) if j != i], axis=0)
        
        preds_ = get_preds(X_train, y_train, X_test, y_test, device)
        y_preds.append(preds_)
        y_true.append(y_test)    
    
    return torch.cat(y_true), torch.cat(y_preds)


def get_preds(X_train, y_train, X_test, y_test, device ):
    # Perform Ridge Regression for all electrodes at once
    best_alphas, best_scores = ridge_regression(X_train, y_train, device)  # Shape: (n_features, n_electrodes)
    # Make predictions on the test set for all electrodes simultaneously
    
    n_samples, n_electrodes, n_times = y_test.shape
    n_features = y_train.size(1)
    
    # Preallocate tensors for predictions
    preds_ = torch.zeros(X_test.size(0), n_electrodes, n_times, device=X_test.device, dtype=torch.float32)
    
    # Iterate over electrodes (outer loop only)
    for electrode in range(n_electrodes):
        alpha = best_alphas[electrode]  # Best alpha for this electrode
        # print('electrode:', electrode, 'alpha:', alpha)
        I = torch.eye(X_train.size(1), device=X_train.device, dtype=torch.float32) * alpha  # Regularization term
    
        # Compute the regularized inverse once per electrode
        XtX_reg_inv = torch.inverse(X_train.T @ X_train + I)  # Shape: [features, features]
    
        # Compute weights for all time points in one operation
        W = XtX_reg_inv @ X_train.T @ y_train[:, electrode, :]  # Shape: [features, n_times]
    
        # Compute predictions for X_test for all time points simultaneously
        preds_[:, electrode, :] = X_test @ W  # Shape: [samples, n_times]
    
    return preds_
    
# def get_fold_preds(fold, device):
    
#     print# Define the test and training sets for the current fold
#     X_test = fold['X_test'].to(device)
#     y_test = fold['y_test'].to(device)

#     X_train = fold['X_train'].to(device)#torch.cat([X[j] for j in range(folds) if j != i], axis=0).to(device)
#     y_train = fold['y_train'].to(device)#torch.cat([y[j] for j in range(folds) if j != i], axis=0).to(device)
    
#     # Perform Ridge Regression for all electrodes at once
#     best_alphas, best_scores = ridge_regression(X_train, y_train, device)  # Shape: (n_features, n_electrodes)
#     # Make predictions on the test set for all electrodes simultaneously
    
#     n_samples, n_electrodes, n_times = y_test.shape
#     n_features = y_train.size(1)
    
#     # Preallocate tensors for predictions
#     preds_ = torch.zeros(X_test.size(0), n_electrodes, n_times, device=X_test.device, dtype=torch.float32)
    
#     # Iterate over electrodes (outer loop only)
#     for electrode in range(n_electrodes):
#         alpha = best_alphas[electrode]  # Best alpha for this electrode
#         print(X_train.shape)
#         I = torch.eye(X_train.size(1), device=X_train.device, dtype=torch.float32) * alpha  # Regularization term
    
#         # Compute the regularized inverse once per electrode
#         XtX_reg_inv = torch.inverse(X_train.T @ X_train + I)  # Shape: [features, features]
    
#         # Compute weights for all time points in one operation
#         W = XtX_reg_inv @ X_train.T @ y_train[:, electrode, :]  # Shape: [features, n_times]
    
#         # Compute predictions for X_test for all time points simultaneously
#         preds_[:, electrode, :] = X_test @ W  # Shape: [samples, n_times]
    
#     return preds_, y_test


# def ridge_regression(X, Y, alphas=[10**i for i in range(-10, 10)]):
#     # Ensure X and Y are float32
#     X = X.float()
#     Y = Y.float()

#     # Initialize variables to store the best values for each electrode
#     n_electrodes = Y.size(1)
#     best_weights = torch.zeros(X.size(1), n_electrodes, device=X.device, dtype=torch.float32)
#     best_alphas = torch.zeros(n_electrodes, device=X.device, dtype=torch.float32)
#     best_scores = torch.full((n_electrodes,), float('-inf'), device=X.device)

#     # Iterate over each alpha to find the best one for each electrode
#     for alpha in alphas:
#         # Create the regularization term
#         I = torch.eye(X.size(1), device=X.device, dtype=torch.float32) * alpha
#         # Compute weights for all electrodes with the current alpha
#         W = torch.inverse(X.T @ X + I) @ X.T @ Y  # Shape: (n_features, n_electrodes)

#         # Calculate predictions for all electrodes with current weights
#         y_pred = X @ W  # Shape: (n_samples, n_electrodes)

#         # Calculate scores for all electrodes at once
#         scores = pearson_corrcoef(Y, y_pred)  # Assuming score_fn outputs a score for each electrode

#         # Identify electrodes where the new score is better
#         improved = scores > best_scores

#         # Update best weights, alphas, and scores only where there was an improvement
#         best_weights[:, improved] = W[:, improved]
#         best_alphas[improved] = alpha
#         best_scores[improved] = scores[improved]

#     return best_weights, best_alphas, best_scores

# def ridge_regression_cv(X, y, folds, device="cuda"):
        
#     folds = create_folds(X, y, k=5)
    
#     # Assuming X_train and y_train have been defined as (n_samples, n_features) and (n_samples, n_electrodes) respectively
#     y_preds = []
#     y_true = []
    
#     for i, fold in tqdm(enumerate(folds)):
        
#         # Define the test and training sets for the current fold
#         X_test = fold['X_test'].to(device)
#         y_test = fold['y_test'].to(device)
    
#         X_train = fold['X_train'].to(device)#torch.cat([X[j] for j in range(folds) if j != i], axis=0).to(device)
#         y_train = fold['y_train'].to(device)#torch.cat([y[j] for j in range(folds) if j != i], axis=0).to(device)
    
        
#         # Perform Ridge Regression for all electrodes at once
#         best_weights, best_alphas, best_scores = ridge_regression(X_train, y_train)  # Shape: (n_features, n_electrodes)
#         # Make predictions on the test set for all electrodes simultaneously
#         preds_ = X_test @ best_weights  # Shape: (n_samples, n_electrodes)
#         y_preds.append(preds_)  # Move back to CPU for evaluation
#         y_true.append(y_test)   # Move back to CPU for consistency
    
#         #del best_weights, best_alphas, best_scores, preds_, X_test, y_test, X_train, y_train  # Free up memory on the GPU
#     y_preds_all = torch.cat(y_preds)
#     y_true_all = torch.cat(y_true)

#     return y_true_all, y_preds_all

# def ridge_regression_cv_nc(X, y, folds, device="cuda"):
        
#     # Assuming X_train and y_train have been defined as (n_samples, n_features) and (n_samples, n_electrodes) respectively
#     y_preds = []
#     y_true = []
    
#     # Loop over each fold for cross-validation
#     for i in tqdm(range(folds)):
#         # Define the test and training sets for the current fold
#         X_test = X[i].to(device)
#         y_test = y[i].to(device)
    
#         X_train = torch.cat([X[j] for j in range(folds) if j != i], axis=0).to(device)
#         y_train = torch.cat([y[j] for j in range(folds) if j != i], axis=0).to(device)
    
#         # Perform Ridge Regression for all electrodes at once
#         best_weights, best_alphas, best_scores = ridge_regression(X_train, y_train)  # Shape: (n_features, n_electrodes)
#         # Make predictions on the test set for all electrodes simultaneously
#         preds_ = X_test @ best_weights  # Shape: (n_samples, n_electrodes)
#         y_preds.append(preds_)  # Move back to CPU for evaluation
#         y_true.append(y_test)   # Move back to CPU for consistency
    
#         #del best_weights, best_alphas, best_scores, preds_, X_test, y_test, X_train, y_train  # Free up memory on the GPU
#     y_preds = torch.cat(y_preds)
#     y_true = torch.cat(y_true)

#     return y_true, y_preds

