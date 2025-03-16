import os
os.environ["OMP_NUM_THREADS"] = "1"        # For OpenMP
os.environ["MKL_NUM_THREADS"] = "1"          # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"     # For OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"      # For NumExpr

from tqdm import tqdm
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Tuple, Union, List
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from typing import Tuple

from scipy.stats import pearsonr
def pearson_corr(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return pearsonr(y_true, y_pred)[0] 


def plot_alpha_dist(alphas: np.ndarray, file_path: str) -> None:
    """
    Plots the kernel density estimate (KDE) of the alpha distributions per electrode 
    and saves the plot to the specified file path.
    """
    n_electrodes = alphas.shape[1]
    colors = sns.color_palette("husl", n_electrodes)
    plt.figure(figsize=(10, 6))
    for i in range(n_electrodes):
        sns.kdeplot(alphas[:, i], color=colors[i], label=f'Electrode {i+1}', linewidth=2)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Alpha Distributions per Electrode")
    plt.legend()
    plt.savefig(file_path)


# class RidgeRegression:
#     def __init__(self, scoring: str = 'pearson', alphas: List[float] = [10**i for i in range(-10, 10)]) -> None:
#         self.scoring = scoring
#         self.alphas = alphas

#     def get_scorer(self) -> Callable[[np.ndarray, np.ndarray], float]:
#         """Returns a scorer function based on the selected scoring metric."""
#         if self.scoring == 'pearson':
#             return make_scorer(lambda y, y_pred: pearson_corr(y, y_pred), greater_is_better=True)
#         raise ValueError(f"Scoring method '{self.scoring}' not supported.")
    
#     def select_best_alphas(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Select the best alpha for each electrode using RidgeCV on each time point. 
#         For each time point, pass the full multi-target y to RidgeCV, and aggregate 
#         across time by computing the mode for each electrode.
        
#         Parameters
#         ----------
#         X : ndarray, shape (n_samples, n_features, n_time) or (n_samples, n_features)
#             Feature data.
#         y : ndarray, shape (n_samples, n_electrodes, n_time)
#             Target data.
#         alphas : list of float
#             Candidate regularization parameters.
#         cv : int, default=5
#             Number of folds for CV.
            
#         Returns
#         -------
#         final_alphas : ndarray, shape (n_electrodes,)
#             One selected alpha per electrode (the mode across time).
#         alphas_time : ndarray, shape (n_time, n_electrodes)
#             The best alpha from CV for each time point and electrode.
#         """
#         print("finding optimal penalty terms...")
#         _, n_electrodes, n_time = y.shape

#         if X.ndim == 2:
#             # Container for best alphas for each time point (each row: time, each col: electrode)
#             alphas_time = np.zeros((n_time, n_electrodes), dtype=np.float32)
#             scores =  np.zeros((n_time, n_electrodes), dtype=np.float32)
            
#             # For each time point, fit RidgeCV with all electrodes at once.
#             for t in tqdm(range(n_time)):
#                 y_t = y[:, :, t]           # shape (n_samples, n_electrodes)
#                 ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True, 
#                                    scoring=self.get_scorer())
#                 ridge_cv.fit(X, y_t)
#                 # ridge_cv.alpha_ is an array of shape (n_electrodes,)
#                 alphas_time[t, :] = ridge_cv.alpha_
#                 scores[t,:] = ridge_cv.best_score_

#             final_alphas = np.zeros(n_electrodes, dtype=np.float32)
#             for e in range(n_electrodes):
#                 # mode returns an object; mode[0] gives the most common value.
#                 final_alphas[e] = np.atleast_1d(mode(alphas_time[:, e]).mode)[0] 
#             print('alpha values:',final_alphas)        


#         elif X.ndim== 3:
#             # Container for best alphas for each time point (each row: time, each col: electrode)
#             alphas_time = np.zeros((n_time, n_electrodes), dtype=np.float32)
            
#             # For each time point, fit RidgeCV with all electrodes at once.
#             for t in tqdm(range(n_time)):
#                 X_t = X[:, :, t]           # shape (n_samples, n_features)
#                 y_t = y[:, :, t]           # shape (n_samples, n_electrodes)
#                 ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True, 
#                                    scoring=self.get_scorer())
#                 ridge_cv.fit(X_t, y_t)
#                 # ridge_cv.alpha_ is an array of shape (n_electrodes,)
#                 alphas_time[t, :] = ridge_cv.alpha_
            
#             # Compute the mode across time for each electrode.
#             final_alphas = np.zeros(n_electrodes, dtype=np.float32)
#             for e in range(n_electrodes):
#                 # mode returns an object; mode[0] gives the most common value.
#                 final_alphas[e] = np.atleast_1d(mode(alphas_time[:, e]).mode)[0]
#             print('alpha values:',final_alphas)       

#         else:
#             raise ValueError("X must be a 2D or 3D array.")
#         return final_alphas, alphas_time


#     def cv(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Runs 5-fold cross validation (without shuffling) on the dataset.
#         The model is run 5 times (one per fold) and the predictions for each test fold
#         are concatenated into the final output.
        
#         Parameters:
#         X : np.ndarray
#             The training features. Can be 2D or 3D.
#         y : np.ndarray
#             The target values. Expected shape is (n_samples, n_electrodes, n_time).
#         alphas : list
#             List of regularization strengths to consider.
#         device : str
#             Device (not used in this code snippet, but kept for consistency).
#         random_state : int
#             For reproducibility if shuffling is ever enabled.
            
#         Returns:
#         y_pred_full : np.ndarray
#             The predictions for every sample, with shape (n_samples, n_electrodes, n_time).
#         y_true_full : np.ndarray
#             The ground truth values, reassembled from the CV splits.
#         """
        
#         kf = KFold(n_splits=5, shuffle=False)
#         n_samples = X.shape[0]
        
#         n_electrodes = y.shape[1]
#         # For the time dimension, if X is 2D, we take time from y.
#         if X.ndim == 2:
#             n_time = y.shape[2]
#         elif X.ndim == 3:
#             n_time = X.shape[2]
        
#         # Prepare arrays to hold full predictions and ground truth.
#         y_pred_full = np.zeros((n_samples, n_electrodes, n_time), dtype=np.float32)
#         y_true_full = np.zeros((n_samples, n_electrodes, n_time), dtype=y.dtype)
        
#         # Loop over the 5 folds.
#         for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
#             print(f"Fold {fold+1}: Training on {len(train_idx)} samples, testing on {len(test_idx)} samples.")
#             X_train_fold = X[train_idx]
#             y_train_fold = y[train_idx]
#             X_test_fold  = X[test_idx]
#             y_test_fold  = y[test_idx]
            
#             # Step 1: Determine best alphas per electrode on the training fold.
#             best_alphas, alphas_time = self.select_best_alphas(X_train_fold, y_train_fold)
#             print("Applying best alphas...")
            
#             # Initialize predictions for this fold.
#             fold_pred = np.zeros((len(test_idx), n_electrodes, n_time), dtype=np.float32)
            
#             if X.ndim == 2:
#                 # X is 2D
#                 for t in tqdm(range(n_time), desc=f"Fold {fold+1} time slices"):
#                     X_train_t = X_train_fold              # shape: (n_samples_train, n_features)
#                     X_test_t  = X_test_fold               # shape: (n_samples_test, n_features)
#                     y_train_t = y_train_fold[:, :, t]      # shape: (n_samples_train, n_electrodes)
#                     ridge = Ridge(alpha=best_alphas, fit_intercept=True)
#                     ridge.fit(X_train_t, y_train_t)
#                     fold_pred[:, :, t] = ridge.predict(X_test_t)
#             elif X.ndim == 3:
#                 # X is 3D
#                 for t in tqdm(range(n_time), desc=f"Fold {fold+1} time slices"):
#                     X_train_t = X_train_fold[:, :, t]       # shape: (n_samples_train, n_features)
#                     X_test_t  = X_test_fold[:, :, t]        # shape: (n_samples_test, n_features)
#                     y_train_t = y_train_fold[:, :, t]         # shape: (n_samples_train, n_electrodes)
#                     ridge = Ridge(alpha=best_alphas, fit_intercept=True)
#                     ridge.fit(X_train_t, y_train_t)
#                     fold_pred[:, :, t] = ridge.predict(X_test_t)
            
#             # Save fold predictions in the full prediction array.
#             y_pred_full[test_idx, :, :] = fold_pred
#             y_true_full[test_idx, :, :] = y_test_fold
            
#         return y_pred_full, y_true_full, alphas_time




class RidgeRegression:
    def __init__(self, scoring: str = 'pearson', alphas: list = [10**i for i in range(-10, 10)]) -> None:
        self.scoring = scoring
        self.alphas = alphas

    def get_scorer(self) -> callable:
        """Returns a scorer function based on the selected scoring metric."""
        if self.scoring == 'pearson':
            return make_scorer(lambda y, y_pred: pearson_corr(y, y_pred), greater_is_better=True)
        raise ValueError(f"Scoring method '{self.scoring}' not supported.")

    def select_best_alphas(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the best alpha for each electrode using RidgeCV on each time point.
        For each time point, pass the full multi-target y to RidgeCV, and aggregate 
        across time by computing the mode for each electrode.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features, n_time) or (n_samples, n_features)
            Feature data.
        y : np.ndarray, shape (n_samples, n_electrodes, n_time)
            Target data.
            
        Returns
        -------
        final_alphas : np.ndarray, shape (n_electrodes,)
            One selected alpha per electrode (the mode across time).
        alphas_time : np.ndarray, shape (n_time, n_electrodes)
            The best alpha from CV for each time point and electrode.
        """
        print("Finding optimal penalty terms...")
        _, n_electrodes, n_time = y.shape

        if X.ndim == 2:
            alphas_time = np.zeros((n_time, n_electrodes), dtype=np.float32)
            scores = np.zeros((n_time, n_electrodes), dtype=np.float32)
            
            # Define helper for a single time slice.
            def process_time_slice_2d(t: int) -> Tuple[np.ndarray, np.ndarray]:
                y_t = y[:, :, t]  # shape: (n_samples, n_electrodes)
                ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True, 
                                scoring=self.get_scorer())
                ridge_cv.fit(X, y_t)
                return ridge_cv.alpha_, ridge_cv.best_score_
            
            # Parallelize over time slices.
            results = Parallel(n_jobs=-1, verbose=5)(
                delayed(process_time_slice_2d)(t) for t in range(n_time)
            )
            for t, (alpha_vals, score_vals) in enumerate(results):
                alphas_time[t, :] = alpha_vals
                scores[t, :] = score_vals
            
            final_alphas = np.zeros(n_electrodes, dtype=np.float32)
            for e in range(n_electrodes):
                final_alphas[e] = np.atleast_1d(mode(alphas_time[:, e]).mode)[0]
            print('Alpha values:', final_alphas)
            
        elif X.ndim == 3:
            alphas_time = np.zeros((n_time, n_electrodes), dtype=np.float32)
            
            # Define helper for a single time slice in the 3D case.
            def process_time_slice_3d(t: int) -> np.ndarray:
                X_t = X[:, :, t]  # shape: (n_samples, n_features)
                y_t = y[:, :, t]  # shape: (n_samples, n_electrodes)
                ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True, 
                                scoring=self.get_scorer())
                ridge_cv.fit(X_t, y_t)
                return ridge_cv.alpha_
            
            results = Parallel(n_jobs=-1, verbose=5)(
                delayed(process_time_slice_3d)(t) for t in range(n_time)
            )
            for t, alpha_vals in enumerate(results):
                alphas_time[t, :] = alpha_vals
            
            final_alphas = np.zeros(n_electrodes, dtype=np.float32)
            for e in range(n_electrodes):
                final_alphas[e] = np.atleast_1d(mode(alphas_time[:, e]).mode)[0]
            print('Alpha values:', final_alphas)
            
        else:
            raise ValueError("X must be a 2D or 3D array.")
        
        return final_alphas, alphas_time


    def cv(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs 5-fold cross validation (without shuffling) on the dataset.
        The model is run 5 times (one per fold) and the predictions for each test fold
        are concatenated into the final output.
        """
        kf = KFold(n_splits=5, shuffle=False)
        n_samples = X.shape[0]
        n_electrodes = y.shape[1]
        if X.ndim == 2:
            n_time = y.shape[2]
        elif X.ndim == 3:
            n_time = X.shape[2]
        else:
            raise ValueError("X must be either 2D or 3D.")

        y_pred_full = np.zeros((n_samples, n_electrodes, n_time), dtype=np.float32)
        y_true_full = np.zeros((n_samples, n_electrodes, n_time), dtype=y.dtype)

        # Helper function to process one time slice.
        def process_time_slice(t: int, 
                               X_train_fold: np.ndarray, 
                               X_test_fold: np.ndarray, 
                               y_train_fold: np.ndarray, 
                               best_alphas: np.ndarray, 
                               ndim: int) -> np.ndarray:
            if ndim == 2:
                X_train_t = X_train_fold  # shape: (n_samples_train, n_features)
                X_test_t = X_test_fold    # shape: (n_samples_test, n_features)
                y_train_t = y_train_fold[:, :, t]  # shape: (n_samples_train, n_electrodes)
            else:  # ndim == 3
                X_train_t = X_train_fold[:, :, t]  # shape: (n_samples_train, n_features)
                X_test_t = X_test_fold[:, :, t]    # shape: (n_samples_test, n_features)
                y_train_t = y_train_fold[:, :, t]    # shape: (n_samples_train, n_electrodes)
            ridge = Ridge(alpha=best_alphas, fit_intercept=True)
            ridge.fit(X_train_t, y_train_t)
            return ridge.predict(X_test_t)

        # Loop over the 5 folds.
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold+1}: Training on {len(train_idx)} samples, testing on {len(test_idx)} samples.")
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]

            best_alphas, alphas_time = self.select_best_alphas(X_train_fold, y_train_fold)
            print("Applying best alphas...")

            # Parallelize the inner loop over time points using joblib.
            fold_pred = np.zeros((len(test_idx), n_electrodes, n_time), dtype=np.float32)
            results = Parallel(n_jobs=-1, verbose=5)(
                delayed(process_time_slice)(t, X_train_fold, X_test_fold, y_train_fold, best_alphas, X.ndim)
                for t in range(n_time)
            )
            # Each result corresponds to predictions for one time slice.
            for t, pred in enumerate(results):
                fold_pred[:, :, t] = pred

            y_pred_full[test_idx, :, :] = fold_pred
            y_true_full[test_idx, :, :] = y_test_fold

        return y_pred_full, y_true_full, alphas_time




# class RidgeRegression:
#     def __init__(self, scoring: str = 'pearson', alphas: list = [10**i for i in range(-10, 10)]) -> None:
#         self.scoring = scoring
#         self.alphas = alphas

#     def get_scorer(self) -> callable:
#         """Returns a scorer function based on the selected scoring metric."""
#         if self.scoring == 'pearson':
#             return make_scorer(lambda y, y_pred: pearson_corr(y, y_pred), greater_is_better=True)
#         raise ValueError(f"Scoring method '{self.scoring}' not supported.")

#     def select_best_alphas(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Select the best alpha for each electrode using RidgeCV on each time point.
#         For each time point, pass the full multi-target y to RidgeCV, and aggregate 
#         across time by computing the mode for each electrode.
#         """
#         print("Finding optimal penalty terms...")
#         _, n_electrodes, n_time = y.shape

#         if X.ndim == 2:
#             alphas_time = np.zeros((n_time, n_electrodes), dtype=np.float32)
#             scores = np.zeros((n_time, n_electrodes), dtype=np.float32)

#             for t in tqdm(range(n_time), desc="Selecting alphas (2D X)"):
#                 y_t = y[:, :, t]  # shape: (n_samples, n_electrodes)
#                 ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True,
#                                    scoring=self.get_scorer())
#                 ridge_cv.fit(X, y_t)
#                 alphas_time[t, :] = ridge_cv.alpha_
#                 scores[t, :] = ridge_cv.best_score_

#             final_alphas = np.zeros(n_electrodes, dtype=np.float32)
#             for e in range(n_electrodes):
#                 final_alphas[e] = np.atleast_1d(mode(alphas_time[:, e]).mode)[0]
#             print('Alpha values:', final_alphas)

#         elif X.ndim == 3:
#             alphas_time = np.zeros((n_time, n_electrodes), dtype=np.float32)
#             for t in tqdm(range(n_time), desc="Selecting alphas (3D X)"):
#                 X_t = X[:, :, t]  # shape: (n_samples, n_features)
#                 y_t = y[:, :, t]  # shape: (n_samples, n_electrodes)
#                 ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True,
#                                    scoring=self.get_scorer())
#                 ridge_cv.fit(X_t, y_t)
#                 alphas_time[t, :] = ridge_cv.alpha_
#             final_alphas = np.zeros(n_electrodes, dtype=np.float32)
#             for e in range(n_electrodes):
#                 final_alphas[e] = np.atleast_1d(mode(alphas_time[:, e]).mode)[0]
#             print('Alpha values:', final_alphas)

#         else:
#             raise ValueError("X must be a 2D or 3D array.")
#         return final_alphas, alphas_time

#     def cv(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Runs 5-fold cross validation (without shuffling) on the dataset.
#         The model is run 5 times (one per fold) and the predictions for each test fold
#         are concatenated into the final output.
#         """
#         kf = KFold(n_splits=5, shuffle=False)
#         n_samples = X.shape[0]
#         n_electrodes = y.shape[1]
#         if X.ndim == 2:
#             n_time = y.shape[2]
#         elif X.ndim == 3:
#             n_time = X.shape[2]
#         else:
#             raise ValueError("X must be either 2D or 3D.")

#         y_pred_full = np.zeros((n_samples, n_electrodes, n_time), dtype=np.float32)
#         y_true_full = np.zeros((n_samples, n_electrodes, n_time), dtype=y.dtype)

#         # Define a helper function to process one fold.
#         def process_fold(train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#             X_train_fold = X[train_idx]
#             y_train_fold = y[train_idx]
#             X_test_fold = X[test_idx]
#             y_test_fold = y[test_idx]
#             # Determine best alphas on the training fold.
#             best_alphas, _ = self.select_best_alphas(X_train_fold, y_train_fold)
#             print("Applying best alphas for fold with", len(train_idx), "training samples.")
#             fold_pred = np.zeros((len(test_idx), n_electrodes, n_time), dtype=np.float32)

#             if X.ndim == 2:
#                 for t in range(n_time):
#                     ridge = Ridge(alpha=best_alphas, fit_intercept=True)
#                     ridge.fit(X_train_fold, y_train_fold[:, :, t])
#                     fold_pred[:, :, t] = ridge.predict(X_test_fold)
#             elif X.ndim == 3:
#                 for t in range(n_time):
#                     ridge = Ridge(alpha=best_alphas, fit_intercept=True)
#                     ridge.fit(X_train_fold[:, :, t], y_train_fold[:, :, t])
#                     fold_pred[:, :, t] = ridge.predict(X_test_fold[:, :, t])
#             return test_idx, fold_pred, y_test_fold

#         # Parallelize the fold processing across all folds.
#         results = Parallel(n_jobs=-1, verbose=5)(
#             delayed(process_fold)(train_idx, test_idx) for train_idx, test_idx in kf.split(X)
#         )

#         # Reassemble the results into the full prediction arrays.
#         for test_idx, fold_pred, y_test_fold in results:
#             y_pred_full[test_idx, :, :] = fold_pred
#             y_true_full[test_idx, :, :] = y_test_fold

#         return y_pred_full, y_true_full
