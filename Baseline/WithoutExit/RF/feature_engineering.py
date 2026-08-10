import numpy as np
from downsampling import data_downsampler
class FeatureEngineer:
    

    def extract_features(self, X):  #Input: X shape (W, S, T)
        # compute per-sensor statistics over time axis
        mean = np.mean(X, axis=2)
        std = np.std(X, axis=2)
        energy = np.mean(X ** 2, axis=2)
        mn = np.min(X, axis=2)
        mx = np.max(X, axis=2)

        feats = np.stack([mean, std, energy, mn, mx], axis=2)
        W, S, F = feats.shape
        
        # X_swapped = np.swapaxes(X, 1, 2)  # Shape: [n_windows, n_timestamps, n_sensors]
        # n_windows, n_ts, n_sensors = X_swapped.shape
        # return X_swapped.reshape(n_windows, n_ts * n_sensors) 
        feat_reshape_obj = data_downsampler(1, feats)
        return feat_reshape_obj.flatten_data() 
        
