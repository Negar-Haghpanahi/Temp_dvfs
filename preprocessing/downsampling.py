import numpy as np
class data_downsampler:
    
    def __init__(self, factor , data):
        self.factor = factor
        self.X = data
 
    def downsample(self):
        
        # X shape: [n_windows, n_sensors, n_timestamps]
        self.X = self.X[:, :, ::self.factor]
        return self.X

    def flatten_data(self):
    
        X_swapped = np.swapaxes(self.X, 1, 2)  # Shape: [n_windows, n_timestamps, n_sensors]
        n_windows, n_ts, n_sensors = self.X.shape
        return X_swapped.reshape(n_windows, n_ts * n_sensors)   
