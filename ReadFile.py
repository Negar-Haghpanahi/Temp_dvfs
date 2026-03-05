import pickle
import argparse
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import copy
from scipy.io import arff
import pandas as pd
import os

class LoadData:

    
    def __init__(self):
        
        self.data = None
        self.labels_array = None
        self.n_window = None
        
        
    def Read(self , datasetName = None):
    
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', type=str, default='EMGPhysical', required=False)
        parser.add_argument('--model_type', type=str, help="model", required=False, default="classic")
        args = parser.parse_args()
        
        if datasetName is None:
            dataset_name = args.dataset_name   
        else:
            dataset_name = datasetName
            
        if dataset_name == 'wisdm':
            data = np.load('Datasets//wisdm.npz')
            self.data, self.labels_array = data["X"], data["y"]
            self.n_window, n_channel, n_data = self.data.shape
        elif dataset_name == 'wharDataOriginal':
            with open("Datasets//wharDataOriginal.pkl", "rb") as f:
                x_raw, self.labels_array = pickle.load(f)
                
            self.data = np.squeeze(x_raw, axis=1)
            self.data = np.transpose(self.data, (0,2,1))  # (n_window, n_channel, n_data)
            self.n_window, n_channel, n_data = self.data.shape
        elif dataset_name == 'EEG Eye State':
            data, meta = arff.loadarff("data_loader\\Datasets\\EEG Eye State.arff")

            # Convert to DataFrame
            df = pd.DataFrame(data)
            # Labels (0 = eye open, 1 = eye closed)
            y = df["eyeDetection"].astype(int).values

            # EEG signals
            X_raw = df.drop(columns=["eyeDetection"]).values

            print("X_raw shape:", X_raw.shape)
            print("y shape:", y.shape)
            FS = 128
            WINDOW_SEC = 1 # 2 or 5 seconds
            WINDOW_SIZE = FS * WINDOW_SEC
            STRIDE = WINDOW_SIZE  # non-overlapping
            X_windows = []
            y_windows = []

            for start in range(0, len(X_raw) - WINDOW_SIZE, STRIDE):
                end = start + WINDOW_SIZE

                # window shape: (time, channels)
                window = X_raw[start:end]

                # label = majority label in window
                label = np.round(y[start:end].mean()).astype(int)

                X_windows.append(window)
                y_windows.append(label)

            X_windows = np.array(X_windows)
            self.labels_array = np.array(y_windows)

            print("Windowed X shape:", X_windows.shape)
            print("Windowed y shape:", self.labels_array.shape)
            self.data = np.transpose(X_windows, (0, 2, 1))

            print("Final X shape:", self.data.shape)

        elif dataset_name == 'SelfBack':                
            

            DATA_DIR = r"C:\Users\negar.haghpanahi\Downloads\selfback\selfBACK\wt"
            WINDOW_SIZE = 100
            STRIDE = 100

            activity_map = {
                "upstairs": 0,
                "downstairs": 1,
                "walkslow": 2,
                "walkmod": 3,
                "walkfast": 4,
                "jogging": 5,
                "standing": 6,
                "sitting": 7,
                "lying": 8
            }

            X_windows = []
            y_windows = []

            for fname in os.listdir(DATA_DIR):

                # ignore hidden/system files
                if fname.startswith("."):
                    continue

                full_path = os.path.join(DATA_DIR, fname)

                if not os.path.isfile(full_path):
                    continue

                # Example filename: 001-WalkingSlow
                try:
                    subject, activity = fname.split("_")
                except ValueError:
                    continue

                if activity not in activity_map:
                    continue

                label = activity_map[activity]

                data = np.loadtxt(full_path, delimiter=",")

                # safety check
                if data.ndim != 2 or data.shape[1] != 6:
                    print(f"Skipping {fname}, shape={data.shape}")
                    continue

                # windowing
                for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    window = data[start:end].T   # (6, time)

                    X_windows.append(window)
                    y_windows.append(label)

            self.data = np.array(X_windows, dtype=np.float32)
            self.labels_array = np.array(y_windows, dtype=np.int64)

            print("Final X shape:", self.data.shape)
            print("Final y shape:", self.labels_array.shape)
            
        elif dataset_name == 'ACCGyro':   
            CSV_PATH = r"Datasets//accelerometer_gyro_mobile_phone_dataset.csv"   # <-- change this
            WINDOW_SIZE = 128
            STRIDE = 64   # overlap allowed

            # =========================
            # LOAD CSV
            # =========================
            df = pd.read_csv(CSV_PATH)

            # Sensors (channels)
            sensor_cols = [
                "accX", "accY", "accZ",
                "gyroX", "gyroY", "gyroZ"
            ]

            X_raw = df[sensor_cols].values        # (time, 6)
            y_raw = df["Activity"].values         # (time,)
            timestamps = df["timestamp"].values   # optional

            print("Raw X shape:", X_raw.shape)
            print("Raw y shape:", y_raw.shape)

            # =========================
            # WINDOWING
            # =========================
            X_windows = []
            y_windows = []

            for start in range(0, len(X_raw) - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE

    
                window = X_raw[start:end]

           
                label = np.bincount(y_raw[start:end]).argmax()

                X_windows.append(window)
                y_windows.append(label)

            X_windows = np.array(X_windows)  # (windows, time, channels)
            self.labels_array = np.array(y_windows)

            print("Windowed (time,channels):", X_windows.shape)

        
            self.data = np.transpose(X_windows, (0, 2, 1))

            print("Final X shape:", self.data.shape)
            print("Final y shape:", self.labels_array.shape)

            # =========================
            # SANITY CHECK
            # =========================
            assert self.data.ndim == 3
            assert self.data.shape[1] == 6
            assert self.data.shape[2] == WINDOW_SIZE
            self.n_window, n_channel, n_data = self.data.shape
            
        else:    
            model_type = args.model_type
            file_path =f'Datasets//{dataset_name}_dataLabels.pkl'
            with open(file_path, 'rb') as file:
                data_dict = pickle.load(file)
            self.data = data_dict['data']
            self.labels_array = data_dict['labels']

            self.n_window, n_channel, n_data = self.data.shape
            unique_numbers_set = set(self.labels_array)
            num_activities = len(unique_numbers_set)
            

            m,n = self.data.shape[::2]                                                        
            self.n_window, n_channel, n_data = self.data.shape
        
        
    def SplitData(self ):
       
        X = self.GetData()
        y = self.GetLabel()

        lst = list(range(0, self.GetWindow()))
        X_train_ind, X_test_ind, y_train, y_test = train_test_split(lst, y, test_size=0.20, random_state=42 )

        
        self.X_train = X[X_train_ind, :, :]
        self.X_test  = X[X_test_ind, :, :]

        # labels
        self.y_train = y[X_train_ind]
        self.y_test  = y[X_test_ind]
                    
    def GetData(self): 
        return self.data
    
    def GetLabel(self):
        return self.labels_array
    
    def GetWindow(self):
        return self.n_window
    
    def GetYtrain(self):
        return self.y_train
    
    def GetYtest(self):
        return self.y_test
    
    def GetXtest(self):
        return self.X_test
    
  