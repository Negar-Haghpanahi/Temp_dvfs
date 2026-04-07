import pickle
import argparse
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import copy ,os
from scipy.io import arff
import pandas as pd
from tslearn.datasets import UCR_UEA_datasets

class LoadData:

    
    def __init__(self):
        
        self.data = None
        self.labels_array = None
        self.n_window = None
        
   

    def read_ts_file(self,filepath):
        X = []
        y = []

        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("@") or line.strip() == "":
                    continue

                label, data = line.split(":")
                y.append(float(label))

                channels = []
                for dim in data.strip().split("|"):
                    channels.append(np.array(dim.split(","), dtype=float))

                X.append(np.array(channels))

        return np.array(X), np.array(y)
   
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
            data = np.load('data_loader\\Datasets\\wisdm.npz')
            self.data, self.labels_array = data["X"], data["y"]
            
        elif dataset_name == 'wharDataOriginal':
            with open("data_loader\\Datasets\\wharDataOriginal.pkl", "rb") as f:
                x_raw, self.labels_array = pickle.load(f)
                
            self.data = np.squeeze(x_raw, axis=1)
            self.data = np.transpose(self.data, (0,2,1))  # (n_window, n_channel, n_data)
        
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
            WINDOW_SEC = 1
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

                        
        elif dataset_name == 'PPGDalia' :
            # loader = UCR_UEA_datasets()

            # This automatically downloads if needed
            # X_train_raw, y_train, X_test_raw, y_test = loader.load_dataset("PPGDalia")
            X_train_raw, y_train = self.read_ts_file(r"D:\datasets\PPGDalia\PPGDalia_TRAIN.ts")
            X_test_raw, y_test   = self.read_ts_file(r"D:\datasets\PPGDalia\PPGDalia_TEST.ts")

            print("Raw train shape:", X_train_raw.shape)
            print("Raw test shape:", X_test_raw.shape )
            X_raw = np.concatenate([X_train_raw, X_test_raw], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)

            print("Combined X shape:", X_raw.shape)
            print("Combined y shape:", y.shape)
            WINDOW_SIZE = 200    # timestamps per window
            STRIDE = 100         # overlap allowed

            X_windows = []
            y_windows = []

            for i in range(X_raw.shape[0]):
                series = X_raw[i]          # (channels, time)
                label = y[i]

                T = series.shape[1]

                for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    window = series[:, start:end]

                    X_windows.append(window)
                    y_windows.append(label)

            self.data = np.array(X_windows)
            self.labels_array = np.array(y_windows)

            print("Windowed X shape:", self.data.shape)
            print("Windowed y shape:", self.labels_array.shape)
            # REQUIRED by your pipeline
            assert self.data.ndim == 3
            print("Final format:", self.data.shape)
            # REQUIRED by your pipeline
        elif dataset_name == 'Accelerometer Gyro Mobile Phone':
           
            CSV_PATH = r"data_loader\\Datasets\\accelerometer_gyro_mobile_phone_dataset.csv"   # <-- change this
            WINDOW_SIZE = 128
            STRIDE = 64   
            df = pd.read_csv(CSV_PATH)
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

                # Window: (time, channels)
                window = X_raw[start:end]

                # Majority label in window
                label = np.bincount(y_raw[start:end]).argmax()

                X_windows.append(window)
                y_windows.append(label)

            X_windows = np.array(X_windows)  # (windows, time, channels)
            self.labels_array = np.array(y_windows)

            print("Windowed (time,channels):", X_windows.shape)
            self.data = np.transpose(X_windows, (0, 2, 1))

            print("Final X shape:", self.data.shape)
            print("Final y shape:", self.labels_array.shape)

            assert self.data.ndim == 3
            assert self.data.shape[1] == 6
            assert self.data.shape[2] == WINDOW_SIZE
            
        elif dataset_name == 'SelfBack':
            

            DATA_DIR = "C:\\Users\\Negar\\Downloads\\selfback\\selfBACK\\wt"  # folder containing merged 6-channel files
            WINDOW_SIZE = 100  # eg. 2 sec @ 100 Hz
            STRIDE = 100       # half overlap

            activity_map = {
                "upstairs": 0,
                "downstairs": 1,
                "walkslow": 2,
                "walkmedium": 3,
                "walkfast": 4,
                "jogging": 5,
                "standing": 6,
                "sitting": 7,
                "lying": 8
            }

            X_windows = []
            y_windows = []

            for fname in os.listdir(DATA_DIR):
    
                # ✅ ignore hidden/system files
                if fname.startswith("."):
                    continue

                full_path = os.path.join(DATA_DIR, fname)

                # ✅ ignore directories
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

                # ✅ read whitespace-separated numeric data
                data = np.loadtxt(full_path, delimiter=",")


                # safety check: must be (time, 6)
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
        else:    
            model_type = args.model_type
            file_path =f'data_loader\\Datasets\\{dataset_name}_dataLabels.pkl'
            with open(file_path, 'rb') as file:
                data_dict = pickle.load(file)
            self.data = data_dict['data']
            self.labels_array = data_dict['labels']

            self.n_window, n_channel, n_data = self.data.shape
            unique_numbers_set = set(self.labels_array)
            num_activities = len(unique_numbers_set)
            

            m,n = self.data.shape[::2]                                                        
            self.n_window, n_channel, n_data = self.data.shape
        
        
    def SplitData(self):
       
        X = self.GetData()
        y = self.GetLabel()

        # lst = list(range(0, self.GetWindow()))
        
        # X_train_ind, X_test_ind, y_train_dummy, y_test_dummy = train_test_split(lst, y, test_size=0.20, random_state=42)

        # # data
        # self.X_train = X[X_train_ind,:,:]
        # self.X_test =  X[X_test_ind,:,:]

        # # labels
        # self.y_train = y[X_train_ind]
        # self.y_test = y[X_test_ind]

        # 1. Split first
        indices = list(range(len(y)))
        train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42)

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]
                    
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












