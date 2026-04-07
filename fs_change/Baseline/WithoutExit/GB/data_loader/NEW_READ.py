import numpy as np
import pandas as pd

# -------------------------
# READ EXCEL FILES
# -------------------------
AX = pd.read_csv("C:\\Users\\negar.haghpanahi\\Downloads\\wisdm\\Original\\Ax_0.00.csv", header=None).values
AY = pd.read_csv("C:\\Users\\negar.haghpanahi\\Downloads\\wisdm\\Original\\Ay_0.00.csv", header=None).values
AZ = pd.read_csv("C:\\Users\\negar.haghpanahi\\Downloads\\wisdm\\Original\\Az_0.00.csv", header=None).values
y  = pd.read_csv("C:\\Users\\negar.haghpanahi\\Downloads\\wisdm\\Original\\Labels.csv", header=None).values.squeeze()

print("AX:", AX.shape)
print("AY:", AY.shape)
print("AZ:", AZ.shape)
print("y :", y.shape)


# Stack along CHANNEL axis
X = np.stack([AX, AY, AZ], axis=1)

print("Final X shape:", X.shape)



np.savez_compressed("C:\\Users\\negar.haghpanahi\\OneDrive - Washington State University (email.wsu.edu)\\WSU\\Fall2025-Semster2\\Research\\DVFS\\Dynamic_Early_Exit\\Code\\data_loader\\Datasets\\wisdm.npz", X=X.astype(np.float32), y=y)



















# import os
# import numpy as np
# import pandas as pd

# # ===============================
# # CONFIG
# # ===============================
# BASE_DIR = r"C:\Users\negar.haghpanahi\Desktop\wisdm-dataset\raw"

# FS = 20                  # sampling rate (Hz)
# WINDOW_SEC = 10
# WINDOW_SIZE = FS * WINDOW_SEC  # 200 samples
# STRIDE = WINDOW_SIZE           # non-overlapping windows

# SENSORS = {
#     "phone_accel": ("phone", "accel"),
#     "phone_gyro":  ("phone", "gyro"),
#     "watch_accel": ("watch", "accel"),
#     "watch_gyro":  ("watch", "gyro"),
# }

# # ===============================
# # READ ONE FILE
# # ===============================
# def read_wisdm_file(filepath):
#     cols = ["subject", "activity", "timestamp", "x", "y", "z"]

#     df = pd.read_csv(
#         filepath,
#         names=cols,
#         sep=",",
#         engine="python"
#     )

#     # Remove trailing semicolon from z
#     df["z"] = df["z"].astype(str).str.replace(";", "", regex=False)
#     df["z"] = df["z"].astype(float)

#     return df

# # ===============================
# # WINDOW ONE SUBJECT
# # ===============================
# def window_subject(df):
#     X_windows = []
#     y_windows = []

#     data = df[["x", "y", "z"]].values
#     labels = df["activity"].values

#     for start in range(0, len(data) - WINDOW_SIZE, STRIDE):
#         end = start + WINDOW_SIZE

#         X_windows.append(data[start:end])
#         y_windows.append(pd.Series(labels[start:end]).mode()[0])

#     return np.array(X_windows), np.array(y_windows)

# # ===============================
# # LOAD ALL FILES FOR ONE SENSOR
# # ===============================
# def load_sensor(device, sensor):
#     sensor_dir = os.path.join(BASE_DIR, device, sensor)
#     files = sorted([f for f in os.listdir(sensor_dir) if f.endswith(".txt")])

#     X_all = []
#     y_all = []

#     print(f"\nLoading {device}/{sensor} ({len(files)} files)")

#     for file in files:
#         filepath = os.path.join(sensor_dir, file)
#         df = read_wisdm_file(filepath)

#         X_win, y_win = window_subject(df)

#         X_all.append(X_win)
#         y_all.append(y_win)

#     return np.concatenate(X_all), np.concatenate(y_all)

# # ===============================
# # MAIN PIPELINE
# # ===============================
# X_streams = []
# y_ref = None

# for name, (device, sensor) in SENSORS.items():
#     X_sensor, y_sensor = load_sensor(device, sensor)

#     print(f"{name}: {X_sensor.shape}")

#     X_streams.append(X_sensor)

#     # Use labels from first sensor only
#     if y_ref is None:
#         y_ref = y_sensor

# # ===============================
# # FUSE ALL SENSORS
# # ===============================
# X_fused = np.concatenate(X_streams, axis=2)

# print("\nFINAL DATASET")
# print("X_fused shape:", X_fused.shape)
# print("y shape:", y_ref.shape)
