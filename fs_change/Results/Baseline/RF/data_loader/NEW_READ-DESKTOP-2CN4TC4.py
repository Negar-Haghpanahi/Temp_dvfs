import os
import numpy as np
import pandas as pd

# ===============================
# CONFIG
# ===============================
BASE_DIR = r"C:\Users\Negar\OneDrive - Washington State University (email.wsu.edu)\WSU\Fall2025-Semster2\Research\DVFS\Dynamic_Early_Exit\Code\data_loader\Datasets\wisdm-dataset\raw"
FS = 20
WINDOW_SEC = 10
WINDOW_SIZE = FS * WINDOW_SEC
STRIDE = WINDOW_SIZE

SENSORS = {
    "phone_accel": ("phone", "accel"),
    "phone_gyro":  ("phone", "gyro"),
    "watch_accel": ("watch", "accel"),
    "watch_gyro":  ("watch", "gyro"),
}

RANDOM_SEED = 42
TRAIN_RATIO = 0.8

# ============================================================
# READ ONE RAW FILE
# ============================================================
def read_wisdm_file(filepath):
    cols = ["subject", "activity", "timestamp", "x", "y", "z"]

    df = pd.read_csv(
        filepath,
        header=None,
        names=cols,
        sep=",",
        engine="python"
    )

    # Clean semicolon
    df["z"] = df["z"].astype(str).str.replace(";", "", regex=False)
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(float)

    return df

# ============================================================
# WINDOW ONE SUBJECT FILE
# ============================================================
def window_subject(df):
    X_windows = []
    y_windows = []
    subject_windows = []

    data = df[["x", "y", "z"]].values
    labels = df["activity"].values
    subject_id = df["subject"].iloc[0]

    for start in range(0, len(data) - WINDOW_SIZE, STRIDE):
        end = start + WINDOW_SIZE

        X_windows.append(data[start:end])
        y_windows.append(pd.Series(labels[start:end]).mode()[0])
        subject_windows.append(subject_id)

    return (
        np.array(X_windows),
        np.array(y_windows),
        np.array(subject_windows)
    )

# ============================================================
# LOAD ALL FILES FOR ONE SENSOR
# ============================================================
def load_sensor(device, sensor):
    sensor_dir = os.path.join(BASE_DIR, device, sensor)
    files = sorted(f for f in os.listdir(sensor_dir) if f.endswith(".txt"))

    X_all, y_all, subjects_all = [], [], []

    print(f"\nLoading {device}/{sensor} ({len(files)} files)")

    for file in files:
        df = read_wisdm_file(os.path.join(sensor_dir, file))
        X_win, y_win, subj_win = window_subject(df)

        X_all.append(X_win)
        y_all.append(y_win)
        subjects_all.append(subj_win)

    return (
        np.concatenate(X_all),
        np.concatenate(y_all),
        np.concatenate(subjects_all),
    )

# ============================================================
# MAIN PIPELINE
# ============================================================
X_streams = []
y_ref = None
subjects_ref = None

for name, (device, sensor) in SENSORS.items():
    X_s, y_s, subj_s = load_sensor(device, sensor)

    print(f"{name} shape: {X_s.shape}")

    X_streams.append(X_s)

    if y_ref is None:
        y_ref = y_s
        subjects_ref = subj_s

# ============================================================
# ALIGN STREAMS (IMPORTANT)
# ============================================================
min_windows = min(x.shape[0] for x in X_streams)

print("\nAligning all sensors to", min_windows, "windows")

X_streams = [x[:min_windows] for x in X_streams]
y_ref = y_ref[:min_windows]
subjects_ref = subjects_ref[:min_windows]

# ============================================================
# FUSE SENSORS
# ============================================================
X_fused = np.concatenate(X_streams, axis=2)

print("\nFINAL DATA SHAPE")
print("X_fused:", X_fused.shape)
print("y:", y_ref.shape)
print("subjects:", subjects_ref.shape)

# ============================================================
# SAVE FULL DATASET
# ============================================================
np.savez(
    "wisdm_fused_all.npz",
    X=X_fused,
    y=y_ref,
    subjects=subjects_ref
)

print("\nSaved full dataset → wisdm_fused_all.npz")

# ============================================================
# SUBJECT-WISE TRAIN / TEST SPLIT
# ============================================================
np.random.seed(RANDOM_SEED)

unique_subjects = np.unique(subjects_ref)
np.random.shuffle(unique_subjects)

split_idx = int(TRAIN_RATIO * len(unique_subjects))
train_subjects = unique_subjects[:split_idx]
test_subjects = unique_subjects[split_idx:]

train_mask = np.isin(subjects_ref, train_subjects)
test_mask = np.isin(subjects_ref, test_subjects)

X_train = X_fused[train_mask]
y_train = y_ref[train_mask]

X_test = X_fused[test_mask]
y_test = y_ref[test_mask]

print("\nTRAIN / TEST SPLIT")
print("Train windows:", X_train.shape[0])
print("Test windows:", X_test.shape[0])
print("Shared subjects:",
      set(subjects_ref[train_mask]) & set(subjects_ref[test_mask]))

# ============================================================
# SAVE SPLITS
# ============================================================
np.savez(
    "wisdm_train_test.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

print("\nSaved train/test split → wisdm_train_test.npz")
