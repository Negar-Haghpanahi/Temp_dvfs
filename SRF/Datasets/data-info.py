import pickle
import os


def inspect_dataset(dataset_name, base_path="Datasets"):
    """
    Load a dataset .pkl file and print its structure.

    Expected format inside pkl:
        {
            'data': numpy array of shape (n_windows, n_channels, n_timesteps),
            'labels': array/list of labels
        }
    """

    file_path = os.path.join(base_path, f"{dataset_name}_dataLabels.pkl")

    # --- Check file exists ---
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # --- Load pickle file ---
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)

    # --- Extract data ---
    data = data_dict['data']
    labels = data_dict['labels']

    # --- Shape info ---
    n_windows, n_channels, n_timesteps = data.shape

    # --- Label info ---
    unique_labels = set(labels)
    num_activities = len(unique_labels)

    # --- Print summary ---
    print("\n📊 Dataset Summary")
    print("=" * 40)
    print(f"Dataset name       : {dataset_name}")
    print(f"File path          : {file_path}")
    print("-" * 40)
    print(f"Number of windows  : {n_windows}")
    print(f"Number of sensors  : {n_channels}")
    print(f"Datapoints/window  : {n_timesteps}")
    print(f"Number of classes  : {num_activities}")
    print("=" * 40)

    return {
        "n_windows": n_windows,
        "n_channels": n_channels,
        "n_timesteps": n_timesteps,
        "num_classes": num_activities
    }


# ===============================
# 🔹 Example usage
# ===============================
if __name__ == "__main__":
    dataset_list = [
        "EMGPhysical",
        "Shoaib",
        "Epilepsy",
        "PAMAP2",
        "WesadChest",
        "SelfRegulationSCP1"
    ]

    for dataset in dataset_list:
        inspect_dataset(dataset)