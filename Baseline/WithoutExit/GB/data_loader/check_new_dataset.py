import pickle
import numpy as np

with open("data_loader\\Datasets\\wharDataOriginal.pkl", "rb") as f:
    data = pickle.load(f)

print("Top-level type:", type(data))

if isinstance(data, dict):
    print("Keys:", data.keys())
    for k, v in data.items():
        print(f"\nKey: {k}")
        print("Type:", type(v))
        if hasattr(v, "shape"):
            print("Shape:", v.shape)
elif isinstance(data, (list, tuple)):
    print("Length:", len(data))
    for i, v in enumerate(data):
        print(f"\nIndex {i}")
        print("Type:", type(v))
        if hasattr(v, "shape"):
            print("Shape:", v.shape)
