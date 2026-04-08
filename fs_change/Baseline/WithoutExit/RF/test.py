import pickle, os, time, argparse, csv, numpy as np , pprint
from data_loader.ReadFile import LoadData


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="wisdm")
    # p.add_argument("--n_est", type=int, default=80)
    # p.add_argument("--max_depth", type=int, default=30)
    
    p.add_argument("--fs_base", type=float, default=10.0)
    p.add_argument("--sensor_wakeup_sec", type=float, default=0.0)
    p.add_argument("--print_trace", action="store_true")
    
    return p.parse_args()
    


if __name__ == "__main__":
    args = parse_args()
    print("time s ", time.time())
    time.sleep(2)
    print("time end_2" , time.time()) 

 

    with open(f"C:\\Users\\negar.haghpanahi\\OneDrive - Washington State University (email.wsu.edu)\\WSU\\Fall2025-Semster2\\Research\\DVFS\\Dynamic_Early_Exit\\CLONE-RSP0\\Temp_dvfs\\fs_change\\Baseline\\WithoutExit\\RF\\PKL_Saved_Files\\{args.dataset_name}_trained_model.pkl", "rb") as f:
        all_models = pickle.load(f, encoding="latin1")
        
    model = all_models[0]["models"]

    print("Model type:", type(model))
    print("n_estimators:", model.n_estimators)
    print("configured max_depth:", model.max_depth)

    depths = [tree.tree_.max_depth for tree in model.estimators_]
    print("actual tree depths:", depths[:10])
    print("max fitted tree depth:", max(depths))
    print("min fitted tree depth:", min(depths))

    print("\nAll important RF params:")
    pprint.pprint(model.get_params())