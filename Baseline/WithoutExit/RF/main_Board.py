import pickle, os, time, argparse, csv, numpy as np
from data_loader.ReadFile import LoadData
from test_Board import Test
from sensor_control import initialize_bmi160, sensor_on, sensor_sleep, auto_calibrate

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="wharDataOriginal")
    #p.add_argument("--n_est", type=int, default=80)
    #p.add_argument("--max_depth", type=int, default=30)
    
    p.add_argument("--fs_base", type=float, default=50.0)
    p.add_argument("--sensor_wakeup_sec", type=float, default=0.0)
    p.add_argument("--print_trace", action="store_true")
    
    return p.parse_args()

def write_content_to_file(file, content, header):
    writer = csv.writer(file)
    for line in content:
        row = [line.get(key, "") for key in header]
        writer.writerow(row)

def add_header(file, header):
    writer = csv.writer(file)
    writer.writerow(header)

if __name__ == "__main__":
    args = parse_args()
    print("time s ", time.time())
    time.sleep(2)
    print("time end_2" , time.time()) 

    classData = LoadData()
    classData.Read(args.dataset_name)
    classData.SplitData()

    with open(f"PKL_Saved_Files/{args.dataset_name}_trained_model.pkl", "rb") as f:
        all_models = pickle.load(f, encoding="latin1")

    x_path = f"PKL_Saved_Files/{args.dataset_name}_X_test.npy"
    y_path = f"PKL_Saved_Files/{args.dataset_name}_y_test.npy"

    if os.path.exists(x_path) and os.path.exists(y_path):
        X_test = np.load(x_path, allow_pickle=True)
        y_test = np.load(y_path, allow_pickle=True)
    else:
        X_test = classData.GetXtest()
        y_test = classData.GetYtest()

    model = all_models[0]["models"]



    initialize_bmi160()
    time.sleep(0.1)
    auto_calibrate()
    time.sleep(0.1)
    try:
        
        all_result = Test(X_test, y_test, model, args)

    finally:
        # sensor_sleep(verbose=True)
        os.system("pkill -f 'python3 data-logger.py'")

    output_file = f"PKL_Saved_Files/{args.dataset_name}_accuracy_results.csv"

    header = [
        "t_start", "t_end", "total",
        "window_sched_sec", "sensor_total_on_sec", "sensor_total_off_sec",
        "true_label", "prediction", "correctness",
        "exit_level", "window_num", "data%"
    ]

    with open(output_file, "w", newline="") as f:
        add_header(f, header)
        write_content_to_file(f, all_result, header)
