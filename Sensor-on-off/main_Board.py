
import pickle ,os , time
from ReadFile import LoadData
import argparse , csv , numpy as np
from test_Board import TestBoardControlled
from sensor_control import initialize_bmi160, auto_calibrate, sensor_on, sensor_sleep

def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--dataset_name", type=str, default="Shoaib")
        p.add_argument("--n_est", type=int, default=75)
        p.add_argument("--max_depth", type=int, default=70)
        p.add_argument("--num_exits", type=int, default=3)

        p.add_argument("--tree_splits", type=float, nargs="+", default=[0.31, 0.54, 1])
        p.add_argument("--proportions", type=float, nargs="+", default=[0.39, 0.57, 1])   # split_points
        p.add_argument("--th_combination", type=float, nargs="+", default=[0.48647753726382825, 1.945910149055313])    # th_list
        return p.parse_args()
    

def write_content_to_file(file, content, header): 
    writer = csv.writer(file)
    for line in content:
        row = [line[key] for key in header]
        writer.writerow(row)


def add_header(file, header):
    writer = csv.writer(file)
    writer.writerow(header)
    
    
if __name__ =="__main__":
    
    time.sleep(2)
    
    args = parse_args()
    classData = LoadData()
    classData.Read(args.dataset_name)
    classData.SplitData()

    
    with open(f"PKL_Saved_Files/margin1.5/RF/{args.dataset_name}_trained_model.pkl", "rb") as f:
        all_models = pickle.load(f,encoding='latin1')
        
   
    x_path = f"PKL_Saved_Files/margin1.5/RF/{args.dataset_name}_X_test.npy"
    y_path = f"PKL_Saved_Files/margin1.5/RF/{args.dataset_name}_y_test.npy"

    if os.path.exists(x_path) and os.path.exists(y_path):
        X_test = np.load(x_path, allow_pickle=True)
        y_test = np.load(y_path, allow_pickle=True)
    else:
        
        classData = LoadData()
        classData.Read(args.dataset_name)
        classData.SplitData()
        X_test = classData.GetXtest()
        y_test = classData.GetYtest()
        
    model = all_models[0]['models']  
    initialize_bmi160()
    print("BMI160 Initialized")
    auto_calibrate()

    
    window_len = int(X_test.shape[2])

    # ------------------------------
    # Run board-controlled evaluation
    # ------------------------------
    all_result = TestBoardControlled(X_test=X_test,y_test=y_test,model=model,args=args,sensor_on=sensor_on,sensor_sleep=sensor_sleep,fs_base=args.fs_base,window_len=window_len,sensor_wakeup_sec=None,print_trace=args.print_trace,)

    output_file = (
        f"PKL_Saved_Files/margin1.5/RF/"
        f"{args.dataset_name}_accuracy_results_test.csv"
    )

    num_exits = len(model.split_points)

    header = ["t_start", "t_end"]
    for i in range(1, num_exits + 1):
        header.append(f"t{i}_acq")
    for i in range(1, num_exits + 1):
        header.append(f"t{i}_compute")

    header.extend([
        "window_sched_sec",
        "sensor_total_on_sec",
        "sensor_total_off_sec",
        "compute_total_sec",
        "total",
        "true_label",
        "prediction",
        "correctness",
        "exit_level",
        "window_num",
        "data%",
    ])

    with open(output_file, "w", newline="") as f:
        add_header(f, header)
        write_content_to_file(f, all_result, header)

    os.system("pkill -f 'python3 data_logger.py'")   
