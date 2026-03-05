
import pickle ,os , time
from ReadFile import LoadData
import argparse , csv , numpy as np
from test_Board import Test


def parse_args():
    

    parser = argparse.ArgumentParser(description = "RF-H Inference")
    parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    parser.add_argument("--n_est", type=int,default=26, help = "The number of estimators")
    parser.add_argument("--max_depth", type=int, default=21, help = "The max depth")
    parser.add_argument("--num_exits", type=int,  default=2 ,help = "The number of exits")
    parser.add_argument("--tree_splits", type=list, default=[0.5, 1] ,help = "Tree splits")
    parser.add_argument("--proportions", type=list, default=[0.25, 1] ,help = "Data proportions",  nargs="+")
    parser.add_argument("--th_combination", type=list, default=[1.56], help = "Threshold combination", nargs="+")
    
    return parser.parse_args()

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

    
    with open(f"PKL_Saved_Files/{args.dataset_name}_trained_model.pkl", "rb") as f:
        all_models = pickle.load(f,encoding='latin1')
        
   
    x_path = f"PKL_Saved_Files/{args.dataset_name}_X_test.npy"
    y_path = f"PKL_Saved_Files/{args.dataset_name}_y_test.npy"

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
    all_result = Test(X_test, y_test, model, args)

    output_file = f"PKL_Saved_Files/{args.dataset_name}_accuracy_results.csv"

    num_exits = len(model.split_points)

    header = ["t_start", "t_end"]
    for i in range(1, num_exits + 1):
        header.append(f"t{i}")
    header.extend(["total", "true_label", "prediction", "correctness",
            "exit_level", "window_num", "data%"])
    
    with open(output_file, "w", newline="") as f:
        add_header(f, header)
        write_content_to_file(f, all_result, header)

    
    os.system("pkill -f 'python3 data_logger.py'")       
