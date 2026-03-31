

import pickle ,os , time
from ReadFile import LoadData
import argparse , csv
from Test_Board import Test
from sensor_control import initialize_bmi160, auto_calibrate, sensor_on, sensor_sleep


def parse_args():
     parser = argparse.ArgumentParser(description = "RF-H Inference")
     parser.add_argument("--dataset_name", type=str, default= "EMGPhysical", help = "The Dataset name")
     parser.add_argument("--n_est", type=int,default=37, help = "The number of estimators")
     parser.add_argument("--max_depth", type=int, default=28, help = "The max depth")
     parser.add_argument("--num_exits", type=int,  default=2,help = "The number of exits")
     parser.add_argument("--tree_splits", type=list, default=[0.5, 1] ,help = "Tree splits")
     parser.add_argument("--proportions", type=list, default=[0.33, 1],help = "Data proportions",  nargs="+")
     parser.add_argument("--th_combination", type=list, default=[1.31], help = "Threshold combination", nargs="+")
    
    



     return parser.parse_args()

def write_content_to_file(file, content, header): # the content is a list of dictionaries
    writer = csv.writer(file)
    for line in content:
        row = [line[key] for key in header]
        writer.writerow(row)


def add_header(file, header):
    writer = csv.writer(file)
    writer.writerow(header)
    
    
if __name__ =="__main__":
    
    t1 = time.time()
    time.sleep(5) # ************************************************************* for how long
    t2 = time.time()
    T = t2-t1 # ********************************************* save it
    print("t1: ", t1)
    print("t2: ", t2)
    
    args = parse_args()
    classData = LoadData()
    classData.Read(args.dataset_name)
    classData.SplitData()

    
    with open(f"PKL_Saved_Files/{args.dataset_name}_trained_model.pkl", "rb") as f:
        all_models = pickle.load(f,encoding='latin1')
        
    with open(f"PKL_Saved_Files/{args.dataset_name}_trained_results.pkl", "rb") as f:
        current_config_training_metrics = pickle.load(f)
     
     
    if args.dataset_name == 'Shoaib':
        fs_base = 50
    if args.dataset_name == 'Epilepsy':
        fs_base = 250
    if args.dataset_name == 'EMGPhysical':
        fs_base = 100
    if args.dataset_name == 'SelfRegulationSCP1':
        fs_base = 256
    if args.dataset_name == 'WESADchest':
        fs_base = 700
    if args.dataset_name == 'PAMAP2':
        fs_base = 100
     
    initialize_bmi160()
    print("BMI160 Initialized")
    auto_calibrate()
     
        
    all_result = Test(classData.GetTestX() , classData.GetYtest() ,all_models ,  args , fs_base , classData.n_data ,args.proportions ) 
    
 
    output_file = f'PKL_Saved_Files/{args.dataset_name}_accuracy_results.csv'
    # header = ['t_start','t1', 't2', 't3', 't4', 'total', 'true_label', 'prediction', 'correctness', 'exit_taken', 'data%']
        # Base header keys
    header = [
        't_start',
        't_end'
    ]

    # Dynamically add 't1', 't2', ... based on the number of exits + 1 (for total stages)
    # If num_exits = 3, this adds 't1', 't2', 't3'
    for i in range(1, args.num_exits + 1):
        header.append(f't{i}')

    # Add remaining keys
    header.extend([
        'total', 'true_label', 'prediction', 'correctness', 
        'exit_level', 'window_num', 'data%',"sensor_total_on_sec", "sensor_total_off_sec"
    ])
    with open(output_file, "w", newline="") as f1:
        add_header(f1, header)
        write_content_to_file(f1, all_result, header)
        
    os.system("pkill -f 'python3 data_logger.py'")            



  # parser = argparse.ArgumentParser(description = "RF-H Inference")
    # parser.add_argument("--dataset_name", type=str, default= "PAMAP2", help = "The Dataset name")
    # parser.add_argument("--n_est", type=int,default=26, help = "The number of estimators")
    # parser.add_argument("--max_depth", type=int, default=21, help = "The max depth")
    # parser.add_argument("--num_exits", type=int,  default=2 ,help = "The number of exits")
    # parser.add_argument("--tree_splits", type=list, default=[0.5, 1] ,help = "Tree splits")
    # parser.add_argument("--proportions", type=list, default=[0.25, 1] ,help = "Data proportions",  nargs="+")
    # parser.add_argument("--th_combination", type=list, default=[1.56], help = "Threshold combination", nargs="+")


    # parser = argparse.ArgumentParser(description = "RF-H Inference")
    # parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    # parser.add_argument("--n_est", type=int,default=87, help = "The number of estimators")
    # parser.add_argument("--max_depth", type=int, default=22, help = "The max depth")
    # parser.add_argument("--num_exits", type=int,  default=3,help = "The number of exits")
    # parser.add_argument("--tree_splits", type=list, default=[0.37, 0.48, 1] ,help = "Tree splits")
    # parser.add_argument("--proportions", type=list, default=[0.25, 0.37, 1],help = "Data proportions",  nargs="+")
    # parser.add_argument("--th_combination", type=list, default=[1.38, 1.33], help = "Threshold combination", nargs="+")


    # parser = argparse.ArgumentParser(description = "RF-H Inference")
    # parser.add_argument("--dataset_name", type=str, default= "EMGPhysical", help = "The Dataset name")
    # parser.add_argument("--n_est", type=int,default=37, help = "The number of estimators")
    # parser.add_argument("--max_depth", type=int, default=28, help = "The max depth")
    # parser.add_argument("--num_exits", type=int,  default=2,help = "The number of exits")
    # parser.add_argument("--tree_splits", type=list, default=[0.5, 1] ,help = "Tree splits")
    # parser.add_argument("--proportions", type=list, default=[0.33, 1],help = "Data proportions",  nargs="+")
    # parser.add_argument("--th_combination", type=list, default=[1.31], help = "Threshold combination", nargs="+")
    
    
    # parser = argparse.ArgumentParser(description = "RF-H Inference")
    # parser.add_argument("--dataset_name", type=str, default= "WESADchest", help = "The Dataset name")
    # parser.add_argument("--n_est", type=int,default=4, help = "The number of estimators")
    # parser.add_argument("--max_depth", type=int, default=58, help = "The max depth")
    # parser.add_argument("--num_exits", type=int,  default=3,help = "The number of exits")
    # parser.add_argument("--tree_splits", type=list, default=[0.5, 0.6, 1] ,help = "Tree splits")
    # parser.add_argument("--proportions", type=list, default=[0.44, 0.62, 1],help = "Data proportions",  nargs="+")
    # parser.add_argument("--th_combination", type=list, default=[0.33, 0.82], help = "Threshold combination", nargs="+")
    
    # parser = argparse.ArgumentParser(description = "RF-H Inference")
    # parser.add_argument("--dataset_name", type=str, default= "SelfRegulationSCP1", help = "The Dataset name")
    # parser.add_argument("--n_est", type=int,default=18, help = "The number of estimators")
    # parser.add_argument("--max_depth", type=int, default=16, help = "The max depth")
    # parser.add_argument("--num_exits", type=int,  default=3,help = "The number of exits")
    # parser.add_argument("--tree_splits", type=list, default=[0.36, 0.54, 1],help = "Tree splits")
    # parser.add_argument("--proportions", type=list, default=[0.3, 0.44, 1],help = "Data proportions",  nargs="+")
    # parser.add_argument("--th_combination", type=list, default=[0.46, 0.66], help = "Threshold combination", nargs="+")
    
    # parser = argparse.ArgumentParser(description = "RF-H Inference")
    # parser.add_argument("--dataset_name", type=str, default= "Shoaib", help = "The Dataset name")
    # parser.add_argument("--n_est", type=int,default=77, help = "The number of estimators")
    # parser.add_argument("--max_depth", type=int, default=52, help = "The max depth")
    # parser.add_argument("--num_exits", type=int,  default=2 ,help = "The number of exits")
    # parser.add_argument("--tree_splits", type=list, default=[0.41, 1] ,help = "Tree splits")
    # parser.add_argument("--proportions", type=list, default=[0.25, 1] ,help = "Data proportions",  nargs="+")
    # parser.add_argument("--th_combination", type=list, default=[1.73], help = "Threshold combination", nargs="+")
