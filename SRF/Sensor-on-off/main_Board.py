

import pickle ,os , time
from ReadFile import LoadData
import argparse , csv
from Test_Board import Test
from sensor_control import initialize_bmi160, auto_calibrate, sensor_on, sensor_sleep


def parse_args():
     parser = argparse.ArgumentParser(description = "RF-H Inference")
     parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
     parser.add_argument("--n_est", type=int,default=87, help = "The number of estimators")
     parser.add_argument("--max_depth", type=int, default=22, help = "The max depth")
     parser.add_argument("--num_exits", type=int,  default=3,help = "The number of exits")
     parser.add_argument("--tree_splits", type=list, default=[0.37, 0.48, 1] ,help = "Tree splits")
     parser.add_argument("--proportions", type=list, default=[0.25, 0.37, 1],help = "Data proportions",  nargs="+")
     parser.add_argument("--th_combination", type=list, default=[1.38, 1.33], help = "Threshold combination", nargs="+")


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
        list_window = [2, 10, 24, 29, 30, 31, 39, 41, 50, 56, 57, 61, 71, 77, 78, 79, 83, 87, 91, 102, 111, 132, 133, 136, 146, 149, 156, 159, 164, 166, 175, 177, 189, 193, 210, 211, 212, 214, 216, 229, 245, 248, 257, 258, 262, 265, 272, 275, 276, 278, 285, 288, 297, 303, 305, 313, 317, 319, 322, 330, 333, 337, 347, 356, 362, 363, 375, 394, 404, 405, 412, 413, 423, 454, 457, 468, 488, 495, 504, 507, 508, 517, 531, 533, 539, 552, 555, 577, 584, 587, 593, 599, 600, 601, 605, 610, 611, 617, 625, 626]
    if args.dataset_name == 'Epilepsy':
        fs_base = 250
        list_window = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54]
    if args.dataset_name == 'EMGPhysical':
        fs_base = 100
        list_window = [0, 4, 5, 7, 9, 10, 11, 12, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 42, 43, 44, 45, 46, 47, 49, 51, 53, 55, 57, 58, 59, 63, 65, 67, 68, 69, 70, 71, 72, 73, 76, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 98, 99, 100, 101, 102, 103, 106, 109, 110, 113, 114, 115, 116, 117, 118, 119, 120, 123, 126, 129, 130, 131, 132, 133, 138, 139, 140, 143, 144, 145, 146, 148, 149, 150, 152, 153, 155]
    if args.dataset_name == 'SelfRegulationSCP1':
        fs_base = 256
        list_window = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 103, 104, 105, 107, 108, 109, 110, 111]  
    if args.dataset_name == 'WESADchest':
        fs_base = 700
        list_window = [23, 29, 44, 49, 51, 59, 65, 68, 77, 116, 124, 125, 169, 204, 219, 221, 232, 241, 271, 275, 276, 299, 344, 353, 360, 373, 382, 384, 396, 400, 413, 428, 432, 455, 464, 481, 530, 532, 535, 537, 554, 570, 579, 588, 613, 622, 623, 637, 664, 698, 714, 715, 791, 820, 851, 865, 886, 904, 926, 929, 937, 948, 949, 952, 972, 974, 1004, 1022, 1030, 1036, 1037, 1047, 1051, 1052, 1056, 1065, 1070, 1105, 1124, 1153, 1160, 1178, 1194, 1206, 1207, 1209, 1214, 1229, 1267, 1311, 1324, 1340, 1343, 1378, 1388, 1414, 1417, 1420, 1437, 1474]
    if args.dataset_name == 'PAMAP2':
        fs_base = 100
        list_window = [0, 3, 5, 9, 15, 17, 18, 22, 24, 25, 30, 31, 33, 39, 42, 45, 46, 55, 56, 57, 63, 70, 72, 73, 76, 77, 78, 82, 84, 90, 93, 94, 101, 104, 108, 110, 114, 124, 126, 131, 132, 137, 152, 155, 157, 165, 167, 172, 175, 176, 181, 184, 192, 193, 194, 203, 211, 220, 222, 223, 225, 227, 229, 233, 238, 247, 248, 255, 258, 268, 271, 277, 284, 287, 301, 307, 316, 321, 324, 332, 334, 335, 337, 346, 361, 369, 378, 380, 383, 388, 389, 390, 391, 392, 395, 397, 398, 401, 405, 406]
     
    initialize_bmi160()
    print("BMI160 Initialized")
    auto_calibrate()
     
        
    all_result = Test(list_window ,classData.GetTestX() , classData.GetYtest() ,all_models ,  args , fs_base , classData.n_data ,args.proportions ) 
    
 
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
