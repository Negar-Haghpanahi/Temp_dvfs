

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
        list_window = [2, 10, 31, 33, 39, 49, 54, 55, 63, 65, 72, 76, 77, 78, 81, 84, 86, 97, 101, 109, 120, 131, 133, 135, 145, 148, 155, 174, 192, 198, 209, 210, 211, 213, 215, 231, 234, 248, 250, 254, 259, 290, 292, 299, 300, 306, 323, 328, 329, 336, 338, 340, 342, 344, 350, 352, 363, 367, 375, 382, 390, 393, 408, 416, 417, 430, 437, 446, 448, 453, 457, 478, 493, 503, 522, 527, 532, 533, 536, 548, 549, 569, 571, 581, 583, 591, 596, 597, 627, 636, 651, 654, 660, 668, 672, 686, 706, 712, 714, 716]
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
        list_window = [0, 2, 4, 5, 6, 9, 10, 11, 12, 15, 16, 18, 19, 22, 24, 26, 27, 29, 30, 31, 32, 33, 35, 36, 38, 41, 42, 45, 46, 51, 55, 56, 60, 62, 65, 66, 67, 68, 69, 73, 75, 76, 78, 82, 85, 90, 93, 95, 96, 97, 98, 100, 101, 104, 109, 112, 113, 114, 115, 118, 119, 122, 123, 124, 125, 127, 128, 132, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 148, 154, 155, 156, 158, 159, 161, 162, 165, 166, 168, 169, 170, 173, 182, 183, 185, 186, 187, 191, 193]
    if args.dataset_name == 'SelfRegulationSCP1':
        fs_base = 256
        list_window = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 103, 104, 105, 107, 108, 109, 110, 111]  
    if args.dataset_name == 'WESADchest':
        fs_base = 700
        list_window = [23, 29, 44, 49, 51, 59, 65, 68, 77, 116, 124, 125, 169, 204, 219, 221, 232, 241, 271, 275, 276, 299, 344, 353, 360, 373, 382, 384, 396, 400, 413, 428, 432, 455, 464, 481, 530, 532, 535, 537, 554, 570, 579, 588, 613, 622, 623, 637, 664, 698, 714, 715, 791, 820, 851, 865, 886, 904, 926, 929, 937, 948, 949, 952, 972, 974, 1004, 1022, 1030, 1036, 1037, 1047, 1051, 1052, 1056, 1065, 1070, 1105, 1124, 1153, 1160, 1178, 1194, 1206, 1207, 1209, 1214, 1229, 1267, 1311, 1324, 1340, 1343, 1378, 1388, 1414, 1417, 1420, 1437, 1474]
    if args.dataset_name == 'PAMAP2':
        fs_base = 100
        list_window = [0, 9, 11, 15, 17, 19, 22, 24, 25, 30, 31, 33, 39, 42, 46, 55, 56, 57, 66, 70, 72, 73, 75, 76, 77, 78, 79, 82, 84, 90, 93, 101, 113, 116, 117, 126, 132, 137, 148, 157, 165, 173, 176, 181, 185, 192, 196, 211, 220, 229, 231, 237, 238, 245, 247, 261, 275, 277, 278, 281, 284, 285, 286, 287, 289, 294, 297, 305, 311, 332, 335, 336, 352, 362, 364, 365, 368, 374, 377, 378, 383, 388, 391, 393, 404, 405, 407, 412, 418, 423, 426, 427, 428, 432, 434, 437, 438, 440, 441, 442]
     
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
