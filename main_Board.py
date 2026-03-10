
import pickle ,os , time
from ReadFile import LoadData
import argparse , csv , numpy as np
from test_Board import Test

    

def parse_args():
         p = argparse.ArgumentParser()
         p.add_argument("--dataset_name", type=str, default="Shoaib")
         p.add_argument("--n_est", type=int, default=200)
         p.add_argument("--max_depth", type=int, default=5)
         p.add_argument("--num_exits", type=int, default=4)

         p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
         p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
         p.add_argument("--th_combination", type=float, nargs="+", default=[0.4864, 1.45943, 0.9729550])    # th_list
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
    print(args.dataset_name)
    classData.Read(args.dataset_name)
    classData.SplitData()

    
    with open(f"PKL_Saved_Files/margin1.38/GB/{args.dataset_name}_trained_model.pkl", "rb") as f:
        all_models = pickle.load(f,encoding='latin1')
        
   
    x_path = f"PKL_Saved_Files/margin1.38/GB/{args.dataset_name}_X_test.npy"
    y_path = f"PKL_Saved_Files/margin1.38/GB/{args.dataset_name}_y_test.npy"

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

    output_file = f"PKL_Saved_Files/margin1.38/GB/{args.dataset_name}_accuracy_results.csv"

    num_exits = len(model.split_points)

    header = ["t_start", "t_end"]
    for i in range(1, num_exits + 1):
        header.append(f"t{i}")
    header.extend(["total", "true_label", "prediction", "correctness",
            "exit_level", "window_num", "data%"])
    
    with open(output_file, "w", newline="") as f:
        add_header(f, header)
        write_content_to_file(f, all_result, header)

    
    os.system("pkill -f 'python3 data-logger.py'") 



# FOR GB:
    # def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="wisdm")
    #     p.add_argument("--n_est", type=int, default=120)
    #     p.add_argument("--max_depth", type=int, default=5)
    #     p.add_argument("--num_exits", type=int, default=4)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.44793, 1.34381, 0.89587])    # th_list
    #     return p.parse_args()
    
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="wharDataOriginal")
#     p.add_argument("--n_est", type=int, default=120)
#     p.add_argument("--max_depth", type=int, default=5)
#     p.add_argument("--num_exits", type=int, default=4)

#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[0.5198603, 1.559581,1.03972])    # th_list
#     return p.parse_args()


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="ACCGyro")
#     p.add_argument("--n_est", type=int, default=150)
#     p.add_argument("--max_depth", type=int, default=4)
#     p.add_argument("--num_exits", type=int, default=4)
#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.47 ,0.59, 1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.3, 0.41 ,0.56, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[0.6931, 0.1732, 0.34657])    # th_list
#     return p.parse_args()

#    def parse_args():
#         p = argparse.ArgumentParser()
#         p.add_argument("--dataset_name", type=str, default="Epilepsy")
#         p.add_argument("--n_est", type=int, default=250)
#         p.add_argument("--max_depth", type=int, default=4)
#         p.add_argument("--num_exits", type=int, default=2)

#         p.add_argument("--tree_splits", type=float, nargs="+", default=[0.47, 1])
#         p.add_argument("--proportions", type=float, nargs="+", default=[0.36, 1])   # split_points
#         p.add_argument("--th_combination", type=float, nargs="+", default=[0.693147])    # th_list
#         return p.parse_args()
    
    #  def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="Shoaib")
    #     p.add_argument("--n_est", type=int, default=200)
    #     p.add_argument("--max_depth", type=int, default=5)
    #     p.add_argument("--num_exits", type=int, default=4)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.4864, 1.45943, 0.9729550])    # th_list
    #     return p.parse_args()


#===============================================================================================


    # def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="Epilepsy")
    #     p.add_argument("--n_est", type=int, default=80)
    #     p.add_argument("--max_depth", type=int, default=20)
    #     p.add_argument("--num_exits", type=int, default=3)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.31, 0.54, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.39, 0.57, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.34657359027997264, 1.3862943611198906])    # th_list
    #     return p.parse_args()
    
    # def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="Shoaib")
    #     p.add_argument("--n_est", type=int, default=75)
    #     p.add_argument("--max_depth", type=int, default=70)
    #     p.add_argument("--num_exits", type=int, default=3)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.31, 0.54, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.39, 0.57, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.48647753726382825, 1.945910149055313])    # th_list
    #     return p.parse_args()

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="ACCGyro")
#     p.add_argument("--n_est", type=int, default=60)
#     p.add_argument("--max_depth", type=int, default=60)
#     p.add_argument("--num_exits", type=int, default=4)

#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.47 ,0.59, 1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.3, 0.41 ,0.57, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[0.6931471805599453, 0.17328679513998632, 0.34657359027997264])    # th_list
#     return p.parse_args()


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="wharDataOriginal")
#     p.add_argument("--n_est", type=int, default=60)
#     p.add_argument("--max_depth", type=int, default=15)
#     p.add_argument("--num_exits", type=int, default=2)

#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.33,  1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.34, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[1.0397207708399179])    # th_list
#     return p.parse_args()



# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="wisdm")
#     p.add_argument("--n_est", type=int, default=60)
#     p.add_argument("--max_depth", type=int, default=15)
#     p.add_argument("--num_exits", type=int, default=2)

#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.33 , 1.0])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.34, 1.0])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[1.0397207708399179])    # th_list
#     return p.parse_args()






