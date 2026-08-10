import argparse, pickle, time
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from feature_engineering import FeatureEngineer
from data_loader.ReadFile import LoadData
from utils.model_size_calculator import ModelSizeCalculator  
from utils.logger import setup_logger
from sklearn.ensemble import GradientBoostingClassifier

logger = setup_logger("TrainSaveBoard")

def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--dataset_name", type=str, default="Shoaib")
        p.add_argument("--n_est", type=int, default=200)
        p.add_argument("--max_depth", type=int, default=5)
        return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    loader = LoadData()
    loader.Read(args.dataset_name)
    loader.SplitData()

    X_train, y_train = loader.X_train, loader.y_train
    X_test, y_test = loader.X_test, loader.y_test

    n_classes = len(np.unique(y_train))
    size_calculator = ModelSizeCalculator(n_classes=n_classes)  

    rf_params = {
                "n_estimators": args.n_est,
                "max_depth": args.max_depth,
                "learning_rate":  0.1, 
                "random_state": 42,
                 #"n_jobs": -1
            }
    t0 = time.time()
    fe = FeatureEngineer()
    rf = GradientBoostingClassifier(**rf_params)
    New_feat=fe.extract_features(X_train)
    rf.fit(New_feat, y_train)
    fit_time_sec = time.time() - t0


    # ========== CALCULATE MODEL SIZE ==========
    try:
        # baseline RF -> use the baseline wrapper method
        model_size_info = size_calculator.calculate_baseline_model_size(rf)
    except Exception as e:
        logger.error(f"Error calculating model size for config ")
        model_size_info = None


    models_info = []
 
    # model_size_info = None
    try:
        logger.info("Model size summary:")
        logger.info(f"  Total Nodes: {model_size_info['total_nodes']:,}")
        logger.info(f"  Total Memory: {model_size_info['total_memory_kb']:.2f} KB")
        logger.info(f"  Num Exits: {model_size_info['num_exits']}")

        for ex in model_size_info["per_exit_details"]:
            logger.info(
                f"  Exit {ex['exit']}: {ex['nodes']:,} nodes, "
                f"{ex['memory_kb']:.2f} KB, {ex['n_estimators']} trees, "
                f"depth={ex['avg_depth']:.1f}"
            )
    except Exception as e:
        logger.error(f"Model size calculation failed: {e}")

    models_info.append({
            'models': rf
        })
    
    # ===== SAVE =====
    save_dir = Path("Baseline/WithoutExit/GB/PKL_Saved_Files")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f"{args.dataset_name}_trained_model.pkl", "wb") as f:
        pickle.dump(models_info, f)

    # Save training metadata if I need in the future
    training_summary = {
        "dataset_name": args.dataset_name,
        "n_classes": int(n_classes),
        "rf_params": rf_params,
        "fit_time_sec": float(fit_time_sec),
      
        "model_size_info": model_size_info,  
    }

    with open(save_dir / f"{args.dataset_name}_trained_results.pkl", "wb") as f:
        pickle.dump(training_summary, f)

   
    np.save(save_dir / f"{args.dataset_name}_X_test.npy", X_test)
    np.save(save_dir / f"{args.dataset_name}_y_test.npy", y_test)

    print(f"Saved model to: {save_dir / (args.dataset_name + '_trained_model.pkl')}")
    print(f"Saved results to: {save_dir / (args.dataset_name + '_trained_results.pkl')}")
    
    
    
    
    
    
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
    # p.add_argument("--dataset_name", type=str, default="wharDataOriginal")
    # p.add_argument("--n_est", type=int, default=120)
    # p.add_argument("--max_depth", type=int, default=5)
    # p.add_argument("--num_exits", type=int, default=4)

#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[0.5198603, 1.559581,1.03972])    # th_list
#     return p.parse_args()


# def parse_args():
#     p = argparse.ArgumentParser()
    # p.add_argument("--dataset_name", type=str, default="ACCGyro")
    # p.add_argument("--n_est", type=int, default=150)
    # p.add_argument("--max_depth", type=int, default=4)
#     p.add_argument("--num_exits", type=int, default=4)
#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.47 ,0.59, 1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.3, 0.41 ,0.56, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[0.6931, 0.1732, 0.34657])    # th_list
#     return p.parse_args()

   # def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="Epilepsy")
    #     p.add_argument("--n_est", type=int, default=250)
    #     p.add_argument("--max_depth", type=int, default=4)
    #     p.add_argument("--num_exits", type=int, default=2)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.47, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.36, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.693147])    # th_list
    #     return p.parse_args()
    
    #  def parse_args():
    #     p = argparse.ArgumentParser()
        # p.add_argument("--dataset_name", type=str, default="Shoaib")
        # p.add_argument("--n_est", type=int, default=200)
        # p.add_argument("--max_depth", type=int, default=5)
    #     p.add_argument("--num_exits", type=int, default=4)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.4864, 1.45943, 0.9729550])    # th_list
    #     return p.parse_args()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ===================================================================================================
    # FOR RF
    
    
    # def parse_args():
        # p = argparse.ArgumentParser()
        # p.add_argument("--dataset_name", type=str, default="Epilepsy")
        # p.add_argument("--n_est", type=int, default=80)
        # p.add_argument("--max_depth", type=int, default=20)
        # p.add_argument("--num_exits", type=int, default=3)

        # p.add_argument("--tree_splits", type=float, nargs="+", default=[0.31, 0.54, 1])
        # p.add_argument("--proportions", type=float, nargs="+", default=[0.39, 0.57, 1])   # split_points
        # p.add_argument("--th_combination", type=float, nargs="+", default=[0.34657359027997264, 1.3862943611198906])    # th_list
        # return p.parse_args()
    
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