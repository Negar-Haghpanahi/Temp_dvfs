from data_loader.load_data import LoadData
from generateConfiguration import generate_configurations_Baseline_NoExit_GB
from CSV_logger import ExitCSVLogger
import argparse
import os ,time 
import numpy as np
from utils.runtime_logger import ConfigRuntimeCSVLogger
from utils.logger import setup_logger
from utils.model_size_calculator import ModelSizeCalculator
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import FeatureEngineer
from sklearn.metrics import accuracy_score

logger = setup_logger("DynamicEarlyExit")

def run_all_configs(X_train, y_train, X_test, y_test, configs, start_factor=None):
    # one logger for everything (max exits across all configs)
    results_csv_path = f"Baseline/WithoutExit/GB/Results/{configs[0]['dataset_name']}_Dynamic_start_fs_GB.csv"
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)
    
    runtime_csv_path = f"Baseline/WithoutExit/GB/Results/Summary/{configs[0]['dataset_name']}_runtime_per_config_GB.csv"
    runtime_logger = ConfigRuntimeCSVLogger(runtime_csv_path) 
       
    csv_logger = ExitCSVLogger(results_csv_path, max_exits=1)
    
    n_classes = len(np.unique(y_train))
    size_calculator = ModelSizeCalculator(n_classes=n_classes)
    
    logger.info(f"Starting dataset: {configs[0]['dataset_name']}")
    logger.info(f"Total configs: {len(configs)}")
    logger.info(f"Number of classes: {n_classes}")
    
    confg_id = 0
    for cfg_id, config in enumerate(configs):
    
        
            window_len = X_test.shape[2]
            fs_base = config["fs_base"]
            rf_params = {
                "n_estimators": config.get("n_estimators", 100),
                "max_depth": config.get("max_depth", 3),
                "learning_rate": config.get("learning_rate", 0.1), # add this only for gb
                "random_state": 42,
                 #"n_jobs": -1
            }
            t0 = time.perf_counter()
            fe = FeatureEngineer()
            rf = GradientBoostingClassifier(**rf_params)
            New_feat=fe.extract_features(X_train)
            rf.fit(New_feat, y_train)
            fit_time_sec = time.perf_counter() - t0

    
            # ========== CALCULATE MODEL SIZE ==========
            try:
                # baseline RF -> use the baseline wrapper method
                model_size_info = size_calculator.calculate_baseline_model_size(rf)
            except Exception as e:
                logger.error(f"Error calculating model size for config {confg_id}: {e}")
                model_size_info = None

            # train acc
            t0 = time.perf_counter()
            train_preds = rf.predict(New_feat)
            train_acc = accuracy_score(y_train, train_preds)
            train_pred_time_sec = time.perf_counter() - t0

            # test acc (NO return_debug here)
            t0 = time.perf_counter()
            X_test_feat = fe.extract_features(X_test)
            test_preds = rf.predict(X_test_feat)
            test_acc = accuracy_score(y_test, test_preds)
            test_pred_time_sec = time.perf_counter() - t0

            # baseline debugs can be None -> CSV_logger will auto-fill baseline info
            debugs = None

            # runtime logger stays the same
            runtime_logger.append(
                dataset=config["dataset_name"],
                config_id=confg_id,
                train_acc=train_acc,
                test_acc=test_acc,
                fit_time_sec=fit_time_sec,
                train_pred_time_sec=train_pred_time_sec,
                test_pred_time_sec=test_pred_time_sec,
                model_size_info=model_size_info
            )

            rows = csv_logger.build_rows_for_config(
                config_id=confg_id,
                config_dict=config,
                y_test=y_test,
                preds=test_preds,
                debugs=debugs,
                train_acc=train_acc,
                test_acc=test_acc,
                fs_base=fs_base,
                window_len=window_len,
                split_points=[1.0],          # baseline = full window
                model_size_info=model_size_info,
            )
            csv_logger.append_rows(rows)
            confg_id += 1
    logger.info(f"Starting dataset: {configs[0]['dataset_name']}")
    logger.info(f"Total configs: {len(configs)}")
    logger.info(f"Saving CSV to: {results_csv_path}")        
    print(f"\nSaved ALL configs to: {results_csv_path}")

if __name__ == '__main__':
    
    NAMES = ['Epilepsy', 'wisdm','ACCGyro','wharDataOriginal', 'Shoaib']
    for dataset_name in NAMES:
        print(f"DataSet is : {dataset_name}")
        # load data
        loader = LoadData()
        loader.Read(dataset_name)
        loader.SplitData()

        X_train, y_train = loader.X_train, loader.y_train
        X_test, y_test = loader.X_test, loader.y_test

        configs = generate_configurations_Baseline_NoExit_GB(dataset_name)
        # configs = generate_configurations_gb(datasetnames)

        run_all_configs(X_train, y_train, X_test, y_test, configs)

        
        
        print("===================")
    
    