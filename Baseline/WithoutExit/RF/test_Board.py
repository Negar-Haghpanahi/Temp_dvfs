import time
from feature_engineering import FeatureEngineer
from sklearn.metrics import accuracy_score
from sensor_control import initialize_bmi160, sensor_on, sensor_sleep

def Test(X_test, y_test, model, args):
    

    all_results = []

    window_len = X_test.shape[2]
    window_time = 3  #float(window_len) / float(args.fs_base)
    
    fe = FeatureEngineer()
    for w in range(min(100 ,len(X_test))):
        print(f"w is --> {w}")
        
        
        t_start = time.time()
        sensor_on(verbose=True)
        time.sleep(window_time- 0.2 )
        
        
        X_test_feat = fe.extract_features(X_test[w:w+1])
        t_before = time.time()
        pred = int(model.predict(X_test_feat)[0])
        t_after = time.time()
        print("total time infernce is --> ", t_after-t_before)
     
        t_end = time.time()
        
        row = {
            "t_start": float(t_start),
            "t_end": float(t_end),
            "total": float(t_end - t_start),
            "window_sched_sec": float(window_time),
            "sensor_total_on_sec": float(window_time),
            "sensor_total_off_sec": 0.0,
            "true_label": int(y_test[w]),
            "prediction": int(pred),
            "correctness": int(pred == int(y_test[w])),
            "exit_level": 1,
            "window_num": int(w),
            "data%": 100.0
        }


        all_results.append(row)

    return all_results
