from Inference_Board import RunInference
import time


def full_window_time_sec(window_len, fs_base):
    return float(window_len) / float(fs_base)


def stage_acquisition_times(split_points, window_len, fs_base):
  
    T_window = full_window_time_sec(window_len, fs_base)
    prev = 0.0
    out = []
    for p in split_points:
        p = float(p)
        seg_prop = max(0.0, p - prev)
        out.append(seg_prop * T_window)
        prev = p
    return out


def Test(X_test , y_test , models ,args,  fs_base , window_len ,split_points ):
    
    all_per_sample_results = []
    


    T_window = full_window_time_sec(window_len, fs_base)
    acq_times = stage_acquisition_times(split_points, window_len, fs_base)
    
    # Now, loop through each data type for inference and add type-specific metrics
    for w in range(len(X_test)):
        
        sensor_total_on_sec = 0.0
        compute_total_sec = 0.0
        
        start_time_infernce = time.time()
        inference_obj = RunInference(X_test=X_test[w], y_test=y_test[w], models=models, stages=args.proportions , window_num= w)

        sub_forest_entropy, prediction = inference_obj.predict_proba()

        _, _, _, per_sample_results_for_window = inference_obj.check_exit(sub_forest_entropy, args.th_combination, prediction, y_test[w], start_time_infernce)
        
        # all_stage_exit_accuracies = inference_obj.ExitAtAllStage()
        
        end_time_inference = time.time()

     
        for sample_dict in per_sample_results_for_window:
            sample_dict['t_end'] = end_time_inference
            

        
        all_per_sample_results.extend(per_sample_results_for_window)
        

    return all_per_sample_results