from Inference_Board import RunInference
import time
from  sensor_control import initialize_bmi160 , sensor_on , sensor_sleep



def Test(X_test , y_test , models ,args,  fs_base , window_len ,split_points):
    
    all_per_sample_results = []
    


    
    
    # Now, loop through each data type for inference and add type-specific metrics
    for w in range(min(200 ,len(X_test))):
        
        #print("w is  --> ", w) 
        
        start_time_infernce = time.time()
        inference_obj = RunInference( fs_base , window_len ,split_points ,  X_test=X_test[w], y_test=y_test[w], models=models, stages=args.proportions , window_num= w )

        sub_forest_entropy, prediction = inference_obj.predict_proba()

        _, _, _, per_sample_results_for_window = inference_obj.check_exit(sub_forest_entropy, args.th_combination, prediction, y_test[w], start_time_infernce)
        
        # all_stage_exit_accuracies = inference_obj.ExitAtAllStage()
        
        end_time_inference = time.time()

     
        for sample_dict in per_sample_results_for_window:
            sample_dict['t_end'] = end_time_inference
            

        
        all_per_sample_results.extend(per_sample_results_for_window)
        

    return all_per_sample_results
