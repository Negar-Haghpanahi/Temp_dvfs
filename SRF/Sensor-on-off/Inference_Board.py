import time
import concurrent.futures
import numpy as np

from sensor_control import sensor_on, sensor_sleep


def _predict_for_tree(args):
    tree_batch, x_stage, classes_ = args
    scores = np.zeros((1, len(classes_)), dtype=float)
    x_stage = np.atleast_2d(x_stage)

    for tree in tree_batch:
        if tree is None:
            continue
        pred = tree.predict(x_stage)[0]
        matches = np.where(classes_ == pred)[0]
        if matches.size > 0:
            scores[0, matches[0]] += 1.0
    return scores


def full_window_time_sec(window_len, fs_base):
    return float(window_len) / float(fs_base)


def stage_acquisition_times(split_points, window_len, fs_base):
    """Return incremental acquisition time for each stage."""
    total_window_sec = 2 #full_window_time_sec(window_len, fs_base)
    prev = 0.0
    out = []
    for p in split_points:
        p = float(p)
        seg_prop = max(0.0, p - prev)
        out.append(seg_prop * total_window_sec)
        prev = p
    return out


class RunInference:
    def __init__(self, fs_base, window_len, split_points, X_test, y_test, models, stages, window_num):
        self.fs_base = fs_base
        self.window_len = window_len
        self.split_points = list(split_points)
        self.X_test = np.asarray(X_test)
        self.y_test = int(np.asarray(y_test).ravel()[0])
        self.models = models
        self.stages = list(stages)
        self.window_num = window_num

        self.T_window = full_window_time_sec(self.window_len, self.fs_base)
        self.stage_acq_times = stage_acquisition_times(self.split_points, self.window_len, self.fs_base)

        self.classes_ = self._infer_classes()
        self.n_classes_ = len(self.classes_)

        self.stage_entropies = []
        self.stage_predictions = []
        self.stage_finish_times = []
        self.sensor_total_on_sec = 0.0
        self.sensor_total_off_sec = 0.0
        self.compute_total_sec = 0.0

    def _infer_classes(self):
        for stage in self.models:
            model = stage["model"] if isinstance(stage, dict) else stage
            if hasattr(model, "classes_"):
                return np.asarray(model.classes_)
        return np.asarray([self.y_test])

    @staticmethod
    def entropy(probabilities):
        epsilon = 1e-12
        probs = np.clip(probabilities, epsilon, 1.0)
        return -np.sum(probs * np.log(probs), axis=1)

    def _predict_stage_votes(self, stage_model, x_stage):
        trees = getattr(stage_model, "trees_", [])
        if len(trees) == 0:
            return np.zeros((1, self.n_classes_), dtype=float), 0

        batch_size = 10
        tree_batches = [trees[i:i + batch_size] for i in range(0, len(trees), batch_size)]
        args = [(batch, x_stage, self.classes_) for batch in tree_batches]

        max_workers = min(len(args), 4) if len(args) > 0 else 1
        scores = np.zeros((1, self.n_classes_), dtype=float)

        if len(args) == 1:
            scores += _predict_for_tree(args[0])
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for out in executor.map(_predict_for_tree, args):
                    scores += out

        valid_tree_count = sum(1 for t in trees if t is not None)
        return scores, valid_tree_count

    def run_window(self, threshold_list_of_keys, start_time):
        """
        Sensor turns ON once at the beginning of the window.
        For each stage:
          1) wait only for the additional segment acquisition time
          2) compute the stage prediction
          3) check exit immediately
          4) if exit happens, turn sensor OFF immediately
          5) sleep the remaining part of the window with sensor OFF
        """
        t_start_absolute = float(start_time)

        result = {
            "t_start": t_start_absolute,
            "exit_level": -1,
            "total": -1.0,
            "prediction": -1,
            "true_label": self.y_test,
            "correctness": False,
            "window_num": self.window_num,
            "data%": self.stages[-1],
            "sensor_total_on_sec": 0.0,
            "sensor_total_off_sec": 0.0,
        }
        for i in range(1, len(self.models) + 1):
            result[f"t{i}"] = -1.0

        n_features_total = self.X_test.shape[0]
        cumulative_scores = np.zeros((1, self.n_classes_), dtype=float)
        cumulative_tree_count = 0

        sensor_is_on = False
        t_sensor_on_start = None

        try:
            sensor_on()
            sensor_is_on = True
            t_sensor_on_start = time.time()

            for stage_idx, stage_obj in enumerate(self.models):
                seg_wait = float(self.stage_acq_times[stage_idx])

                # Simulate incremental sensing for this stage while the sensor is ON.
                if seg_wait > 0:
                    time.sleep(seg_wait)

                t_compute_start = time.time()

                n_features_to_select = int(np.ceil(n_features_total * float(self.stages[stage_idx])))
                n_features_to_select = max(1, min(n_features_to_select, n_features_total))
                x_stage = self.X_test[:n_features_to_select]

                stage_model = stage_obj["model"] if isinstance(stage_obj, dict) else stage_obj
                stage_scores, stage_tree_count = self._predict_stage_votes(stage_model, x_stage)

                cumulative_scores += stage_scores
                cumulative_tree_count += stage_tree_count

                denominator = max(cumulative_tree_count, 1)
                proba = cumulative_scores / denominator
                stage_entropy = float(self.entropy(proba)[0])
                stage_prediction = int(self.classes_[np.argmax(proba, axis=1)[0]])

                t_stage_finish = time.time()
                self.compute_total_sec += (t_stage_finish - t_compute_start)

                self.stage_entropies.append(stage_entropy)
                self.stage_predictions.append(stage_prediction)
                self.stage_finish_times.append(t_stage_finish)

                stage_num = stage_idx + 1
                result[f"t{stage_num}"] = t_stage_finish

                is_final_stage = (stage_idx == len(self.models) - 1)
                should_exit = is_final_stage

                if not is_final_stage and stage_idx < len(threshold_list_of_keys):
                    threshold = float(threshold_list_of_keys[stage_idx])
                    if stage_entropy < threshold:
                        should_exit = True

                if should_exit:
                    t_before_sleep_command = time.time()

                    if sensor_is_on:
                        sensor_sleep()
                        t_after_sleep_command = time.time()
                        self.sensor_total_on_sec = t_after_sleep_command - t_sensor_on_start
                        sensor_is_on = False
                    else:
                        t_after_sleep_command = t_before_sleep_command

                    
                        
                
                    observed_fraction = float(self.stages[stage_idx])
                    remaining_off_time = max(0.0, self.T_window * (1.0 - observed_fraction))
                    self.sensor_total_off_sec = remaining_off_time

                    if remaining_off_time > 0:
                        time.sleep(remaining_off_time)

                    t_end_absolute = time.time()

                    result["exit_level"] = stage_num
                    result["prediction"] = stage_prediction
                    result["correctness"] = (result["true_label"] == result["prediction"])
                    result["data%"] = self.stages[stage_idx]
                    result["sensor_total_on_sec"] = self.sensor_total_on_sec
                    result["sensor_total_off_sec"] = self.sensor_total_off_sec
                    result["total"] = t_end_absolute - t_start_absolute

                    return result

            # Safety fallback: should never reach here because final stage always exits.
            t_end_absolute = time.time()
            result["exit_level"] = len(self.models)
            result["prediction"] = self.stage_predictions[-1] if self.stage_predictions else -1
            result["correctness"] = (result["true_label"] == result["prediction"])
            result["data%"] = self.stages[-1]
            result["sensor_total_on_sec"] = self.sensor_total_on_sec
            result["sensor_total_off_sec"] = self.sensor_total_off_sec
            result["total"] = t_end_absolute - t_start_absolute
            return result

        finally:
            # Ensure the sensor is not left ON if an exception happens.
            if sensor_is_on:
                try:
                    sensor_sleep()
                    if t_sensor_on_start is not None:
                        self.sensor_total_on_sec = max(self.sensor_total_on_sec, time.time() - t_sensor_on_start)
                except Exception:
                    pass

    # Compatibility wrappers so the rest of your code does not break immediately.
    def predict_proba(self):
        return self.stage_entropies, self.stage_predictions

    def check_exit(self, sub_rf_entropy, threshold_list_of_keys, predictions, y_test, start_time):
        result = self.run_window(threshold_list_of_keys=threshold_list_of_keys, start_time=start_time)
        return [], [self.window_num], [], [result]






# from xml.parsers.expat import model

# import numpy as np
# from sklearn.utils.multiclass import unique_labels
# from sklearn.metrics import accuracy_score
# from ECE import ECE_computation , calculate_confidence_max_prob
# import concurrent.futures , time
# from sensor_control import sensor_on , sensor_sleep


# def _predict_for_tree(args):
#     tree_batch, X_test_temp, n_samples, classes_ = args
#     scores_per_batch = np.zeros((n_samples, len(classes_)))
#     X_test_temp=np.atleast_2d(X_test_temp)
#     for tree in tree_batch:
#         predictions = tree.predict(X_test_temp)
#         for sample_idx in range(n_samples):
#             predicted_class = predictions[sample_idx]
        
#             if np.isin(predicted_class, classes_): 
#                 class_idx = np.where(classes_ == predicted_class)[0][0]
#                 scores_per_batch[sample_idx, class_idx] += 1   
#     return scores_per_batch



# def full_window_time_sec(window_len, fs_base):
#     return float(window_len) / float(fs_base)


# def stage_acquisition_times(split_points, window_len, fs_base):
  
#     T_window = 2  #full_window_time_sec(window_len, fs_base)
#     prev = 0.0
#     out = []
#     for p in split_points:
#         p = float(p)
#         seg_prop = max(0.0, p - prev)
#         out.append(seg_prop * T_window)
#         prev = p
#     return out




# class RunInference:

#     def __init__(self,  fs_base , window_len ,split_points , X_test, y_test, models , stages, window_num):
        
#         self.X_test = X_test 
#         self.y_test = y_test 
#         self.trees_ = None
#         self.n_samples = 1 #self.X_test.shape[0] # it should be 1
#         self.classes_ = None
#         self.models = models
#         self.stages = stages
#         self.srf_entropy = None 
#         self._final_passed_indices_after_check_exit = [] 
#         self._all_exited_indices_after_check_exit = set() 
#         self.window_num = window_num 
#         self.stage_durations = []
#         self.indices_in = []
#         self.indices_out = []
#         self.absolute_time_per_stage = []
#         self.sensor_on = None
#         self.sensor_sleep = None    
#         self.fs_base = fs_base
#         self.window_len = window_len
#         self.split_points = split_points
#         self.sensor_total_on_sec = []
#         self.compute_total_sec = 0.0
#         self.T_window = None
        
#     def entropy(self , probabilities):  
#         epsilon = 1e-5  
#         return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
    
#     def predict_proba(self):
        
#         y_processed = np.array(self.y_test).ravel().astype(int)
#         self.classes_ = unique_labels(y_processed)
#         self.absolute_time_per_stage = []
#         self.n_classes_ = len(self.classes_)
#         n_features_total = self.X_test.shape[0]
#         self.all_scores = np.zeros((self.n_samples, self.n_classes_))
#         self.all_entropy = []
#         self.all_predict = []
#         self.connected_prediction = []
#         self.list_of_Prob = []
        
#         self.stage_durations = [] # Reset time tracking for each call
#         cumulative_time = 0.0 # No longer needed, but good for local tracking
#         BATCH_SIZE = 10
        
#         #*********************
        
#         self.T_window = 2 # full_window_time_sec(self.window_len, self.fs_base)
#         acq_times = stage_acquisition_times(self.split_points, self.window_len, self.fs_base)
        
#         sensor_on()
        
#         for j in range(len(self.models)):
            
#             seg_wait = float(acq_times[j])
#             #time.sleep(seg_wait)
#             self.sensor_total_on_sec.append(seg_wait)
            
#             t_stage_start = time.time()
#             n_features_to_select = int(np.ceil(n_features_total * self.stages[j])) # Use ceil to ensure at least 1
#             n_features_to_select = min(n_features_to_select, n_features_total) # Cap
#             selected_feature_indices = np.arange(n_features_to_select)
        
#             X_test_temp = self.X_test[:n_features_to_select]
#             X_test_temp=np.atleast_2d(X_test_temp)
            
#             srf = self.models[j]['model']
#             self.trees_ = srf.trees_ 
#             scores = np.zeros((self.n_samples, self.n_classes_))  
            
#             # do the prediction parallel
#             tree_batches = [self.trees_[i:i + BATCH_SIZE] for i in range(0, len(self.trees_), BATCH_SIZE)]
#             list_args = [(batch, X_test_temp, 1, self.classes_) for batch in tree_batches]
                
#             max_workers = min(len(list_args), 4)   
#             with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#                 outputs = list(executor.map(_predict_for_tree, list_args))
            
#             for scores_per_tree in outputs:
#                 scores += scores_per_tree

#             self.all_scores += scores
             
#             self.connected_prediction.append(np.argmax(self.all_scores , axis=1)[0])
            
#             denominator = len(self.trees_) * (j + 1)
            
#             proba = self.all_scores / denominator
#             self.list_of_Prob.append(proba)
#             self.srf_entropy =  self.entropy(proba)
            
#             self.all_entropy.append( self.srf_entropy[0])
#             self.all_predict.append(np.argmax(scores , axis=1)[0])
#             t_stage_end = time.time()
#             time_spent_at_stage = t_stage_end - t_stage_start
        
#         # Store the individual duration (this is the tN)
#             self.absolute_time_per_stage.append(None)
        
#         return self.all_entropy , self.all_predict  
    
 
#     def ExitAtAllStage(self):
 
#         accuracies_exit_all = {}
        
#         for j , prediction_per_stage in enumerate(self.connected_prediction):
            
#             accuracy = accuracy_score(self.y_test, prediction_per_stage)
#             accuracies_exit_all[f"accuracy_exit_all_Sample_RF {j+1}"] = f"{accuracy:.4f}"
        
#         return accuracies_exit_all



   
#     def check_exit(self , sub_rf_entropy , threshold_list_of_keys , predictions , y_test, start_time):
    
#         t_start_absolute = start_time
        
#         num_total_stages = len(self.models) 
#         num_thresholds = len(threshold_list_of_keys) 
        
#         # --- Per-Sample Result Initialization ---
#         per_sample_result = {
#             't_start': -1.0, # Will be set to t_start_absolute later
#             'exit_level': -1, 
#             'total': -1.0, 
#             'prediction': -1, 
#             'true_label': y_test, 
#             'correctness': False,
#             'window_num': self.window_num,
#             'data%': self.stages[-1]
#         }
#         # Initialize ALL individual stage time tracking fields to -1.0
#         for i in range(1, num_total_stages + 1):
#             per_sample_result[f't{i}'] = -1.0 
#         # ---------------------------------------

#         exited_stage = -1
        
#         # Loop for stages 1 to N-1
#         for stage_idx in range(num_total_stages - 1):
            
#             stage_num = stage_idx + 1 
        
#             # FIX 3: Get the absolute time when this stage's prediction finished
#             t_exit_absolute = time.time()
#             # FIX 4: Record the absolute time (tN)
#             per_sample_result[f't{stage_num}'] = t_exit_absolute

#             sample_entropy_value = sub_rf_entropy[stage_idx] 
#             if stage_idx == 0:
#                 time_spent_at_stage = self.sensor_total_on_sec[stage_idx]
#             elif stage_idx == 1:
#                 time_spent_at_stage = self.sensor_total_on_sec[stage_idx] + self.sensor_total_on_sec[stage_idx - 1]
#             elif stage_idx == 2:
#                 time_spent_at_stage = self.sensor_total_on_sec[stage_idx] + self.sensor_total_on_sec[stage_idx - 1] + self.sensor_total_on_sec[stage_idx - 2]
                
#             # exit condition check
            
#             if sample_entropy_value < threshold_list_of_keys[stage_idx]:
                
#                 # Sample Exits
#                 time.sleep(time_spent_at_stage-0.2)
#                 sensor_sleep()
#                 remaining_off_time = max(0.0, self.T_window - time_spent_at_stage)
#                 time.sleep(remaining_off_time)


#                 t_exit_absolute = time.time()# FIX 2: Capture absolute exit time

#                 exited_stage = stage_num
                
#                 # Record final exit metrics
#                 per_sample_result['exit_level'] = stage_num
#                 per_sample_result['prediction'] = predictions[stage_idx] 
                
#                 # FIX 3: Total time is (Absolute Exit time - Absolute Start time)
#                 per_sample_result['total'] = t_exit_absolute - t_start_absolute 
                
#                 per_sample_result['correctness'] = (per_sample_result['true_label'] == per_sample_result['prediction'])
#                 per_sample_result['data%'] = self.stages[stage_num - 1]
#                 per_sample_result['sensor_total_on_sec'] = time_spent_at_stage
#                 per_sample_result['sensor_total_off_sec'] = remaining_off_time
#             break 
                
#         # --------------------------------------------------------------------------------
#         # 2. FORCED EXIT AT THE FINAL RF STAGE (RF N)
#         # --------------------------------------------------------------------------------

#         final_rf_idx = num_total_stages - 1 
#         final_stage_num = num_total_stages 

#         if exited_stage == -1: # Only proceed if the sample has not exited yet
            
#             # FIX 6: Get the absolute time for the final stage
#             t_exit_absolute = time.time()
            
#             final_prediction = predictions[final_rf_idx]
            
#             exited_stage = final_stage_num
            
#             final_prediction = predictions[final_rf_idx]
            
#             per_sample_result[f't{stage_num}'] = t_exit_absolute
            
#             # Record exit metrics
#             per_sample_result['exit_level'] = final_stage_num
#             per_sample_result['prediction'] = final_prediction
            
#             # FIX 5: Total time is (Absolute Exit time - Absolute Start time)
#             per_sample_result['total'] = t_exit_absolute - t_start_absolute
            
#             # Record the individual stage time for the final stage (tN)
#             per_sample_result[f't{final_stage_num}'] = t_exit_absolute 
            
#             per_sample_result['correctness'] = (per_sample_result['true_label'] == per_sample_result['prediction'])
#             per_sample_result['data%'] = self.stages[stage_num]
#             per_sample_result['sensor_total_on_sec'] = sum(self.sensor_total_on_sec)
#             per_sample_result['sensor_total_off_sec'] = 0.0
#         # --------------------------------------------------------------------------------
#         # 3. Finalize and Return
#         # --------------------------------------------------------------------------------
        
#         # FIX 6: Ensure t_start in the dict is the absolute start time
#         per_sample_result['t_start'] = t_start_absolute 
        
#         indices_in = []
#         indices_out = [self.window_num]
#         all_keys_inference_metrics = [] 
#         per_sample_results_list = [per_sample_result]
        
#         return indices_in, indices_out, all_keys_inference_metrics, per_sample_results_list
       
