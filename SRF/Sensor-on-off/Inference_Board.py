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
    total_window_sec = full_window_time_sec(window_len, fs_base)
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

                if seg_wait > 0:
                    time.sleep(seg_wait-0.01)    # added times off

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
                    if sensor_is_on:
                        sensor_sleep()
                        t_sensor_off = time.time()
                        self.sensor_total_on_sec = t_sensor_off - t_sensor_on_start
                        sensor_is_on = False

                    time_spent_so_far = time.time() - t_start_absolute
                    remaining_off_time = max(0.0, self.T_window - time_spent_so_far-0.01)   #added time off
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

            return result

        finally:
            if sensor_is_on:
                try:
                    sensor_sleep()
                except Exception:
                    pass 


    def predict_proba(self):
        return self.stage_entropies, self.stage_predictions

    def check_exit(self, sub_rf_entropy, threshold_list_of_keys, predictions, y_test, start_time):
        result = self.run_window(threshold_list_of_keys=threshold_list_of_keys, start_time=start_time)
        return [], [self.window_num], [], [result]
