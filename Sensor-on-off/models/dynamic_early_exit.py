import numpy as np
from sklearn.ensemble import RandomForestClassifier
from features.feature_engineering import FeatureEngineer
from preprocessing.generateConfiguration import entropy
from Controller.controller import EntropyGapController
from preprocessing.downsampling import data_downsampler
from itertools import product
import logging ,os , csv , time

logger = logging.getLogger("DynamicEarlyExit")

def slice_segment(X_full, start_prop, end_prop):
    T = X_full.shape[2]
    a = int(round(T * start_prop))
    b = int(round(T * end_prop))
    a = max(0, min(a, T))
    b = max(0, min(b, T))
    if b <= a:
        b = min(T, a + 1)
    return X_full[:, :, a:b]

def concat_time(X_acc, X_new):
    return np.concatenate([X_acc, X_new], axis=2)


class DynamicEarlyExitRF:
    def __init__(self, split_points, th_list,gamma_list, tree_splits_list, rf_params=None, factors=(1, 4 , 8, 64), start_factors=(1, 4 ,8, 64), default_start_factor=1):
        self.split_points = list(split_points)
        self.num_stages = len(self.split_points)
        self.th_list = list(th_list)
        self.factors = list(factors)
        self.start_factors = list(start_factors)
        self.default_start_factor = int(default_start_factor)
        self.tree_splits = tree_splits_list 
        self.fe = FeatureEngineer()
        self.ctrl = EntropyGapController()
        self.rf_params = rf_params
        self.models = []
        self.gamma_list = gamma_list
        self.gamma_calib_report   = []
        self.number_of_tress_per_forest = []
        allocated_trees = 0
        for k in range(len(self.tree_splits)):
            if k == len(self.tree_splits) - 1:
                num_estimators = self.rf_params.get("n_estimators", 300) - allocated_trees
            else:
                num_estimators = max(1 , int(self.tree_splits[k] * self.rf_params.get("n_estimators", 300))- allocated_trees)
                allocated_trees += num_estimators

            self.number_of_tress_per_forest.append(num_estimators)


    def _exit_stage_from_debug(self, debug):
       
        return int(debug["stages"][-1]["stage"])
    
    
    def next_start_factor_policy(self ,debug, num_classes,weak_ratio=0.02, strong_ratio=0.10):
        
        stages = debug["stages"]
        last = stages[-1]
        exit_stage = int(last["stage"])

        H_exit = last.get("entropy", None)
        tau_exit = last.get("tau", None)

        # if forced last stage or missing tau => treat hard
        if H_exit is None or tau_exit is None:
            return 1

        H_exit = float(H_exit)
        tau_exit = float(tau_exit)
        delta = max(0.0, tau_exit - H_exit)

        weak = weak_ratio * tau_exit
        strong = strong_ratio * tau_exit

        # used_full = any(int(st.get("factor_used", -1)) == 1 for st in stages) #If a sample ever required factor 1 (full rate) inside its path, it’s a “hard sample”.

        if exit_stage == 1:
            if delta >= strong:
                f = 64
            elif delta >= weak:
                f = 8
            else:
                f = 1
                
        elif exit_stage == 2:
            if delta >= strong:
                f = 8
            elif delta >= weak:
                f = 4                       
            else:
                f = 1
        
        elif exit_stage == 3:
            if delta >= strong:
                f = 4
            else:
                f = 1
                
        else:
            f = 1

        return f
        
    # ---------- TRAIN ----------
    def fit(self, X_train_full, y_train):
        
        self.num_classes = int(len(np.unique(y_train)))
        self.models = []
        K = len(self.split_points)

        segments = []
        segments.append(slice_segment(X_train_full, 0.0, self.split_points[0]))
        for i in range(1, K):
            segments.append(slice_segment(X_train_full, self.split_points[i-1], self.split_points[i]))

        for k in range(K):
            X_rows, y_rows = [], []

            per_segment_factor_sets = []
            per_segment_factor_sets.append(self.start_factors) 
            for _ in range(1, k+1):
                per_segment_factor_sets.append(self.factors)    

            # enumerate combinations
            for factor_tuple in product(*per_segment_factor_sets):
                
                f0 = factor_tuple[0]
                x_acc = data_downsampler(f0, segments[0]).downsample()

           
                for seg_idx in range(1, k+1):
                    f = factor_tuple[seg_idx]
                    seg_ds = data_downsampler(f, segments[seg_idx]).downsample()
                    x_acc = concat_time(x_acc, seg_ds)

                feat = self.fe.extract_features(x_acc)
                X_rows.append(feat)
                y_rows.append(y_train)

            X_stage = np.vstack(X_rows)
            y_stage = np.concatenate(y_rows)

            stage_params = dict(self.rf_params)
            stage_params["n_estimators"] = self.number_of_tress_per_forest[k]

            rf = RandomForestClassifier(**stage_params)
            rf.fit(X_stage, y_stage)
            self.models.append(rf)
            
            
     
    def predict_one_stage(self, x_full_one, stage_idx,x_acc=None,start_factor=None,factor_next=None,H_prev=None,sample_id=-1,print_trace=False,):
        """
        Run exactly one stage, so board code can control sensor ON/OFF outside.
        """
        if start_factor is None:
            start_factor = self.default_start_factor

        start_factor = int(start_factor)
        K = len(self.split_points)

        stage_t0 = time.time()

        rf = self.models[stage_idx]
        sp = self.split_points[stage_idx]

        if stage_idx == 0:
            seg0 = slice_segment(x_full_one, 0.0, sp)
            x_acc = data_downsampler(start_factor, seg0).downsample()
            factor_used_here = start_factor
        else:
            prev_sp = self.split_points[stage_idx - 1]
            new_seg = slice_segment(x_full_one, prev_sp, sp)

            if factor_next is None:
                factor_next = 1

            new_seg_ds = data_downsampler(int(factor_next), new_seg).downsample()
            x_acc = concat_time(x_acc, new_seg_ds)
            factor_used_here = int(factor_next)

        feat = self.fe.extract_features(x_acc)
        proba = rf.predict_proba(feat)[0]
        H = entropy(proba)
        pred = int(np.argmax(proba))

        p_sorted = np.sort(proba)[::-1]
        p_clipped = np.clip(p_sorted, 1e-6, 1.0)
        m = float(p_clipped[0] / p_clipped[1])

        if stage_idx < K - 1:
            tau = float(self.th_list[stage_idx])
            gamma = float(self.gamma_list[stage_idx])
            exit_now = (H <= tau) and (m >= gamma)
        else:
            tau = None
            gamma = None
            exit_now = True

        continued_with_factor = None
        factor_next_out = factor_next

        if (not exit_now) and (stage_idx < K - 1):
            next_factor = self.ctrl.choose_factor(H, tau, self.num_classes, H_prev=H_prev)
            factor_next_out = int(next_factor)
            continued_with_factor = int(next_factor)

        stage_t1 = time.time()
        stage_time_sec = stage_t1 - stage_t0

        stage_info = {
            "stage": int(stage_idx + 1),
            "split_point": float(sp),
            "factor_used": int(factor_used_here),
            "entropy": float(H),
            "tau": None if tau is None else float(tau),
            "exit": bool(exit_now),
            "continued_with_factor": continued_with_factor,
            "margin": float(m),
            "gamma": None if gamma is None else float(gamma),
            "stage_time_sec": float(stage_time_sec),
        }

        if print_trace:
            if tau is not None:
                print(
                    f"[Sample {sample_id}] Stage {stage_idx+1} | sp={sp} | "
                    f"factor_used={factor_used_here} | H={H:.4f} | tau={tau:.4f} | "
                    f"margin={m:.4f} | exit={exit_now}"
                )
            else:
                print(
                    f"[Sample {sample_id}] Stage {stage_idx+1} (LAST) | sp={sp} | "
                    f"factor_used={factor_used_here} | H={H:.4f} | forced exit=True"
                )

        return pred, stage_info, x_acc, factor_next_out, H, exit_now     
            
    # ---------- INFERENCE (ONE SAMPLE) ----------
    def predict_one(self, x_full_one, sample_id=-1, print_trace=False, start_factor=None):
        debug = {"stages": [], "start_factor": start_factor}
        K = len(self.split_points)

        if start_factor is None:
            start_factor = self.default_start_factor

        start_factor = int(start_factor)

        x_acc = None
        factor_next = None
        H_prev = None

        for k in range(K):
            pred, stage_info, x_acc, factor_next, H_prev, exit_now = self.predict_one_stage(
                x_full_one=x_full_one,
                stage_idx=k,
                x_acc=x_acc,
                start_factor=start_factor,
                factor_next=factor_next,
                H_prev=H_prev,
                sample_id=sample_id,
                print_trace=print_trace,
            )

            debug["stages"].append(stage_info)

            if exit_now:
                return pred, debug

        return pred, debug

    # ---------- INFERENCE (BATCH) ----------
    def predict(self, X_test, print_trace=False, return_debugs=False,start_factor=None, cross_sample_adapt=False):
        preds, debugs = [], []

        if start_factor is None:
            start_factor_state = self.default_start_factor
        else:
            start_factor_state = int(start_factor)

        for i in range(len(X_test)):
            x_one = X_test[i:i+1]

            pred, dbg = self.predict_one(
                x_one,
                sample_id=i,
                print_trace=print_trace,
                start_factor=start_factor_state
            )

            preds.append(pred)
            debugs.append(dbg)

            if cross_sample_adapt and (i < len(X_test) - 1):
                start_factor_state = int(self.next_start_factor_policy(dbg, self.num_classes))

                if print_trace:
                    print(f"[Cross-sample] next sample start_factor -> {start_factor_state}")

        preds = np.array(preds)

        if return_debugs:
            return preds, debugs
        return preds

    # ------ MARGIN ANALYSIS
    # def collect_margin_analysis(self,X_val_full,y_val,config_id,dataset_name,test_acc, exit_stage_map, csv_path,max_stages, use_factor=1, ):
       


    #     K = len(self.split_points)
    #     os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    #     fixed_cols = [
    #         "config_id", "dataset", "num_exits", "test_acc",
    #         "sample_id", "true_label", "actual_exit_stage",
    #     ]
    #     stage_cols = []
    #     for s in range(1, max_stages):   # max_stages-1 non-last stages
    #         stage_cols += [
    #             f"stage_{s}_margin",
    #             f"stage_{s}_entropy",
    #             f"stage_{s}_tau",
    #             f"stage_{s}_pred",
    #             f"stage_{s}_correct",
    #             f"stage_{s}_exit_entropy",
    #         ]

    #     fieldnames   = fixed_cols + stage_cols
    #     write_header = not os.path.exists(csv_path)

    #     N        = X_val_full.shape[0]
    #     all_rows = []

    #     for i in range(N):
    #         x_one = X_val_full[i:i+1]
    #         true  = int(y_val[i])

           
    #         row = {
    #             "config_id":         config_id,
    #             "dataset":           dataset_name,
    #             "num_exits":         K,
    #             "test_acc":          round(float(test_acc), 6),
    #             "sample_id":         i,
    #             "true_label":        true,
    #             "actual_exit_stage": exit_stage_map.get(i, -1),
    #         }
    #         for s in range(1, max_stages):
    #             row[f"stage_{s}_margin"]       = -1
    #             row[f"stage_{s}_entropy"]      = -1
    #             row[f"stage_{s}_tau"]          = -1
    #             row[f"stage_{s}_pred"]         = -1
    #             row[f"stage_{s}_correct"]      = -1
    #             row[f"stage_{s}_exit_entropy"] = -1

           
    #         seg0  = slice_segment(x_one, 0.0, self.split_points[0])
    #         x_acc = data_downsampler(use_factor, seg0).downsample()

    #         for k in range(K - 1):
    #             if k >= 1:
    #                 new_seg    = slice_segment(x_one,
    #                                            self.split_points[k - 1],
    #                                            self.split_points[k])
    #                 new_seg_ds = data_downsampler(use_factor, new_seg).downsample()
    #                 x_acc      = concat_time(x_acc, new_seg_ds)

    #             model_k = self.models[k]
    #             tau_k   = float(self.th_list[k])

    #             feat  = self.fe.extract_features(x_acc)
    #             proba = model_k.predict_proba(feat)[0]
    #             pred  = int(np.argmax(proba))

    #             p_sorted  = np.sort(proba)[::-1]
    #             p_clipped = np.clip(p_sorted, 1e-6, 1.0)
    #             m         = float(p_clipped[0] / p_clipped[1])
    #             H         = float(entropy(proba))
                
                

    #             s = k + 1
    #             row[f"stage_{s}_margin"]       = round(m, 6)
    #             row[f"stage_{s}_entropy"]      = round(H, 6)
    #             row[f"stage_{s}_tau"]          = round(tau_k, 6)
    #             row[f"stage_{s}_pred"]         = pred
    #             row[f"stage_{s}_correct"]      = int(pred == true)
    #             row[f"stage_{s}_exit_entropy"] = int(H <= tau_k)

    #         all_rows.append(row)

       
    #     with open(csv_path, "a", newline="", encoding="utf-8") as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    #         if write_header:
    #             writer.writeheader()
    #         writer.writerows(all_rows)

    #     logger.info(f"[MarginAnalysis] cfg{config_id}: appended {N} rows → {csv_path}")
    #     print(f"[MarginAnalysis] cfg{config_id}: appended {N} rows → {csv_path}")

    #     # self.gamma_list         = [1.0] * (K - 1)   fix this after read the csv
    #     # self.gamma_calib_report = []