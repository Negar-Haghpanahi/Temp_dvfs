import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from features.feature_engineering import FeatureEngineer
from preprocessing.generateConfiguration import entropy
import logging

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
   

    def __init__(self, split_points, th_list, gamma_list, tree_splits_list, rf_params=None):
        self.split_points = list(split_points)
        self.num_stages = len(self.split_points)

        self.th_list = list(th_list)
        self.gamma_list = list(gamma_list)

        self.tree_splits = tree_splits_list
        self.rf_params = rf_params or {}

        self.fe = FeatureEngineer()
        self.models = []
        self.number_of_tress_per_forest = []

        
        allocated_trees = 0
        for k in range(len(self.tree_splits)):
            if k == len(self.tree_splits) - 1:
                num_estimators = self.rf_params.get("n_estimators", 300) - allocated_trees
            else:
                num_estimators = max(
                    1,
                    int(self.tree_splits[k] * self.rf_params.get("n_estimators", 300)) - allocated_trees
                )
                allocated_trees += num_estimators
            self.number_of_tress_per_forest.append(num_estimators)

    def fit(self, X_train_full, y_train):
        self.num_classes = int(len(np.unique(y_train)))
        self.models = []
        K = len(self.split_points)

        
        segments = []
        segments.append(slice_segment(X_train_full, 0.0, self.split_points[0]))
        for i in range(1, K):
            segments.append(slice_segment(X_train_full, self.split_points[i - 1], self.split_points[i]))

        
        for k in range(K):
            x_acc = segments[0]
            for seg_idx in range(1, k + 1):
                x_acc = concat_time(x_acc, segments[seg_idx])

            feat = self.fe.extract_features(x_acc)

            stage_params = dict(self.rf_params)
            stage_params["n_estimators"] = self.number_of_tress_per_forest[k]

            rf = GradientBoostingClassifier(**stage_params)
            rf.fit(feat, y_train)
            self.models.append(rf)

    def predict_one(self, x_full_one, sample_id=-1, print_trace=False, start_factor=None):
       
        debug = {"stages": [], "start_factor": 1}
        K = len(self.split_points)

        # stage 1 accumulated segment
        seg0 = slice_segment(x_full_one, 0.0, self.split_points[0])
        x_acc = seg0

        for k in range(K):
            rf = self.models[k]
            sp = self.split_points[k]

           
            if k >= 1:
                prev_sp = self.split_points[k - 1]
                new_seg = slice_segment(x_full_one, prev_sp, sp)
                x_acc = concat_time(x_acc, new_seg)

            feat = self.fe.extract_features(x_acc)
            proba = rf.predict_proba(feat)[0]
            H = entropy(proba)
            pred = int(np.argmax(proba))

            p_sorted = np.sort(proba)[::-1]
            p_clipped = np.clip(p_sorted, 1e-6, 1.0)
            m = float(p_clipped[0] / p_clipped[1])

            if k < K - 1:
                tau = float(self.th_list[k])
                gamma = float(self.gamma_list[k])
                exit_now = (H <= tau) and (m >= gamma)
            else:
                tau = None
                gamma = None
                exit_now = True  # last forced

            stage_info = {
                "stage": k + 1,
                "split_point": float(sp),
                "factor_used": 1,                 # always full-res
                "entropy": float(H),
                "tau": None if tau is None else float(tau),
                "margin": float(m),
                "gamma": None if gamma is None else float(gamma),
                "exit": bool(exit_now),
                "continued_with_factor": None if exit_now else 1,  # always continue full-res
            }

            if print_trace:
                if tau is not None:
                    print(
                        f"[Sample {sample_id}] Stage {k+1} | sp={sp} | "
                        f"factor_used=1 | H={H:.4f} | tau={tau:.4f} | "
                        f"m={m:.4f} | gamma={gamma:.4f} | exit={exit_now}"
                    )
                else:
                    print(
                        f"[Sample {sample_id}] Stage {k+1} (LAST) | sp={sp} | "
                        f"factor_used=1 | H={H:.4f} | forced exit=True"
                    )

            debug["stages"].append(stage_info)

            if exit_now:
                return pred, debug

        return pred, debug

    def predict(self, X_test, print_trace=False, return_debugs=False, start_factor=None, cross_sample_adapt=False):
       
        preds, debugs = [], []
        for i in range(len(X_test)):
            x_one = X_test[i:i + 1]
            pred, dbg = self.predict_one(
                x_one,
                sample_id=i,
                print_trace=print_trace,
                start_factor=1
            )
            preds.append(pred)
            debugs.append(dbg)

        preds = np.array(preds)
        if return_debugs:
            return preds, debugs
        return preds