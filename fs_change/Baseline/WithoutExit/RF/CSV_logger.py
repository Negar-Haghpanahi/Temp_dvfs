# CSV_logger.py
import csv
import os


class ExitCSVLogger:
    def __init__(self, save_path, max_exits):
        self.save_path = save_path
        self.max_exits = int(max_exits)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        self.exit_cols = [f"exit_{i+1}" for i in range(self.max_exits)]
        self.factor_used_cols = [f"factor_used_{i+1}" for i in range(self.max_exits)]
        self.continued_cols = [f"continued_{i+1}" for i in range(self.max_exits)]

        self.split_point_cols = [f"split_point_{i+1}" for i in range(self.max_exits)]
        self.stage_time_cols = [f"stage_time_{i+1}_sec" for i in range(self.max_exits)]

        self.model_size_cols = [
            "total_nodes",
            "total_memory_kb",
            "total_pickle_kb",
        ]

        self.exit_nodes_cols = [f"exit_{i+1}_nodes" for i in range(self.max_exits)]
        self.exit_memory_cols = [f"exit_{i+1}_memory_kb" for i in range(self.max_exits)]
        self.exit_pickle_cols = [f"exit_{i+1}_pickle_kb" for i in range(self.max_exits)]
        self.exit_estimators_cols = [f"exit_{i+1}_n_estimators" for i in range(self.max_exits)]
        self.exit_depth_cols = [f"exit_{i+1}_avg_depth" for i in range(self.max_exits)]

        self.fieldnames = (
            ["config_id", "dataset", "Num_exits", "max_depth", "n_estimators",
             "train_acc", "test_acc",
             "sample_id", "true_label", "pred_label",
             "start_factor",
             "exit_stage",
             "fs_base", "window_len",
             "split_points",
             "factor_path_used", "factor_path_continued"]
            + self.exit_cols
            + self.factor_used_cols
            + self.continued_cols
            + self.stage_time_cols
            + self.model_size_cols
            + self.exit_nodes_cols
            + self.exit_memory_cols
            + self.exit_pickle_cols
            + self.exit_estimators_cols
            + self.exit_depth_cols
        )

        need_header = (not os.path.exists(self.save_path)) or (os.path.getsize(self.save_path) == 0)
        if need_header:
            with open(self.save_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                w.writeheader()

    def append_rows(self, rows):
        with open(self.save_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            w.writerows(rows)

    @staticmethod
    def _path_from_cols(row, prefix, upto):
        vals = []
        for s in range(1, upto + 1):
            v = int(row.get(f"{prefix}_{s}", -1))
            if v != -1:
                vals.append(v)
        return "->".join(map(str, vals))

    @staticmethod
    def _make_baseline_debug():
        """
        Baseline no-exit RF:
        pretend it "exits" once at stage 1 using full window.
        """
        return {
            "start_factor": 1,
            "stages": [
                {"stage": 1, "factor_used": 1, "continued_with_factor": -1, "exit": True}
            ]
        }

    def build_rows_for_config(
        self,
        config_id,
        config_dict,
        y_test,
        preds,
        debugs,
        train_acc,
        test_acc,
        fs_base,
        window_len,
        split_points=None,
        model_size_info=None,
    ):
        dataset = config_dict.get("dataset_name", "UNKNOWN")

        # Baseline configs may not have Num_exits. Default to 1.
        num_exits = int(config_dict.get("Num_exits", 1))

        max_depth = config_dict.get("max_depth", None)
        n_estimators = config_dict.get("n_estimators", None)

        rows = []
        full_window_sec = float(window_len) / float(fs_base)

        # If caller didn't provide split_points (baseline), use [1.0]
        if split_points is None:
            split_points = [1.0]
        split_points = [float(x) for x in split_points]

        # If caller didn't provide debugs (baseline), synthesize them
        if debugs is None or len(debugs) == 0:
            debugs = [self._make_baseline_debug() for _ in range(len(preds))]

        # ---- Model size info handling (same for all rows) ----
        model_size_data = {}
        if model_size_info is not None:
            model_size_data['total_nodes'] = model_size_info.get('total_nodes', -1)
            model_size_data['total_memory_kb'] = model_size_info.get('total_memory_kb', -1)
            model_size_data['total_pickle_kb'] = model_size_info.get('total_pickle_kb', -1)

            per_exit = model_size_info.get('per_exit_details', [])
            for exit_detail in per_exit:
                exit_num = exit_detail.get('exit', 1)
                model_size_data[f'exit_{exit_num}_nodes'] = exit_detail.get('nodes', -1)
                model_size_data[f'exit_{exit_num}_memory_kb'] = exit_detail.get('memory_kb', -1)
                model_size_data[f'exit_{exit_num}_pickle_kb'] = exit_detail.get('pickle_size_kb', -1)
                model_size_data[f'exit_{exit_num}_n_estimators'] = exit_detail.get('n_estimators', -1)
        else:
            model_size_data['total_nodes'] = -1
            model_size_data['total_memory_kb'] = -1
            model_size_data['total_pickle_kb'] = -1
            for i in range(self.max_exits):
                model_size_data[f'exit_{i+1}_nodes'] = -1
                model_size_data[f'exit_{i+1}_memory_kb'] = -1
                model_size_data[f'exit_{i+1}_pickle_kb'] = -1
                model_size_data[f'exit_{i+1}_n_estimators'] = -1

        # ---- Per-sample rows ----
        for i, dbg in enumerate(debugs):
            stages = dbg.get("stages", [])
            stage_map = {int(st["stage"]): st for st in stages if "stage" in st}
            stages_sorted = [stage_map[s] for s in sorted(stage_map.keys())]

            exit_flags = {f"exit_{j+1}": -1 for j in range(self.max_exits)}
            factor_used_cols = {f"factor_used_{j+1}": -1 for j in range(self.max_exits)}
            continued_cols = {f"continued_{j+1}": -1 for j in range(self.max_exits)}
            stage_time_cols = {f"stage_time_{j+1}_sec": -1.0 for j in range(self.max_exits)}

            exit_stage = -1

            for st in stages_sorted:
                s = int(st["stage"])
                if s <= self.max_exits:
                    if st.get("factor_used", None) is not None:
                        factor_used_cols[f"factor_used_{s}"] = int(st["factor_used"])

                    cf = st.get("continued_with_factor", None)
                    if cf is not None:
                        continued_cols[f"continued_{s}"] = int(cf)

                if bool(st.get("exit", False)) and exit_stage == -1:
                    exit_stage = s
                    if s <= self.max_exits:
                        exit_flags[f"exit_{s}"] = 1

            # compute per-stage time up to exit_stage
            if exit_stage != -1:
                prev_p = 0.0
                for s in range(1, exit_stage + 1):
                    if s - 1 >= len(split_points):
                        break
                    p = float(split_points[s - 1])
                    fct = int(factor_used_cols.get(f"factor_used_{s}", -1))
                    if fct == -1:
                        prev_p = p
                        continue
                    seg_prop = p - prev_p
                    if seg_prop < 0:
                        break
                    stage_time_cols[f"stage_time_{s}_sec"] = seg_prop * full_window_sec
                    prev_p = p

            start_factor = dbg.get("start_factor", -1)

            row = {
                "config_id": int(config_id),
                "dataset": dataset,
                "Num_exits": num_exits,
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "sample_id": int(i),
                "true_label": int(y_test[i]),
                "pred_label": int(preds[i]),
                "start_factor": int(start_factor),
                "exit_stage": int(exit_stage),
                "fs_base": float(fs_base),
                "window_len": int(window_len),
                "split_points": split_points,
            }

            row.update(exit_flags)
            row.update(factor_used_cols)
            row.update(continued_cols)
            row.update(stage_time_cols)
            row.update(model_size_data)

            if exit_stage == -1:
                row["factor_path_used"] = ""
                row["factor_path_continued"] = ""
            else:
                row["factor_path_used"] = self._path_from_cols(row, "factor_used", upto=exit_stage)
                row["factor_path_continued"] = self._path_from_cols(row, "continued", upto=max(0, exit_stage - 1))

            rows.append(row)

        return rows