import csv
import os
from collections import defaultdict

class ConfigRuntimeCSVLogger:
    """
    Logger for runtime and model size metrics per configuration
    """
    
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Define all columns
        self.fieldnames = [
            'dataset',
            'config_id',
            "n_estimators",
            "max_depth",
            "th_list",
            "tree_splits_list",
            "split_points",
            
            'train_acc',
            'test_acc',
            'fit_time_sec',
            'train_pred_time_sec',
            'test_pred_time_sec',
            'total_runtime_sec',
            # Model size columns
            'total_nodes',
            'total_leaves',
            'total_memory_kb',
            'total_pickle_kb',
            'num_exits',
            # 'avg_nodes_per_exit',
            # Per-exit node counts (will add dynamically based on max exits)
            'exit_1_nodes',
            'exit_2_nodes',
            'exit_3_nodes',
            'exit_4_nodes',
            # Per-exit memory (KB)
            'exit_1_memory_kb',
            'exit_2_memory_kb',
            'exit_3_memory_kb',
            'exit_4_memory_kb',
            # Per-exit pickle size (KB)
            'exit_1_pickle_kb',
            'exit_2_pickle_kb',
            'exit_3_pickle_kb',
            'exit_4_pickle_kb',
            # Per-exit tree counts
            'exit_1_n_estimators',
            'exit_2_n_estimators',
            'exit_3_n_estimators',
            'exit_4_n_estimators',
            # Per-exit physical tree counts (for GB)
            # 'exit_1_n_physical_trees',
            # 'exit_2_n_physical_trees',
            # 'exit_3_n_physical_trees',
            # 'exit_4_n_physical_trees',
            # Per-exit average depth
            # 'exit_1_avg_depth',
            # 'exit_2_avg_depth',
            # 'exit_3_avg_depth',
            # 'exit_4_avg_depth',

            'exit_1_pct','exit_2_pct','exit_3_pct','exit_4_pct',
        
                    ]
        
        self.rows_buffer = []
        self._write_header()
    
    def _write_header(self):
        """Write CSV header if file doesn't exist"""
        if not os.path.exists(self.save_path) or os.path.getsize(self.save_path) == 0:
            with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                writer.writeheader()
    
    def append(
        self,
        dataset,
        config_id,

        # --- NEW: add these ---
        n_estimators=None,
        max_depth=None,
        th_list=None,
        tree_splits_list=None,
        split_points=None,

        # existing args
        train_acc=None,
        test_acc=None,
        fit_time_sec=0.0,
        train_pred_time_sec=0.0,
        test_pred_time_sec=0.0,
        model_size_info=None,
        exit_pct=None,
        gamma_calib_report=None,
    ):
        total_runtime_sec = fit_time_sec + train_pred_time_sec + test_pred_time_sec

        row = {
            'dataset': dataset,
            'config_id': config_id,
            "n_estimators": -1 if n_estimators is None else int(n_estimators),
            "max_depth": -1 if max_depth is None else int(max_depth),
            "th_list": th_list,
            "tree_splits_list": tree_splits_list,
            "split_points": split_points,
            
            
            'train_acc': train_acc,
            'test_acc': test_acc,
            'fit_time_sec': fit_time_sec,
            'train_pred_time_sec': train_pred_time_sec,
            'test_pred_time_sec': test_pred_time_sec,
            'total_runtime_sec': total_runtime_sec,
        }

        # --- exit percentage per stage ---
        # columns: exit_1_pct ... exit_4_pct
        if exit_pct is not None:
            for k, v in exit_pct.items():
                row[k] = float(v)

        # --- gamma calibration report ---
        # columns like: gamma_1, gamma_1_exit_rate, gamma_1_exit_acc, gamma_1_found, ...
        if gamma_calib_report is not None:
            for item in gamma_calib_report:
                s = int(item["stage"])
                row[f"gamma_{s}"] = float(item.get("gamma", -1))
                row[f"gamma_{s}_exit_rate"] = float(item.get("exit_rate", 0))
                row[f"gamma_{s}_exit_acc"] = float(item.get("exit_acc", 0))
                row[f"gamma_{s}_found"] = int(bool(item.get("found", False)))

        # --- model size info (keep your existing logic) ---
        if model_size_info is not None:
            row['total_nodes'] = model_size_info['total_nodes']
            row['total_leaves'] = model_size_info['total_leaves']
            row['total_memory_kb'] = model_size_info['total_memory_kb']
            row['total_pickle_kb'] = model_size_info.get('total_pickle_kb', -1)

            for exit_detail in model_size_info['per_exit_details']:
                exit_num = exit_detail['exit']
                row[f'exit_{exit_num}_nodes'] = exit_detail['nodes']
                row[f'exit_{exit_num}_memory_kb'] = exit_detail['memory_kb']
                row[f'exit_{exit_num}_pickle_kb'] = exit_detail.get('pickle_size_kb', -1)
                row[f'exit_{exit_num}_n_estimators'] = exit_detail['n_estimators']

        self.rows_buffer.append(row)

        with open(self.save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            writer.writerow(row)
            
    def get_summary(self):
        """Get summary statistics from buffered rows"""
        if not self.rows_buffer:
            return {}
        
        summary = defaultdict(list)
        for row in self.rows_buffer:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    summary[key].append(value)
        
        stats = {}
        for key, values in summary.items():
            stats[key] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            }
        
        return stats