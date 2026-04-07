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
        ]
        
        self.rows_buffer = []
        self._write_header()
    
    def _write_header(self):
        """Write CSV header if file doesn't exist"""
        if not os.path.exists(self.save_path) or os.path.getsize(self.save_path) == 0:
            with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                writer.writeheader()
    
    def append(self, dataset, config_id, train_acc, test_acc, 
               fit_time_sec, train_pred_time_sec, test_pred_time_sec,
               model_size_info=None):
        """
        Append a configuration's metrics
        
        Args:
            dataset: str, dataset name
            config_id: int, configuration ID
            train_acc: float, training accuracy
            test_acc: float, test accuracy
            fit_time_sec: float, training time in seconds
            train_pred_time_sec: float, train prediction time in seconds
            test_pred_time_sec: float, test prediction time in seconds
            model_size_info: dict, output from ModelSizeCalculator.calculate_early_exit_model_size()
        """
        total_runtime_sec = fit_time_sec + train_pred_time_sec + test_pred_time_sec
        
        row = {
            'dataset': dataset,
            'config_id': config_id,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'fit_time_sec': fit_time_sec,
            'train_pred_time_sec': train_pred_time_sec,
            'test_pred_time_sec': test_pred_time_sec,
            'total_runtime_sec': total_runtime_sec,
        }
        
        # Add model size information if provided
        if model_size_info is not None:
            row['total_nodes'] = model_size_info['total_nodes']
            row['total_leaves'] = model_size_info['total_leaves']
            row['total_memory_kb'] = model_size_info['total_memory_kb']
            row['total_pickle_kb'] = model_size_info.get('total_pickle_kb', -1)
            # row['total_physical_trees'] = model_size_info.get('total_physical_trees', -1)
            row['num_exits'] = model_size_info.get('num_exits', -1)
            # row['avg_nodes_per_exit'] = model_size_info['avg_nodes_per_exit']
            
            # Add per-exit details
            for exit_detail in model_size_info['per_exit_details']:
                exit_num = exit_detail['exit']
                row[f'exit_{exit_num}_nodes'] = exit_detail['nodes']
                row[f'exit_{exit_num}_memory_kb'] = exit_detail['memory_kb']
                row[f'exit_{exit_num}_pickle_kb'] = exit_detail.get('pickle_size_kb', -1)
                row[f'exit_{exit_num}_n_estimators'] = exit_detail['n_estimators']
                # row[f'exit_{exit_num}_n_physical_trees'] = exit_detail.get('n_physical_trees', -1)
                # row[f'exit_{exit_num}_avg_depth'] = exit_detail['avg_depth']
        
        self.rows_buffer.append(row)
        
        # Write to file
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