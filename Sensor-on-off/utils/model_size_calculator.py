import numpy as np
import pickle
import sys

class ModelSizeCalculator:
    """
    Calculate model size (nodes and memory) for Random Forest or Gradient Boosting models
    
    IMPORTANT: Handles GB's 2D estimators_ array correctly!
    - For GB multi-class: estimators_ has shape (n_estimators, n_classes)
    - We report n_estimators (boosting iterations) but calculate memory for ALL trees
    """
    
    def __init__(self, n_classes):
        """
        Args:
            n_classes: Number of classes in the classification problem
        """
        self.n_classes = n_classes
        # Memory per node calculation:
        # - feature index: 4 bytes (int32)
        # - threshold: 8 bytes (float64)
        # - impurity: 8 bytes (float64)
        # - n_node_samples: 8 bytes (int64)
        # - weighted_n_node_samples: 8 bytes (float64)
        # - left child: 8 bytes (int64)
        # - right child: 8 bytes (int64)
        # - value array: 8 bytes × n_classes (float64)
        self.bytes_per_node = 52 + (8 * n_classes)
    
    def _is_gradient_boosting(self, model):
        """Check if model is GradientBoostingClassifier"""
        return hasattr(model, 'estimators_') and \
               isinstance(model.estimators_, np.ndarray) and \
               model.estimators_.ndim == 2
    
    def calculate_single_model_size(self, model):
        """
        Calculate nodes and memory for a single RF or GB model
        
        Args:
            model: sklearn RandomForestClassifier or GradientBoostingClassifier
            
        Returns:
            dict with:
                - n_estimators: int (boosting iterations for GB, trees for RF)
                - n_physical_trees: int (actual tree objects stored)
                - total_nodes: int
                - total_leaves: int
                - total_memory_kb: float
                - avg_depth: float
                - max_depth: int
        """
        if not hasattr(model, 'estimators_'):
            raise ValueError("Model doesn't have estimators_")
        
        is_gb = self._is_gradient_boosting(model)
        
        if is_gb:
            # GB: estimators_ is (n_estimators, n_classes) array
            n_estimators = model.estimators_.shape[0]  # Boosting iterations
            n_physical_trees = model.estimators_.size   # Total tree objects
            trees = model.estimators_.flatten()
        else:
            # RF: estimators_ is 1D list/array
            trees = model.estimators_
            n_estimators = len(trees)
            n_physical_trees = len(trees)
        
        # Count nodes, leaves, and depths across ALL physical trees
        total_nodes = 0
        total_leaves = 0
        depths = []
        
        for tree in trees:
            # FIXED: After flatten(), trees are already DecisionTreeRegressor objects
            # No subscripting needed! Just access tree.tree_ directly
            if hasattr(tree, 'tree_'):
                total_nodes += tree.tree_.node_count
                total_leaves += tree.tree_.n_leaves
                depths.append(tree.tree_.max_depth)
        
        # Calculate memory based on ALL physical trees
        total_memory_bytes = total_nodes * self.bytes_per_node
        total_memory_kb = total_memory_bytes / 1024
        
        # Calculate pickle size (actual serialized size)
        try:
            pickled_data = pickle.dumps(model)
            pickle_bytes = len(pickled_data)
            pickle_kb = pickle_bytes / 1024
        except Exception as e:
            # If pickle fails, set to -1
            pickle_kb = -1
        
        # Calculate depth statistics
        avg_depth = np.mean(depths) if depths else 0
        max_depth = max(depths) if depths else 0
        
        return {
            'n_estimators': n_estimators,              # What user configured
            'n_physical_trees': n_physical_trees,      # Actual tree objects
            'is_gradient_boosting': is_gb,
            'total_nodes': total_nodes,
            'total_leaves': total_leaves,
            'total_memory_kb': total_memory_kb,        # Calculated from nodes
            'pickle_size_kb': pickle_kb,               # Actual pickle size
            'avg_nodes_per_tree': total_nodes / n_physical_trees if n_physical_trees > 0 else 0,
            'avg_depth': avg_depth,
            'max_depth': max_depth
        }
    
    def calculate_early_exit_model_size(self, early_exit_model):
        """
        Calculate size for DynamicEarlyExitRF/GB model with multiple exits
        
        Args:
            early_exit_model: Your DynamicEarlyExitRF or DynamicEarlyExitGB instance
                              (must have .models attribute which is a list of RF/GB models)
        
        Returns:
            dict with:
                - total_nodes: Total nodes across ALL exits and ALL physical trees
                - total_memory_kb: Total memory across ALL exits
                - num_exits: Number of exit points
                - per_exit_details: List of dicts, one per exit
        """
        if not hasattr(early_exit_model, 'models'):
            raise ValueError("Model doesn't have 'models' attribute")
        
        total_nodes = 0
        total_leaves = 0
        total_memory_kb = 0
        total_pickle_kb = 0
        total_physical_trees = 0
        per_exit_details = []
        
        for exit_idx, exit_model in enumerate(early_exit_model.models):
            # Calculate size for this exit's model
            exit_size = self.calculate_single_model_size(exit_model)
            
            # Accumulate totals
            total_nodes += exit_size['total_nodes']
            total_leaves += exit_size['total_leaves']
            total_memory_kb += exit_size['total_memory_kb']
            total_pickle_kb += exit_size['pickle_size_kb'] if exit_size['pickle_size_kb'] > 0 else 0
            total_physical_trees += exit_size['n_physical_trees']
            
            # Store per-exit details
            per_exit_details.append({
                'exit': exit_idx + 1,
                'n_estimators': exit_size['n_estimators'],  # Boosting iterations (user's config)
                'n_physical_trees': exit_size['n_physical_trees'],  # Actual trees stored
                'is_gradient_boosting': exit_size['is_gradient_boosting'],
                'nodes': exit_size['total_nodes'],
                'leaves': exit_size['total_leaves'],
                'memory_kb': exit_size['total_memory_kb'],
                'pickle_size_kb': exit_size['pickle_size_kb'],
                'avg_nodes_per_tree': exit_size['avg_nodes_per_tree'],
                'avg_depth': exit_size['avg_depth'],
                'max_depth': exit_size['max_depth']
            })
        
        return {
            'total_nodes': total_nodes,
            'total_leaves': total_leaves,
            'total_memory_kb': total_memory_kb,
            'total_pickle_kb': total_pickle_kb,
            'total_physical_trees': total_physical_trees,
            'num_exits': len(early_exit_model.models),
            'avg_nodes_per_exit': total_nodes / len(early_exit_model.models) if len(early_exit_model.models) > 0 else 0,
            'per_exit_details': per_exit_details
        }
    
    def format_size_summary(self, size_info):
        """
        Format size information as a string for logging
        
        Args:
            size_info: Output from calculate_early_exit_model_size()
            
        Returns:
            str: Formatted summary
        """
        summary = []
        summary.append(f"Total Model Size:")
        summary.append(f"  - Total Nodes: {size_info['total_nodes']:,}")
        summary.append(f"  - Calculated Memory: {size_info['total_memory_kb']:.2f} KB")
        summary.append(f"  - Pickle Size: {size_info.get('total_pickle_kb', -1):.2f} KB")
        summary.append(f"  - Num Exits: {size_info['num_exits']}")
        summary.append(f"  - Avg Nodes per Exit: {size_info['avg_nodes_per_exit']:.2f}")
        
        if 'total_physical_trees' in size_info:
            summary.append(f"  - Total Physical Trees: {size_info['total_physical_trees']}")
        
        # Show overhead if pickle size is available
        if size_info.get('total_pickle_kb', -1) > 0 and size_info['total_memory_kb'] > 0:
            overhead = size_info['total_pickle_kb'] - size_info['total_memory_kb']
            overhead_pct = (overhead / size_info['total_memory_kb']) * 100
            summary.append(f"  - Pickle Overhead: {overhead:.2f} KB ({overhead_pct:.1f}%)")
        
        summary.append(f"\nPer-Exit Breakdown:")
        
        for exit_detail in size_info['per_exit_details']:
            model_type = "GB" if exit_detail['is_gradient_boosting'] else "RF"
            
            # For GB, show both configured estimators and physical trees
            if exit_detail['is_gradient_boosting']:
                tree_info = (f"{exit_detail['n_estimators']} iterations "
                           f"({exit_detail['n_physical_trees']} physical trees)")
            else:
                tree_info = f"{exit_detail['n_estimators']} trees"
            
            pickle_info = f", pickle={exit_detail.get('pickle_size_kb', -1):.2f}KB" if exit_detail.get('pickle_size_kb', -1) > 0 else ""
            
            summary.append(
                f"  Exit {exit_detail['exit']} ({model_type}): "
                f"{exit_detail['nodes']:,} nodes, "
                f"calc_mem={exit_detail['memory_kb']:.2f}KB{pickle_info}, "
                f"{tree_info}, "
                f"avg_depth={exit_detail['avg_depth']:.1f}"
            )
        
        return "\n".join(summary)