import random
import pandas as pd
import numpy as np
import itertools


class Config:
    
    def __init__(self):
        self.depths =  [i for i in range(50,100,10)]
        self.treenum_options = [i for i in range(10,101,10)]  
        self.RESTRICTED_POINTS = []

# this func creats the same split as Sirine has
def generate_percentages(num_exits): # this will generate multiple tree splits or data percentages
    #wid = 0.9 / num_exits
    if num_exits == 2:  
        wid = 0.5
    elif num_exits == 3:
        wid = 0.25
    else:
        wid = 0.16
    list_of_percentages = []

    for _ in range(10): 
        per_list = []
        current = 0
        #current += 0.5 # first exit
        current += random.uniform(0.3,0.5) # first exit
        per_list.append(round(current, 2))
        for _ in range(num_exits - 2):
            current += random.uniform(0.1, wid)
            per_list.append(round(current, 2))
        per_list.append(1)
        list_of_percentages.append(tuple(per_list))  # convert to tuple for hashing

    unique_set = set(list_of_percentages)
    return [list(p) for p in unique_set]  # convert tuples back to lists


def entropy(proba, eps=1e-12):
    p = np.clip(proba, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def max_entropy(num_classes):
        probabilities = np.ones(num_classes) / num_classes
        return -np.sum(probabilities * np.log(probabilities))
    

def func_threshold_combinations(num_exits, dataset_max_entropy):
        num_bins = 4
        bin_width = dataset_max_entropy / num_bins
        thresholds = [(i + 1) * bin_width for i in range(num_bins)]  # e.g., [0.5, 1.0, 1.5, 2.0]

        # Generate permutations
        permutations = [list(p) for p in itertools.permutations(thresholds, num_exits -1 )] 

        # Shuffle to randomize selection
        random.shuffle(permutations)

        return permutations

def margin(proba, eps=1e-12):
    p = np.clip(proba, eps, 1.0)
    p_sorted = np.sort(p)[::-1]
    if len(p_sorted) < 2:
        return 0.0
    return float(p_sorted[0] / p_sorted[1])

def max_margin():
    return 1.0

# def func_gamma_monotone(num_exits, gammas=None):
#     """
#     Generate margin threshold lists (gamma_list) of length num_exits-1.
#     We make gamma decrease with stage (strict early, relaxed later).
#     """
#     if gammas is None:
#         gammas = [0.30, 0.20, 0.10, 0.05]  
#     combs = []
#     for combo in itertools.combinations(gammas, num_exits - 1):
#         combs.append(sorted(list(combo), reverse=True))
#     random.shuffle(combs)
#     return combs


def generate_configurations(dataset_name):
    
    l=[]
    dataset_name = dataset_name[0]    
   
    random.seed(42)
    exits = [2,3,4] 
    if dataset_name == 'Shoaib':
        n_classes = 7
        my_list = [(50,60), (60,60), (70,75), (50,80)]
        fs_base = 50
    if dataset_name == 'Epilepsy':
        n_classes = 4
        my_list = [(10,85), (15,90), (20,80), (25,80)] 
        fs_base = 250
    if dataset_name == 'EMGPhysical':
        n_classes = 4
        my_list = [(25,40), (35,40), (40,30), (35,30)]
        fs_base = 100
    if dataset_name == 'SelfRegulationSCP1':
        n_classes = 2
        my_list = [(10,20), (20,20), (30,15), (20,15)]
        fs_base = 256
    if dataset_name == 'WESADchest':
        n_classes = 3
        my_list = [(20,8), (40,6), (60,4)]
        fs_base = 700
    if dataset_name == 'PAMAP2':
        n_classes = 5
        my_list = [(15,30), (20,20), (30,10), (30,20)]
        fs_base = 100
        
    if dataset_name == 'wisdm':
        n_classes = 6
        my_list = [(30,80), (20,70), (15,60), (40,85)]
        fs_base = 20
            
    if dataset_name == 'ACCGyro':
        n_classes = 2
        my_list = [(50,60), (60,60), (70,75), (50,80)]
        fs_base = 10
    
    if dataset_name == 'wharDataOriginal':
        n_classes = 8
        my_list = [(30,80), (20,70), (15,60), (40,85)]
        fs_base = 50    
    
    for (depth, n_est) in my_list:
        for num_exits in exits:
            entropy_max = max_entropy(n_classes)
            th_combinations_list = func_threshold_combinations(num_exits, entropy_max)
            #th_combinations = [th_combinations_list[0], th_combinations_list[1]]
            th_combinations = random.sample(th_combinations_list,2) # extract randomly to ensure that the total set of final configs has ascending, descending, and random order
            # th_combinations = th_combinations_list[0]

            list_tree_splits_l = generate_percentages(num_exits)
            # #list_tree_splits = [list_tree_splits_l[0], list_tree_splits_l[1]]
            list_tree_splits = random.sample(list_tree_splits_l,1)

            list_data_percentages_l = generate_percentages(num_exits)
            #list_data_percentages = [list_data_percentages_l[0], list_data_percentages_l[1]]
            list_data_percentages = random.sample(list_data_percentages_l,1)
            
            
            # gamma_combinations_list = func_gamma_monotone(num_exits)
            # gamma_combinations = random.sample(gamma_combinations_list, 2)


            for tree_splits in list_tree_splits:
                for proportions in list_data_percentages:
                    dictionary = {'dataset_name': dataset_name, 'Num_exits': num_exits, 'max_depth': depth, 'tree_splits': tree_splits, 'split_points': proportions, 'fs_base': fs_base, 'th': th_combinations, 'n_estimators': n_est}
                    l.append(dictionary)



    return l


def generate_configurations_gb(dataset_name):
    l=[]
    dataset_name = dataset_name[0]    
   
    random.seed(42)
    exits = [2,3,4] 
    if dataset_name == 'Shoaib':
        n_classes = 7
        my_list = [(3,100), (4,150), (5,200), (3,250)]  # (max_depth, n_estimators)
        fs_base = 50
    if dataset_name == 'Epilepsy':
        n_classes = 4
        my_list = [(3,150), (4,200), (5,150), (4,250)] 
        fs_base = 250
    if dataset_name == 'EMGPhysical':
        n_classes = 4
        my_list = [(3,100), (4,150), (5,100), (4,100)]
        fs_base = 100
    if dataset_name == 'SelfRegulationSCP1':
        n_classes = 2
        my_list = [(3,100), (4,100), (5,80), (4,80)]
        fs_base = 256
    if dataset_name == 'WESADchest':
        n_classes = 3
        my_list = [(3,50), (4,80), (5,100)]
        fs_base = 700
    if dataset_name == 'PAMAP2':
        n_classes = 5
        my_list = [(3,150), (4,100), (5,100), (4,150)]
        fs_base = 100
        
    if dataset_name == 'wisdm':
        n_classes = 6
        my_list = [(3,200), (4,150), (5,120), (3,250)]
        fs_base = 20
            
    if dataset_name == 'ACCGyro':
        n_classes = 2
        my_list = [(3,100), (4,150), (5,200), (3,250)]
        fs_base = 10
    
    if dataset_name == 'wharDataOriginal':
        n_classes = 8
        my_list = [(3,200), (4,150), (5,120), (3,250)]
        fs_base = 50    
    
    for (depth, n_est) in my_list:
        for num_exits in exits:
            entropy_max = max_entropy(n_classes)
            th_combinations_list = func_threshold_combinations(num_exits, entropy_max)
            th_combinations = random.sample(th_combinations_list, 2)

            list_tree_splits_l = generate_percentages(num_exits)
            list_tree_splits = random.sample(list_tree_splits_l, 1)

            list_data_percentages_l = generate_percentages(num_exits)
            list_data_percentages = random.sample(list_data_percentages_l, 1)
            
            # gamma_combinations_list = func_gamma_monotone(num_exits)
            # gamma_combinations = random.sample(gamma_combinations_list, 2)

            for tree_splits in list_tree_splits:
                for proportions in list_data_percentages:
                    dictionary = {
                        'dataset_name': dataset_name, 
                        'Num_exits': num_exits, 
                        'max_depth': depth, 
                        'tree_splits': tree_splits, 
                        'split_points': proportions, 
                        'fs_base': fs_base, 
                        'th': th_combinations, 
                        # 'gamma': gamma_combinations, 
                        'n_estimators': n_est,
                        'learning_rate': 0.1  # Added learning_rate parameter for GB
                    }
                    l.append(dictionary)

    return l


def generate_configurations_Baseline_NoExit(dataset_name):
    l=[]
    # dataset_name = dataset_name[0]    
   
    random.seed(42)
    if dataset_name == 'Shoaib':
        n_classes = 7
        my_list = [(3,100), (4,150), (5,200), (3,250)]  
        fs_base = 50
    if dataset_name == 'Epilepsy':
        n_classes = 4
        my_list = [(3,150), (4,200), (5,150), (4,250)] 
        fs_base = 250
        
    if dataset_name == 'wisdm':
        n_classes = 6
        my_list = [(3,200), (4,150), (5,120), (3,250)]
        fs_base = 20
            
    if dataset_name == 'ACCGyro':
        n_classes = 2
        my_list = [(3,100), (4,150), (5,200), (3,250)]
        fs_base = 10
    
    if dataset_name == 'wharDataOriginal':
        n_classes = 8
        my_list = [(3,200), (4,150), (5,120), (3,250)]
        fs_base = 50  
            
    for (depth, n_est) in my_list:
        dictionary = {'dataset_name': dataset_name, 'max_depth': depth, 'fs_base': fs_base, 'n_estimators': n_est}
        l.append(dictionary)

    return l