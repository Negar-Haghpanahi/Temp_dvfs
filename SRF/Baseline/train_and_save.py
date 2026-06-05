import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import argparse
from DataLoad import LoadData


def Train_and_Save(dataset_name, n_estimators, tree_depth):
    file_path = f'Baseline\\Datasets\\{dataset_name}_dataLabels.pkl'
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)

    data = data_dict['data']          # data
    labels_array = data_dict['labels'] # labels

    n_window, n_channel, n_data = data.shape  # data shape
    lst = np.arange(n_window)


    loader = LoadData()
    loader.Read(dataset_name)
    loader.SplitData()

    X_train = loader.GetTrainX()
    X_val   = loader.GetValX()
    X_test  = loader.GetTestX()

    y_train = loader.GetYtrain()
    y_val   = loader.GetYval()
    y_test  = loader.GetYtest()

   

    nb_clss = len(np.unique(y_train))

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=tree_depth)

    # Training
    clf.fit(X_train, y_train)

    

    # Save the model
    with open(f"Baseline\\saved_pkl\\{dataset_name}_trained_model.pkl", "wb") as f:
        pickle.dump(clf,f)

    # Save the X_train and y_train and nb_classes
    np.save(f"Baseline\\saved_pkl\\{dataset_name}_test_data.npy", X_test)
    np.save(f"Baseline\\saved_pkl\\{dataset_name}_test_labels.npy", y_test) 
    np.save(f"Baseline\\saved_pkl\\{dataset_name}_nb_classes.npy", np.array([nb_clss]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "RF-V Training")
    parser.add_argument("--dataset_name", type=str, default= "SelfRegulationSCP1", help = "The Dataset name")
    parser.add_argument("--n_est", type=int, default=19, help = "The number of estimators")
    parser.add_argument("--max_depth", type=int, default=10, help = "The max depth")
    args = parser.parse_args()
    Train_and_Save(args.dataset_name, args.n_est, args.max_depth)
  

