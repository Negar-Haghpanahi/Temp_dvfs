from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train , dataset_name=None):
   
    if dataset_name == 'SelfRegulationSCP1':
        depth = 16
        n_est = 18

    if dataset_name == 'WESADchest':
        depth = 58
        n_est = 4
  
    if dataset_name == "Shoaib":
        depth =   52
        n_est = 77 
       
    if dataset_name == "PAMAP2":
        depth =  21
        n_est = 26 
       
    if dataset_name == "Epilepsy":
        depth = 22
        n_est = 87
    
    if dataset_name == "EMGPhysical":
        depth = 28
        n_est = 37
    
    else:
        depth = 24
        n_est = 200
           
    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model