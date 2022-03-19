# import  logistic regression , random forest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# import confusion matrix , classification report, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np 
# compute false negative rate

import pandas as pd
def prepare_training_data(data):
    #print(f'data cols {data.columns}')
    # define sample count 
    pos_count = len(data[data['is_bundled']==1])
    sample_count = int(pos_count*0.8) 

    # sample 
    data_1 = data[data['is_bundled']==1].sample(sample_count)
    data_0 = data[data['is_bundled']==0].sample(sample_count)
    data_train = pd.concat([data_0,data_1]).values
    # prepare data 
    X = data.drop(['is_bundled'], axis=1)
    y = data['is_bundled']
    return X,y

def get_model_results(model,data):
    # prepare data
    X = data.drop(['is_bundled'], axis=1)
    y = data[['is_bundled']]
    # predict
    y_pred = model.predict(X)
    # get index of false negatives
    temp_table = y.copy()
    temp_table['pred'] = y_pred
    fn_index = temp_table[(temp_table['is_bundled']==1)&(temp_table['pred']==0)].index



    # get confusion matrix
    cm = confusion_matrix(y, y_pred)
    # get classification report
    repport = classification_report(y, y_pred)

    # get model features
    features =model.feature_names_in_

    return  cm, repport , features , fn_index

def run_experiment(data,config):
    model_type = config['model_type']
    feat_gen_func = config['feat_gen_func']
    model_params = config['model_params']
    cach_data = config['cach_data']
    data = data.copy()
    

    # define model
    if model_type == 'logistic':
        model = LogisticRegression(**model_params)
    elif model_type == 'randomforest':
        model = RandomForestClassifier(**model_params)
    
    # prepare data
    if isinstance(cach_data, pd.DataFrame): 
        data_enriched = cach_data
    else:
        data_enriched = feat_gen_func(data)
        cach_data = data_enriched
        
    
    X,y = prepare_training_data(data_enriched)
    # fit model
    model.fit(X, y)
    # get results
    cm, repport , features ,fn_index= get_model_results(model,data_enriched)
    return  cm, repport , features ,fn_index, cach_data

