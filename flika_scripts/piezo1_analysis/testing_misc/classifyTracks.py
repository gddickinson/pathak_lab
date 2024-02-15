#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:23:00 2022

@author: george
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from importlib import reload
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from pathlib import Path
from scipy import stats
from sklearn import datasets, decomposition, metrics
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import power_transform, PowerTransformer, StandardScaler


from tqdm import tqdm
import os, glob


def predict_SPT_class(train_data_path, pred_data_path, exptName, level):
    """Computes predicted class for SPT data where
        1:Mobile, 2:Intermediate, 3:Trapped

    Args:
        train_data_path (str): complete path to training features data file in .csv format, ex. 'C:/data/tdTomato_37Degree_CytoD_training_feats.csv'
                               should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('tdTomato_37Degree'),
                               a 'TrackID' column with unique numerical IDs for each track, and an 'Elected_Label' column derived from human voting.
        pred_data_path (str): complete path to features that need predictions in .csv format, ex. 'C:/data/newconditions/gsmtx4_feature_data.csv'
                               should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('GsMTx4'),
                               and a 'TrackID' column with unique numerical IDs for each track.
    
    Output:
        .csv file of dataframe of prediction_file features with added SVMPredictedClass column output to pred_data_path parent folder
    """
    def prepare_box_cox_data(data):
        data = data.copy()
        for col in data.columns:
            minVal = data[col].min()
            if minVal <= 0:
                data[col] += (np.abs(minVal) + 1e-15)
        return data
    
    train_feats = pd.read_csv(Path(train_data_path), sep=',')
    train_feats = train_feats.loc[train_feats['Experiment'] == 'tdTomato_37Degree']
    train_feats = train_feats[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension', 'Elected_Label']]
    train_feats = train_feats.replace({"Elected_Label":  {"mobile":1,"confined":2, "trapped":3}})
    X = train_feats.iloc[:, 2:-1]
    y = train_feats.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    X_train_, X_test_ = prepare_box_cox_data(X_train), prepare_box_cox_data(X_test)
    X_train_, X_test_ = pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(X_train_), columns=X.columns), pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(X_test_), columns=X.columns)
    
    for col_name in X_train.columns:
        X_train.eval(f'{col_name} = @X_train_.{col_name}')
        X_test.eval(f'{col_name} = @X_test_.{col_name}')
    
    pipeline_steps = [("pca", decomposition.PCA()), ("scaler", StandardScaler()), ("SVC", SVC(kernel="rbf"))]
    check_params = {
        "pca__n_components" : [3],
        "SVC__C" : [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
        "SVC__gamma" : [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 1., 5., 10., 50.0],
    }
    
    pipeline = Pipeline(pipeline_steps)
    create_search_grid = GridSearchCV(pipeline, param_grid=check_params, cv=10)
    create_search_grid.fit(X_train, y_train)
    pipeline.set_params(**create_search_grid.best_params_)
    pipeline.fit(X_train, y_train)
    X = pd.read_csv(Path(pred_data_path), sep=',')
    
    X = X.rename(columns={"radius_gyration" : "radiusGyration",
                      "track_number" : "TrackID",
                      "netDispl" : "NetDispl",
                      "asymmetry" : "Asymmetry",   
                      "kurtosis" : "Kurtosis"                     
                      }) ###GD EDIT
    
    X['Experiment'] = exptName    ###GD EDIT
    
    X = X[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension']]
    X_label = X['Experiment'].iloc[0]
    X_feats = X.iloc[:, 2:]
    X_feats_ = pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(prepare_box_cox_data(X_feats)), columns=X_feats.columns)
    
    for col_name in X_feats.columns:
        X_feats.eval(f'{col_name} = @X_feats_.{col_name}')
    
    X_pred = pipeline.predict(X_feats)
    X['SVMPredictedClass'] = X_pred.astype('int')
    X_outpath = Path(pred_data_path).parents[0] / f'{Path(pred_data_path).stem}_SVMPredicted{level}.csv'
    #X.to_csv(X_outpath, sep=',', index=False)
    
    #add classes to RG file ####GD EDIT
    tracksDF = pd.read_csv(Path(pred_data_path), sep=',')
    tracksDF['Experiment'] = X['Experiment']
    tracksDF['SVM'] = X['SVMPredictedClass']
    
    tracksDF.to_csv(X_outpath, sep=',', index=None)


def classifyTracks(tracksList, train_data_path, level=''):
    
    for pred_data_path in tqdm(tracksList):
        exptName = os.path.basename(pred_data_path).split('_MMStack')[0]
        predict_SPT_class(train_data_path, pred_data_path, exptName, level)
    


if __name__ == '__main__':
    ##### RUN ANALYSIS        
    #path = '/Users/george/Data/10msExposure2s'
    #path = '/Users/george/Data/10msExposure2s_fixed'
    #path = '/Users/george/Data/10msExposure2s_test'    
    #path = '/Users/george/Data/10msExposure2s_new'
    #path = '/Users/george/Data/tdt'
    path = '/Users/george/Data/nonbapta_dyetitration' 
  
    
    #get folder paths
    tracksList = glob.glob(path + '/**/*_tracksRG.csv', recursive = True)   
    
    #training data path
    trainpath = '/Users/george/Data/from_Gabby/gabby_scripts/workFlow/training_data/tdTomato_37Degree_CytoD_training_feats.csv'
    
    #run analysis
    classifyTracks(tracksList, trainpath)