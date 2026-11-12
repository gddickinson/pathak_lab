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
from sklearn.metrics import confusion_matrix
try:
    from sklearn.metrics import plot_confusion_matrix
except ImportError:  # removed in scikit-learn 1.2
    from sklearn.metrics import ConfusionMatrixDisplay as plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import power_transform, PowerTransformer, StandardScaler


from tqdm import tqdm
import os, glob, sys

# predict_SPT_class lives in 2_analysis/svmClassification.py (this file used to hold a copy of it)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '2_analysis'))
from svmClassification import predict_SPT_class
from pipeline_defaults import FEATURE_TRANSFORM


def classifyTracks(tracksList, train_data_path, level='', feature_transform=FEATURE_TRANSFORM):
    
    for pred_data_path in tqdm(tracksList):
        exptName = os.path.basename(pred_data_path).split('_MMStack')[0]
        predict_SPT_class(train_data_path, pred_data_path, exptName, level, feature_transform=feature_transform)
    


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