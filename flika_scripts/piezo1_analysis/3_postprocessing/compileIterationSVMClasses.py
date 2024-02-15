#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:17:58 2022

@author: george
"""


import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob


def compileSVMresults(path):
    resultsDF = pd.DataFrame()
    fileList = glob.glob(path + '/**/*_locsID_tracksRG_SVMPredicted_cutoff_*.csv', recursive = True)
    for file in tqdm(fileList):
        i = 0
        tempDF = pd.read_csv(file)        
        exptName = tempDF['Experiment'][0]
        cutoff = os.path.splitext(file)[0].split('_')[-1]
        n_SVM_1 = len(tempDF[tempDF['SVM'] == 1].groupby(['track_number']) )
        n_SVM_2 = len(tempDF[tempDF['SVM'] == 2].groupby(['track_number']) )
        n_SVM_3 = len(tempDF[tempDF['SVM'] == 3].groupby(['track_number']) )       
        d = {'exptName':exptName, 'cutoff':cutoff, 'n_SVM_1':n_SVM_1, 'n_SVM_2':n_SVM_2, 'n_SVM_3':n_SVM_3}        
        row = pd.DataFrame(data=d, index=[i])
        i +=1
        
        resultsDF = resultsDF.append(row)
    
    resultsDF['cutoff'] = resultsDF['cutoff'].astype('int')
    resultsDF = resultsDF.sort_values('cutoff')
    
    statsDF = resultsDF.groupby('cutoff').agg(['mean','std'])
          
    return resultsDF, statsDF



if __name__ == '__main__':
    ##### RUN ANALYSIS         
    #path = '/Users/george/Data/tdt_iterate' 
    path = '/Users/george/Data/simulated_200Frames_highDiff'
    
    resultsDF, statsDF = compileSVMresults(path)
    
    resultsDF.plot('cutoff')
    #statsDF.plot()
    
    savename1 = os.path.join(path,'SVM-count_Result')
    savename2 = os.path.join(path,'SVM-count_Stats')
    resultsDF.to_csv(savename1)
    statsDF.to_csv(savename2)
    
    
    
    
    
    