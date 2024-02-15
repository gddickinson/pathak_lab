#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:48:04 2022

@author: george
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob

pixelSize = 0.108


def getStats(df):
    stats = df.agg(
        {
        'n_segments':['mean','std'],
        'track_length':['mean','std'],
        'radius_gyration':['mean','std'],
        'radius_gyration_scaled_nSegments':['mean','std'],
        'radius_gyration_scaled_trackLength':['mean','std'],               
        'asymmetry':['mean','std'],
        'skewness':['mean','std'],
        'kurtosis':['mean','std'],
        'radius_gyration_scaled':['mean','std'],
        'intensity':['mean','std'],
        'lag':['mean','std', 'sum'],
        'fracDimension':['mean','std'],
        'netDispl':['mean','std'],
        'Straight':['mean','std'],
        'nnDist': ['mean','std'],
        'SVM':['mean']
        }
        )
    
    exptName = df['Experiment'][0]
    
    #create df with mean values
    summaryDF_mean = stats.iloc[0].to_frame(name=exptName).T    
    summaryDF_mean = summaryDF_mean.rename(columns={
            'n_segments':'n_segments_mean',
            'track_length':'track_length_mean',            
            'radius_gyration':'radius_gyration_mean',
            'asymmetry':'asymmetry_mean',
            'skewness':'skewness_mean',
            'kurtosis':'kurtosis_mean',
            'radius_gyration_scaled':'radius_gyration_scaled_mean',
            'radius_gyration_scaled_nSegments':'radius_gyration_scaled_nSegments_mean',
            'radius_gyration_scaled_trackLength':'radius_gyration_scaled_trackLength_mean',                     
            'intensity':'intensity_mean',
            'lag':'lag_mean',
            'fracDimension':'fracDimension_mean',
            'netDispl':'netDispl_mean',
            'Straight':'Straight_mean',
            'nnDist': 'nnDist_mean',         
            'SVM': 'SVM_mean'                
            })    
    
    #create df with std values
    summaryDF_std = stats.iloc[1].to_frame(name=exptName).T
    summaryDF_std = summaryDF_std.rename(columns={
            'n_segments':'n_segments_std',
            'track_length':'track_length_std',              
            'radius_gyration':'radius_gyration_std',
            'asymmetry':'asymmetry_std',
            'skewness':'skewness_std',
            'kurtosis':'kurtosis_std',
            'radius_gyration_scaled':'radius_gyration_scaled_std',
            'radius_gyration_scaled_nSegments':'radius_gyration_scaled_nSegments_std',
            'radius_gyration_scaled_trackLength':'radius_gyration_scaled_trackLength_std',
            'intensity':'intensity_std',
            'lag':'lag_std',
            'fracDimension':'fracDimension_std',
            'netDispl':'netDispl_std',
            'Straight':'Straight_std', 
            'nnDist': 'nnDist_std',              
            })  
    
    resultDF = pd.concat([summaryDF_mean,summaryDF_std], axis=1)
    resultDF['n_tracks'] = len(df)
    
    #set column order
    resultDF = resultDF[[
            'n_tracks',
            'n_segments_mean',
            'n_segments_std',
            'track_length_mean',
            'track_length_std',            
            'radius_gyration_mean',
            'radius_gyration_std',            
            'asymmetry_mean',
            'asymmetry_std',            
            'skewness_mean',
            'skewness_std',            
            'kurtosis_mean',
            'kurtosis_std',            
            'radius_gyration_scaled_mean',
            'radius_gyration_scaled_std', 
            'radius_gyration_scaled_nSegments_mean',
            'radius_gyration_scaled_nSegments_std',            
            'radius_gyration_scaled_trackLength_mean',
            'radius_gyration_scaled_trackLength_std',                      
            'intensity_mean',
            'intensity_std',            
            'lag_mean',
            'lag_std',            
            'fracDimension_mean',
            'fracDimension_std',            
            'netDispl_mean',
            'netDispl_std',            
            'Straight_mean',
            'Straight_std',  
            'nnDist_mean',              
            'nnDist_std',             
            'SVM_mean'  
        
        ]]
    return resultDF

def compileTrackResults(exptFolder, SVMclass = 'ALL'):
    trackDF_list = glob.glob(exptFolder + '/**/*_SVMPredicted3_SVM-{}.csv'.format(SVMclass), recursive = True)  

    statsTable = pd.DataFrame()

       
    #add mean values to each experiments df
    for trackFile in tqdm(trackDF_list):
        tempDF = pd.read_csv(trackFile)
        experimentName = tempDF.iloc[0]['Experiment']
        tempDF = tempDF[['track_number','n_segments', 'track_length', 'radius_gyration', 'asymmetry', 'skewness', 'kurtosis','radius_gyration_scaled','radius_gyration_scaled_nSegments', 'radius_gyration_scaled_trackLength','intensity', 'lag', 'meanLag', 'fracDimension', 'netDispl', 'Straight', 'nnDist','SVM']]
        
        #get mean values for each track
        resultDF = tempDF.groupby('track_number', as_index=False).mean()
        
        resultDF['Experiment'] = experimentName
        
        #export track tables        
        resultDF.to_csv(trackFile.split('.csv')[0]+'_trackStats.csv')
        
        #stats for all tracks
        statsTable = statsTable.append(getStats(resultDF))



    #export stats tables   
    exptType = exptFolder.split('/')[-1]
    statsTable.to_csv(os.path.join(exptFolder,exptType+'SVM-{}_trackStats.csv'.format(SVMclass)))
  

    print('{} stats exported'.format(exptType))    
        

if __name__ == '__main__':
    ##### RUN ANALYSIS        
    #path = '/Users/george/Data/10msExposure2s'
    #path = '/Users/george/Data/10msExposure2s_fixed'
    path = '/Users/george/Data/nonbapta_dyetitration' 
 
    #path = '/Users/george/Data/10msExposure2s_new'    
    
    #get expt folder list
    exptList = glob.glob(path + '/*', recursive = True)   

    for exptFolder in tqdm(exptList):
        compileTrackResults(exptFolder, SVMclass = '1')
        compileTrackResults(exptFolder, SVMclass = '2')
        compileTrackResults(exptFolder, SVMclass = '3')
