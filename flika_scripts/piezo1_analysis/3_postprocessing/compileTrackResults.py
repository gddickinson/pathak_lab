#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:48:04 2022

@author: george
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

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
        'nnDist_inFrame': ['mean','std'],
        'velocity': ['mean','std'],
        'direction_Relative_To_Origin': ['mean','std'],
        'nnCountInFrame_within_3_pixels': ['mean','std'],
        'nnCountInFrame_within_5_pixels': ['mean','std'],
        'nnCountInFrame_within_10_pixels': ['mean','std'],
        'nnCountInFrame_within_20_pixels': ['mean','std'],
        'nnCountInFrame_within_30_pixels': ['mean','std']
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
            'nnDist_inFrame': 'nnDist_inFrame_mean',
            'velocity': 'velocity_mean',
            'direction_Relative_To_Origin': 'direction_Relative_To_Origin_mean',            
            'nnCountInFrame_within_3_pixels': 'nnCountInFrame_within_3_pixels_mean',
            'nnCountInFrame_within_5_pixels': 'nnCountInFrame_within_5_pixels_mean',
            'nnCountInFrame_within_10_pixels': 'nnCountInFrame_within_10_pixels_mean',
            'nnCountInFrame_within_20_pixels': 'nnCountInFrame_within_20_pixels_mean',
            'nnCountInFrame_within_30_pixels': 'nnCountInFrame_within_30_pixels_mean'
                
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
            'nnDist_inFrame': 'nnDist_inFrame_std',
            'velocity': 'velocity_std',
            'direction_Relative_To_Origin': 'direction_Relative_To_Origin_std',  
            'nnCountInFrame_within_3_pixels': 'nnCountInFrame_within_3_pixels_std',
            'nnCountInFrame_within_5_pixels': 'nnCountInFrame_within_5_pixels_std',
            'nnCountInFrame_within_10_pixels': 'nnCountInFrame_within_10_pixels_std',
            'nnCountInFrame_within_20_pixels': 'nnCountInFrame_within_20_pixels_std',
            'nnCountInFrame_within_30_pixels': 'nnCountInFrame_within_30_pixels_std'
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
            'nnDist_inFrame_mean',
            'nnDist_inFrame_std',
            'velocity_mean',
            'velocity_std',
            'direction_Relative_To_Origin_mean',
            'direction_Relative_To_Origin_std',                        
            'nnCountInFrame_within_3_pixels_mean',
            'nnCountInFrame_within_3_pixels_std',
            'nnCountInFrame_within_5_pixels_mean',
            'nnCountInFrame_within_5_pixels_std',
            'nnCountInFrame_within_10_pixels_mean',
            'nnCountInFrame_within_10_pixels_std',
            'nnCountInFrame_within_20_pixels_mean',
            'nnCountInFrame_within_20_pixels_std',
            'nnCountInFrame_within_30_pixels_mean',
            'nnCountInFrame_within_30_pixels_std'       
        ]]
    return resultDF

def compileTrackResults(exptFolder, analysisStage = 'NNcount'):
     
    experiment = os.path.basename(exptFolder)
    print('-------------------')
    print('expt:  {}'.format(experiment))

    conditionList = glob.glob(exptFolder + '/*', recursive = False) 

    for conditionType in conditionList:
        condition = os.path.basename(conditionType)
        #skip ACTINRICM folders
        if 'ACTINRICM' in condition or 'DICERRICM' in condition:
            continue
        print('-------------------')            
        print('condition:  {}'.format(condition))

        #make empty stats table for condition
        statsTable = pd.DataFrame()
        #get files with track analysis
        trackDF_list = glob.glob(conditionType + '/*_{}.csv'.format(analysisStage), recursive = False)    
        #print(trackDF_list)
    
    
        #add mean values to each experiments df
        for trackFile in tqdm(trackDF_list):
            tempDF = pd.read_csv(trackFile)
            experimentName = tempDF.iloc[0]['Experiment']
            
            #exclude unlinked points
            tempDF = tempDF[~pd.isnull(tempDF['track_number'])]
        
            
            #print(experimentName)
            tempDF = tempDF[['track_number','n_segments', 'track_length', 'radius_gyration',
                             'asymmetry', 'skewness', 'kurtosis','radius_gyration_scaled',
                             'radius_gyration_scaled_nSegments', 'radius_gyration_scaled_trackLength',
                             'intensity', 'lag', 'fracDimension', 'netDispl', 'Straight',
                             'nnDist','SVM', 'nnDist_inFrame','velocity', 'direction_Relative_To_Origin',
                             'nnCountInFrame_within_3_pixels', 'nnCountInFrame_within_5_pixels','nnCountInFrame_within_10_pixels',
                             'nnCountInFrame_within_20_pixels','nnCountInFrame_within_30_pixels']]
            
            #use absolute straightness values for stats
            tempDF['Straight'] = tempDF['Straight'].abs()
            
            #get mean values for each track
            resultDF = tempDF.groupby('track_number', as_index=False).mean()
            #add experiment name to df
            resultDF['Experiment'] = experimentName            
            #export track tables        
            resultDF.to_csv(trackFile.split('.csv')[0]+'_trackMeans.csv')
            print('{} trackMeans exported'.format(experimentName))
            
            #stats for all tracks
            statsTable = statsTable.append(getStats(resultDF))

        #export stats tables   
        statsTable.to_csv(os.path.join(exptFolder,'{}_trackStats.csv'.format(condition)))
          
        print('{} stats exported'.format(condition))    
        

if __name__ == '__main__':
    ##### RUN ANALYSIS        
    path = r'/Users/george/Data/testingCompilationGabbyFolderStructure' 
       
    #get expt folder list
    exptList = glob.glob(path + '/*', recursive = False)   

    for exptFolder in tqdm(exptList):
        compileTrackResults(exptFolder, analysisStage = 'NNcount')

