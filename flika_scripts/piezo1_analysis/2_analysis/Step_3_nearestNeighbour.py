#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:13:24 2022

@author: george
"""

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Import required libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, glob
from sklearn.neighbors import KDTree
import math
from scipy import stats, spatial
from matplotlib import pyplot as plt



def getNearestNeighbors(train,test,k=2):
    """
    Find k nearest neighbors for each point in test set from train set.

    Args:
    train (array): Training data points
    test (array): Test data points
    k (int): Number of nearest neighbors to find

    Returns:
    tuple: Distances and indices of k nearest neighbors
    """
    tree = KDTree(train, leaf_size=5)
    if k > len(train):
        #no neighbours to count return nan
        a = np.empty((k,k))
        a[:] = np.nan
        return np.nan, np.nan
    else:
        dist, ind = tree.query(test, k=k)
    #dist.reshape(np.size(dist),)
    return dist, ind

def getNN(tracksDF):
    """
    Calculate nearest neighbor distances for each point in the DataFrame.

    Args:
    tracksDF (DataFrame): DataFrame containing track information

    Returns:
    DataFrame: Original DataFrame with added nearest neighbor distance column
    """
    #sort by frame
    tracksDF = tracksDF.sort_values(by=['frame'])
    #make empty list to store NN distances & indexes
    nnDistList = []
    nnIndexList = []
    #get list of frames in tracksDF to iterate over
    frames = tracksDF['frame'].unique().tolist()
    #get nn for each centroid position by frame
    for i, frame in enumerate(frames):
        #filter by frame
        frameXY = tracksDF[tracksDF['frame'] == frame][['x [nm]','y [nm]']].to_numpy()
        #nearest neighbour
        distances, indexes = getNearestNeighbors(frameXY,frameXY, k=2)
        #append distances and indexes of 1st neighbour to list
        if (np.isnan(distances).any()):
            nnDistList.append(np.nan)
        else:
            nnDistList.extend(distances[:,1])
        #nnIndexList.extend(indexes[:,1])
        print('\r' + 'NN analysis complete for frame{} of {}'.format(i,len(frames)), end='\r')

    #add results to dataframe
    tracksDF['nnDist_inFrame'] =  nnDistList
    #tracksDF['nnIndex_inFrame_all'] = nnIndexList

    tracksDF = tracksDF.sort_index()
    print('\r' + 'NN-analysis added', end='\r')

    return tracksDF

def calcNNforFiles(tracksList):
    """
    Calculate nearest neighbor distances for a list of track files.

    Args:
    tracksList (list): List of paths to track files
    """
    for trackFile in tqdm(tracksList):

        ##### load data
        tracksDF = pd.read_csv(trackFile)

        #add nearest neigbours to df
        tracksDF = getNN(tracksDF)

        #just keep nn info
        tracksDF = tracksDF[['id', 'frame', 'x [nm]', 'y [nm]', 'nnDist_inFrame']]

        #save
        saveName = os.path.splitext(trackFile)[0] + '_NN.csv'
        tracksDF.to_csv(saveName)
        print('\n new tracks file exported to {}'.format(saveName))


def addNNtoSVMFiles(svmFileList):
    """
   Add nearest neighbor information to SVM classification files.

   Args:
   svmFileList (list): List of paths to SVM classification files
   """
    for svmFile in tqdm(svmFileList):
        ##### load svm
        svmDF = pd.read_csv(svmFile)
        ##### load locID_NN
        nnFile = svmFile.split('_SVM-ALL.csv')[0] + '_locsID_NN.csv'
        nnDF = pd.read_csv(nnFile)
        nnDF = nnDF[['id','nnDist_inFrame']]
        nnDF['nnDist_inFrame'] = nnDF['nnDist_inFrame'] / 108
        #nnDF['x2'] = nnDF['x [nm]'] / 108
        #nnDF['y2'] = nnDF['y [nm]'] / 108

        #join df based on id
        svmDF = svmDF.join(nnDF.set_index('id'), on='id')

        #save
        saveName = os.path.splitext(svmFile)[0] + '_NN.csv'
        svmDF.to_csv(saveName)
        print('\n NN-file exported to {}'.format(saveName))



if __name__ == '__main__':
    ##### RUN ANALYSIS

    # Set path to data directory
    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'
    #path = '/Users/george/Data/gabby_missingIntensities'

    # Get folder paths for localization files
    tracksList = glob.glob(path + '/**/*_locsID.csv', recursive = True)   #using locs file to measure distances to all detected locs (some removed during linking/feature calc if tracks too short)

    # Run nearest neighbor analysis
    calcNNforFiles(tracksList)

    # Add nn to SVM files based on id
    svmFileList = glob.glob(path + '/**/*_SVM-ALL.csv', recursive = True)
    addNNtoSVMFiles(svmFileList)


