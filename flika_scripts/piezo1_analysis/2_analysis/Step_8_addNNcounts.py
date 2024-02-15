#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:48:41 2023

@author: george
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob

from sklearn.neighbors import KDTree
#from scipy.spatial import distance_matrix

# def getNearestNeighbors(train,test,k=2):
#     tree = KDTree(train, leaf_size=5)
#     dist, ind = tree.query(test, k=k)
#     #dist.reshape(np.size(dist),)
#     return dist, ind

def countNNwithRadius(train,test,r=1):
    tree = KDTree(train, leaf_size=5)
    count = tree.query_radius(test, r=r, count_only=True)
    return count

# def flatten(l):
#     return [item for sublist in l for item in sublist]


def getNNinFrame(tracksDF, radiusList=[3,5,10,20,30]):
    #sort by frame
    tracksDF = tracksDF.sort_values(by=['frame'])

    for r in radiusList:

        #make empty list to store NN distances & indexes
        countList = []
        #get list of frames in tracksDF to iterate over
        frames = tracksDF['frame'].unique().tolist()

        #get nn for each centroid position by frame
        for i, frame in enumerate(frames):
            #filter by frame
            frameXY = tracksDF[tracksDF['frame'] == frame][['x','y']].to_numpy()
            #nearest neighbour
            count = countNNwithRadius(frameXY,frameXY,r=r)
            #subtract 1 from count (to self)
            count = count -1
            #append distances and indexes of 1st neighbour to list
            countList.extend(count)
            #nnIndexList.extend(indexes[:,1])
            print('\r' + 'NN analysis complete for frame{} of {} of r= {}'.format(i,len(frames),r), end='\r')

        #add results to dataframe
        tracksDF['nnCountInFrame_within_{}_pixels'.format(r)] =  countList

    tracksDF = tracksDF.sort_index()
    #print('\r' + 'NNcount-analysis added', end='\r')

    return tracksDF



if __name__ == '__main__':

    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'
    #path = '/Users/george/Data/gabby_missingIntensities'

    #add nn count
    fileList = glob.glob(path + '/**/*_velocity_AllLocs.csv', recursive = True)

    for file in tqdm(fileList):

        df = pd.read_csv(file)

        newDF = getNNinFrame(df)

        saveName = os.path.splitext(file)[0]+'_NNcount.csv'
        newDF.to_csv(saveName, index=None)
