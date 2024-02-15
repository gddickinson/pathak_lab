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



def addDiffusiontoDF(df):

        newDF = pd.DataFrame()

        trackList = df['track_number'].unique().tolist()

        #iterate through individual tracks
        for track in tqdm(trackList):
            trackDF = df[df['track_number']==track]
            #set positions relative to origin of 0,0
            minFrame = trackDF['frame'].min()
            origin_X = float(trackDF[trackDF['frame'] == minFrame]['x'])
            origin_Y = float(trackDF[trackDF['frame'] == minFrame]['y'])
            trackDF['zeroed_X'] = trackDF['x'] - origin_X
            trackDF['zeroed_Y'] = trackDF['y'] - origin_Y
            #generate lag numbers
            trackDF['lagNumber'] = trackDF['frame'] - minFrame
            #calc distance from origin
            trackDF['distanceFromOrigin'] = np.sqrt(  (np.square(trackDF['zeroed_X']) + np.square(trackDF['zeroed_Y']))   )


            #add track results to df
            newDF = pd.concat([newDF, trackDF])

        #get squared values
        newDF['d_squared'] = np.square(newDF['distanceFromOrigin'])
        newDF['lag_squared'] = np.square(newDF['lag'])

        return newDF


if __name__ == '__main__':

    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'
    #path = '/Users/george/Data/gabby_missingIntensities'

    #add nn to SVM files based on id
    fileList = glob.glob(path + '/**/*_SVMPredicted_NN.csv', recursive = True)

    for file in tqdm(fileList):

        df = pd.read_csv(file)

        newDF = addDiffusiontoDF(df)

        saveName = os.path.splitext(file)[0]+'_diffusion.csv'
        newDF.to_csv(saveName, index=None)
