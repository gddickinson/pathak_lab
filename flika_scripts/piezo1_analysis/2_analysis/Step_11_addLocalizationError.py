#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:13:24 2022

@author: george
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob


def addLocalizationError(df):
    #add track mean X,Y positions
    df['mean_X'] = df.groupby('track_number')[['x']].transform('mean')
    df['mean_Y'] = df.groupby('track_number')[['y']].transform('mean')
    #add euclid distance of loc from mean x and y
    df['distanceFromMean'] = np.sqrt( (df['mean_X']-df['x'])**2 + (df['mean_Y']-df['y'])**2)
    #add mean loc distance for track
    df['meanLocDistanceFromCenter'] = df.groupby('track_number')[['distanceFromMean']].transform('mean')
    return df

if __name__ == '__main__':
    ##### RUN ANALYSIS
    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'

    #add localization error info to dfs
    files = glob.glob(path + '/**/*_SVM-ALL.csv', recursive = True)

    #add localization error info
    fileList = glob.glob(path + '/**/*_trapped-AllFrames.csv', recursive = True)

    for file in tqdm(fileList):

        df = pd.read_csv(file)

        newDF = addLocalizationError(df)

        saveName = os.path.splitext(file)[0]+'_locErr.csv'
        newDF.to_csv(saveName, index=None)


