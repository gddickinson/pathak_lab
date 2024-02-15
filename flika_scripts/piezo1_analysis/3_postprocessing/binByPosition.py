#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:53:09 2023

@author: george
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import os, glob
from scipy.signal import find_peaks, peak_widths

def binByPosition(df, bin_size=10):
    #drop unlinked points
    unlinked_df = df[~df['track_number'].isna()]


    return unlinked_df

if __name__ == '__main__':

    path = '/Users/george/Desktop/peakAnalysis'

    #add nn count
    fileList = glob.glob(path + '/**/*_BGsubtract.csv', recursive = True)

    for file in tqdm(fileList):

        #load analysis df
        df = pd.read_csv(file)

        #bin by xy positions and calculate new lags/velocities
        newDF = binByPosition(df)

        newDF = newDF[newDF['track_number'] == 190]
        newDF = newDF[['frame', 'zeroed_X', 'zeroed_Y']]

        bins = list(range(min(newDF['frame']),max(newDF['frame']),10))
        binned = pd.cut(newDF['frame'], 10, retbins=True)


        #saveName = os.path.splitext(file)[0]+'_binByPosition.csv'
        #newDF.to_csv(saveName, index=None)
