#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:05:03 2023

@author: george
"""

import pandas as pd

def runningSum(df, colName, group = ['Experiment', 'track_number']):
    '''' Adds running sum of selected column to df - grouped by experiment and track_number'''
    df['{}_running_sum'.format(colName)] = df.groupby(group)[colName].transform(pd.Series.cumsum)
    df = df.sort_index()
    return df


if __name__ == '__main__':
    #set filename for csv table
    file = '/Users/george/Desktop/testing/GB_131_2021_08_10_HTEndothelial_BAPTA_plate1_9_cropped_trackid20_1327_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity_AllLocs_NNcount_BGsubtract.csv'
    #load df
    df = pd.read_csv(file)
    #perform runningSum on selected column
    colName = 'lag'
    df = runningSum(df, colName)
    #save updated df
    saveName = file.split('.csv')[0] + '_{}_runningSum.csv'.format(colName)
    df.to_csv(saveName, index=None)
    print('new df saved to {}'.format(saveName))

