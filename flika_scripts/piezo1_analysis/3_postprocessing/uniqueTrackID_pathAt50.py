#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:05:03 2023

@author: george
"""
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd

def uniqueTrackID(df):
    '''' Adds unigue track ID for worksheet with multiple experiments'''
    #get unique numerical ID for each experiment as str
    df['track_ID'] = (df.groupby(['Experiment']).cumcount()==0).astype(int)
    df['track_ID'] = df['track_ID'].cumsum()
    df['track_ID'] = df['track_ID'].astype(str)
    #get track number asstr
    df['track_number'] = df['track_number'].astype(int)
    df['track_number'] = df['track_number'].astype(str)
    #join experiment and track numbers seperated by "_"
    df['track_ID'] = df[['track_ID','track_number']].apply(lambda x: "_".join(x), axis=1)
    df['track_number'] = df['track_number'].astype(float)
    return df

def getPathLengthAtLag(df, lag=50):
    '''adds lag_running_sum at lag x to every tracks row'''
    filtered = df[df['lagNumber']==lag]
    filtered = filtered[['track_ID', 'lag_running_sum']]
    filtered = filtered.rename(columns={'lag_running_sum': 'lagSumToLag{}'.format(lag)})
    df = df.merge(filtered, on='track_ID', how='outer')
    return df


if __name__ == '__main__':
    #set filename for csv table
    file = '/Users/george/Desktop/testing/testSheet.csv'

    #load df
    df = pd.read_csv(file)
    #split into tracks/unlinked
    unlinked = df[df['track_number'].isna()]
    linked = df[~df['track_number'].isna()]

    #add new ID
    newDF = uniqueTrackID(linked)
    #add column with path length at lag x for every row in track
    finalLag=50
    newDF = getPathLengthAtLag(newDF, lag = finalLag)

    #add back unlinked points
    newDF = pd.concat([newDF, unlinked])

    #save updated df
    saveName = file.split('.csv')[0] + '_newID_lagSumToLag{}.csv'.format(finalLag)
    newDF.to_csv(saveName, index=None)
    print('new df saved to {}'.format(saveName))

