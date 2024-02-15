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



def addVelocitytoDF(df):
        newDF = pd.DataFrame()

        trackList = df['track_number'].unique().tolist()

        #iterate through individual tracks
        for track in tqdm(trackList):
            trackDF = df[df['track_number']==track]

            #add differantial for distance
            diff = np.diff(trackDF['distanceFromOrigin'].to_numpy()) / np.diff(trackDF['lagNumber'].to_numpy())
            diff = np.insert(diff,0,0)
            trackDF['dy-dt: distance'] = diff

            #add track results to df
            newDF = pd.concat([newDF, trackDF])


        #add delta-t for each lag
        newDF['dt'] = np.insert(newDF['frame'].to_numpy()[1:],-1,0) - newDF['frame'].to_numpy()
        newDF['dt'] = newDF['dt'].mask(newDF['dt'] <= 0, None)
        #instantaneous velocity
        newDF['velocity'] = newDF['lag']/newDF['dt']
        #direction relative to 0,0 origin : 360 degreeas
        degrees = np.arctan2(newDF['zeroed_Y'].to_numpy(), newDF['zeroed_X'].to_numpy())/np.pi*180
        degrees[degrees < 0] = 360+degrees[degrees < 0]
        newDF['direction_Relative_To_Origin'] =  degrees
        #add mean track velocity
        newDF['meanVelocity'] = newDF.groupby('track_number')['velocity'].transform('mean')

        return newDF


if __name__ == '__main__':

    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'
    #path = '/Users/george/Data/gabby_missingIntensities'

    #add nn to SVM files based on id
    fileList = glob.glob(path + '/**/*_NN_diffusion.csv', recursive = True)

    for file in tqdm(fileList):

        df = pd.read_csv(file)

        newDF = addVelocitytoDF(df)

        saveName = os.path.splitext(file)[0]+'_velocity.csv'
        newDF.to_csv(saveName, index=None)
