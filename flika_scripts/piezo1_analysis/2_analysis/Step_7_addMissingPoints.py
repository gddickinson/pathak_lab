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



def addMissingLocsToDF(df, locsDF, pixelSize = 108):
        #get IDs of locs in df
        IDlist = df['id']
        #convert nm to pixels
        locsDF['x'] = locsDF['x [nm]'] / pixelSize
        locsDF['y'] = locsDF['y [nm]'] / pixelSize
        #just keep columns to join
        locsDF = locsDF[['id','frame','x', 'y']]
        #filter for missing locs
        missingLocs = locsDF[~locsDF['id'].isin(IDlist)]

        #append df
        df = pd.concat([df,missingLocs])

        return df


if __name__ == '__main__':

    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'
    #path = '/Users/george/Data/gabby_missingIntensities'


    #add nn to SVM files based on id
    dfFileList = glob.glob(path + '/**/*_diffusion_velocity.csv', recursive = True)

    for dfFile in tqdm(dfFileList):

        df = pd.read_csv(dfFile)
        locsDF = pd.read_csv(dfFile.split('_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity.csv')[0] + '_locsID.csv')

        newDF = addMissingLocsToDF(df, locsDF)

        saveName = os.path.splitext(dfFile)[0]+'_AllLocs.csv'
        newDF.to_csv(saveName, index=None)


