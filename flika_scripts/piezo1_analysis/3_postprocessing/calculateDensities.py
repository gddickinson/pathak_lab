#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:54:31 2023

@author: george
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob


def calcDensity(df, n = 10):
    #get 1st n frames of df
    df = df[df['frame']<n]
    #print(df)
    #print(len(df))
    #get x,y boundary
    minX = np.min(df['x [nm]'])
    minY = np.min(df['y [nm]'])
    maxX = np.max(df['x [nm]'])
    maxY = np.max(df['y [nm]'])


    #quatiles
    xLen = maxX - minX
    yLen = maxY - minY

    x_quat = xLen/4
    y_quat = yLen/4

    #quad boundary
    xMaxLimit = maxX - x_quat
    xMinLimit = minX + x_quat
    yMaxLimit = maxY - y_quat
    yMinLimit = minY + y_quat

    print("x min: {}, x max: {}, y min: {}, y max: {}".format( xMinLimit , xMaxLimit, yMinLimit, yMaxLimit))

    #crop to center quad of
    df = df[(df['x [nm]'] > xMinLimit) & (df['x [nm]'] < xMaxLimit)]
    df = df[(df['y [nm]'] > yMinLimit) & (df['y [nm]'] < yMaxLimit)]
    #print(len(df))

    if n > 1:
        #get mean number of locs per frame
        n_locs = len(df)  / n
    else:
        n_locs = len(df)

    return n_locs, x_quat, y_quat


if __name__ == '__main__':

    path = '/Users/george/Desktop/filterTest_2'

    locsFiles = glob.glob(path + '/**/*_locs.csv', recursive = True)

    saveName = os.path.join(path, 'densityResult.csv')

    resultDF = pd.DataFrame()

    file_list = []
    n_locs_list = []
    xSize_list = []
    ySize_list = []

    n_frames = 10

    for file in tqdm(locsFiles):

        df = pd.read_csv(file)

        n_locs, xSize, ySize  = calcDensity(df, n_frames)

        file_list.append(file)
        n_locs_list.append(n_locs)
        xSize_list.append(xSize)
        ySize_list.append(ySize)

        print('{}: {} locs'.format(os.path.basename(file), n_locs))


    resultDF['file'] = file_list
    resultDF['n locs (mean of 1st {} frames)'.format(n_frames)] = n_locs_list
    resultDF['x_crop size (nm)'] = xSize_list
    resultDF['y_crop size (nm)'] = ySize_list

    resultDF['center quad area (square nm)'] = resultDF['x_crop size (nm)']*resultDF['y_crop size (nm)']
    resultDF['density (locs /square nm)'] = resultDF['n locs (mean of 1st {} frames)'.format(n_frames)]/resultDF['center quad area (square nm)']

    resultDF.to_csv(saveName)
