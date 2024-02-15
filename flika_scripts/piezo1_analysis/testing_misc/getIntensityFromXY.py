#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 08:59:45 2023

@author: george
"""

import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import skimage.io as skio

def getIntensity(A,locsDF, frames=[0], pixelSize=108):
    #convert frames to zero start
    locsDF['frame'] = locsDF['frame'] -1
    #filter df for frames to analyse
    locsDF = locsDF[locsDF['frame'].isin(frames)]
    #intensities retrieved from image stack using point data (converted from nm floats to pixel ints)
    locsDF['x'] = locsDF['x [nm]'] / pixelSize
    locsDF['y'] = locsDF['y [nm]'] / pixelSize

    #array dimensions
    n, w, h = A.shape

    #init intensity list
    intensities = []

    for i in range(len(locsDF)):
        frame = round(locsDF['frame'][i])
        x = round(locsDF['x'][i])
        y = round(locsDF['y'][i])

        #set x,y bounds for 3x3 pixel square
        xMin = x - 1
        xMax = x + 2

        yMin = y - 1
        yMax = y + 2

        #deal with edge cases
        if xMin < 0:
            xMin = 0
        if xMax > w:
            xMax = w

        if yMin <0:
            yMin = 0
        if yMax > h:
            yMax = h

        #get mean pixels values for 3x3 square
        intensities.append(np.mean(A[frame][yMin:yMax,xMin:xMax]))

    #make df for result
    intensityDF = pd.DataFrame({'intensity': intensities})
    #round to 2 decimal places
    intensityDF['intensity'] = intensityDF['intensity'].round(2)

    return intensityDF


if __name__ == '__main__':
    ##### RUN ANALYSIS
    path = r'/Users/george/Desktop/testing'

    #get tiff folder list
    tiffList = glob.glob(path + '/**/*.tif', recursive = True)

    for file in tqdm(tiffList):
        #derive locs file name from tiff file
        locsFile = file.split('.')[0] + '_locs.csv'
        #load tiff array
        A = skio.imread(file)
        #load locs df
        locsDF = pd.read_csv(locsFile)
        #get intensities
        intensityDF = getIntensity(A,locsDF)
        #save result
        saveName = file.split('.')[0] + '_intensity.csv'
        intensityDF.to_csv(saveName, index=None)

