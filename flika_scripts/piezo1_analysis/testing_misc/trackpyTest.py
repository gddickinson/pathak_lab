#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:22:45 2023

@author: george
"""
#%matplotlib qt 

import numba
import numpy as np
import pandas as pd
import skimage.io as skio
from tqdm import tqdm
import os, glob
import trackpy as tp

tiffFile = '/Users/george/Data/trackpyTest/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10.tif'
locsFile = '/Users/george/Data/trackpyTest/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10_locs.csv'

pixelSize = 0.108

#load tiff
tiffStack = skio.imread(tiffFile, plugin='tifffile')

# #test trackpy locate function
# f = tp.locate(tiffStack[0], 11, invert=False)
# #show head
# print(f.head)
# #plot localizations
# tp.annotate(f, tiffStack[0])


#load locs
locs = pd.read_csv(locsFile)
#convert to pixels
locs['x'] = locs['x [nm]'] * pixelSize/10
locs['y'] = locs['y [nm]'] * pixelSize/10

#drop unneeded cols
locs = locs[['frame', 'x', 'y', 'id', 'x [nm]', 'y [nm]']]

#plot 1 frame
tp.annotate(locs[locs['frame']==1], tiffStack[0])

#check subpixel accuracy
#tp.subpx_bias(locs)

#link locs
#tp.quiet()

#minimum number of link segments (need at least 2 to avoid colinearity in feature calc)
minLinkSegments = 2

#max number of gap frames to skip
gapSize = 3

#max distance in pixels to allow a linkage
distance = 3

t = tp.link(locs, distance, memory=gapSize)
t1 = tp.filter_stubs(t, minLinkSegments)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

ax = tp.plot_traj(t1)

#test of adaptive search
maxDistance = 11

t_adaptive = tp.link(locs, maxDistance, adaptive_stop=0.1, adaptive_step=0.95, memory=gapSize)

t_adaptive1 = tp.filter_stubs(t_adaptive, minLinkSegments)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t_adaptive['particle'].nunique())
print('After:', t_adaptive1['particle'].nunique())
ax = tp.plot_traj(t_adaptive)

#adding prediction
pred = tp.predict.NearestVelocityPredict()
t_predict = pred.link_df(locs, distance, memory=gapSize, adaptive_stop=0.1, adaptive_step=0.95)

t_predict1 = tp.filter_stubs(t_predict, minLinkSegments)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t_predict['particle'].nunique())
print('After:', t_predict1['particle'].nunique())
ax = tp.plot_traj(t_predict)


#get intensity values for x,y coordinates
def getIntensities(dataArray, pts):
    #intensities retrieved from image stack using point data (converted from floats to ints)
    
    n, w, h = dataArray.shape
    
    #clear intensity list
    intensities = []
    
    for point in pts:
        frame = int(round(point[0]))
        x = int(round(point[1]))
        y = int(round(point[2]))
        
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
        
        #get mean pixels values for 3x3 square - background subtract using frame min intensity as estimate of background
        intensities.append((np.mean(dataArray[frame][xMin:xMax,yMin:yMax]) - np.min(dataArray[frame])))
    
    return intensities

pts = t_predict1[['frame','x','y']]
pts['frame'] = pts['frame']-1
pts = pts.to_numpy()
intensities = getIntensities(tiffStack, pts)



