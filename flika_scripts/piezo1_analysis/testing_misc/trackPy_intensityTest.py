#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:03:36 2023

@author: george
"""
%matplotlib qt 
%gui qt

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numba

import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob


import json, codecs
from distutils.version import StrictVersion
from qtpy.QtCore import QUrl, QRect, QPointF, Qt
from qtpy.QtGui import QDesktopServices, QIcon, QPainterPath, QPen, QColor
from qtpy.QtWidgets import QHBoxLayout, QGraphicsPathItem, qApp
from qtpy import uic

import flika
from flika import global_vars as g
from flika.window import Window
from flika.process.file_ import save_file_gui, open_file_gui, open_file

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox

import sys
sys.path.append(r'/Users/george/.FLIKA/plugins/pynsight_GDedit')
from insight_writer import write_insight_bin
from gaussianFitting import fitGaussian, gaussian, generate_gaussian
from SLD_histogram import SLD_Histogram
from MSD_Plot import MSD_Plot

from flika import start_flika

from sklearn.neighbors import KDTree
import math

from scipy import stats, spatial
from matplotlib import pyplot as plt

import seaborn as sns
from importlib import reload
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from pathlib import Path
from scipy import stats
from sklearn import datasets, decomposition, metrics
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import power_transform, PowerTransformer, StandardScaler

import skimage.io as skio

import trackpy as tp 

def getIntensities(dataArray, pts):
    #intensities retrieved from image stack using point data (converted from floats to ints)
    
    n, w, h = dataArray.shape
    
    print(dataArray.shape)
    
    #clear intensity list
    intensities = []
    
    for point in pts:
        frame = round(point[0])
        x = round(point[1])
        y = round(point[2])
        
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
        intensities.append((np.mean(dataArray[frame][yMin:yMax,xMin:xMax]) - np.min(dataArray[frame])))
    
        # plt.imshow(dataArray[frame])
        # plt.scatter([xMin,xMax, xMin, xMax],[yMin,yMax, yMax, yMin])
        # plt.scatter(x,y)
        
        # plt.figure(2)
        
        # plt.imshow(dataArray[frame][yMin:yMax,xMin:xMax])
              
        
    
    return intensities



path = '/Users/george/Data/trackpyTest'


fileName = '/Users/george/Data/trackpyTest/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10.tif'
 
pixelSize = 0.108 

skipFrames = 1 

distanceToLink = 3 

level='' 

#linkingType = 'standard'
#linkingType = 'adaptive'
#linkingType = 'velocityPredict'    
linkingType = 'adaptive + velocityPredict' 

maxDistance=6

#max distance in pixels to allow a linkage
distance = 3
#max number of gap frames to skip
gapSize = 1
maxSearchDistance = 6

#pixels in nm
pixelSize = pixelSize *1000
    
#set file & save names
pointsFileName = os.path.splitext(fileName)[0] + '_locsID{}.csv'.format(level)
#lagsHistoSaveName = os.path.splitext(pointsFileName)[0] + '_lagsHisto{}.txt'.format(level)  
tracksSaveName = os.path.splitext(pointsFileName)[0] + '_tracks{}.csv'.format(level) 

#turn off trackpy messages
tp.quiet()
#load locs file
locs = pd.read_csv(pointsFileName)        
#convert coordinates to pixels
locs['x'] = locs['x [nm]'] / pixelSize
locs['y'] = locs['y [nm]'] / pixelSize       
#drop unneeded cols
locs = locs[['frame', 'x', 'y', 'id', 'x [nm]', 'y [nm]']]

#link points
if linkingType=='standard':
    # standard linking
    tracks = tp.link(locs, distanceToLink, memory=skipFrames)
    
if linkingType=='adaptive':
    # adaptive linking
    tracks = tp.link(locs, maxDistance, adaptive_stop=0.1, adaptive_step=0.95, memory=gapSize) 

if linkingType=='velocityPredict':
    # adaptive linking using velocity prediction
    pred = tp.predict.NearestVelocityPredict()
    tracks = pred.link_df(locs, distance, memory=gapSize)   

if linkingType=='adaptive + velocityPredict':
    # adaptive linking using velocity prediction
    pred = tp.predict.NearestVelocityPredict()
    tracks = pred.link_df(locs, distance, memory=gapSize, adaptive_stop=0.1, adaptive_step=0.95)           
       
#get background subtracted intensity for each point
A = skio.imread(fileName, plugin='tifffile')
pts = tracks[['frame','x','y']]
#pts['frame'] = pts['frame']-1
pts = pts.to_numpy()
intensities = getIntensities(A, pts)
tracks['intensity'] = intensities


# =============================================================================
# plt.imshow(A[99])
# testFrame = tracks[tracks['frame']==99]
# test_pts = testFrame[['frame','x','y']]
# 
# test_pts = test_pts.to_numpy()
# plt.scatter(test_pts[:,1], test_pts[:,2])
# =============================================================================


#rename cols to match pynsight
tracks['track_number'] = tracks['particle']         
#sort by track_number and frame
tracks_sort = tracks.sort_values(by=['track_number', 'frame'])        
#reorder columns and drop particles
#tracks = tracks[['track_number','frame', 'x', 'y','intensity', 'id', 'x [nm]', 'y [nm]']]        
#Save tracks
#tracks.to_csv(tracksSaveName)            
#print('tracks file {} saved'.format(tracksSaveName))