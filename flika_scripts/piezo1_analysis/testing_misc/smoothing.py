#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:48:14 2023

@author: george
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

plotGraphs = True

def movingaverage(interval, window_size=10):
    window= np.ones(int(window_size))/float(window_size)
    smooth = np.convolve(interval, window, 'valid')
    start= int(((len(interval) - len(smooth)))/2)
    end = len(interval) - (start + len(smooth))
    return np.pad(smooth,(start,end),'edge')

# Create a test trace
# x = np.linspace(0,2*np.pi,100)
# y = np.sin(x) + np.random.random(100) * 0.2

# Load track to background subtract
file = '/Users/george/Desktop/multipleTrackTest/GB_228_2023_05_04_HTEndothelial_BAPTA_plate2_ultraslow_2uMYoda1_1_MMStack_Default.ome_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity_AllLocs_NNcount_BGsubtract_trapped-AllFrames.csv'
df = pd.read_csv(file)

#get roi intensity for all frames using trapped puncta
smoothDF = df[df['SVM']==3]
firstTrack = smoothDF['track_number'].iloc[0]
smoothDF = smoothDF[smoothDF['track_number']==firstTrack]
x = smoothDF['frame'].to_numpy()
y = smoothDF['roi_1'].to_numpy()

# Smooth the background
smoothing = int(len(y)/10)
smoothed_y = movingaverage(y, window_size=smoothing)
smoothDF['smooth'] = smoothed_y

# filter for track to be background subtracted
trackNumber = 1 #GABBY - CHANGE THE TRACK NUMBER HERE
df = df[df['track_number']==trackNumber]

frames = df['frame'].to_numpy()
smoothDF = smoothDF[smoothDF['frame'].isin(frames)]
df['smooth'] = smoothDF['smooth'].to_numpy()
df['intensity - smoothed roi_1'] = df['intensity'] - df['smooth']

if plotGraphs:
    # Plot the backgrtound and smoothing
    plt.scatter(x,y, label='Original background')
    plt.plot(x,smoothed_y, label='Smoothed background', c='r')
    plt.legend()
    plt.show()
    #plot intensity
    plt.figure()
    plt.scatter(df['frame'], df['intensity'], label='Original intensity')
    plt.scatter(df['frame'], df['intensity - smoothed roi_1'], label='Smoothed background subtracted intensity', c='r')
    plt.legend()
    plt.show()

# save the result
saveName = os.path.splitext(file)[0]+'_track{}.csv'.format(trackNumber)
df.to_csv(saveName, index=None)

