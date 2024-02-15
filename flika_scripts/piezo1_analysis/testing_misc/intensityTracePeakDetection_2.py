#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:55:40 2023

@author: george
"""

import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths
from matplotlib import pyplot as plt
import pandas as pd

data_path = '/Users/george/Desktop/peakAnalysis/GB_168_2022_03_14_HTEndothelial_BAPTA_plate1_yoda1_11_cropped_track6183_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity_AllLocs_NNcount.csv'

#load data
df = pd.read_csv(data_path)
#drop unlinked points
df = df[~df['track_number'].isna()]

#filter
#df = df[df['n_segments'] > 300]
df = df[df['track_number'] == 190]
#df = df[df['track_number'] == 7]

# get traces for each roi
traceList = []

#get individual track traces
grouped = df.groupby('track_number')['intensity'].apply(list)
trackNumbers = df.groupby('track_number')['track_number'].apply(list)

for group in grouped:
    traceList.append(np.array(group))



#smoothing
smoothed = []
kernel_size = 100
kernel = np.ones(kernel_size) / kernel_size
for trace in traceList:
    data_convolved = np.convolve(trace, kernel, mode='valid')
    #pad edges
    data_convolved = np.pad(data_convolved, int(kernel_size/2), mode='minimum')
    smoothed.append(data_convolved)

#trace = traceList[0]
trace = smoothed[0]

threshold = (trace.std()) + (trace.mean())

peaks, properties = find_peaks(trace, height= threshold, width=10)
results_half = peak_widths(trace, peaks, rel_height=0.5)
results_half[0]  # widths
results_full = peak_widths(trace, peaks, rel_height=1)
results_full[0]  # widths

plt.plot(traceList[0])
plt.plot(trace)
plt.plot(peaks, trace[peaks], "o", color="C3")
plt.hlines(*results_half[1:], color="C3")
#plt.hlines(*results_full[1:], color="C3")
plt.show()
