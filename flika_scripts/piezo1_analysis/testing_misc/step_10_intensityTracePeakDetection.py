#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:55:40 2023

@author: george
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import os, glob
from scipy.signal import find_peaks, peak_widths

def detectIntensityPeaks(df, kernel_size = 100):
    #drop unlinked points
    unlinked_df = df[~df['track_number'].isna()]

    # get traces for each roi
    traceList = []
    trackNumbers = []

    #get individual track traces
    grouped = unlinked_df.groupby('track_number')['intensity'].apply(list)
    trackNumbers = unlinked_df['track_number'].unique().tolist()

    for group in grouped:
        traceList.append(np.array(group))


    #smoothing
    smoothed = []
    kernel = np.ones(kernel_size) / kernel_size
    for trace in traceList:
        data_convolved = np.convolve(trace, kernel, mode='valid')
        #pad edges
        data_convolved = np.pad(data_convolved, int(kernel_size/2), mode='minimum')
        smoothed.append(data_convolved)

    peak_list = []
    props_list = []
    width_list = []

    for smooth in smoothed:

        threshold = (trace.std()) + (trace.mean())

        peaks, properties = find_peaks(smooth, height= threshold, width=10)
        peak_list.append(peaks)
        props_list.append(properties)
        results_half = peak_widths(smooth, peaks, rel_height=0.5)
        width_list.append(results_half)
        #results_half[0]  # widths
        #results_full = peak_widths(smooth, peaks, rel_height=1)
        #results_full[0]  # widths


        # plt.plot(trace)
        # plt.plot(smooth)
        # plt.plot(peaks, smooth[peaks], "o", color="C3")
        # plt.hlines(*results_half[1:], color="C3")
        # #plt.hlines(*results_full[1:], color="C3")
        # plt.show()


    #add results by track
    for i, trackID in enumerate(trackNumbers):
        #df.loc[df['track_number'] == trackID, 'smoothed intensity'] = smoothed[i]
        df.loc[df['track_number'] == trackID, '# of peaks'] = len(peak_list[i])
        #df.loc[df['track_number'] == trackID, 'peak half width'] = width_list[i]
        #df.loc[df['track_number'] == trackID, 'peak props'] = props_list[i]
    return df

if __name__ == '__main__':

    path = '/Users/george/Desktop/peakAnalysis'

    #add nn count
    fileList = glob.glob(path + '/**/*_BGsubtract.csv', recursive = True)

    for file in tqdm(fileList):

        #load analysis df
        df = pd.read_csv(file)

        #add peak detection result
        newDF = detectIntensityPeaks(df)

        saveName = os.path.splitext(file)[0]+'_peaks.csv'
        newDF.to_csv(saveName, index=None)
