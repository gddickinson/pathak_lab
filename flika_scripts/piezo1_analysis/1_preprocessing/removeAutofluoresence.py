#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:38:45 2023

@author: george
"""
#%matplotlib qt

from matplotlib import  pyplot as plt
import numpy as np
import skimage.io as skio
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square, remove_small_objects, binary_dilation, disk
from skimage.measure import label, points_in_poly, regionprops, find_contours
import glob, os
from tqdm import tqdm
import pandas as pd

'''https://scikit-image.org/docs/stable/auto_examples/index.html'''


def cropHighIntensityRegions(fileName, thresh = 250, autoThresh = False, minSize = 90, plotResult = False, addNoise = False, noiseScale = 1.0, dilationRadius = 6):
    ''''takes tiff stack as input and returns copy with high intensity regions cropped out'''
    #load image stack
    A = skio.imread(fileName)

    #get max intensity
    maxIntensity = np.max(A, axis=0)

    #threshold to binary image
    if autoThresh:
        thresh = threshold_otsu(maxIntensity) # automatic detection of thresh
    binary = closing(maxIntensity > thresh, square(6))

    #dilate bright objects
    dilate = binary_dilation(binary,disk(dilationRadius, dtype=bool))

    # label image regions and remove small objects
    label_image = label(dilate)
    largeRegions = remove_small_objects(label_image, min_size = minSize)

    #generate mask and crop image
    mask = largeRegions > 0

    # Crop out the regions defined by the mask array
    cropped_stack = A
    for img in cropped_stack:
        if addNoise:
            #get mean and sd of image intensities
            mu = np.mean(img) * noiseScale
            sigma = np.std(img)
            #make random array
            randomValues = np.random.normal(mu, sigma, mask.size)
            randomArray = np.reshape(randomValues, mask.shape)
            #mask
            img[mask] = randomArray[mask]
        else:
            img[mask] = 0

    #plot results
    if plotResult:
        fig1, axs1 = plt.subplots(1, 5, figsize=(20, 5))
        axs1[0].imshow(A[0])
        axs1[0].set_title('1st frame')

        axs1[1].imshow(maxIntensity)
        axs1[1].set_title('max Intensity')

        axs1[2].imshow(binary)
        axs1[2].set_title('thresholded >{}'.format(thresh))

        axs1[3].imshow(largeRegions)
        axs1[3].set_title('objects >{}'.format(minSize))

        axs1[4].imshow(cropped_stack[0])
        axs1[4].set_title('1st frame after crop')

    return cropped_stack


def add_within_region_column(labels, df):
    '''Add a new column to a DataFrame indicating if each point is within any labelled region'''
    df['within_region'] = False
    for region_label in np.unique(labels):
        if region_label == 0:
            continue
        mask = labels == region_label
        df['within_region'] = df['within_region'] | mask[df['y'], df['x']]
    return df

def removeLocsFromHighIntensityRegions(fileName, locs_fileName, thresh = 250, autoThresh = False, minSize = 90, dilationRadius = 6, pixelSize = 108, plotResult=False):
    ''''takes tiff stack and locs file as inputs - returns new locs file with locs in high intensity regions removed'''
    #load image stack
    A = skio.imread(fileName)

    #get max intensity
    maxIntensity = np.max(A, axis=0)

    #threshold to binary image
    if autoThresh:
        thresh = threshold_otsu(maxIntensity) # automatic detection of thresh
    binary = closing(maxIntensity > thresh, square(6))

    #dilate bright objects
    dilate = binary_dilation(binary,disk(dilationRadius, dtype=bool))

    # label image regions and remove small objects
    label_image = label(dilate)
    largeRegions = remove_small_objects(label_image, min_size = minSize)

    #load locs
    df = pd.read_csv(locs_fileName)
    #add xy positions in pixels rounded to nearest pixel
    df['x'] = df['x [nm]'] / pixelSize
    df['y'] = df['y [nm]'] / pixelSize

    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)

    #add column indicating if point in any of the regions
    df = add_within_region_column(largeRegions, df)

    #filter df
    df_excluded = df[df['within_region']]
    df_included = df[~df['within_region']]

    #plot results
    if plotResult:
        plt.scatter(df_included['x'],df_included['y'], color='white', s=0.1)
        plt.scatter(df_excluded['x'],df_excluded['y'], color='red', s=0.1)
        plt.imshow(A[0])

    return df_included.drop(['x','y','within_region'], axis=1)



if __name__ == '__main__':
    ##### RUN ANALYSIS
    path = r'/Users/george/Data/thresholdTest'

    #get tiff folder list
    tiffList = glob.glob(path + '/**/*.tif', recursive = True)

    for file in tqdm(tiffList):
        '''############### crop autofluorescence from image stack ###############'''
# =============================================================================
#         cropped_stack = cropHighIntensityRegions(file)
#         #save new image stack
#         saveName = file.split('.')[0] + '_crop.tif'
#         skio.imsave(saveName, cropped_stack)
# =============================================================================

        '''###### remove localizations from regions of high fluorescence from locs file ##########'''
        locsFile = os.path.splitext(file)[0].split('.tiff')[0] + '_locs.csv'
        locsDF = removeLocsFromHighIntensityRegions(file, locsFile, plotResult=True)
        #save locs
        saveName = os.path.splitext(file)[0].split('.tiff')[0] + '_highFluroRemoved_locs.csv'
        locsDF.to_csv(saveName, index=None)






