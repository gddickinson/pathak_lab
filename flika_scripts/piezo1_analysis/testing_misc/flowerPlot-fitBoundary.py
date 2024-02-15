#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:31:07 2023

@author: george
"""
%matplotlib qt 
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import os, glob
from matplotlib.patches import Ellipse, Arrow
import matplotlib.transforms as transforms
from math import cos, sin, degrees
import skimage.io as skio


'''https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/'''


def confidence_ellipse(x, y, ax, nstd=2.0, facecolor='none', **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    centre = (np.mean(x), np.mean(y))

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    
    ellipse = Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), facecolor=facecolor, **kwargs)
    
    if width > height:
        r = width
    else:
        r = height
    
    arrow = Arrow(np.mean(x), np.mean(y), r*cos(theta), r*sin(theta), width=1)
    
    ax.add_patch(ellipse)
    ax.add_patch(arrow)
    majorAxis_deg  = degrees(theta)
    #majorAxis_deg = (majorAxis_deg + 360) % 360 # +360 for implementations where mod returns negative numbers
    
    #print(majorAxis_deg)
    #print(ellipse.properties()['angle'])
    
    return ax, majorAxis_deg 



#fileName = '/Users/george/Data/trackpyTest/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity_AllLocs_NNcount.csv'

fileName = '/Users/george/Data/htag_cutout_wt/wt_bin10/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_1_MMStack_Default_bin10_locsID_tracksRG_SVMPredicted_NN_diffusion.csv'

df = pd.read_csv(fileName)
df = df.dropna()
df['track_number'] = df['track_number'].astype(int)

#filtering
longTracks = df[df['n_segments']>60]
longTracks = df[df['SVM']==1]

trackList= longTracks['track_number'].unique().tolist()

fig1, axs1 = plt.subplots(1, 1, figsize=(5, 5))
axs1.set_aspect('equal', adjustable='box')
axs1.scatter(longTracks['zeroed_X'],longTracks['zeroed_Y'])

n = trackList[6]
track = longTracks[longTracks['track_number'] == n]

axs1.scatter(track['zeroed_X'],track['zeroed_Y'])

#fit ellipse to single track
fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
axs2.set_aspect('equal', adjustable='box')
axs2.axvline(c='grey', lw=1)
axs2.axhline(c='grey', lw=1)

_, degree = confidence_ellipse(track['zeroed_X'], track['zeroed_Y'], axs2, edgecolor='red')
axs2.scatter(track['zeroed_X'],track['zeroed_Y'])


fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
axs3.set_aspect('equal', adjustable='box')
axs3.scatter(longTracks['zeroed_X'],longTracks['zeroed_Y'])
axs3.axvline(c='grey', lw=1)
axs3.axhline(c='grey', lw=1)

degreeList = []

for n in trackList:
    track = longTracks[longTracks['track_number'] == n]
    _, degree = confidence_ellipse(track['zeroed_X'], track['zeroed_Y'], axs3, edgecolor='red') 
    degreeList.append(degree)

fig4, axs4 = plt.subplots(1, 1, figsize=(5, 5))
axs4.hist(degreeList,10)

correctedDegList = []
for deg in degreeList:
    if deg < 0:
        deg = deg + 180
    correctedDegList.append(deg)
    
fig5, axs5 = plt.subplots(1, 1, figsize=(5, 5))
axs5.hist(correctedDegList,10)

####### ACTIN #########
'''https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html'''


from skimage.filters import threshold_otsu
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches
import math
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

actinFileName = '/Users/george/Data/htag_cutout_wt/DICRICMACTIN/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_1_ACTIN.tif'
actin_img = skio.imread(actinFileName)[0]

thresh = threshold_otsu(actin_img)
binary = actin_img > thresh
edges = canny(binary, sigma=3)

bw = closing(actin_img > thresh, square(5))
# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
image_label_overlay = label2rgb(label_image, image=actin_img, bg_label=0)

fig6, [axs6,axs7,axs8] = plt.subplots(1, 3, figsize=(15, 5))
axs6.imshow(actin_img, origin='lower')
axs6.scatter(longTracks['x'],longTracks['y'])

axs7.imshow(cleared, origin='lower')
axs8.imshow(image_label_overlay, origin='lower')

orientationList = []

for props in regionprops(label_image):
    # take regions with large enough areas
    if props.area >= 50:
        y0, x0 = props.centroid
        orientation = props.orientation
        orientationList.append(degrees(orientation))
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
    
        axs8.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        axs8.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        axs8.plot(x0, y0, '.g', markersize=15)
        
        #plot bounding box
        #minr, minc, maxr, maxc = props.bbox
        #bx = (minc, maxc, maxc, minc, minc)
        #by = (minr, minr, maxr, maxr, minr)
        #axs8.plot(bx, by, '-b', linewidth=2.5)        


correctedDegList_actin = []
for deg in orientationList:
    if deg < 0:
        deg = deg + 180
    correctedDegList_actin.append(deg)

fig7, axs9 = plt.subplots(1, 1, figsize=(5, 5))
axs9.hist(correctedDegList_actin,10)
