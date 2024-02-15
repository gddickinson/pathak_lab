##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:55:40 2023

@author: george
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import os, glob
import skimage.io as skio
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
from flika.process.file_ import save_file_gui, open_file_gui, open_file
from flika.roi import open_rois

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox

from flika import start_flika

from sklearn.neighbors import KDTree


def countNNwithRadius(train,test,r=1):
    tree = KDTree(train, leaf_size=5)
    count = tree.query_radius(test, r=r, count_only=True)
    return count


def getNNinFrame(tracksDF, radiusList=[3,5,10,20,30]):
    #sort by frame
    tracksDF = tracksDF.sort_values(by=['frame'])

    for r in radiusList:

        #make empty list to store NN distances & indexes
        countList = []
        #get list of frames in tracksDF to iterate over
        frames = tracksDF['frame'].unique().tolist()

        #get nn for each centroid position by frame
        for i, frame in enumerate(frames):
            #filter by frame
            frameXY = tracksDF[tracksDF['frame'] == frame][['x','y']].to_numpy()
            #nearest neighbour
            count = countNNwithRadius(frameXY,frameXY,r=r)
            #subtract 1 from count (to self)
            count = count -1
            #append distances and indexes of 1st neighbour to list
            countList.extend(count)
            #nnIndexList.extend(indexes[:,1])
            print('\r' + 'NN analysis complete for frame{} of {} of r= {}'.format(i,len(frames),r), end='\r')

        #add results to dataframe
        tracksDF['nnCountInFrame_within_{}_pixels'.format(r)] =  countList

    tracksDF = tracksDF.sort_values(['track_number','frame'], ascending=True)
    #print('\r' + 'NNcount-analysis added', end='\r')

    return tracksDF

def getIntensities(dataArray, pts):
    #intensities retrieved from image stack using point data (converted from floats to ints)

    n, h, w = dataArray.shape

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

        #get mean pixels values for 3x3 square
        #intensities.append((np.mean(dataArray[frame][yMin:yMax,xMin:xMax]) - np.min(dataArray[frame]))) #no longer subtracting min
        intensities.append(np.mean(dataArray[frame][yMin:yMax,xMin:xMax]))

    return intensities

def addLagDisplacementToDF(tracksDF):
    #align x and y locations of link
    tracksDF = tracksDF.assign(x2=tracksDF.x.shift(-1))
    tracksDF = tracksDF.assign(y2=tracksDF.y.shift(-1))

    #calculate lag Distance
    tracksDF['x2-x1_sqr'] = np.square(tracksDF['x2']-tracksDF['x'])
    tracksDF['y2-y1_sqr'] = np.square(tracksDF['y2']-tracksDF['y'])
    tracksDF['distance'] = np.sqrt((tracksDF['x2-x1_sqr']+tracksDF['y2-y1_sqr']))

    #mask final track position lags
    tracksDF['mask'] = True
    tracksDF.loc[tracksDF.groupby('track_number').tail(1).index, 'mask'] = False  #mask final location

    #get lags for all track locations (not next track)
    tracksDF['lag'] = tracksDF['distance'].where(tracksDF['mask'])

    #add track mean lag distance to all rows
    tracksDF['meanLag'] = tracksDF.groupby('track_number')['lag'].transform('mean')

    #add track length for each track row
    tracksDF['track_length'] = tracksDF.groupby('track_number')['lag'].transform('sum')

    #add 'radius_gyration' (scaled by mean lag displacement)
    tracksDF['radius_gyration_scaled'] = tracksDF['radius_gyration']/tracksDF['meanLag']

    #add 'radius_gyration' (scaled by n_segments)
    tracksDF['radius_gyration_scaled_nSegments'] = tracksDF['radius_gyration']/tracksDF['n_segments']

    #add 'radius_gyration' (scaled by track_length)
    tracksDF['radius_gyration_scaled_trackLength'] = tracksDF['radius_gyration']/tracksDF['track_length']

    print('\r' + 'lags added', end='\r')

    return tracksDF

def addPointsToUnbinnedTracks(df, tiffFile, roi_1, cameraEstimate, bin_size = 10):
    #remove unlinked points
    df = df[~df['track_number'].isna()]
    unlinked = df[df['track_number'].isna()]

    #get track_ids
    trackList = df['track_number'].unique()

    #add frame number scaled to unbinned recording
    df['frame_unbinned'] = df['frame'] * bin_size

    #load tiff
    A = skio.imread(tiffFile)

    newDF = pd.DataFrame()

    for i, track_number in enumerate(trackList):
        #inititate df to store track results
        tempDF = pd.DataFrame()

        #get track data
        trackDF = df[df['track_number'] == int(track_number)]

        # Extract x,y,frame data for each point
        points = np.column_stack((trackDF['frame_unbinned'].to_list(), trackDF['x'].to_list(), trackDF['y'].to_list()))

        #interpolate points for missing frames
        allFrames = range(int(min(points[:,0])), int(max(points[:,0]))+1)
        xinterp = np.interp(allFrames, points[:,0], points[:,1])
        yinterp = np.interp(allFrames, points[:,0], points[:,2])

        points = np.column_stack((allFrames, xinterp, yinterp))

        #get mean intensities from unbinned tiff
        intensities = getIntensities(A, points)

        #populate tempDF
        tempDF['frame'] = allFrames
        tempDF['track_number'] = track_number
        tempDF['x'] = xinterp
        tempDF['y'] = yinterp
        tempDF['intensity'] = intensities

        #set positions relative to origin of 0,0
        minFrame = tempDF['frame'].min()
        origin_X = float(tempDF[tempDF['frame'] == minFrame]['x'])
        origin_Y = float(tempDF[tempDF['frame'] == minFrame]['y'])
        tempDF['zeroed_X'] = tempDF['x'] - origin_X
        tempDF['zeroed_Y'] = tempDF['y'] - origin_Y
        #generate lag numbers
        tempDF['lagNumber'] = tempDF['frame'] - minFrame
        #calc distance from origin
        tempDF['distanceFromOrigin'] = np.sqrt(  (np.square(tempDF['zeroed_X']) + np.square(tempDF['zeroed_Y']))   )

        #add differantial for distance
        diff = np.diff(tempDF['distanceFromOrigin'].to_numpy()) / np.diff(tempDF['lagNumber'].to_numpy())
        diff = np.insert(diff,0,0)
        tempDF['dy-dt: distance'] = diff

        #copy over SVM properties from binned
        props_list = ['radius_gyration', 'asymmetry',
        'skewness', 'kurtosis','fracDimension', 'netDispl',
        'Straight', 'Experiment', 'SVM',
        'nnDist_inFrame']

        for prop in props_list:
            tempDF[prop] = trackDF[prop]

        tempDF['n_segments'] = len(allFrames)

        #add lag props
        tempDF = addLagDisplacementToDF(tempDF)

        #add background values for each frame
        for frame, value in enumerate(roi_1):
            tempDF.loc[tempDF['frame'] == frame, 'roi_1'] = value
        for frame, value in enumerate(cameraEstimate):
            tempDF.loc[tempDF['frame'] == frame, 'camera black estimate'] = value

        newDF = pd.concat([newDF,tempDF])


    #get squared values
    newDF['d_squared'] = np.square(newDF['distanceFromOrigin'])
    newDF['lag_squared'] = np.square(newDF['lag'])

    #add delta-t for each lag
    newDF['dt'] = np.insert(newDF['frame'].to_numpy()[1:],-1,0) - newDF['frame'].to_numpy()
    newDF['dt'] = newDF['dt'].mask(newDF['dt'] <= 0, None)
    #instantaneous velocity
    newDF['velocity'] = newDF['lag']/newDF['dt']
    #direction relative to 0,0 origin : 360 degreeas
    degrees = np.arctan2(newDF['zeroed_Y'].to_numpy(), newDF['zeroed_X'].to_numpy())/np.pi*180
    degrees[degrees < 0] = 360+degrees[degrees < 0]
    newDF['direction_Relative_To_Origin'] =  degrees
    #add mean track velocity
    newDF['meanVelocity'] = newDF.groupby('track_number')['velocity'].transform('mean')

    #add background subtracted intensity
    newDF['intensity - mean roi1'] = newDF['intensity'] - np.mean(newDF['roi_1'])
    newDF['intensity - mean roi1 and black'] = newDF['intensity'] - np.mean(newDF['roi_1']) - np.mean(newDF['camera black estimate'])

    #drop intermediate cols
    newDF = newDF.drop(columns=['x2', 'y2', 'x2-x1_sqr', 'y2-y1_sqr', 'distance', 'mask'])

    #add unlinked points back
    unlinked = unlinked[newDF.columns]
    newDF = pd.concat([newDF, unlinked])
    return newDF

def getTrackIDs(df, grouping='none'):
    if grouping == 'none':
        #return unique IDs
        return df['track_number'].unique()

    elif grouping == 'in pixel':
        #combine tracks for SVM3 at same location by rounding XY
        df['meanXloc'] = round(df.groupby(['track_number'])['x'].transform('mean'),0)
        df['meanYloc'] = round(df.groupby(['track_number'])['y'].transform('mean'),0)

        tempDF = df[['track_number', 'n_segments', 'meanXloc', 'meanYloc']]
        tempDF = tempDF.sort_values('n_segments')
        tempGroup = tempDF.drop_duplicates(subset=['meanXloc','meanYloc'], keep='last').reset_index(drop = True)
        # this tracklist is used to filter out smaller duplicate fragments at a SVM3 site
        return tempGroup['track_number'].unique()

    elif grouping == 'hcluster':
        #use hierarchical clustering to group type 3 sites by distance
        import scipy.cluster.hierarchy as hcluster
        thresh = 3
        data = df[['x','y']].to_numpy()
        df['cluster'] = hcluster.fclusterdata(data, thresh, criterion="distance")
        tempDF = df[['track_number', 'n_segments', 'cluster']]
        tempDF = tempDF.sort_values('n_segments')
        tempGroup = tempDF.drop_duplicates(subset=['cluster'], keep='last').reset_index(drop = True)
        # this tracklist is used to filter out smaller duplicate fragments at a SVM3 site
        return tempGroup['track_number'].unique()


def movingaverage(interval, window_size=10):
    window= np.ones(int(window_size))/float(window_size)
    smooth = np.convolve(interval, window, 'valid')
    start= int(((len(interval) - len(smooth)))/2)
    end = len(interval) - (start + len(smooth))
    return np.pad(smooth,(start,end),'edge')

def addMissingPoints(df, tiffFile, roi_1, cameraEstimate, tracksToKeep='all', grouping='none'):
    #add blank column to record ROI over meanXY intensity values
    df['intensity_roiOnMeanXY'] = None
    df['intensity_roiOnMeanXY - mean roi1'] = None
    df['intensity_roiOnMeanXY - mean roi1 and black'] = None

    df['roi_1 smoothed'] = None
    df['intensity_roiOnMeanXY - smoothed roi_1'] = None
    df['intensity - smoothed roi_1'] = None

    #remove unlinked points
    df = df[~df['track_number'].isna()]
    unlinked = df[df['track_number'].isna()]

    #filter SVM type 3
    df_mobile = df[df['SVM'] != 3]
    df = df[df['SVM'] == 3]

    #get SVM3 track_ids
    trackList = getTrackIDs(df, grouping=grouping)

    #load tiff
    A = skio.imread(tiffFile)

    #total length of recording
    n_frames,w,h = A.shape

    newDF = pd.DataFrame()

    #iterate over tracks - add intensity value for every frame in recording
    for i, track_number in enumerate(trackList):
        #inititate df to store track results
        tempDF = pd.DataFrame()

        #get track data
        trackDF = df[df['track_number'] == int(track_number)]

        # Extract x,y,frame data for each point
        points = np.column_stack((trackDF['frame'].to_list(), trackDF['x'].to_list(), trackDF['y'].to_list()))

        #interp[olate missing frames
        interpFrames = range(int(min(points[:,0])), int(max(points[:,0]))+1)
        xinterp = np.interp(interpFrames, points[:,0], points[:,1])
        yinterp = np.interp(interpFrames, points[:,0], points[:,2])

        # add frames to ends based on mean site position
        allFrames = np.array(range(0, n_frames))
        allX = np.pad(xinterp,(int(min(points[:,0])), n_frames - int(max(points[:,0]))-1), 'mean')
        allY = np.pad(yinterp,(int(min(points[:,0])), n_frames - int(max(points[:,0]))-1), 'mean')

        points = np.column_stack((allFrames, allX, allY))

        # fit roi over mean site position
        xVals = np.zeros_like(allX)
        yVals = np.zeros_like(allY)
        xVals.fill(np.mean(allX))
        yVals.fill(np.mean(allY))
        pointsMeanXY = np.column_stack((allFrames, xVals, yVals))

        #get mean intensities from unbinned tiff
        intensities = getIntensities(A, points)
        intensitiesMeanXY = getIntensities(A, pointsMeanXY)

        #populate tempDF
        tempDF['frame'] = allFrames
        tempDF['track_number'] = track_number
        tempDF['x'] = allX
        tempDF['y'] = allY
        tempDF['intensity'] = intensities
        tempDF['intensity_roiOnMeanXY'] = intensitiesMeanXY

        #set positions relative to origin of 0,0
        minFrame = tempDF['frame'].min()
        origin_X = float(tempDF[tempDF['frame'] == minFrame]['x'])
        origin_Y = float(tempDF[tempDF['frame'] == minFrame]['y'])
        tempDF['zeroed_X'] = tempDF['x'] - origin_X
        tempDF['zeroed_Y'] = tempDF['y'] - origin_Y
        #generate lag numbers
        tempDF['lagNumber'] = tempDF['frame'] - minFrame
        #calc distance from origin
        tempDF['distanceFromOrigin'] = np.sqrt(  (np.square(tempDF['zeroed_X']) + np.square(tempDF['zeroed_Y']))   )

        #add differantial for distance
        diff = np.diff(tempDF['distanceFromOrigin'].to_numpy()) / np.diff(tempDF['lagNumber'].to_numpy())
        diff = np.insert(diff,0,0)
        tempDF['dy-dt: distance'] = diff

        #copy over SVM properties
        props_list = ['radius_gyration', 'asymmetry',
        'skewness', 'kurtosis','fracDimension', 'netDispl',
        'Straight', 'Experiment', 'SVM',
        'nnDist_inFrame']

        for prop in props_list:
            tempDF[prop] = trackDF[prop].iloc[0]


        tempDF['n_segments'] = len(allFrames)

        #add lag props
        tempDF = addLagDisplacementToDF(tempDF)

        #add background values for each frame
        for frame, value in enumerate(roi_1):
            tempDF.loc[tempDF['frame'] == frame, 'roi_1'] = value
        for frame, value in enumerate(cameraEstimate):
            tempDF.loc[tempDF['frame'] == frame, 'camera black estimate'] = value

        # Smooth the roi1 signal
        smoothing = int(len(roi_1)/10)
        smoothed_roi_1 = movingaverage(roi_1, window_size=smoothing)

        for frame, value in enumerate(smoothed_roi_1):
            tempDF.loc[tempDF['frame'] == frame, 'roi_1 smoothed'] = value


        newDF = pd.concat([newDF, tempDF])


    #get squared values
    newDF['d_squared'] = np.square(newDF['distanceFromOrigin'])
    newDF['lag_squared'] = np.square(newDF['lag'])

    #add delta-t for each lag
    newDF['dt'] = np.insert(newDF['frame'].to_numpy()[1:],-1,0) - newDF['frame'].to_numpy()
    newDF['dt'] = newDF['dt'].mask(newDF['dt'] <= 0, None)
    #instantaneous velocity
    newDF['velocity'] = newDF['lag']/newDF['dt']
    #direction relative to 0,0 origin : 360 degreeas
    degrees = np.arctan2(newDF['zeroed_Y'].to_numpy(), newDF['zeroed_X'].to_numpy())/np.pi*180
    degrees[degrees < 0] = 360+degrees[degrees < 0]
    newDF['direction_Relative_To_Origin'] =  degrees
    #add mean track velocity
    newDF['meanVelocity'] = newDF.groupby('track_number')['velocity'].transform('mean')

    #drop intermediate cols
    newDF = newDF.drop(columns=['x2', 'y2', 'x2-x1_sqr', 'y2-y1_sqr', 'distance', 'mask'])

    #add mobile + unlinked points back
    unlinked = unlinked[newDF.columns]
    df_mobile = df_mobile[newDF.columns]
    if tracksToKeep == 'all':
        newDF = pd.concat([newDF, df_mobile])
    newDF = pd.concat([newDF, unlinked])

    #add background subtracted intensity
    newDF['intensity - mean roi1'] = newDF['intensity'] - np.mean(newDF['roi_1'])
    newDF['intensity - mean roi1 and black'] = newDF['intensity'] - np.mean(newDF['roi_1']) - np.mean(newDF['camera black estimate'])

    newDF['intensity_roiOnMeanXY - mean roi1'] = newDF['intensity_roiOnMeanXY'] - np.mean(newDF['roi_1'])
    newDF['intensity_roiOnMeanXY - mean roi1 and black'] = newDF['intensity_roiOnMeanXY'] - np.mean(newDF['roi_1']) - np.mean(newDF['camera black estimate'])

    newDF['intensity - smoothed roi_1'] = newDF['intensity'] - newDF['roi_1 smoothed']
    newDF['intensity_roiOnMeanXY - smoothed roi_1'] = newDF['intensity_roiOnMeanXY'] - newDF['roi_1 smoothed']

    newDF = newDF.sort_values(by='track_number')
    return newDF



if __name__ == '__main__':

# =============================================================================
# #######################################################################################
# ##### USE THIS SECTION TO ADD INTERPOLATED POINTS TO BINNED/UNBINNED RECORDINGS   #####
# #######################################################################################
#
#     path = '/Users/george/Desktop/unbinnedTest'
#
#     #bin size in int
#     binSize = 10
#     #this suffix will work if binning done using the script I wrote
#     binSuffix = '_bin{}'.format(binSize)
#
#     #get file names
#     fileList = glob.glob(path + '/**/*_BGsubtract.csv', recursive = True)
#
#     for file in tqdm(fileList):
#         #load analysis df
#         df = pd.read_csv(file)
#         tiffFile_original = os.path.splitext(file)[0].split('_locs')[0] + '.tif'
#         tiffFile = os.path.splitext(file)[0].split(binSuffix)[0] + '.tif'
#
#         roiFileName = 'ROI_' + os.path.basename(file).split('_locs')[0] + '.txt'
#         roiFolder = os.path.dirname(file)
#         roiFile = os.path.join(roiFolder,roiFileName)
#
#         fa = start_flika()
#         data_window = open_file(tiffFile)
#         #get min values for each frame
#         cameraEstimate = np.min(data_window.image, axis=(1,2))
#         #load rois
#         rois = open_rois(roiFile)
#         #get trace for each roi
#         roi_1 = rois[0].getTrace()
#
#         fa.close()
#
#         #add interpolated points
#         newDF = addPointsToUnbinnedTracks(df, tiffFile, roi_1, cameraEstimate, bin_size = binSize)
#
#         #add NN
#         newDF = getNNinFrame(newDF)
#
#         saveName = os.path.splitext(file)[0]+'_unbinned.csv'
#         newDF.to_csv(saveName, index=None)
# =============================================================================


########################################################################################
##### USE THIS SECTION TO ADD INTERPOLATED POINTS TO TRAPPED PUNCTA SITES (SVM 3)  #####
########################################################################################

    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'

    #get file names
    fileList = glob.glob(path + '/**/*_BGsubtract.csv', recursive = True)

    for file in tqdm(fileList):
        #load analysis df
        df = pd.read_csv(file)
        tiffFile = os.path.splitext(file)[0].split('_locs')[0] + '.tif'

        roiFileName = 'ROI_' + os.path.basename(file).split('_locs')[0] + '.txt'
        roiFolder = os.path.dirname(file)
        roiFile = os.path.join(roiFolder,roiFileName)

        fa = start_flika()
        data_window = open_file(tiffFile)
        #get min values for each frame
        cameraEstimate = np.min(data_window.image, axis=(1,2))
        #load rois
        rois = open_rois(roiFile)
        #get trace for each roi
        roi_1 = rois[0].getTrace()

        fa.close()

        #add missing points to trapped sites
        newDF = addMissingPoints(df, tiffFile, roi_1, cameraEstimate, tracksToKeep='all', grouping='hcluster')

        #add NN
        newDF = getNNinFrame(newDF)

        saveName = os.path.splitext(file)[0]+'_trapped-AllFrames.csv'
        newDF.to_csv(saveName, index=None)





