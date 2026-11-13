#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:25:24 2022

@author: george
"""

#%matplotlib qt
#%gui qt

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Import required libraries
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
from sklearn.metrics import confusion_matrix
try:
    from sklearn.metrics import plot_confusion_matrix
except:
    from sklearn.metrics import ConfusionMatrixDisplay as plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import power_transform, PowerTransformer, StandardScaler

import skimage.io as skio

# Qt-free linking and track features live next to this script (also used by the web interface)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pointLinking import PointLinker, addID, load_points, flatten, savetracksCSV
from svmClassification import predict_SPT_class
from pipeline_defaults import FEATURE_TRANSFORM, PIXEL_SIZE_NM, SKIP_FRAMES, LINK_DISTANCE, MIN_SEGMENTS
from trackFeatures import (RadiusGyrationAsymmetrySkewnessKurtosis, FractalDimension, NetDisplacementEfficiency,
                           SummedSinesCosines, getRadiusGyrationForAllTracksinDF, getFeaturesForAllTracksinDF,
                           addLagDisplacementToDF, getNearestNeighbors, getNN, addFeaturesToTracksDF)

# Set recursion limit
recursionLimit = 1000000
sys.setrecursionlimit(recursionLimit)

#print(sys.getrecursionlimit())

# Function definitions

def getSigma():
    ''' This function isn't complete.  I need to cut out a 20x20 pxl window around large amplitude particles '''
    I = g.m.currentWindow.image
    xorigin = 8
    yorigin = 9
    sigma = 2
    amplitude = 50
    p0 = [xorigin, yorigin, sigma, amplitude]
    p, I_fit, _ = fitGaussian(I, p0)
    xorigin, yorigin, sigma, amplitude = p
    return sigma


def convolve(I, sigma):
    from scipy.signal import convolve2d
    G = generate_gaussian(17, sigma)
    newI = np.zeros_like(I)
    for t in np.arange(len(I)):
        print(t)
        newI[t] = convolve2d(I[t], G, mode='same', boundary='fill', fillvalue=0)
    return newI


def get_points(I):
    import scipy.ndimage
    s = scipy.ndimage.generate_binary_structure(3, 1)
    s[0] = 0
    s[2] = 0
    labeled_array, num_features = scipy.ndimage.measurements.label(I, structure=s)
    objects = scipy.ndimage.measurements.find_objects(labeled_array)

    all_pts = []
    for loc in objects:
        offset = np.array([a.start for a in loc])
        pts = np.argwhere(labeled_array[loc] != 0) + offset
        ts = np.unique(pts[:, 0])
        for t in ts:
            pts_t = pts[pts[:, 0] == t]
            x = np.mean(pts_t[:, 1])
            y = np.mean(pts_t[:, 2])
            all_pts.append([t, x, y])
    all_pts = np.array(all_pts)
    return all_pts


def cutout(pt, Movie, width):
    """Cut out a region around a point in a movie"""
    assert width % 2 == 1  # mx must be odd
    t, x, y = pt
    t=int(t)
    mid = int(np.floor(width / 2))
    x0 = int(x - mid)
    x1 = int(x + mid)
    y0 = int(y - mid)
    y1 = int(y + mid)
    mt, mx, my = Movie.shape
    if y0 < 0: y0 = 0
    if x0 < 0: x0 = 0
    if y1 >= my: y1 = my - 1
    if x1 >= mx: x1 = mx - 1
    corner = [x0, y0]
    I = Movie[t, x0:x1 + 1, y0:y1 + 1]
    return I, corner


def refine_pts(pts, blur_window, sigma, amplitude):
    """Refine point locations"""
    global halt_current_computation
    if blur_window is None:
        g.alert("Before refining points, you must select a 'blurred window'")
        return None, False
    new_pts = []
    old_frame = -1
    for pt in pts:
        new_frame = int(pt[0])
        if old_frame != new_frame:
            old_frame = new_frame
            blur_window.imageview.setCurrentIndex(old_frame)
            qApp.processEvents()
            if halt_current_computation:
                halt_current_computation = False
                return new_pts, False
        width = 9
        mid = int(np.floor(width / 2))
        I, corner = cutout(pt, blur_window.image, width)
        xorigin = mid
        yorigin = mid
        p0 = [xorigin, yorigin, sigma, amplitude]
        fit_bounds = [(0, 9), (0, 9), (0, 4), (0, 1000)]
        p, I_fit, _ = fitGaussian(I, p0, fit_bounds)
        xfit = p[0] + corner[0]
        yfit = p[1] + corner[1]
        #                t,  old x, old y, new_x, new_y, sigma, amplitude
        new_pts.append([pt[0], pt[1], pt[2], xfit, yfit, p[2], p[3]])
    new_pts = np.array(new_pts)
    return new_pts, True


class Points(PointLinker):
    """Class to handle point data and track linking (linking itself is in pointLinking.PointLinker)"""


    def clearTracks(self):
        if self.window is not None and not self.window.closed:
            for pathitem in self.pathitems:
                self.window.imageview.view.removeItem(pathitem)
        self.pathitems = []

    def showTracks(self):
        # clear self.pathitems
        self.clearTracks()

        frame = self.window.imageview.currentIndex
        if frame<len(self.tracks_by_frame):
            tracks = self.tracks_by_frame[frame]
            pen = QPen(Qt.green, .4)
            pen.setCosmetic(True)
            pen.setWidth(2)
            for track_idx in tracks:
                pathitem = QGraphicsPathItem(self.window.imageview.view)
                pathitem.setPen(pen)
                self.window.imageview.view.addItem(pathitem)
                self.pathitems.append(pathitem)
                pts = self.txy_pts[self.tracks[track_idx]]
                x = pts[:, 1]+.5; y = pts[:,2]+.5
                path = QPainterPath(QPointF(x[0],y[0]))
                for i in np.arange(1, len(pts)):
                    path.lineTo(QPointF(x[i],y[i]))
                pathitem.setPath(path)



def skip_refinePoints(txy_pts):
    """Skip point refinement"""
    if txy_pts is None:
        return None
    new_pts = []
    for pt in txy_pts:
                    #    t,  old x, old y, new_x, new_y, sigma, amplitude
        new_pts.append([pt[0], pt[1], pt[2], pt[1], pt[2], -1, -1])
    pts_refined = np.array(new_pts)
    return pts_refined


def loadtracksjson(filename):
    """Load tracks from JSON file"""
    obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
    pts = json.loads(obj_text)
    txy_pts = np.array(pts['txy_pts'])
    txy_pts = txy_pts

    txy_pts = skip_refinePoints(txy_pts)
    points = Points(txy_pts)
    points.tracks = pts['tracks']
    points.get_tracks_by_frame()
    return points



def getIntensities(dataArray, pts):
    """Get intensities for points from image data"""
    #intensities retrieved from image stack using point data (converted from floats to ints)

    n, w, h = dataArray.shape #!TODO check w and h are right way around - maybe transposed from flika import (only affects edge cases)

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

def calcFeaturesforFiles(tracksList, minNumberSegments=1):
    """Calculate features for a list of track files"""
    for trackFile in tqdm(tracksList):
            try:
                ##### load data
                tracksDF = pd.read_csv(trackFile)

                tracksDF = addFeaturesToTracksDF(tracksDF, minNumberSegments)

                #saveRg DF
                saveName = os.path.splitext(trackFile)[0] + 'RG.csv'
                tracksDF.to_csv(saveName)
                print('\n new tracks file exported to {}'.format(saveName))

            except Exception as e:
                print(e)
                print('error in RG analysis, skipping {}'.format(trackFile))


def classifyTracks(tracksList, train_data_path, level='', feature_transform=FEATURE_TRANSFORM):
    """Classify tracks using SVM (feature_transform: 'none' or 'boxcox', see pipeline_defaults.py)"""
    for pred_data_path in tqdm(tracksList):
        exptName = os.path.basename(pred_data_path).split('_MMStack')[0]
        predict_SPT_class(train_data_path, pred_data_path, exptName, level=level, feature_transform=feature_transform)


def filterDFandLocs_SVM3(dfFile, locsIdentifer='_tracksRG_SVMPredicted.csv', colName='SVM'):
    """Filter DataFrame and locs file for SVM class 3"""
    #load df with SVM
    df = pd.read_csv(dfFile)
    #load original locs file
    locsFileName = dfFile.split(locsIdentifer)[0] + '.csv'
    locs = pd.read_csv(locsFileName)
    locs['id'] = locs['id'].astype('int')

    #filter by SVM class
    filteredDF = df[df[colName]==3]
    filteredID_list = filteredDF['id'].tolist()


    #filter locs file to get locs not allocated to filtered df
    remainingLocs = locs[~locs['id'].isin(filteredID_list)]
    #remainingLocs = remainingLocs.set_index('id')

    #save df and locs files
    filteredDF.to_csv(dfFile.split('.csv')[0] + '_SVM-3.csv', index=None)
    remainingLocs.to_csv(locsFileName.split('.csv')[0] + '2.csv', index=None)
    return



def filterDFandLocs_SVM2(dfFile, locsIdentifer='_tracks2RG2_SVMPredicted2.csv', colName='SVM'):
    """Filter DataFrame and locs file for SVM class 2"""
    #load df with SVM
    df = pd.read_csv(dfFile)

    #load original locs file
    locsFileName = dfFile.split(locsIdentifer)[0] + '.csv'
    locs = pd.read_csv(locsFileName)

    #load SVM3 df
    df_SVM3_file = dfFile.split('_locsID2_tracks2RG2_SVMPredicted2.csv')[0] + '_locsID_tracksRG_SVMPredicted_SVM-3.csv'
    df_SVM3 = pd.read_csv(df_SVM3_file)

    #filter by SVM class
    df_SVM2 = df[df[colName]==2]
    df_SVM3_extras = df[df[colName]==3]

    #combine SVM3 tracks
    maxTrackNumber = max(df_SVM3['track_number'])
    df_SVM3_extras['track_number'] = df_SVM3_extras['track_number'] + maxTrackNumber
    df_SVM3 = pd.concat([df_SVM3, df_SVM3_extras])

    df_SVM2and3 = pd.concat([df_SVM2, df_SVM3])

    #get ids of SVM tracks
    filteredID_list = df_SVM2and3['id'].tolist()

    #filter locs file to get locs not allocated to filtered df
    remainingLocs = locs[~locs['id'].isin(filteredID_list)]
    remainingLocs['id'] = remainingLocs['id'].astype('int')

    #save df and locs files
    df_SVM2.to_csv(dfFile.split('.csv')[0] + '_SVM-2.csv', index=None)
    df_SVM3.to_csv(df_SVM3_file.split('_locsID_tracksRG_SVMPredicted_SVM-3.csv')[0] + '_locsID2_tracks2RG2_SVMPredicted2_SVM-3.csv', index=None)
    remainingLocs.to_csv(locsFileName.split('_locsID2.csv')[0] + '_locsID3.csv', index=None)

    return

def linkFiles(tiffList, pixelSize = PIXEL_SIZE_NM, frameLength = 1, skipFrames = 1, distanceToLink = 3, level=''):
    """Link files using flika (pixelSize in nm)"""
    for fileName in tqdm(tiffList):

        #set file & save names
        pointsFileName = os.path.splitext(fileName)[0] + '_locsID{}.csv'.format(level)
        lagsHistoSaveName = os.path.splitext(pointsFileName)[0] + '_lagsHisto{}.txt'.format(level)
        tracksSaveName = os.path.splitext(pointsFileName)[0] + '_tracks{}.csv'.format(level)

        #import tiff to flika
        data_window = open_file(fileName)
        #import points
        txy_pts = load_points(pointsFileName, pixelsize=pixelSize)
        p = Points(txy_pts)
        #link points
        p.link_pts(skipFrames,distanceToLink)
        #get background subtracted intensity for each point
        p.getIntensities(data_window.imageArray())
        #save tracks
        tracks = p.tracks
        savetracksCSV(p, tracksSaveName, pointsFileName, pixelSize=pixelSize)
        #export SLD histogram
        #SLD_hist = SLD_Histogram(p, pixelSize, frameLength)
        #SLD_hist.export_histogram(autoSave = True, autoFileName = lagsHistoSaveName)

        #close flika windows
        #SLD_hist.close()
        g.m.clear()
    return

def linkFilesNoFlika(tiffList, pixelSize = PIXEL_SIZE_NM, frameLength = 1, skipFrames = 1, distanceToLink = 3, level='', allFiles=[], includeRecursionProblem=False):
    """Link files without using flika (pixelSize in nm)"""
    for fileName in tqdm(tiffList):
        print('linking : {}'.format(fileName))

        #set file & save names
        pointsFileName = os.path.splitext(fileName)[0] + '_locsID{}.csv'.format(level)
        lagsHistoSaveName = os.path.splitext(pointsFileName)[0] + '_lagsHisto{}.txt'.format(level)
        tracksSaveName = os.path.splitext(pointsFileName)[0] + '_tracks{}.csv'.format(level)

        # #skip linking if link file already exists
        # if tracksSaveName in allFiles:
        #     print('skipping: {}'.format(tracksSaveName))
        #     continue

        #import tiff to flika
        A = skio.imread(fileName, plugin='tifffile')
        #orient to match flika array
        A = np.rot90(A, axes=(1,2))
        A = np.fliplr(A)
        #import points
        txy_pts = load_points(pointsFileName, pixelsize=pixelSize)
        p = Points(txy_pts)
        #link points
        p.link_pts(skipFrames,distanceToLink)
        #get background subtracted intensity for each point
        p.getIntensities(A)
        #save tracks
        if includeRecursionProblem:
          tracks = p.tracks
          savetracksCSV(p, tracksSaveName, pointsFileName, pixelSize=pixelSize)
        else:
            if p.recursiveFailure == False:
                tracks = p.tracks
                savetracksCSV(p, tracksSaveName, pointsFileName, pixelSize=pixelSize)
            else:
                print('recursion error, skipped {}'.format(fileName))

    return


def importJSON(tiffList, pixelSize = PIXEL_SIZE_NM, level=''):
    """Import tracks from JSON files (pixelSize in nm)"""
    for fileName in tqdm(tiffList):

        #set file & save names
        jsonFileName = os.path.splitext(fileName)[0] + '{}.json'.format(level)
        #pointsFileName = os.path.splitext(fileName)[0] + '_locsID{}.csv'.format(level)
        lagsHistoSaveName = os.path.splitext(fileName)[0] + '_lagsHisto{}.txt'.format(level)
        tracksSaveName = os.path.splitext(fileName)[0] + '_locsID_tracks{}.csv'.format(level)

        #import tiff to flika
        data_window = open_file(fileName)
        #import points

        p =loadtracksjson(jsonFileName)
        print('tracks loaded')
        #link points
        #p.link_pts(skipFrames,distanceToLink)

        #get background subtracted intensity for each point
        p.getIntensities(data_window.imageArray())

        #save tracks
        tracks = p.tracks
        savetracksCSV(p, tracksSaveName, None, noLocsFile=True, pixelSize=pixelSize)
        #export SLD histogram
        #SLD_hist = SLD_Histogram(p, pixelSize, frameLength)
        #SLD_hist.export_histogram(autoSave = True, autoFileName = lagsHistoSaveName)

        #close flika windows
        #SLD_hist.close()
        g.m.clear()


def linkFiles_trackpy(tiffList, pixelSize = PIXEL_SIZE_NM, skipFrames = 1, distanceToLink = 3, level='', linkingType='standard', maxDistance=5):
    """Link files using trackpy (pixelSize in nm)"""
    #try loading trackpy
    try:
        import trackpy as tp
    except:
        print("trackpy installation not detected. Install instructions at 'http://soft-matter.github.io/trackpy/v0.5.0/installation.html' ")
        return

    for fileName in tqdm(tiffList):

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
            tracks = pred.link_df(locs, maxDistance, memory=gapSize, adaptive_stop=0.1, adaptive_step=0.95)

        #get background subtracted intensity for each point
        A = skio.imread(fileName, plugin='tifffile')
        pts = tracks[['frame','x','y']]
        #pts['frame'] = pts['frame']-1
        pts = pts.to_numpy()
        intensities = getIntensities(A, pts)
        tracks['intensity'] = intensities

        #rename cols to match pynsight
        tracks['track_number'] = tracks['particle']
        #sort by track_number and frame
        tracks = tracks.sort_values(by=['track_number', 'frame'])
        #reorder columns and drop particles
        tracks = tracks[['track_number','frame', 'x', 'y','intensity', 'id', 'x [nm]', 'y [nm]']]
        #Save tracks
        tracks.to_csv(tracksSaveName)
        print('tracks file {} saved'.format(tracksSaveName))
    return

if __name__ == '__main__':
    ##### RUN ANALYSIS

    # Set paths and parameters
    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'
    #path = '/Users/george/Data/test'
    #path = '/Users/george/Data/gabby_missingIntensities'

    # Get file lists
    #tiffList = glob.glob(path + '/**/*_bin10.tif', recursive = True)
    #tiffList = glob.glob(path + '/**/*_crop200.tif', recursive = True)
    tiffList = glob.glob(path + '/**/*.tif', recursive = True)

    allFiles = glob.glob(path + '/**/*', recursive = True)

    #training data path
    trainpath = '/Users/george/Data/from_Gabby/gabby_scripts/workFlow/training_data/tdTomato_37Degree_CytoD_training_feats.csv'

    #minimum number of link segments (need at least 2 to avoid colinearity in feature calc)
    #minLinkSegments = 2
    minLinkSegments = MIN_SEGMENTS    #gabby: 6 (pipeline_defaults.py)

    #max number of gap frames to skip
    #gapSize = 2
    gapSize = SKIP_FRAMES #gabby: 18 (pipeline_defaults.py)

    #max distance in pixels to allow a linkage
    #distance = 5
    distance = LINK_DISTANCE #gabby: 3 (pipeline_defaults.py)

    #pixel size in nm (default: pipeline_defaults.PIXEL_SIZE_NM, 108 nm for the lab's camera)
    pixelSize_new = PIXEL_SIZE_NM

    #SVM feature transform: 'none' (default, as published; more accurate on the lab's training set) or 'boxcox' (see pipeline_defaults.py)
    featureTransform = FEATURE_TRANSFORM

    #trackpy options
    linkingType = 'standard'
    #linkingType = 'adaptive'
    #linkingType = 'velocityPredict'
    #linkingType = 'adaptive + velocityPredict'
    maxSearchDistance = 6 #for adaptive search

    ##########################################################################
    #STEP 2.1 Add IDs to locs file
    ##########################################################################
    ## get expt folder list
    locsList = glob.glob(path + '/**/*_locs.csv', recursive = True)

    for file in tqdm(locsList):
        addID(file)


    # #loop through linkage cut off didstances
    # for distance in tqdm(range(3,4)):


    ##########################################################################
    #STEP 3 link points
    ##########################################################################

    # ## LINK USING FLIKA (Kyle's code)
    # fa = start_flika()
    # ##run linking on all tiffs in directory
    # linkFiles(tiffList, skipFrames = gapSize, distanceToLink = distance, pixelSize = pixelSize_new)
    # fa.close()

    # ## LINK USING Kyle's pynsight code outside of Flika
    linkFilesNoFlika(tiffList, skipFrames = gapSize, distanceToLink = distance, pixelSize = pixelSize_new, allFiles=allFiles)


    # LINK USING TRACKPY
    #linkFiles_trackpy(tiffList, skipFrames = gapSize, distanceToLink = distance, pixelSize = pixelSize_new, linkingType=linkingType, maxDistance=maxSearchDistance)


    ##########################################################################
    ## IMPORT POINTS AND TRACKS FROM JSON (INSTEAD OF PREVIOUS STEPS)
    #fa = start_flika()
    #importJSON(tiffList, pixelSize = pixelSize_new)
    #fa.close()
    ##########################################################################


    ##########################################################################
    #STEP  4 Calculate RG and Features
    ##########################################################################
    #get folder paths
    tracksList = glob.glob(path + '/**/*_locsID_tracks.csv', recursive = True)

    #run analysis - filter for track lengths > minLinkSements
    calcFeaturesforFiles(tracksList, minNumberSegments=minLinkSegments)

    ##########################################################################
    #STEP  5 Classify Tracks
    ##########################################################################
    #get folder paths
    tracksList = glob.glob(path + '/**/*_tracksRG.csv', recursive = True)

    #run analysis
    #classifyTracks(tracksList, trainpath, level='_cutoff_{}'.format(distance)) #uncomment if running multiple cut off values
    classifyTracks(tracksList, trainpath, level=''.format(distance), feature_transform=featureTransform)


    ##########################################################################
    #STEP 6 If using thunderStorm add 'X [nm]', Y [nm]', 'intensity [photons]' cols, if dropped
    ##########################################################################
    #get folder paths
    fileList = glob.glob(path + '/**/*_SVMPredicted.csv', recursive = True)
    locsFileList = glob.glob(path + '/**/*_locsID.csv', recursive = True)

    for i,file in enumerate(fileList):
        # load results file and corresponding locs file
        resultFile = pd.read_csv(file)
        locsFile = pd.read_csv(locsFileList[i])[['id','x [nm]','y [nm]','intensity [photon]']]

        #join tables based on id
        joinDF = resultFile.merge(locsFile, on='id', how='inner')

        #overwrite result file with merged table
        joinDF.to_csv(file)
