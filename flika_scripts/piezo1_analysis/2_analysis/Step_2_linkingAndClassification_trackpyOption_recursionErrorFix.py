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
from pointLinking import PointLinker
from trackFeatures import (RadiusGyrationAsymmetrySkewnessKurtosis, FractalDimension, NetDisplacementEfficiency,
                           SummedSinesCosines, getRadiusGyrationForAllTracksinDF, getFeaturesForAllTracksinDF,
                           addLagDisplacementToDF, getNearestNeighbors, getNN, addFeaturesToTracksDF)

# Set recursion limit
recursionLimit = 1000000
sys.setrecursionlimit(recursionLimit)

#print(sys.getrecursionlimit())

# Function definitions

def addID(file):
    """Add ID column to CSV file"""
    df = pd.read_csv(file)
    df['id'] = df['id'].astype('int')
    df['frame'] = df['frame'].astype('int') -1 #reseting first frame to zero to maych flika display
    df.to_csv(file.split('.csv')[0] + 'ID.csv')


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


def load_points(filename, pixelsize = 108):
    """Load points from CSV file"""
    #filename = open_file_gui("Open points from thunderstorm csv, filetypes='*.csv")
    if filename is None:
        return None

    pointsDF = pd.read_csv(filename)
    pointsDF['frame'] = pointsDF['frame'].astype(int)
    pointsDF['x'] = pointsDF['x [nm]'] / pixelsize
    pointsDF['y'] = pointsDF['y [nm]'] / pixelsize
    pointsDF = pointsDF[['frame','x','y']]
    all_pts = pointsDF.to_numpy()

    print('--- Point Data loaded ---')
    print(pointsDF)

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


def flatten(l):
    """Flatten a list of lists"""
    return [item for sublist in l for item in sublist]

def savetracksCSV(points, filename, locsFileName, noLocsFile=False):
    """Save tracks to CSV file"""
    tracks = points.tracks
    if isinstance(tracks[0][0], int):
        tracks = [[np.ndarray.item(a) for a in b] for b in tracks]
    txy_pts = points.txy_pts.tolist()

    txy_intensities = points.intensities

    txy_indexes = flatten(points.pts_idx_by_frame)


    #filter tracks list to only include linked tracks
    linkedTracks = [item for item in tracks if len(item) > 1]


    #get xy coordinates and intensities for linked tracks
    trackNumber = []

    txy_intensitiesByTrack = []

    txy_indexesByTrack = []


    # for i, indices in enumerate(linkedTracks):
    #     trackNumber.append( i )
    #     txy_ptsByTrack.append(list(txy_pts[j] for j in indices))
    #     #txy_intensitiesByTrack.append(list(txy_intensities[k] for k in indices))


    frameList = []
    xList = []
    yList = []

    for i, indices in enumerate(linkedTracks):
        ptsList = list(txy_pts[j] for j in indices)

        intensitiesList =list(txy_intensities[k] for k in indices)

        if noLocsFile == False:
            indexList =list(txy_indexes[l] for l in indices)

        for pts in ptsList:
            trackNumber.append(i)
            frameList.append(pts[0])
            xList.append(pts[1])
            yList.append(pts[2])


        for intensity in intensitiesList:
            txy_intensitiesByTrack.append(intensity)

        if noLocsFile == False:
            for ind in indexList:
                txy_indexesByTrack.append(ind)


    #make dataframe of tracks, xy coordianates and intensities for linked tracks
    if noLocsFile == False:
        dict = {'track_number': trackNumber, 'frame':frameList, 'x': xList, 'y':yList, 'intensity': txy_intensitiesByTrack, 'file_id': txy_indexesByTrack}
    else:
        dict = {'track_number': trackNumber, 'frame':frameList, 'x': xList, 'y':yList, 'intensity': txy_intensitiesByTrack}

    linkedtrack_DF = pd.DataFrame(dict)

    if noLocsFile == False:
        #match id to locs file values (starting at 1)

        #linkedtrack_DF['file_id'] = linkedtrack_DF['file_id'] + 1
        locs_DF = pd.read_csv(locsFileName)
        id_list = locs_DF['id'].tolist()
        file_id_list = linkedtrack_DF['file_id'].tolist()

        idsForlinkedPoints = [id_list[i] for i in file_id_list]

        linkedtrack_DF['id'] = idsForlinkedPoints
    else:
        linkedtrack_DF['id'] = linkedtrack_DF.index

    #convert back to nm
    linkedtrack_DF['x [nm]'] = linkedtrack_DF['x'] * 108
    linkedtrack_DF['y [nm]'] = linkedtrack_DF['y'] * 108

    #cast frames as int
    linkedtrack_DF['frame'] = linkedtrack_DF['frame'].astype('int')

    #round intensity
    linkedtrack_DF['intensity'] = linkedtrack_DF['intensity'].round(2)

    #drop file-id
    if noLocsFile == False:
        linkedtrack_DF = linkedtrack_DF.drop(['file_id'], axis=1)

    #save df as csv
    #linkedtrack_DF = linkedtrack_DF.sort_values('id')
    linkedtrack_DF.to_csv(filename, index=True)


    print('tracks file {} saved'.format(filename))


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


    def getIntensities(self, dataArray):
        #intensities retrieved from image stack using point data (converted from floats to ints)

        n, w, h = dataArray.shape

        #clear intensity list
        self.intensities = []

        for point in tqdm(self.txy_pts):
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

            #get mean pixels values for 3x3 square
            #self.intensities.append((np.mean(dataArray[frame][xMin:xMax,yMin:yMax]) - np.min(dataArray[frame]))) #no longer subtracting min
            self.intensities.append(np.mean(dataArray[frame][xMin:xMax,yMin:yMax]))

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


def predict_SPT_class(train_data_path, pred_data_path, exptName, level):
    """Computes predicted class for SPT data where
        1:Mobile, 2:Intermediate, 3:Trapped

    Args:
        train_data_path (str): complete path to training features data file in .csv format, ex. 'C:/data/tdTomato_37Degree_CytoD_training_feats.csv'
                               should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('tdTomato_37Degree'),
                               a 'TrackID' column with unique numerical IDs for each track, and an 'Elected_Label' column derived from human voting.
        pred_data_path (str): complete path to features that need predictions in .csv format, ex. 'C:/data/newconditions/gsmtx4_feature_data.csv'
                               should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('GsMTx4'),
                               and a 'TrackID' column with unique numerical IDs for each track.

    Output:
        .csv file of dataframe of prediction_file features with added SVMPredictedClass column output to pred_data_path parent folder
    """
    def prepare_box_cox_data(data):
        data = data.copy()
        for col in data.columns:
            minVal = data[col].min()
            if minVal <= 0:
                data[col] += (np.abs(minVal) + 1e-15)
        return data

    train_feats = pd.read_csv(Path(train_data_path), sep=',')
    train_feats = train_feats.loc[train_feats['Experiment'] == 'tdTomato_37Degree']
    train_feats = train_feats[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension', 'Elected_Label']]
    train_feats = train_feats.replace({"Elected_Label":  {"mobile":1,"confined":2, "trapped":3}})
    X = train_feats.iloc[:, 2:-1]
    y = train_feats.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    X_train_, X_test_ = prepare_box_cox_data(X_train), prepare_box_cox_data(X_test)
    X_train_, X_test_ = pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(X_train_), columns=X.columns), pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(X_test_), columns=X.columns)

    for col_name in X_train.columns:
        X_train.eval(f'{col_name} = @X_train_.{col_name}')
        X_test.eval(f'{col_name} = @X_test_.{col_name}')

    pipeline_steps = [("pca", decomposition.PCA()), ("scaler", StandardScaler()), ("SVC", SVC(kernel="rbf"))]
    check_params = {
        "pca__n_components" : [3],
        "SVC__C" : [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
        "SVC__gamma" : [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 1., 5., 10., 50.0],
    }

    pipeline = Pipeline(pipeline_steps)
    create_search_grid = GridSearchCV(pipeline, param_grid=check_params, cv=10)
    create_search_grid.fit(X_train, y_train)
    pipeline.set_params(**create_search_grid.best_params_)
    pipeline.fit(X_train, y_train)
    X = pd.read_csv(Path(pred_data_path), sep=',')

    X = X.rename(columns={"radius_gyration" : "radiusGyration",
                      "track_number" : "TrackID",
                      "netDispl" : "NetDispl",
                      "asymmetry" : "Asymmetry",
                      "kurtosis" : "Kurtosis"
                      }) ###GD EDIT

    X['Experiment'] = exptName    ###GD EDIT

    X = X[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension']]
    X_label = X['Experiment'].iloc[0]
    X_feats = X.iloc[:, 2:]
    X_feats_ = pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(prepare_box_cox_data(X_feats)), columns=X_feats.columns)

    for col_name in X_feats.columns:
        X_feats.eval(f'{col_name} = @X_feats_.{col_name}')

    X_pred = pipeline.predict(X_feats)
    X['SVMPredictedClass'] = X_pred.astype('int')
    X_outpath = Path(pred_data_path).parents[0] / f'{Path(pred_data_path).stem}_SVMPredicted{level}.csv'
    #X.to_csv(X_outpath, sep=',', index=False)

    #add classes to RG file ####GD EDIT
    tracksDF = pd.read_csv(Path(pred_data_path), sep=',')
    tracksDF['Experiment'] = X['Experiment']
    tracksDF['SVM'] = X['SVMPredictedClass']

    tracksDF.to_csv(X_outpath, sep=',', index=None)


def classifyTracks(tracksList, train_data_path, level=''):
    """Classify tracks using SVM"""
    for pred_data_path in tqdm(tracksList):
        exptName = os.path.basename(pred_data_path).split('_MMStack')[0]
        predict_SPT_class(train_data_path, pred_data_path, exptName, level=level)


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

def linkFiles(tiffList, pixelSize = 0.108, frameLength = 1, skipFrames = 1, distanceToLink = 3, level=''):
    """Link files using flika"""
    for fileName in tqdm(tiffList):

        #set file & save names
        pointsFileName = os.path.splitext(fileName)[0] + '_locsID{}.csv'.format(level)
        lagsHistoSaveName = os.path.splitext(pointsFileName)[0] + '_lagsHisto{}.txt'.format(level)
        tracksSaveName = os.path.splitext(pointsFileName)[0] + '_tracks{}.csv'.format(level)

        #import tiff to flika
        data_window = open_file(fileName)
        #import points
        txy_pts = load_points(pointsFileName)
        p = Points(txy_pts)
        #link points
        p.link_pts(skipFrames,distanceToLink)
        #get background subtracted intensity for each point
        p.getIntensities(data_window.imageArray())
        #save tracks
        tracks = p.tracks
        savetracksCSV(p, tracksSaveName, pointsFileName)
        #export SLD histogram
        #SLD_hist = SLD_Histogram(p, pixelSize, frameLength)
        #SLD_hist.export_histogram(autoSave = True, autoFileName = lagsHistoSaveName)

        #close flika windows
        #SLD_hist.close()
        g.m.clear()
    return

def linkFilesNoFlika(tiffList, pixelSize = 0.108, frameLength = 1, skipFrames = 1, distanceToLink = 3, level='', allFiles=[], includeRecursionProblem=False):
    """Link files without using flika"""
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
        txy_pts = load_points(pointsFileName)
        p = Points(txy_pts)
        #link points
        p.link_pts(skipFrames,distanceToLink)
        #get background subtracted intensity for each point
        p.getIntensities(A)
        #save tracks
        if includeRecursionProblem:
          tracks = p.tracks
          savetracksCSV(p, tracksSaveName, pointsFileName)
        else:
            if p.recursiveFailure == False:
                tracks = p.tracks
                savetracksCSV(p, tracksSaveName, pointsFileName)
            else:
                print('recursion error, skipped {}'.format(fileName))

    return


def importJSON(tiffList, pixelSize = 0.108, level=''):
    """Import tracks from JSON files"""
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
        savetracksCSV(p, tracksSaveName, None, noLocsFile=True)
        #export SLD histogram
        #SLD_hist = SLD_Histogram(p, pixelSize, frameLength)
        #SLD_hist.export_histogram(autoSave = True, autoFileName = lagsHistoSaveName)

        #close flika windows
        #SLD_hist.close()
        g.m.clear()


def linkFiles_trackpy(tiffList, pixelSize = 0.108, skipFrames = 1, distanceToLink = 3, level='', linkingType='standard', maxDistance=5):
    """Link files using trackpy"""
    #try loading trackpy
    try:
        import trackpy as tp
    except:
        print("trackpy installation not detected. Install instructions at 'http://soft-matter.github.io/trackpy/v0.5.0/installation.html' ")
        return

    #pixels in nm
    pixelSize = pixelSize *1000

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
    minLinkSegments = 6    #gabby

    #max number of gap frames to skip
    #gapSize = 2
    gapSize = 18 #gabby

    #max distance in pixels to allow a linkage
    #distance = 5
    distance = 3 #gabby

    #pixel size
    pixelSize_new = 0.108

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
    classifyTracks(tracksList, trainpath, level=''.format(distance))


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












