#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:25:24 2022

@author: george
"""

#%matplotlib qt
#%gui qt

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
from sklearn.metrics import confusion_matrix
try:
    from sklearn.metrics import plot_confusion_matrix
except:
    from sklearn.metrics import ConfusionMatrixDisplay as plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import power_transform, PowerTransformer, StandardScaler

import skimage.io as skio

recursionLimit = 1000000
sys.setrecursionlimit(recursionLimit)

#print(sys.getrecursionlimit())


def addID(file):
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
    return [item for sublist in l for item in sublist]

def savetracksCSV(points, filename, locsFileName, noLocsFile=False):
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


class Points(object):
    def __init__(self, txy_pts):
        self.frames = np.unique(txy_pts[:, 0]).astype(int)
        self.txy_pts = txy_pts
        self.window = None
        self.pathitems = []
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []  # this array has the same structure as points_by_array but contains the index of the original txy_pts argument
        self.intensities = [] #GD edit
        self.recursiveFailure = False

    def link_pts(self, maxFramesSkipped, maxDistance):
        print('Linking points')
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []  # this array has the same structure as points_by_array but contains the index of the original txy_pts argument
        for frame in np.arange(0, np.max(self.frames) + 1):
            indicies = np.where(self.txy_pts[:, 0] == frame)[0]
            pos = self.txy_pts[indicies, 1:]
            self.pts_by_frame.append(pos)
            self.pts_remaining.append(np.ones(pos.shape[0], dtype=bool))
            self.pts_idx_by_frame.append(indicies)

        tracks = []
        for frame in self.frames:
            for pt_idx in np.where(self.pts_remaining[frame])[0]:
                self.pts_remaining[frame][pt_idx] = False
                abs_pt_idx = self.pts_idx_by_frame[frame][pt_idx]
                track = [abs_pt_idx]
                track = self.extend_track(track, maxFramesSkipped, maxDistance)
                tracks.append(track)
        self.tracks = tracks

    def extend_track(self, track, maxFramesSkipped, maxDistance, i=0):
        #this limits the amount of possible recursion - 10000 set by trial and error
        if i >= 10000:
            self.recursiveFailure = True
            return track

        pt = self.txy_pts[track[-1]]
        # pt can move less than two pixels in one frame, two frames can be skipped
        for dt in np.arange(1, maxFramesSkipped+2):
            frame = int(pt[0]) + dt
            if frame >= len(self.pts_remaining):
                return track
            candidates = self.pts_remaining[frame]
            nCandidates = np.count_nonzero(candidates)
            if nCandidates == 0:
                continue
            else:
                distances = np.sqrt(np.sum((self.pts_by_frame[frame][candidates] - pt[1:]) ** 2, 1))
            if any(distances < maxDistance):
                next_pt_idx = np.where(candidates)[0][np.argmin(distances)]
                abs_next_pt_idx = self.pts_idx_by_frame[frame][next_pt_idx]
                track.append(abs_next_pt_idx)
                self.pts_remaining[frame][next_pt_idx] = False
                track = self.extend_track(track, maxFramesSkipped, maxDistance, i=i+1)
                return track
        return track

    def get_tracks_by_frame(self):
        tracks_by_frame = [[] for frame in np.arange(np.max(self.frames)+1)]
        for i, track in enumerate(self.tracks):
            frames = self.txy_pts[track][:,0].astype(int)
            for frame in frames:
                tracks_by_frame[frame].append(i)
        self.tracks_by_frame = tracks_by_frame

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
    if txy_pts is None:
        return None
    new_pts = []
    for pt in txy_pts:
                    #    t,  old x, old y, new_x, new_y, sigma, amplitude
        new_pts.append([pt[0], pt[1], pt[2], pt[1], pt[2], -1, -1])
    pts_refined = np.array(new_pts)
    return pts_refined


def loadtracksjson(filename):
    obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
    pts = json.loads(obj_text)
    txy_pts = np.array(pts['txy_pts'])
    txy_pts = txy_pts

    txy_pts = skip_refinePoints(txy_pts)
    points = Points(txy_pts)
    points.tracks = pts['tracks']
    points.get_tracks_by_frame()
    return points

# Radius of Gyration and Asymmetry
def RadiusGyrationAsymmetrySkewnessKurtosis(trackDF):
    # Drop any skipped frames and convert trackDF to XY array
    points_array = np.array(trackDF[['x', 'y']].dropna())
    # get Rg etc using Vivek's codes
    center = points_array.mean(0)
    normed_points = points_array - center[None, :]
    radiusGyration_tensor = np.einsum('im,in->mn', normed_points, normed_points)/len(points_array)
    eig_values, eig_vectors = np.linalg.eig(radiusGyration_tensor)
    radius_gyration_value = np.sqrt(np.sum(eig_values))
    asymmetry_numerator = pow((eig_values[0] - eig_values[1]), 2)
    asymmetry_denominator = 2 * (pow((eig_values[0] + eig_values[1]), 2))
    asymmetry_value = - math.log(1 - (asymmetry_numerator / asymmetry_denominator))
    maxcol = list(eig_values).index(max(eig_values))
    dominant_eig_vect = eig_vectors[:, maxcol]
    points_a = points_array[:-1]
    points_b = points_array[1:]
    ba = points_b - points_a
    proj_ba_dom_eig_vect = np.dot(ba, dominant_eig_vect) / np.power(np.linalg.norm(dominant_eig_vect), 2)
    skewness_value = stats.skew(proj_ba_dom_eig_vect)
    kurtosis_value = stats.kurtosis(proj_ba_dom_eig_vect)
    return radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value


# Fractal Dimension
def FractalDimension(points_array):
    ####Vivek's code
    #Check if points are on the same line:
    x0, y0 = points_array[0]
    points = [ (x, y) for x, y in points_array if ( (x != x0) or (y != y0) ) ]
    slopes = [ ((y - y0) / (x - x0)) if (x != x0) else None for x, y in points ]
    if all( s == slopes[0] for s in slopes):
        raise ValueError("Fractal Dimension cannot be calculated for points that are collinear")
    total_path_length = np.sum(pow(np.sum(pow(points_array[1:, :] - points_array[:-1, :], 2), axis=1), 0.5))
    stepCount = len(points_array)
    candidates = points_array[spatial.ConvexHull(points_array).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    maxIndex = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    largestDistance = dist_mat[maxIndex]
    fractal_dimension_value = math.log(stepCount) / math.log(stepCount * largestDistance * math.pow(total_path_length, -1))
    return fractal_dimension_value

# Net Displacement
def NetDisplacementEfficiency(points_array):
    ####Vivek's code
    net_displacement_value = np.linalg.norm(points_array[0]-points_array[-1])
    netDispSquared = pow(net_displacement_value, 2)
    points_a = points_array[1:, :]
    points_b = points_array[:-1, :]
    dist_ab_SumSquared = sum(pow(np.linalg.norm(points_a-points_b, axis=1), 2))
    efficiency_value = netDispSquared / ((len(points_array)-1) * dist_ab_SumSquared)
    return net_displacement_value, efficiency_value


# Bending & Straightness Features
def SummedSinesCosines(points_array):
    ## Vivek's code
    # Look for repeated positions in consecutive frames
    compare_against = points_array[:-1]
    # make a truth table identifying duplicates
    duplicates_table = points_array[1:] == compare_against
    # Sum the truth table across the rows, True = 1, False = 0
    duplicates_table = duplicates_table.sum(axis=1)
    # If both x and y are duplicates, value will be 2 (True + True == 2)
    duplicate_indices = np.where(duplicates_table == 2)
    # Remove the consecutive duplicates before sin, cos calc
    points_array = np.delete(points_array, duplicate_indices, axis=0)
    # Generate three sets of points
    points_set_a = points_array[:-2]
    points_set_b = points_array[1:-1]
    points_set_c = points_array[2:]
    # Generate two sets of vectors
    ab = points_set_b - points_set_a
    bc = points_set_c - points_set_b
    # Evaluate sin and cos values
    cross_products = np.cross(ab, bc)
    dot_products = np.einsum('ij,ij->i', ab, bc)
    product_magnitudes_ab_bc = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)
    cos_vals = dot_products / product_magnitudes_ab_bc
    cos_mean_val = np.mean(cos_vals)
    sin_vals = cross_products / product_magnitudes_ab_bc
    sin_mean_val = np.mean(sin_vals)
    return sin_mean_val, sin_vals, cos_mean_val, cos_vals

def getRadiusGyrationForAllTracksinDF(tracksDF):
    tracksToTest = tracksDF['track_number'].tolist()
    idTested = []
    radius_gyration_list=[]
    asymmetry_list=[]
    skewness_list=[]
    kurtosis_list=[]
    trackIntensity_mean = []
    trackIntensity_std = []

    for i in range(len(tracksToTest)):
        idToTest = tracksToTest[i]
        if idToTest not in idTested:
            radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value = RadiusGyrationAsymmetrySkewnessKurtosis(tracksDF[tracksDF['track_number']==idToTest])
            idTested.append(idToTest)
            #print(radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value)
            print('\r' + 'RG analysis complete for track {} of {}'.format(idToTest,max(tracksToTest)), end='\r')

        radius_gyration_list.append(radius_gyration_value)
        asymmetry_list.append(asymmetry_value)
        skewness_list.append(skewness_value)
        kurtosis_list.append(kurtosis_value)

        trackIntensity_mean.append(np.mean(tracksDF[tracksDF['track_number']==idToTest]['intensity']))
        trackIntensity_std.append(np.std(tracksDF[tracksDF['track_number']==idToTest]['intensity']))


    tracksDF['radius_gyration'] = radius_gyration_list
    tracksDF['asymmetry'] = asymmetry_list
    tracksDF['skewness'] = skewness_list
    tracksDF['kurtosis'] = kurtosis_list
    tracksDF['track_intensity_mean'] = trackIntensity_mean
    tracksDF['track_intensity_std'] = trackIntensity_std

    return tracksDF

def getFeaturesForAllTracksinDF(tracksDF):
    tracksToTest = tracksDF['track_number'].tolist()
    idTested = []
    fracDim_list = []
    netDispl_list = []
    straight_list = []


    for i in range(len(tracksToTest)):
        idToTest = tracksToTest[i]
        if idToTest not in idTested:
            #get single track
            trackDF = tracksDF[tracksDF['track_number']==idToTest]
            # Drop any skipped frames and convert trackDF to XY array
            points_array = np.array(trackDF[['x', 'y']].dropna())

            #fractal_dimension calc
            fractal_dimension_value = FractalDimension(points_array)
            #net_Dispacement calc
            net_displacement_value, _ = NetDisplacementEfficiency(points_array)
            #straightness calc
            _, _, cos_mean_val, _ = SummedSinesCosines(points_array)

            #update ID tested
            idTested.append(idToTest)
            #print(radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value)
            print('\r' + 'features analysis complete for track {} of {}'.format(idToTest,max(tracksToTest)), end='\r')

        #add feature values to lists
        fracDim_list.append(fractal_dimension_value)
        netDispl_list.append(net_displacement_value)
        straight_list.append(cos_mean_val)

    #update tracksDF
    tracksDF['fracDimension'] = fracDim_list
    tracksDF['netDispl'] = netDispl_list
    tracksDF['Straight'] = straight_list

    return tracksDF

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


def getNearestNeighbors(train,test,k=2):
    tree = KDTree(train, leaf_size=5)
    if k > len(train):
        #no neighbours to count return nan
        a = np.empty((k,k))
        a[:] = np.nan
        return np.nan, np.nan
    else:
        dist, ind = tree.query(test, k=k)
    #dist.reshape(np.size(dist),)
    return dist, ind

def getNN(tracksDF):
    #sort by frame
    tracksDF = tracksDF.sort_values(by=['frame'])
    #make empty list to store NN distances & indexes
    nnDistList = []
    nnIndexList = []
    #get list of frames in tracksDF to iterate over
    frames = tracksDF['frame'].unique().tolist()
    #get nn for each centroid position by frame
    for i, frame in enumerate(frames):
        #filter by frame
        frameXY = tracksDF[tracksDF['frame'] == frame][['x','y']].to_numpy()
        #nearest neighbour
        distances, indexes = getNearestNeighbors(frameXY,frameXY, k=2)
        #append distances and indexes of 1st neighbour to list
        if (np.isnan(distances).any()):
            nnDistList.append(np.nan)
            nnIndexList.append(np.nan)
        else:
            nnDistList.extend(distances[:,1])
            nnIndexList.extend(indexes[:,1])
        print('\r' + 'NN analysis complete for frame{} of {}'.format(i,len(frames)), end='\r')

    #add results to dataframe
    tracksDF['nnDist'] =  nnDistList
    tracksDF['nnIndex_inFrame'] = nnIndexList

    tracksDF = tracksDF.sort_index()
    print('\r' + 'NN-analysis added', end='\r')

    return tracksDF


def getIntensities(dataArray, pts):
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
    for trackFile in tqdm(tracksList):
            try:
                ##### load data
                tracksDF = pd.read_csv(trackFile)

                #add number of segments for each Track row
                tracksDF['n_segments'] = tracksDF.groupby('track_number')['track_number'].transform('count')


                if minNumberSegments !=0:
                #filter by number of track segments
                    tracksDF = tracksDF[tracksDF['n_segments'] > minNumberSegments]

                #add Rg values to df
                tracksDF = getRadiusGyrationForAllTracksinDF(tracksDF)

                #add features to df
                tracksDF = getFeaturesForAllTracksinDF(tracksDF)

                #add lags to df
                tracksDF = addLagDisplacementToDF(tracksDF)

                #add nearest neigbours to df
                tracksDF = getNN(tracksDF)

                #### DROP any Unnamed columns #####
                tracksDF = tracksDF[['track_number', 'frame', 'id', 'x','y', 'intensity', 'n_segments', 'track_length','radius_gyration', 'asymmetry', 'skewness',
                                     'kurtosis', 'radius_gyration_scaled','radius_gyration_scaled_nSegments','radius_gyration_scaled_trackLength', 'track_intensity_mean', 'track_intensity_std', 'lag', 'meanLag',
                                     'fracDimension', 'netDispl', 'Straight', 'nnDist', 'nnIndex_inFrame']]


                #round values
                tracksDF = tracksDF.round({'track_length': 3,
                                           'radius_gyration': 3,
                                           'asymmetry': 3,
                                           'skewness': 3,
                                           'kurtosis': 3,
                                           'radius_gyration_scaled': 3,
                                           'radius_gyration_scaled_nSegments': 3,
                                           'radius_gyration_scaled_trackLength': 3,
                                           'track_intensity_mean': 2,
                                           'track_intensity_std': 2,
                                           'lag': 3,
                                           'meanLag': 3,
                                           'fracDimension': 3,
                                           'netDispl': 3,
                                           'Straight': 3,
                                           'nnDist': 3
                                           })


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

    for pred_data_path in tqdm(tracksList):
        exptName = os.path.basename(pred_data_path).split('_MMStack')[0]
        predict_SPT_class(train_data_path, pred_data_path, exptName, level=level)


def filterDFandLocs_SVM3(dfFile, locsIdentifer='_tracksRG_SVMPredicted.csv', colName='SVM'):
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
    df_SVM3 = df_SVM3.append(df_SVM3_extras)

    df_SVM2and3 = df_SVM2.append(df_SVM3)

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
    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'
    #path = '/Users/george/Data/test'
    #path = '/Users/george/Data/gabby_missingIntensities'

    #get folder paths
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












