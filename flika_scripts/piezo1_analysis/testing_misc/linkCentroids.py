#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:05:54 2022

@author: george
"""

%matplotlib qt 
%gui qt


import numpy as np
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

import pandas as pd
from tqdm import tqdm

from flika import start_flika

#sys.setrecursionlimit(10000)

halt_current_computation = False

def launch_docs():
    url='https://github.com/kyleellefsen/pynsight'
    QDesktopServices.openUrl(QUrl(url))



def Export_pts_from_MotilityTracking():
    tracks = g.m.trackPlot.all_tracks
    t_out = []
    x_out = []
    y_out = []
    for i in np.arange(len(tracks)):
        track = tracks[i]
        t_out.extend(track['frames'])
        x_out.extend(track['x_cor'])
        y_out.extend(track['y_cor'])
    p_out = np.array([t_out, x_out, y_out]).T
    filename = r'C:\Users\kyle\Desktop\trial8_pts.txt'
    np.savetxt(filename, p_out)


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

def savetracksCSV(points, filename):    
    tracks = points.tracks
    if isinstance(tracks[0][0], np.int64):
        tracks = [[np.asscalar(a) for a in b] for b in tracks]
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
        
        indexList =list(txy_indexes[l] for l in indices)       
        
        for pts in ptsList:
            trackNumber.append(i)
            frameList.append(pts[0])
            xList.append(pts[1])
            yList.append(pts[2])
            

        for intensity in intensitiesList:
            txy_intensitiesByTrack.append(intensity)
            
        for ind in indexList:
            txy_indexesByTrack.append(ind)
        

    #make dataframe of tracks, xy coordianates and intensities for linked tracks  
    dict = {'track_number': trackNumber, 'frame':frameList, 'x': xList, 'y':yList, 'intensity': txy_intensitiesByTrack, 'id': txy_indexesByTrack}              
              
    
    linkedtrack_DF = pd.DataFrame(dict)
    
    #match id to locs file values (starting at 1)
    linkedtrack_DF['id'] = linkedtrack_DF['id'] + 1
    
    #convert back to nm
    #linkedtrack_DF['x [nm]'] = linkedtrack_DF['x'] * 108
    #linkedtrack_DF['y [nm]'] = linkedtrack_DF['y'] * 108 
    
    #cast frames as int
    linkedtrack_DF['frame'] = linkedtrack_DF['frame'].astype('int')
    
    #round intensity
    linkedtrack_DF['intensity'] = linkedtrack_DF['intensity'].round(2) 
    
    #save df as csv
    #linkedtrack_DF = linkedtrack_DF.sort_values('id')
    linkedtrack_DF.to_csv(filename, index=True)
    
    print('tracks file {} saved'.format(filename))
    

class Points(object):
    def __init__(self, txy_pts):
        self.frames = np.unique(txy_pts[:, 0]).astype(np.int)
        self.txy_pts = txy_pts
        self.window = None
        self.pathitems = []
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []  # this array has the same structure as points_by_array but contains the index of the original txy_pts argument
        self.intensities = [] #GD edit


    def link_pts(self, maxFramesSkipped, maxDistance):
        print('Linking points')
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []  # this array has the same structure as points_by_array but contains the index of the original txy_pts argument
        for frame in np.arange(0, np.max(self.frames) + 1):
            indicies = np.where(self.txy_pts[:, 0] == frame)[0]
            pos = self.txy_pts[indicies, 1:]
            self.pts_by_frame.append(pos)
            self.pts_remaining.append(np.ones(pos.shape[0], dtype=np.bool))
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

    def extend_track(self, track, maxFramesSkipped, maxDistance):
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
                track = self.extend_track(track, maxFramesSkipped, maxDistance)
                return track
        return track

    def get_tracks_by_frame(self):
        tracks_by_frame = [[] for frame in np.arange(np.max(self.frames)+1)]
        for i, track in enumerate(self.tracks):
            frames = self.txy_pts[track][:,0].astype(np.int)
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
        
        for point in self.txy_pts:
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
            
            #get mean pixels values for 3x3 square - background subtract using frame min intensity as estimate of background
            self.intensities.append((np.mean(dataArray[frame][xMin:xMax,yMin:yMax]) - np.min(dataArray[frame])))




def linkFiles(tiffList, pointsFileID = '_locsID.csv', pixelSize = 0.108, frameLength = 1, skipFrames = 1, distanceToLink = 3):
    for fileName in tqdm(tiffList):
        
        #set file & save names
        pointsFileName = os.path.splitext(fileName)[0] + pointsFileID
        lagsHistoSaveName = os.path.splitext(pointsFileName)[0] + '_lagsHisto.txt'  
        tracksSaveName = os.path.splitext(pointsFileName)[0] + '_tracks.csv' 
        
        #import tiff to flilka
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
        savetracksCSV(p, tracksSaveName)
        #export SLD histogram
        SLD_hist = SLD_Histogram(p, pixelSize, frameLength)
        SLD_hist.export_histogram(autoSave = True, autoFileName = lagsHistoSaveName)
        
        #close flika windows        
        SLD_hist.close()
        g.m.clear()    
    

        
if __name__ == '__main__':

    fa = start_flika()
            
    #path = '/Users/george/Data/10msExposure2s'
    #path = '/Users/george/Data/10msExposure2s_fixed'
    #path = '/Users/george/Data/10msExposure2s_test'
    #path = '/Users/george/Data/10msExposure2s_new'
    #path = '/Users/george/Data/tdt'
    path = '/Users/george/Data/nonbapta_dyetitration' 

    
    #get folder paths
    tiffList = glob.glob(path + '/**/*_bin10.tif', recursive = True)
    #tiffList = glob.glob(path + '/**/*_crop100.tif', recursive = True)    
    #tiffList = glob.glob(path + '/**/*.tif', recursive = True)      
    
    #run linking on all tiffs in directory
    linkFiles(tiffList)
    
    fa.close()
    



