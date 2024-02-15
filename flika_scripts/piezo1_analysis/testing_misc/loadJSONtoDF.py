#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:20:55 2023

@author: george
"""

import pandas as pd
import json, codecs
import numpy as np
from qtpy.QtCore import QUrl, QRect, QPointF, Qt
from qtpy.QtGui import QDesktopServices, QIcon, QPainterPath, QPen, QColor
from qtpy.QtWidgets import QHBoxLayout, QGraphicsPathItem, qApp
from qtpy import uic


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


if __name__ == '__main__':
    #file = '/Users/george/Desktop/alan_jsonFiles/AL_59_2020-07-07-TIRFM_Diff_tdt-MEFs_B_3.json'    
    file = '/Users/george/Desktop/alan_jsonFiles/AL_62_2020-07-13-TIRFM_Diff_tdt-MEFs_A_4.json'
    points = loadtracksjson(file)



