# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:38:20 2020

@author: george.dickinson@gmail.com
"""
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
from flika.window import Window
import flika.global_vars as g
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
from os.path import expanduser
import os
import math
import sys


flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

from flika.process.file_ import open_file


import pandas as pd
from matplotlib import pyplot as plt



df = pd.read_csv('/Users/george/Data/testing/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity.csv')

trackDF = df[df['track_number'] == 3]

flika.start_flika()

win = open_file('/Users/george/Data/testing/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10.tif')

points = np.column_stack(( trackDF['frame'].to_list(), trackDF['x'].to_list(), trackDF['y'].to_list() ))
A = win.imageArray()

d = 10

A_pad = np.pad(A,((0,0),(d,d),(d,d)),'constant', constant_values=0)

frames = int(A.shape[0])
A_crop = np.zeros((frames,d,d))
x_limit = d/2 
y_limit = d/2

for point in points:
    minX = int(point[1] - x_limit + d)
    maxX = int(point[1] + x_limit + d)
    minY = int(point[2] - y_limit + d)
    maxY = int(point[2] + y_limit + d)
    crop = A_pad[int(point[0]),minX:maxX,minY:maxY]
    A_crop[int(point[0])] = crop
    
cropWin = Window(A_crop)

trace = np.mean(A_crop, axis=(0,1))





