#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:57:07 2023

@author: george
"""
import skimage.io as skio
from matplotlib import pyplot as plt
import flika
from flika import global_vars as g
from flika.window import Window
from flika.process.file_ import save_file_gui, open_file_gui, open_file
import numpy as np


from flika import start_flika

fileName = r'/Users/george/Data/ultraSlow/test/GB_228_2023_05_04_HTEndothelial_BAPTA_plate2_ultraslow_7_MMStack_Default.ome.tif'

A = skio.imread(fileName, plugin='tifffile')

fig1 = plt.figure('1')
plt.imshow(A[0])

fa = start_flika()
data_window = open_file(fileName)

B = data_window.imageArray()
fa.close()

fig2 = plt.figure('2')
plt.imshow(B[0])


C = np.rot90(B, axes=(1,2))
C = np.fliplr(C)

fig3 = plt.figure('3')
plt.imshow(C[0])
