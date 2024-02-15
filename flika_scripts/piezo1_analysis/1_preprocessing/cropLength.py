#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:35:40 2022

@author: george
"""

from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
from flika.process.file_ import save_file_gui, open_file_gui, open_file, save_file
from flika.process.stacks import trim
from flika.app.application import FlikaApplication

from qtpy.QtGui import QColor

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox

from flika import *

%gui qt

import os, sys
import glob
from tqdm import tqdm


def cropTiffUsingFlika(tiffFile, cropSize):
    print('processing {}'.format(os.path.basename(tiffFile)))
    data_window = open_file(tiffFile)
    crop_window = trim(0, cropSize, increment=1, delete=False, keepSourceWindow=False)
    savename = tiffFile.split('.')[0] + '_crop{}.tif'.format(int(cropSize))
    save_file(savename) 
    print('saved {}'.format(os.path.basename(savename)))  
    g.m.clear()
    

def processFolder(path, cropSize = 20):
    #get folder paths
    tiffList = glob.glob(path + '/**/*.tif', recursive = True)
    
    for tiffFile in tqdm(tiffList):
        cropTiffUsingFlika(tiffFile, cropSize)
              


if __name__ == '__main__':
    #set top folder level for analysis
    #path = '/Users/george/Data/10msExposure2s'
    #path = '/Users/george/Data/10msExposure2s_fixed'
    #path = '/Users/george/Data/10msExposure2s_test'
    path = '/Users/george/Data/nonbapta_dyetitration/25pm'
    
    fa = start_flika()
    processFolder(path, cropSize = 1000)
    fa.close()

    
    


