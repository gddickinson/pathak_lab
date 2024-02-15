#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:53:54 2023

@author: george
"""

import numpy as np
from skimage import io

fileName = '/Users/george/Data/simulated_20frames_iterate/sim3.tif'

im = io.imread(fileName)

im = im.astype('int32')

#io.imshow(im[0])


io.imsave(fileName,im)