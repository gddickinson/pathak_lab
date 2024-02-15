#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:54:39 2023

@author: george
"""

import glob

path = r'/Users/george/Desktop/subFolderTest/'
for path in glob.glob(f'{path}/*/**/', recursive=True):
    print(path)


subDir_list = glob.glob(f'{path}/*/**/', recursive=True)
