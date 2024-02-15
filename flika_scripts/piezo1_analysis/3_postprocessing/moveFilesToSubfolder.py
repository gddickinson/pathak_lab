#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:50:51 2023

@author: george
"""

import os
import glob
from pathlib import Path
import shutil

# set top path of directory
path = r'/Users/george/Desktop/subFolderTest/top'

# Get all csv files in directory
csv_file_list = glob.glob(path + '/**/*.csv', recursive = True)

# print files
print('------------------------')
print('files to move:')
for item in csv_file_list: print(item)
print('------------------------')

# Copy the excel files to the new directory
for csv_file_path in csv_file_list:

    # Get Directory folder to create subfolder
    subfolder_location = Path(csv_file_path).parents[1]
    #set subfolder name
    subfolder_name = 'newFolder'
    # new subfolder location
    subfolder = os.path.join(subfolder_location,subfolder_name)
    #if subfolder doesn't exist create it
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    # copy file to subfolder
    shutil.copy(csv_file_path, subfolder)

print('finished')
