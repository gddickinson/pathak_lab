#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:45:10 2023

@author: george
"""

import os
import glob
import pandas as pd

# set path of top directory
dir_path = r'/Users/george/Desktop/testing_3'

# suffixes of files to count
suffixList = [
    '_locs.csv',
    '_NNcount.csv',
    '_SVMPredicted.csv'
    ]

#create columns to use in results df based on suffix list
cols=[]
for suffix in suffixList:
    cols.append('file name {}'.format(suffix))
    cols.append('number of rows {}'.format(suffix))

# init results table
results = pd.DataFrame(columns=cols)

# Get list of subdirectories
#subDir_list = [ f.path for f in os.scandir(dir_path) if f.is_dir() ]
subDir_list = glob.glob(f'{dir_path}/*/**/', recursive=True)

#loop through subdirectories
for subDir in subDir_list:
    tempResult = {}
    #get files to count
    files=[]
    for suffix in suffixList:
        searchTerm = '/*{}'.format(suffix)
        files.append(glob.glob(subDir + searchTerm))

    # loop through files
    for i,fileName in enumerate(files):
        df = pd.read_csv(fileName[0])
        row_count = df.shape[0]
        tempResult.update({'file name {}'.format(suffixList[i]) : [fileName[0]], 'number of rows {}'.format(suffixList[i]) :[row_count]})

    #convert temp results dict to df
    tempDF = pd.DataFrame.from_dict(tempResult)
    #append new row to results DF
    results = pd.concat([ results, tempDF])


#export results df to csv
saveName = os.path.join(dir_path,'countResult.csv')
results.to_csv(saveName)

print('finished: results file written to {}'.format(saveName))
