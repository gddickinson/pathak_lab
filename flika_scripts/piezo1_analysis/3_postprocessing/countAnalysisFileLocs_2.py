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


# init results table
results = pd.DataFrame()

# Get list of subdirectories
subDir_list = glob.glob(f'{dir_path}/*/**/', recursive=True)

#loop through subdirectories
for subDir in subDir_list:
    tempResult = {}
    #get files to count
    files=glob.glob(subDir + '*.csv')

    # loop through files
    for i,fileName in enumerate(files):
        df = pd.read_csv(fileName)
        row_count = df.shape[0]
        tempResult.update({'file name {}'.format(i) : [fileName], 'number of rows {}'.format(i) :[row_count]})

    #convert temp results dict to df
    tempDF = pd.DataFrame.from_dict(tempResult)
    #append new row to results DF
    results = pd.concat([ results, tempDF])


#export results df to csv
saveName = os.path.join(dir_path,'countResult.csv')
results.to_csv(saveName)

print('finished: results file written to {}'.format(saveName))
