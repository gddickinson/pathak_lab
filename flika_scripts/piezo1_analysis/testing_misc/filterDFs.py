#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:05:03 2023

@author: george
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd

from tqdm import tqdm
import os, glob


def filterDF(df, colName, filterOp, filterValue1=None, filterValue2=None):
    if filterOp == 'over':
        filteredDF = df[df['{}'.format(colName)] > filterValue1]
    if filterOp == 'under':
        filteredDF = df[df['{}'.format(colName)] < filterValue1]
    if filterOp == 'equals':
        filteredDF = df[df['{}'.format(colName)] == filterValue1]
    if filterOp == 'notequal':
        filteredDF = df[df['{}'.format(colName)] != filterValue1]
    if filterOp == 'between':
        filteredDF = df[(df['{}'.format(colName)] > filterValue1) & (df['{}'.format(colName)] < filterValue2)]

    return filteredDF


if __name__ == '__main__':

    path = '/Users/george/Desktop/filterTest'

    colName = 'intensity [photon]'
    filterValue1 = 50
    filterValue2 = 100
    filterOp = 'between'

    #get filenames for all files to filter under path folder
    fileList = glob.glob(path + '/**/*_locs.csv', recursive = True)

    for file in tqdm(fileList):

        df = pd.read_csv(file)

        filteredDF = filterDF(df, colName, filterOp, filterValue1, filterValue2)

        if filterValue2 == None:
            saveName = os.path.splitext(file)[0]+'_{}{}{}.csv'.format(colName, filterOp, filterValue1)
        else:
            saveName = os.path.splitext(file)[0]+'_{}{}{}-{}.csv'.format(colName, filterOp, filterValue1, filterValue2)
        #if you want to overwrite the file uncomment line below
        #saveName = file

        filteredDF.to_csv(saveName, index=None)
