#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:56:03 2022

@author: george
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob



def addID(file):
    df = pd.read_csv(file)
    df['id'] = df['id'].astype('int')
    df['frame'] = df['frame'].astype('int') -1 #reseting first frame to zero to maych flika display
    df.to_csv(file.split('.csv')[0] + 'ID.csv')


if __name__ == '__main__':
    ##### RUN ANALYSIS        
    #path = '/Users/george/Data/10msExposure2s'
    #path = '/Users/george/Data/10msExposure2s_fixed'
    #path = '/Users/george/Data/10msExposure2s_test'    
    #path = '/Users/george/Data/10msExposure2s_new'
    #path = '/Users/george/Data/tdt'
    path = '/Users/george/Data/10msExposure10s_fixed'
    
    
    #get expt folder list
    locsList = glob.glob(path + '/**/*_locs.csv', recursive = True)   

    for file in tqdm(locsList):
        addID(file)