#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:17:50 2023

@author: george
"""

import pandas as pd


path = '/Users/george/Data/htag_cutout/GB_199_2022_09_01_tracks/gof/GB_199_2022_09_01_HTEndothelial_locsID_tracksRG_SVMPredicted_NN.csv'

df = pd.read_csv(path)


svms = [1,2,3]

for svm in svms:
    df_SVM = df[df['SVM'] == svm]
    saveName = path.split('.')[0] + '_SVM{}.csv'.format(svm)
    df_SVM.to_csv(saveName)
