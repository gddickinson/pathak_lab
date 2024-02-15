#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:03:12 2023

@author: george
"""
import pandas as pd


file = '/Users/george/Desktop/missingPointsTest/GB_228_2023_05_04_HTEndothelial_BAPTA_plate2_ultraslow_7_MMStack_Default_binned_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity_AllLocs_NNcount_BGsubtract.csv'

df = pd.read_csv(file)

df = df[df['SVM']==3]


def getTrackIDs(df, grouping='none'):
    if grouping == 'none':
        #return unique IDs
        return df['track_number'].unique()

    elif grouping == 'in pixel':
        #combine tracks for SVM3 at same location by rounding XY
        df['meanXloc'] = round(df.groupby(['track_number'])['x'].transform('mean'),0)
        df['meanYloc'] = round(df.groupby(['track_number'])['y'].transform('mean'),0)

        tempDF = df[['track_number', 'n_segments', 'meanXloc', 'meanYloc']]
        tempDF = tempDF.sort_values('n_segments')
        tempGroup = tempDF.drop_duplicates(subset=['meanXloc','meanYloc'], keep='last').reset_index(drop = True)
        # this tracklist is used to filter out smaller duplicate fragments at a SVM3 site
        return tempGroup['track_number'].unique()

    elif grouping == 'hcluster':
        #use hierarchical clustering to group type 3 sites by distance
        import scipy.cluster.hierarchy as hcluster
        thresh = 3
        data = df[['x','y']].to_numpy()
        df['cluster'] = hcluster.fclusterdata(data, thresh, criterion="distance")
        tempDF = df[['track_number', 'n_segments', 'cluster']]
        tempDF = tempDF.sort_values('n_segments')
        tempGroup = tempDF.drop_duplicates(subset=['cluster'], keep='last').reset_index(drop = True)
        # this tracklist is used to filter out smaller duplicate fragments at a SVM3 site
        return tempGroup['track_number'].unique()


#get SVM3 track_ids
trackList = getTrackIDs(df, grouping='hcluster')


