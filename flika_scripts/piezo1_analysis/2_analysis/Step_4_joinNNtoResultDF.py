#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:13:24 2022

@author: george
"""

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Import required libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, glob

def addNNtoDF(df, nnDF, pixelSize = 108):
    """
    Add nearest neighbor information to the main DataFrame.

    Args:
    df (DataFrame): Main DataFrame containing track information
    nnDF (DataFrame): DataFrame containing nearest neighbor information
    pixelSize (float): Pixel size in nanometers (default: 108)

    Returns:
    DataFrame: Main DataFrame with added nearest neighbor information
    """
    # Select only 'id' and 'nnDist_inFrame' columns from nnDF
    nnDF = nnDF[['id', 'nnDist_inFrame']]

    # Convert nearest neighbor distance from nm to pixels
    nnDF['nnDist_inFrame'] = nnDF['nnDist_inFrame'] / pixelSize

    # Join DataFrames based on 'id'
    df = df.join(nnDF.set_index('id'), on='id')

    return df

if __name__ == '__main__':
    # Set path to data directory
    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'

    # Get list of result files
    dfFileList = glob.glob(path + '/**/*_locsID_tracksRG_SVMPredicted.csv', recursive = True)

    # Process each file
    for dfFile in tqdm(dfFileList):
        # Load main DataFrame
        df = pd.read_csv(dfFile)

        # Load nearest neighbor DataFrame
        nnDF = pd.read_csv(dfFile.split('_locsID_tracksRG_SVMPredicted.csv')[0] + '_locsID_NN.csv')

        # Add nearest neighbor information to main DataFrame
        newDF = addNNtoDF(df, nnDF)

        # Save updated DataFrame
        saveName = os.path.splitext(dfFile)[0] + '_NN.csv'
        newDF.to_csv(saveName, index=None)
