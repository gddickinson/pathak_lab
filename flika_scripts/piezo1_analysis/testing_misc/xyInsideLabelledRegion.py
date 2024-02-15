#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 14:36:37 2023

@author: george
"""

from skimage import measure
import numpy as np

# Generate example labelled image
image = np.zeros((10, 10), dtype=np.uint8)
image[2:5, 2:5] = 1
image[6:8, 6:8] = 2
labels = measure.label(image)

# Check if (x,y) coordinates are located within a labelled region
def is_within_region(labels, x, y, region_label):
    if labels[x, y] == region_label:
        return True
    else:
        return False

# Example usage
print(is_within_region(labels, 3, 3, 1))  # True
print(is_within_region(labels, 7, 7, 2))  # True
print(is_within_region(labels, 5, 5, 1))  # False


#############################################

from skimage import measure
import numpy as np
import pandas as pd

# Generate example labelled image
image = np.zeros((10, 10), dtype=np.uint8)
image[2:5, 2:5] = 1
image[6:8, 6:8] = 2
labels = measure.label(image)

# Add a new column to a DataFrame indicating if each point is within any labelled region
def add_within_region_column(labels, df):
    df['within_region'] = False
    for region_label in np.unique(labels):
        if region_label == 0:
            continue
        mask = labels == region_label
        df['within_region'] = df['within_region'] | mask[df['x'], df['y']]
    return df

# Example usage
df = pd.DataFrame({'x': [3, 7, 5], 'y': [3, 7, 5]})
df = add_within_region_column(labels, df)
print(df)
