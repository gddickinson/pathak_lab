#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:01:40 2023

@author: george
"""

import numpy as np
from matplotlib import pyplot as plt



def gauss(width,height, sigma=50.0, normalise=False, rangeMax = 255):
    mu_x, mu_y = width / 2, height / 2

    # Create a grid of x and y positions
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate the 2D Gaussian function
    gaussian = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))

    if normalise:
        # Normalize the Gaussian function to the range [0, rangeMax]
        gaussian = (gaussian * rangeMax / np.max(gaussian)).astype(np.uint8)

    return gaussian

gaussian = gauss(1000,1000,sigma=100, normalise=True, rangeMax = 100000)

plt.imshow(gaussian,'plasma')

