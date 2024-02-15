#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:32:36 2022

@author: george
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob

from sklearn.neighbors import KDTree
import math

from scipy import stats, spatial
from matplotlib import pyplot as plt


# Radius of Gyration and Asymmetry
def RadiusGyrationAsymmetrySkewnessKurtosis(trackDF):
    # Drop any skipped frames and convert trackDF to XY array
    points_array = np.array(trackDF[['x', 'y']].dropna())  
    # get Rg etc using Vivek's codes
    center = points_array.mean(0)
    normed_points = points_array - center[None, :]
    radiusGyration_tensor = np.einsum('im,in->mn', normed_points, normed_points)/len(points_array)
    eig_values, eig_vectors = np.linalg.eig(radiusGyration_tensor)
    radius_gyration_value = np.sqrt(np.sum(eig_values))
    asymmetry_numerator = pow((eig_values[0] - eig_values[1]), 2)
    asymmetry_denominator = 2 * (pow((eig_values[0] + eig_values[1]), 2))
    asymmetry_value = - math.log(1 - (asymmetry_numerator / asymmetry_denominator))
    maxcol = list(eig_values).index(max(eig_values))
    dominant_eig_vect = eig_vectors[:, maxcol]
    points_a = points_array[:-1]
    points_b = points_array[1:]
    ba = points_b - points_a
    proj_ba_dom_eig_vect = np.dot(ba, dominant_eig_vect) / np.power(np.linalg.norm(dominant_eig_vect), 2)
    skewness_value = stats.skew(proj_ba_dom_eig_vect)
    kurtosis_value = stats.kurtosis(proj_ba_dom_eig_vect)
    return radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value


# Fractal Dimension
def FractalDimension(points_array):
    ####Vivek's code    
    #Check if points are on the same line:
    x0, y0 = points_array[0]
    points = [ (x, y) for x, y in points_array if ( (x != x0) or (y != y0) ) ]
    slopes = [ ((y - y0) / (x - x0)) if (x != x0) else None for x, y in points ]
    if all( s == slopes[0] for s in slopes):
        raise ValueError("Fractal Dimension cannot be calculated for points that are collinear")
    total_path_length = np.sum(pow(np.sum(pow(points_array[1:, :] - points_array[:-1, :], 2), axis=1), 0.5))
    stepCount = len(points_array)
    candidates = points_array[spatial.ConvexHull(points_array).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    maxIndex = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    largestDistance = dist_mat[maxIndex]
    fractal_dimension_value = math.log(stepCount) / math.log(stepCount * largestDistance * math.pow(total_path_length, -1))
    return fractal_dimension_value

# Net Displacement
def NetDisplacementEfficiency(points_array):      
    ####Vivek's code   
    net_displacement_value = np.linalg.norm(points_array[0]-points_array[-1])
    netDispSquared = pow(net_displacement_value, 2)
    points_a = points_array[1:, :]
    points_b = points_array[:-1, :]
    dist_ab_SumSquared = sum(pow(np.linalg.norm(points_a-points_b, axis=1), 2))
    efficiency_value = netDispSquared / ((len(points_array)-1) * dist_ab_SumSquared)
    return net_displacement_value, efficiency_value


# Bending & Straightness Features
def SummedSinesCosines(points_array):
    ## Vivek's code
    # Look for repeated positions in consecutive frames
    compare_against = points_array[:-1]
    # make a truth table identifying duplicates
    duplicates_table = points_array[1:] == compare_against
    # Sum the truth table across the rows, True = 1, False = 0
    duplicates_table = duplicates_table.sum(axis=1)
    # If both x and y are duplicates, value will be 2 (True + True == 2)
    duplicate_indices = np.where(duplicates_table == 2)
    # Remove the consecutive duplicates before sin, cos calc
    points_array = np.delete(points_array, duplicate_indices, axis=0)
    # Generate three sets of points
    points_set_a = points_array[:-2]
    points_set_b = points_array[1:-1]
    points_set_c = points_array[2:]
    # Generate two sets of vectors
    ab = points_set_b - points_set_a
    bc = points_set_c - points_set_b
    # Evaluate sin and cos values
    cross_products = np.cross(ab, bc)
    dot_products = np.einsum('ij,ij->i', ab, bc)
    product_magnitudes_ab_bc = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)
    cos_vals = dot_products / product_magnitudes_ab_bc
    cos_mean_val = np.mean(cos_vals)
    sin_vals = cross_products / product_magnitudes_ab_bc
    sin_mean_val = np.mean(sin_vals)
    return sin_mean_val, sin_vals, cos_mean_val, cos_vals

def getRadiusGyrationForAllTracksinDF(tracksDF):
    tracksToTest = tracksDF['track_number'].tolist()
    idTested = []
    radius_gyration_list=[] 
    asymmetry_list=[] 
    skewness_list=[] 
    kurtosis_list=[]  
    trackIntensity_mean = []
    trackIntensity_std = []
    
    for i in range(len(tracksToTest)):
        idToTest = tracksToTest[i]
        if idToTest not in idTested:
            radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value = RadiusGyrationAsymmetrySkewnessKurtosis(tracksDF[tracksDF['track_number']==idToTest])
            idTested.append(idToTest)
            #print(radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value)
            print('\r' + 'RG analysis complete for track {} of {}'.format(idToTest,max(tracksToTest)), end='\r')
            
        radius_gyration_list.append(radius_gyration_value) 
        asymmetry_list.append(asymmetry_value) 
        skewness_list.append(skewness_value)
        kurtosis_list.append(kurtosis_value) 
        
        trackIntensity_mean.append(np.mean(tracksDF[tracksDF['track_number']==idToTest]['intensity']))
        trackIntensity_std.append(np.std(tracksDF[tracksDF['track_number']==idToTest]['intensity']))    
        
            
    tracksDF['radius_gyration'] = radius_gyration_list
    tracksDF['asymmetry'] = asymmetry_list
    tracksDF['skewness'] = skewness_list
    tracksDF['kurtosis'] = kurtosis_list 
    tracksDF['track_intensity_mean'] = trackIntensity_mean
    tracksDF['track_intensity_std'] = trackIntensity_std
    
    return tracksDF

def getFeaturesForAllTracksinDF(tracksDF):
    tracksToTest = tracksDF['track_number'].tolist()
    idTested = []
    fracDim_list = [] 
    netDispl_list = []
    straight_list = []

    
    for i in range(len(tracksToTest)):
        idToTest = tracksToTest[i]
        if idToTest not in idTested:
            #get single track
            trackDF = tracksDF[tracksDF['track_number']==idToTest]            
            # Drop any skipped frames and convert trackDF to XY array
            points_array = np.array(trackDF[['x', 'y']].dropna())              
            
            #fractal_dimension calc
            fractal_dimension_value = FractalDimension(points_array)
            #net_Dispacement calc
            net_displacement_value, _ = NetDisplacementEfficiency(points_array)
            #straightness calc
            _, _, cos_mean_val, _ = SummedSinesCosines(points_array)
            
            #update ID tested
            idTested.append(idToTest)
            #print(radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value)
            print('\r' + 'features analysis complete for track {} of {}'.format(idToTest,max(tracksToTest)), end='\r')
        
        #add feature values to lists
        fracDim_list.append(fractal_dimension_value) 
        netDispl_list.append(net_displacement_value)
        straight_list.append(cos_mean_val)
                
    #update tracksDF        
    tracksDF['fracDimension'] = fracDim_list
    tracksDF['netDispl'] = netDispl_list
    tracksDF['Straight'] = straight_list
    
    return tracksDF

def addLagDisplacementToDF(tracksDF):
    #align x and y locations of link
    tracksDF = tracksDF.assign(x2=tracksDF.x.shift(-1))  
    tracksDF = tracksDF.assign(y2=tracksDF.y.shift(-1))

    #calculate lag Distance
    tracksDF['x2-x1_sqr'] = np.square(tracksDF['x2']-tracksDF['x'])
    tracksDF['y2-y1_sqr'] = np.square(tracksDF['y2']-tracksDF['y'])      
    tracksDF['distance'] = np.sqrt((tracksDF['x2-x1_sqr']+tracksDF['y2-y1_sqr']))
    
    #mask final track position lags
    tracksDF['mask'] = True
    tracksDF.loc[tracksDF.groupby('track_number').tail(1).index, 'mask'] = False  #mask final location
    
    #get lags for all track locations (not next track)    
    tracksDF['lag'] = tracksDF['distance'].where(tracksDF['mask'])

    #add track mean lag distance to all rows
    tracksDF['meanLag'] = tracksDF.groupby('track_number')['lag'].transform('mean')   
    
    #add track length for each track row
    tracksDF['track_length'] = tracksDF.groupby('track_number')['lag'].transform('sum')   

    #add 'radius_gyration' (scaled by mean lag displacement)
    tracksDF['radius_gyration_scaled'] = tracksDF['radius_gyration']/tracksDF['meanLag']

    #add 'radius_gyration' (scaled by n_segments)
    tracksDF['radius_gyration_scaled_nSegments'] = tracksDF['radius_gyration']/tracksDF['n_segments']
      
    #add 'radius_gyration' (scaled by track_length)
    tracksDF['radius_gyration_scaled_trackLength'] = tracksDF['radius_gyration']/tracksDF['track_length']
        
    print('\r' + 'lags added', end='\r')
     
    return tracksDF


def getNearestNeighbors(train,test,k=2):
    tree = KDTree(train, leaf_size=5)   
    dist, ind = tree.query(test, k=k)
    #dist.reshape(np.size(dist),)     
    return dist, ind

def getNN(tracksDF):
    #sort by frame
    tracksDF = tracksDF.sort_values(by=['frame'])
    #make empty list to store NN distances & indexes
    nnDistList = []
    nnIndexList = []
    #get list of frames in tracksDF to iterate over
    frames = tracksDF['frame'].unique().tolist()
    #get nn for each centroid position by frame
    for i, frame in enumerate(frames):
        #filter by frame
        frameXY = tracksDF[tracksDF['frame'] == frame][['x','y']].to_numpy()
        #nearest neighbour
        distances, indexes = getNearestNeighbors(frameXY,frameXY, k=2)   
        #append distances and indexes of 1st neighbour to list
        nnDistList.extend(distances[:,1])
        nnIndexList.extend(indexes[:,1])
        print('\r' + 'NN analysis complete for frame{} of {}'.format(i,len(frames)), end='\r')

    #add results to dataframe
    tracksDF['nnDist'] =  nnDistList
    tracksDF['nnIndex_inFrame'] = nnIndexList

    tracksDF = tracksDF.sort_index()
    print('\r' + 'NN-analysis added', end='\r')

    return tracksDF

def calcFeaturesforFiles(tracksList, minNumberSegments=1):
    for trackFile in tqdm(tracksList):
                
            ##### load data
            tracksDF = pd.read_csv(trackFile)

            #add number of segments for each Track row
            tracksDF['n_segments'] = tracksDF.groupby('track_number')['track_number'].transform('count')
                     
            
            if minNumberSegments !=0:
            #filter by number of track segments
                tracksDF = tracksDF[tracksDF['n_segments'] > minNumberSegments]
    
            #add Rg values to df
            tracksDF = getRadiusGyrationForAllTracksinDF(tracksDF)
            
            #add features to df
            tracksDF = getFeaturesForAllTracksinDF(tracksDF)
            
            #add lags to df
            tracksDF = addLagDisplacementToDF(tracksDF)
            
            #add nearest neigbours to df
            tracksDF = getNN(tracksDF)

            #### DROP any Unnamed columns #####
            tracksDF = tracksDF[['track_number', 'frame', 'id', 'x','y', 'intensity', 'n_segments', 'track_length','radius_gyration', 'asymmetry', 'skewness',
                                 'kurtosis', 'radius_gyration_scaled','radius_gyration_scaled_nSegments','radius_gyration_scaled_trackLength', 'track_intensity_mean', 'track_intensity_std', 'lag', 'meanLag',
                                 'fracDimension', 'netDispl', 'Straight', 'nnDist', 'nnIndex_inFrame']]
        
        
            #round values
            tracksDF = tracksDF.round({'track_length': 3,
                                       'radius_gyration': 3,
                                       'asymmetry': 3,                                       
                                       'skewness': 3,                                      
                                       'kurtosis': 3,
                                       'radius_gyration_scaled': 3,
                                       'radius_gyration_scaled_nSegments': 3,
                                       'radius_gyration_scaled_trackLength': 3,
                                       'track_intensity_mean': 2,
                                       'track_intensity_std': 2,
                                       'lag': 3,
                                       'meanLag': 3,
                                       'fracDimension': 3,
                                       'netDispl': 3,
                                       'Straight': 3,
                                       'nnDist': 3                                                                              
                                       })
        
        
            #saveRg DF
            saveName = os.path.splitext(trackFile)[0] + 'RG.csv'
            tracksDF.to_csv(saveName)
            print('\n new tracks file exported to {}'.format(saveName))    


if __name__ == '__main__':
    ##### RUN ANALYSIS        
    #path = '/Users/george/Data/10msExposure2s'
    #path = '/Users/george/Data/10msExposure2s_fixed'
    #path = '/Users/george/Data/10msExposure2s_test'
    #path = '/Users/george/Data/10msExposure2s_new'
    #path = '/Users/george/Data/tdt'
    path = '/Users/george/Data/nonbapta_dyetitration' 

    
    #get tracks files
    tracksList = glob.glob(path + '/**/*_locsID_tracks.csv', recursive = True)   
    
    #run analysis - filter for track lengths > 5
    calcFeaturesforFiles(tracksList, minNumberSegments=4)
            
            