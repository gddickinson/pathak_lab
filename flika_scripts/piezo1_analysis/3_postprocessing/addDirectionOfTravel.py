#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:10:45 2025

@author: user

Direction of Travel Calculator for Tracked Puncta - With Inverted Y-axis
This script processes linked track data to calculate and add directional information.
"""

import numpy as np
import pandas as pd
import os
import glob
import math
from tqdm import tqdm

def calculate_direction(x1, y1, x2, y2):
    """
    Calculate the direction of travel between two points, with inverted Y-axis.
    Returns angle in degrees (0-360) with 0 being right, 90 being down, etc.
    """
    dx = x2 - x1
    # Invert y-direction to match microscope display (y increases downward)
    dy = -(y2 - y1)  # Negate the difference

    # Handle the case where there's no movement
    if dx == 0 and dy == 0:
        return float('nan')

    # Calculate angle in radians
    angle_rad = math.atan2(dy, dx)

    # Convert to degrees (0-360 range)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def calculate_directional_persistence(angles):
    """
    Calculate directional persistence from a series of movement angles.
    Returns a value between 0 (random) and 1 (straight line).
    """
    if len(angles) < 2 or np.isnan(angles).all():
        return float('nan')

    # Filter out NaN values
    valid_angles = angles[~np.isnan(angles)]

    if len(valid_angles) < 2:
        return float('nan')

    # Calculate angular differences between consecutive movements
    angle_diffs = []
    for i in range(1, len(valid_angles)):
        diff = abs(valid_angles[i] - valid_angles[i-1])
        # Ensure we get the smallest angle between directions
        if diff > 180:
            diff = 360 - diff
        angle_diffs.append(diff)

    # Average the angular differences (0 means perfectly straight, 180 means completely random)
    avg_diff = np.mean(angle_diffs)

    # Convert to persistence (1 means straight line, 0 means random)
    persistence = 1 - (avg_diff / 180)

    return persistence

def add_direction_to_tracks(tracks_file):
    """
    Process a single tracks file to add direction of travel.
    """
    try:
        # Load the tracks data
        df = pd.read_csv(tracks_file)

        # Create columns for the direction
        df['direction_degrees'] = float('nan')
        df['direction_radians'] = float('nan')
        df['direction_x'] = float('nan')
        df['direction_y'] = float('nan')

        # Initialize lists for directional persistence calculation
        track_numbers = df['track_number'].unique()
        track_persistence = {track: float('nan') for track in track_numbers}

        # Process each track
        for track_num in track_numbers:
            # Get the data for this track and sort by frame
            track_data = df[df['track_number'] == track_num].sort_values('frame')

            if len(track_data) < 2:
                continue

            # Extract positions
            x_vals = track_data['x'].values
            y_vals = track_data['y'].values
            track_indices = track_data.index.values

            # Calculate direction for each step
            angles = []
            for i in range(len(x_vals) - 1):
                angle = calculate_direction(x_vals[i], y_vals[i], x_vals[i+1], y_vals[i+1])
                angles.append(angle)

                # Update the direction for the current point
                # This will be the direction from this point to the next
                idx = track_indices[i]
                df.at[idx, 'direction_degrees'] = angle
                df.at[idx, 'direction_radians'] = math.radians(angle)
                df.at[idx, 'direction_x'] = math.cos(math.radians(angle))
                df.at[idx, 'direction_y'] = -math.sin(math.radians(angle))  # Negative for display

            # Calculate directional persistence for this track
            track_persistence[track_num] = calculate_directional_persistence(np.array(angles))

        # Add directional persistence to the dataframe
        df['directional_persistence'] = df['track_number'].map(track_persistence)

        # Save the updated dataframe
        output_file = tracks_file.replace('.csv', '_with_direction.csv')
        df.to_csv(output_file, index=False)
        print(f"Direction analysis completed for {tracks_file}")
        return output_file

    except Exception as e:
        print(f"Error processing {tracks_file}: {e}")
        return None

def process_all_track_files(path, file_pattern='*_tracksRG*.csv'):
    """
    Process all track files in a directory and its subdirectories.
    """
    # Find all track files
    track_files = glob.glob(path + '/**/' + file_pattern, recursive=True)

    if not track_files:
        print(f"No files matching pattern '{file_pattern}' found in {path}")
        return

    print(f"Found {len(track_files)} files to process.")

    # Process each file
    processed_files = []
    for file in tqdm(track_files):
        output_file = add_direction_to_tracks(file)
        if output_file:
            processed_files.append(output_file)

    print(f"Completed direction analysis for {len(processed_files)} files.")



def main():
    # Set the path to your data directory
    #path = input("Enter the path to your data directory: ")

    # Set the file pattern to match
    #file_pattern = input("Enter file pattern to match (default: *_tracksRG*.csv): ") or '*_tracksRG*.csv'

    path = '/Users/george/Desktop/testing'
    file_pattern = '*_tracksRG.csv'

    # Process all files
    process_all_track_files(path, file_pattern)

if __name__ == '__main__':
    main()
