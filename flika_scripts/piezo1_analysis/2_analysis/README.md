# PIEZO1 Analysis Pipeline

A comprehensive pipeline for analyzing PIEZO1 protein dynamics from microscopy data. This pipeline processes raw microscopy data through multiple steps to track proteins, analyze their movement patterns, and classify their behavior.

## Analysis Steps

### 1. Centroid Detection
**Script**: `Step_1_getCentroidsByRunningImageJMacro.py`
- Creates ImageJ macro for batch processing with ThunderSTORM
- Detects protein centroids in microscopy data
- Outputs: `*_locs.csv` files with particle locations

### 2. Particle Linking
**Script**: `Step_2_linkingAndClassification_trackpyOption_recursionErrorFix.py`
- Links detected particles across frames
- Multiple linking options available:
  - Standard linking
  - Adaptive linking
  - Velocity prediction-based linking
- Parameters:
  - `skipFrames`: Maximum gap size (default: 18)
  - `distanceToLink`: Maximum linking distance (default: 3 pixels)
- Outputs: `*_tracks.csv` files with linked trajectories

### 3. Nearest Neighbor Analysis
**Script**: `Step_3_nearestNeighbour.py`
- Calculates nearest neighbor distances for each point
- Uses KDTree for efficient spatial searching
- Outputs: `*_NN.csv` files with neighbor information

### 4. Results Compilation
**Script**: `Step_4_joinNNtoResultDF.py`
- Joins nearest neighbor data with main tracking results
- Converts measurements between pixels and physical units
- Outputs: Combined analysis files

### 5. Diffusion Analysis
**Script**: `Step_5_addDiffusion.py`
- Calculates diffusion metrics for each track
- Computes distances from track origin
- Adds squared displacement values

### 6. Velocity Analysis
**Script**: `Step_6_addVelocity.py`
- Calculates instantaneous and mean velocities
- Adds directional information
- Computes velocity relative to origin

### 7. Missing Points Analysis
**Script**: `Step_7_addMissingPoints.py`
- Identifies and processes unlinked localizations
- Adds missing points to analysis
- Preserves tracking integrity

### 8. Neighbor Count Analysis
**Script**: `Step_8_addNNcounts.py`
- Counts neighbors within specified radii
- Default radii: [3,5,10,20,30] pixels
- Provides density information

### 9. Background Subtraction
**Script**: `Step_9_addBackgroundSubtractedIntensity.py`
- Performs background intensity subtraction
- Uses ROI measurements
- Accounts for camera black level

### 10. Point Interpolation
**Script**: `Step_10_addInterpolatedPointstoBinnedRecording.py`
- Interpolates between detected points
- Handles binned recordings
- Adds missing frames for complete trajectories

### 11. Localization Error
**Script**: `Step_11_addLocalizationError.py`
- Calculates localization precision
- Adds mean positions and distances
- Quantifies tracking uncertainty

## Usage

1. Set up directory structure:
```
data_directory/
├── raw_data.tif
├── ROI_data.txt
└── analysis/
```

2. Run the analysis pipeline:
```python
# Example configuration
path = '/path/to/data'
parameters = {
    'skipFrames': 18,
    'distanceToLink': 3,
    'pixelSize': 0.108,  # microns
    'minLinkSegments': 6
}

# Run analysis steps sequentially
from Step_1_getCentroidsByRunningImageJMacro import getFileList, writeMacro
from Step_2_linkingAndClassification_trackpyOption_recursionErrorFix import linkFilesNoFlika
# ... import additional steps as needed
```

## Key Parameters

- **Pixel Size**: 0.108 μm (default)
- **Minimum Track Length**: 6 segments
- **Maximum Frame Skip**: 18 frames
- **Linking Distance**: 3 pixels
- **Analysis Radii**: [3,5,10,20,30] pixels

## Output Files

The pipeline generates various output files at each step:

- `*_locs.csv`: Raw localizations
- `*_tracks.csv`: Linked trajectories
- `*_NN.csv`: Nearest neighbor data
- `*_diffusion.csv`: Diffusion metrics
- `*_velocity.csv`: Velocity measurements
- `*_BGsubtract.csv`: Background-subtracted data
- `*_trapped-AllFrames.csv`: Complete trajectory data
- `*_locErr.csv`: Localization error metrics

## Requirements

- Python 3.7+
- Required packages:
  ```
  numpy
  pandas
  scikit-image
  scikit-learn
  tqdm
  flika
  trackpy
  ```

## Notes

- Process data sequentially through all steps
- Monitor memory usage with large datasets
- Backup raw data before processing
- Check intermediate results between steps

## References

When using this analysis pipeline, please cite:
```
Bertaccini et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, Cellular and Tissue Imaging. 
bioRxiv 2023.12.22.573117; doi: https://doi.org/10.1101/2023.12.22.573117
```
