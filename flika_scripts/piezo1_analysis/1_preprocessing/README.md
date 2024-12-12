# PIEZO1 Data Preprocessing

This directory contains scripts for preparing microscopy data for PIEZO1 analysis. These preprocessing steps ensure data quality and format consistency before the main analysis pipeline.

## Scripts Overview

### 1. Frame Binning
**Script**: `binByFrame.py`
- Bins frames temporally to improve signal-to-noise ratio
- Uses FLIKA's frame binning functionality
- Parameters:
  ```python
  binSize = 10  # Number of frames to combine
  ```
- Usage:
  ```python
  from binByFrame import processFolder
  processFolder('/path/to/data/', binSize=10)
  ```
- Output: `*_bin{binSize}.tif` files

### 2. Bit Depth Conversion
**Script**: `convert64bitTiffto32bit.py`
- Converts 64-bit TIFF files to 32-bit format
- Ensures compatibility with analysis tools
- Usage:
  ```python
  # Automatic conversion
  im = io.imread(fileName)
  im = im.astype('int32')
  io.imsave(fileName, im)
  ```

### 3. Recording Length Cropping
**Script**: `cropLength.py`
- Trims recordings to specified length
- Useful for standardizing analysis durations
- Parameters:
  ```python
  cropSize = 1000  # Number of frames to keep
  ```
- Usage:
  ```python
  from cropLength import processFolder
  processFolder('/path/to/data/', cropSize=1000)
  ```
- Output: `*_crop{cropSize}.tif` files

### 4. Autofluorescence Removal
**Script**: `removeAutofluoresence.py`
- Removes high-intensity autofluorescent regions
- Filters localizations in affected areas
- Parameters:
  ```python
  thresh = 250        # Intensity threshold
  minSize = 90       # Minimum region size
  dilationRadius = 6 # Region expansion size
  pixelSize = 108    # nm per pixel
  ```
- Features:
  - Automated threshold detection
  - Region dilation for safety margins
  - Optional noise replacement
  - Visualization tools
- Outputs:
  - `*_crop.tif`: Cleaned image stack
  - `*_highFluroRemoved_locs.csv`: Filtered localizations

## Usage Example

```python
# Complete preprocessing workflow
from flika import start_flika

# 1. Start with bit depth conversion if needed
from convert64bitTiffto32bit import *
convert_tiff('raw_data.tif')

# 2. Bin frames
fa = start_flika()
from binByFrame import processFolder
processFolder('/path/to/data/', binSize=10)
fa.close()

# 3. Crop recording length
fa = start_flika()
from cropLength import processFolder
processFolder('/path/to/data/', cropSize=1000)
fa.close()

# 4. Remove autofluorescence
from removeAutofluoresence import removeLocsFromHighIntensityRegions
cleaned_locs = removeLocsFromHighIntensityRegions(
    'data.tif',
    'locs.csv',
    thresh=250,
    minSize=90,
    dilationRadius=6
)
```

## Key Parameters

### Binning
- `binSize`: Number of frames to combine (default: 10)
- Larger values improve SNR but reduce temporal resolution

### Cropping
- `cropSize`: Number of frames to keep
- Choose based on phenomenon timescale and memory constraints

### Autofluorescence Removal
- `thresh`: Intensity threshold (default: 250)
- `autoThresh`: Enable automatic threshold detection
- `minSize`: Minimum region size in pixels (default: 90)
- `dilationRadius`: Region expansion size (default: 6)
- `addNoise`: Replace removed regions with noise
- `noiseScale`: Scale factor for replacement noise

## Tips
- Process files sequentially to manage memory
- Verify results visually using `plotResult=True`
- Back up raw data before preprocessing
- Adjust parameters based on your microscopy settings
- Monitor file sizes after binning/cropping

## Dependencies
- FLIKA
- NumPy
- scikit-image
- tqdm
- pandas
- matplotlib

## Notes
- All scripts preserve original data by creating new files
- Use consistent parameters across dataset
- Consider storage requirements for binned data
- Document parameter choices for reproducibility
