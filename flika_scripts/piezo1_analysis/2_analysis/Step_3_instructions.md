# Nearest Neighbor Analysis for PIEZO1 Tracks

This script performs nearest neighbor (NN) analysis on PIEZO1 track data. It calculates the distance to the nearest neighbor for each localization in each frame and adds this information to the track data.

## Prerequisites

- Python 3.x
- Required Python packages: numpy, pandas, sklearn, scipy, matplotlib

## Usage

1. Set the `path` variable in the script to the directory containing your localization and SVM classification files.

2. Run the script:
python Step_3_nearestNeighbour.py
Copy
## Functionality

The script performs two main operations:

1. Calculates nearest neighbor distances for each localization in each frame.
2. Adds the nearest neighbor information to the SVM classification files.

## Input Files

- *_locsID.csv: Localization files with IDs
- *_SVM-ALL.csv: Files containing SVM classification results

## Output Files

- *_locsID_NN.csv: Localization files with added nearest neighbor distances
- *_SVM-ALL_NN.csv: SVM classification files with added nearest neighbor distances

## Customization

You can modify the following parameters in the script:
- `k` in the `getNearestNeighbors` function: Number of nearest neighbors to find (default is 2)
- The pixel size conversion factor (currently set to 108 nm/pixel)

## Note

This script assumes that the input files follow a specific naming convention. Ensure that your files match the expected format, or modify the file glob patterns in the script accordingly.
