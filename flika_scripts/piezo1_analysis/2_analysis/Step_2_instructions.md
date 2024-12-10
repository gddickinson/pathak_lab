# PIEZO1 Track Analysis Pipeline

This script is part of a pipeline for analyzing PIEZO1 tracks in single-molecule localization microscopy data. It performs several steps including linking localizations into tracks, calculating track features, and classifying tracks using a Support Vector Machine (SVM) model.

## Prerequisites

- Python 3.x
- Required Python packages: numpy, pandas, sklearn, scipy, matplotlib, seaborn, flika, trackpy (optional)
- Custom modules: pynsight_GDedit (ensure this is in your Python path)

## Usage

1. Set the `path` variable to the directory containing your .tif files and localization data.
2. Set the `trainpath` variable to the path of your training data for the SVM classifier.
3. Adjust other parameters as needed (e.g., `minLinkSegments`, `gapSize`, `distance`, `pixelSize_new`).
4. Run the script:
python Step_2_linkingAndClassification_trackpyOption_recursionErrorFix.py
Copy
## Pipeline Steps

1. Add IDs to localization files
2. Link localizations into tracks
3. Calculate track features (e.g., radius of gyration, fractal dimension)
4. Classify tracks using SVM
5. Add additional columns from ThunderStorm results (if applicable)

## Output

The script generates several intermediate and final output files:
- *_locsID.csv: Localization files with added IDs
- *_tracks.csv: Linked tracks
- *_tracksRG.csv: Tracks with calculated features
- *_SVMPredicted.csv: Tracks with SVM classification

## Customization

You can modify various parameters in the script to adjust the analysis:
- `minLinkSegments`: Minimum number of segments in a track
- `gapSize`: Maximum number of frames to skip when linking
- `distance`: Maximum distance for linking localizations
- `pixelSize_new`: Pixel size in micrometers

## Note

This script is part of a larger workflow. Ensure that you have run the previous steps (e.g., ThunderSTORM analysis) before running this script.
