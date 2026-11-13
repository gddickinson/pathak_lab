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
- `pixelSize_new`: Pixel size in nanometres (default `PIXEL_SIZE_NM` = 108 nm, the lab's camera, set in
  `pipeline_defaults.py`); used to convert x [nm] to pixels when loading points and back when saving tracks
- `featureTransform`: SVM feature transform, `'none'` (default: raw features, as in published results; the old
  code's Box-Cox never took effect) or `'boxcox'` (Box-Cox lambdas fitted on the training tracks and reused for
  prediction). 10-fold CV accuracy on the committed training CSV: none 89.5 %, boxcox 86.4 % (held-out 20 %: 89.5 % vs 84.5 %). The default lives in `pipeline_defaults.py`.

## Note

This script is part of a larger workflow. Ensure that you have run the previous steps (e.g., ThunderSTORM analysis) before running this script.
