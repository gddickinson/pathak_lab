# Join Nearest Neighbor Information to Result DataFrame

This script adds nearest neighbor (NN) information to the main result DataFrame containing PIEZO1 track data and SVM predictions. It's the final step in the PIEZO1 track analysis pipeline.

## Prerequisites

- Python 3.x
- Required Python packages: numpy, pandas, tqdm

## Usage

1. Set the `path` variable in the script to the directory containing your result files and nearest neighbor files.

2. Run the script:
python Step_4_joinNNtoResultDF.py

## Functionality

The script performs the following operations:

1. Loads the main result DataFrame (containing track information and SVM predictions).
2. Loads the corresponding nearest neighbor information DataFrame.
3. Adds the nearest neighbor information to the main DataFrame.
4. Saves the updated DataFrame with the added nearest neighbor information.

## Input Files

- *_locsID_tracksRG_SVMPredicted.csv: Main result files containing track information and SVM predictions
- *_locsID_NN.csv: Files containing nearest neighbor information

## Output Files

- *_locsID_tracksRG_SVMPredicted_NN.csv: Updated result files with added nearest neighbor information

## Customization

You can modify the following parameters in the script:
- `pixelSize` in the `addNNtoDF` function: Pixel size in nanometers (default is 108)

## Note

This script assumes that the input files follow a specific naming convention. Ensure that your files match the expected format, or modify the file glob patterns in the script accordingly.

The nearest neighbor distances are converted from nanometers to pixels in this step. Make sure the `pixelSize` parameter is set correctly for your data.
