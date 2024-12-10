# ThunderSTORM Macro Generator

This script generates an ImageJ macro for batch processing of .tif files using ThunderSTORM, a plugin for ImageJ that performs single-molecule localization microscopy (SMLM) analysis.

## Prerequisites

- Python 3.x
- ImageJ with ThunderSTORM plugin installed

## Usage

1. Set the `path` variable in the script to the directory containing your .tif files.

2. Run the script:
python Step_1_getCentroidsByRunningImageJMacro.py

3. The script will generate a macro file named `thunderStorm_macro_auto.ijm` in the specified directory.

4. Open ImageJ and run the generated macro file to process your .tif files with ThunderSTORM.

## Output

For each input .tif file, the macro will generate a corresponding _locs.csv file containing the localization results from ThunderSTORM.

## Customization

You can modify the `macroCommand` variable in the script to adjust ThunderSTORM parameters or add additional processing steps.

## Note

Ensure that the paths in the script are correct for your system and that ImageJ with ThunderSTORM is properly installed and configured.
