#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:11:58 2022

@author: george
"""

# This script creates a macro to run ThunderSTORM for batch processing of files in ImageJ

####TODO! run macro directly using python
#pip install pyimagej
#need to install maven and set PATH (export PATH=/Users/george/opt/apache-maven-3.8.6/bin:$PATH)

import glob
import os


def getFileList(path):
    """
    Get a list of .tif files in the specified directory and its subdirectories.

    Args:
    path (str): The directory path to search for .tif files

    Returns:
    tuple: Two strings, one containing comma-separated file paths,
           and another with corresponding result file paths
    """
    # Get all .tif files in the directory and subdirectories
    files = glob.glob(path + '/**/*.tif', recursive = True)

    # Convert list of files to comma-separated string
    fileStr = str(files).replace('[', '').replace(']', '')

    # Create corresponding result file paths
    resStr = fileStr.replace('.tif', '_locs.csv')

    return fileStr, resStr

macroCommand = """for (i=0; i  < datapaths.length; i++) {
 	//open(datapaths[i]);
 	run("Bio-Formats Importer", "open=" + datapaths[i] +" color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
 	run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
 	run("Export results", "filepath=["+respaths[i]+"] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true");
 	//close();
    while (nImages>0) {
        selectImage(nImages);
        close();
    }
}"""

#### WITH FILTER - only works for first file in batch! can't figure out why ###

# macroCommand = """for (i=0; i  < datapaths.length; i++) {
#  	//open(datapaths[i]);
#  	run("Bio-Formats Importer", "open=" + datapaths[i] +" color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
#  	run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
#     run("Show results table", "action=filter formula=[intensity > 500 & intensity < 600]");
#  	run("Export results", "filepath=["+respaths[i]+"] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true");
#     run("Show results table", "action=reset");
#     //close();
#     while (nImages>0) {
#         selectImage(nImages);
#         close();
#     }
# }"""

# macroCommand_multiemitter = """for (i=0; i  < datapaths.length; i++) {
# 	//open(datapaths[i]);
# 	run("Bio-Formats Importer", "open=" + datapaths[i] +" color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
# 	run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=false nmax=5 fixed_intensity=true expected_intensity=100:500 pvalue=1.0E-6 renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
# 	run("Export results", "filepath=["+respaths[i]+"] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true");
# 	//close();
#     while (nImages>0) {
#         selectImage(nImages);
#         close();
#     }
# }"""


# #gabbys params for ultraslow
# macroCommand_multiemitter = """for (i=0; i  < datapaths.length; i++) {
#  	//open(datapaths[i]);
#  	run("Bio-Formats Importer", "open=" + datapaths[i] +" color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
#  	run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=false nmax=5 fixed_intensity=true expected_intensity=20:500 pvalue=1.0E-6 renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
#  	run("Export results", "filepath=["+respaths[i]+"] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true");
#  	//close();
#     while (nImages>0) {
#         selectImage(nImages);
#         close();
#     }
# }"""

def writeMacro(datapaths, respaths, macroCommand, savepath):
    """
    Write the ImageJ macro to a file.

    Args:
    datapaths (str): Comma-separated string of input file paths
    respaths (str): Comma-separated string of output file paths
    macroCommand (str): The ImageJ macro command
    savepath (str): Path to save the macro file
    """
    stringtowrite = "{}\n{}\n{}".format(datapaths, respaths, macroCommand)

    with open(savepath, "w") as f:
        f.write(stringtowrite)



if __name__ == '__main__':
    # Set top folder level for analysis
    path = '/Users/george/Data/MCS_04_20230906_BAPTA_NSC66_5uM_UltraQuiet_FOV56_1'

    # Set path to save the macro
    savepath = os.path.join(path, 'thunderStorm_macro_auto.ijm')

    # Get list of input files and corresponding output files
    dataPathStr, resPathStr = getFileList(path)

    # Format paths for use in ImageJ macro
    datapaths = 'datapaths = newArray({});'.format(dataPathStr)
    respaths = 'respaths = newArray({});'.format(resPathStr)

    # Write the macro to a file
    writeMacro(datapaths, respaths, macroCommand, savepath)

    #multiemitter option - to modify number of emitters etc modify the macroCommand_multiemitter string above
    #writeMacro(datapaths, respaths, macroCommand_multiemitter, savepath)
