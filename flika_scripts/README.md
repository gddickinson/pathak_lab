# PIEZO1 Analysis Pipeline

A comprehensive analysis pipeline for studying PIEZO1 protein dynamics in microscopy data, as described in [Bertaccini et al. 2023](https://doi.org/10.1101/2023.12.22.573117). This pipeline provides tools for preprocessing microscopy data, analyzing protein movement and behavior, and postprocessing results.

## Directory Structure
```
piezo1_analysis/
├── 1_preprocessing/      # Data preparation
├── 2_analysis/          # Core analysis pipeline
├── 3_postprocessing/    # Results compilation and analysis
└── testing_misc/        # Development and test scripts
```

## Pipeline Overview

### 1. Preprocessing
- **Purpose**: Prepare raw microscopy data for analysis
- **Key Operations**:
  - Frame binning for SNR improvement
  - Bit depth conversion
  - Recording length standardization
  - Autofluorescence removal
- **Output**: Cleaned TIFF files ready for analysis

### 2. Analysis
- **Purpose**: Track and analyze PIEZO1 movement
- **Key Steps**:
  1. Centroid detection (ThunderSTORM)
  2. Particle linking
  3. Nearest neighbor analysis
  4. Diffusion analysis
  5. Velocity calculations
  6. Classification
- **Output**: Tracked particles with movement metrics

### 3. Postprocessing
- **Purpose**: Compile and analyze results
- **Key Features**:
  - Statistical analysis
  - Track compilation
  - Classification summaries
  - Data organization
  - Results visualization
- **Output**: Final analysis results and statistics

## Quick Start

1. **Prepare Data**:
```python
# Preprocess microscopy data
from preprocessing.binByFrame import processFolder
processFolder('/path/to/data/', binSize=10)
```

2. **Run Analysis**:
```python
# Link particles and analyze
from analysis.Step_2_linkingAndClassification import linkFilesNoFlika
linkFilesNoFlika(tiffList, skipFrames=18, distanceToLink=3)
```

3. **Process Results**:
```python
# Compile statistics
from postprocessing.compileTrackResults import compileTrackResults
compileTrackResults(exptFolder, analysisStage='NNcount')
```

## Key Parameters

### Processing
- `pixelSize`: 0.108 μm (default)
- `binSize`: 10 frames (typical)
- `cropSize`: Experiment-dependent

### Analysis
- `skipFrames`: 18 frames max gap
- `distanceToLink`: 3 pixels
- `minLinkSegments`: 6 segments

### Classification
- SVM classes: 1 (mobile), 2 (intermediate), 3 (trapped)
- Intensity thresholds: Experiment-specific
- Neighbor radii: [3,5,10,20,30] pixels

## Requirements

### Software
- Python 3.7+
- FLIKA
- ImageJ/FIJI with ThunderSTORM

### Python Packages
- numpy
- pandas
- scikit-image
- scikit-learn
- trackpy
- tqdm
- scipy
- matplotlib

## Data Organization

### Input Data
```
data/
├── experiment_1/
│   ├── condition_1/
│   │   └── recording.tif
│   └── condition_2/
└── experiment_2/
```

### Analysis Results
```
results/
├── preprocessing/
├── tracking/
└── statistics/
```

## Usage Tips

1. **Preprocessing**:
   - Verify raw data quality
   - Document processing parameters
   - Keep original data intact

2. **Analysis**:
   - Monitor memory usage
   - Process files sequentially
   - Validate intermediate results

3. **Postprocessing**:
   - Back up results regularly
   - Use consistent naming
   - Document parameter choices

## Development

- Test scripts available in `testing_misc/`
- See individual directory READMEs for details
- Follow existing code structure
- Document modifications

## Troubleshooting

Common issues:
- Memory errors: Process files in batches
- File permissions: Check write access
- Missing results: Verify file paths
- Analysis errors: Check parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit pull request

## Citation

When using this pipeline, please cite:
```
Bertaccini et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, 
Cellular and Tissue Imaging. bioRxiv 2023.12.22.573117
```

## Contact

For questions or issues:
- Submit GitHub issues
- See paper for author contact information
