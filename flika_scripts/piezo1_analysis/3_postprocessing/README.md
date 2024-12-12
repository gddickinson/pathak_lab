# PIEZO1 Analysis Postprocessing

Tools for processing, organizing, and analyzing PIEZO1 tracking results. These scripts handle data compilation, statistical analysis, filtering, and file organization.

## Scripts Overview

### Data Analysis

#### 1. Position-based Analysis
- `binByPosition.py`: Spatial binning of particle positions
- `calculateDensities.py`: Calculates particle densities and distributions
  ```python
  n_locs, x_size, y_size = calcDensity(df, n_frames=10)
  ```

#### 2. Track Statistics
- `compileTrackResults.py`: Comprehensive statistical analysis
- `runningSum.py`: Calculates cumulative measurements
  ```python
  df = runningSum(df, colName='lag', group=['Experiment', 'track_number'])
  ```
- `uniqueTrackID_pathAt50.py`: Assigns unique IDs and calculates path lengths
  ```python
  df = uniqueTrackID(df)
  df = getPathLengthAtLag(df, lag=50)
  ```

### Data Organization

#### 3. Classification Management
- `compileIterationSVMClasses.py`: Compiles SVM classification results
- `filterDFbySVM.py`: Separates data by SVM class
- `filterDFbyIntensity.py`: Filters based on intensity values

#### 4. File Management
- `moveFilesToSubfolder.py`: Organizes files into subdirectories 
- `countAnalysisFileLocs.py`: Counts records across analysis files
- `countAnalysisFileLocs_2.py`: Alternative counting implementation

## Features

### Statistical Analysis
- Track measurements:
  - Length and segments
  - Radius of gyration
  - Asymmetry metrics
  - Velocity profiles
  - Nearest neighbor statistics

### Data Management
- File organization:
  ```
  output/
  ├── SVM_classes/
  ├── intensity_filtered/
  └── statistics/
  ```
- Record counting
- Data filtering
- Results compilation

### Track Processing
- Running sums
- Path length calculations
- Unique ID assignment
- Classification sorting

## Usage Examples

### 1. Compile Track Statistics
```python
from compileTrackResults import compileTrackResults

compileTrackResults(
    exptFolder='/path/to/data',
    analysisStage='NNcount'
)
```

### 2. Filter Results
```python
# By SVM class
for svm in [1, 2, 3]:
    df_SVM = df[df['SVM'] == svm]
    df_SVM.to_csv(f'results_SVM{svm}.csv')

# Calculate running sums
df = runningSum(df, 'lag')
```

### 3. Organize Files
```python
# Move files to subdirectories
for csv_file_path in csv_file_list:
    subfolder = os.path.join(subfolder_location, 'analysis_results')
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    shutil.copy(csv_file_path, subfolder)
```

## Output Files

### Analysis Results
- `*_trackMeans.csv`: Track-level statistics
- `*_trackStats.csv`: Experiment-level statistics
- `*_SVMx.csv`: Classification-specific data
- `*_runningSum.csv`: Cumulative measurements
- `countResult.csv`: File statistics

### Statistical Measures
- Basic statistics (mean, std)
- Track characteristics
- Spatial measurements
- Classification metrics
- Density calculations

## Key Parameters

### Track Analysis
- `lag`: Time window for path length calculations (default: 50)
- `n_frames`: Frames for density calculation (default: 10)
- `bin_size`: Spatial binning parameter (default: 10)

### File Processing
- Suffix patterns for file counting
- Directory structure for organization
- Filtering thresholds

## Tips
- Process hierarchically (experiment → condition → file)
- Back up data before filtering/moving
- Verify file paths and permissions
- Monitor memory usage
- Use consistent naming conventions

## Dependencies
- NumPy
- Pandas
- tqdm
- scipy
- pathlib
- shutil

## Notes
- Run scripts sequentially in analysis pipeline
- Verify input data structure
- Monitor disk space
- Document parameter choices
- Keep original data intact

## Error Handling
- Checks for missing files
- Handles unlinked particles
- Validates data structures
- Manages file permissions

## Contributing
Please submit issues and pull requests to improve functionality.

## References
Cite when using:
```
Bertaccini et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, 
Cellular and Tissue Imaging. bioRxiv 2023.12.22.573117
```
