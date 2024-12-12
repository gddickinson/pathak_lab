# Testing and Miscellaneous Scripts

This directory contains experimental, development, and testing scripts created during the PIEZO1 analysis pipeline development. While these scripts may not be part of the final analysis workflow, they may be useful for testing, debugging, or future development.

## Script Overview

### Data Processing Tests
- `addIDtoLocs.py`: Testing ID assignment methods
- `calcuateRGandFeatures.py`: Experimental radius of gyration calculations
- `classifyTracks.py`: Track classification prototypes
- `compileTrackResults_OLD.py`: Previous version of results compilation
- `cropStackToPoints.py`: Testing cropping methods

### Filtering Tests
- `filterDFs.py`: DataFrame filtering experiments
- `groupFilter_DF.py`: Group-based filtering approaches
- `flikaOutputArrayOrientation.py`: Testing FLIKA output orientations

### Visualization Experiments
- `flowerPlot-fitBoundary.py`: Experimental visualization technique
- `polarPlot.py`: Polar coordinate plotting tests

### Analysis Tests
- `gaussFunc.py`: Gaussian function implementations
- `getIntensityFromXY.py`: Testing intensity extraction methods
- `intensityTracePeakDetection.py`: Peak detection algorithms
- `intensityTracePeakDetection_2.py`: Alternative peak detection method
- `meanTest.py`: Statistical testing approaches

### Utility Tests
- `joinTracks.py`: Track joining methods
- `linkCentroids.py`: Centroid linking approaches
- `listSubfolders.py`: Directory traversal utilities
- `loadJSONtoDF.py`: JSON loading experiments
- `smoothing.py`: Signal smoothing implementations
- `xyInsideLabelledRegion.py`: Region detection methods

## Purpose

These scripts were created to:
- Test new analysis approaches
- Debug existing functionality
- Experiment with alternative methods
- Prototype new features
- Validate analysis techniques

## Usage Notes

- Scripts may be incomplete or non-functional
- May contain hardcoded paths or parameters
- Limited error handling
- Minimal documentation
- May require modification for current use

## Development Reference

While not used in production, these scripts may be useful for:
- Understanding development history
- Testing new features
- Debugging similar functionality
- Prototyping improvements
- Learning alternative approaches

## Structure
```
testing_misc/
├── Analysis Tests/
│   ├── calcuateRGandFeatures.py
│   ├── gaussFunc.py
│   └── meanTest.py
├── Visualization Tests/
│   ├── flowerPlot-fitBoundary.py
│   └── polarPlot.py
└── Utility Tests/
    ├── joinTracks.py
    └── linkCentroids.py
```

## Dependencies
- Same as main analysis pipeline
- May require additional testing packages
- Some scripts may have unique requirements

## Contributing
- Feel free to use these scripts as starting points
- Document any improvements or fixes
- Note successful approaches for future reference

## Warning
These scripts are provided as-is for reference and development purposes. They:
- May not follow best practices
- Could contain bugs or errors
- Might use deprecated methods
- May need significant modification
- Are not maintained

## Future Development
When using these scripts for development:
1. Test thoroughly before implementation
2. Document modifications
3. Update dependencies
4. Add error handling
5. Follow current best practices

## Notes
- Keep for reference purposes
- May contain useful code snippets
- Could be basis for new features
- Helpful for understanding development process
- Valuable for debugging similar issues

For production analysis, please use the main pipeline scripts in the parent directories.