# FLIKA Plugins for PIEZO1 Analysis

Custom plugins for the FLIKA image processing platform to analyze and visualize PIEZO1 protein dynamics. These plugins provide interactive tools for particle tracking, visualization, and data analysis.

## Plugin Overview

### Locs and Tracks Plotter
- Main visualization plugin for particle tracking
- Interactive track display and analysis
- Features:
  - Point and track visualization
  - Color-coded tracks based on properties
  - Track selection and filtering
  - Background subtraction
  - Statistical analysis tools

### Overlay Tools
- `overlay/`: Image overlay capabilities
- `overlayMultipleRecordings/`: Multi-recording alignment
- Features:
  - Dual-channel visualization
  - Gamma correction
  - Channel alignment
  - Intensity adjustment

### Translation and Scaling
- Tools for spatial alignment and transformation
- Features:
  - Template-based alignment
  - ROI-based transformations
  - Scale adjustments
  - Rotation controls

### Video Export
- Tools for exporting visualizations
- Features:
  - ROI-based video export
  - Frame rate control
  - Multiple export formats
  - Custom scaling options

## Installation

1. Locate your FLIKA plugins directory:
```bash
~/.FLIKA/plugins/  # Linux/Mac
%APPDATA%/FLIKA/plugins/  # Windows
```

2. Copy plugin folders:
```bash
cp -r flika_plugins/* ~/.FLIKA/plugins/
```

3. Restart FLIKA to load new plugins

## Usage

### Track Visualization
```python
from flika_plugins.locsAndTracksPlotter import LocsAndTracksPlotter

# Initialize plotter
plotter = LocsAndTracksPlotter()

# Plot tracks
plotter.plotTrackData()
```

### Overlay Analysis
```python
from flika_plugins.overlay import Overlay

# Create overlay
overlay = Overlay()
overlay.overlay()
```

### Data Transform
```python
from flika_plugins.translateAndScale import TranslateAndScale

# Initialize transformer
transformer = TranslateAndScale()
transformer.transformData()
```

## Key Features

### Visualization Options
- Track color coding
- Point display controls
- ROI selection
- Intensity scaling
- Background subtraction

### Analysis Tools
- Track statistics
- Diffusion analysis
- Nearest neighbor calculations
- Intensity measurements

### Data Management
- Track filtering
- ROI-based selection
- Data export
- Transform saving

## Dependencies

- FLIKA
- PyQt5
- numpy
- pandas
- scipy
- scikit-image
- pyqtgraph

## Plugin Configuration

Each plugin can be configured through its GUI interface or programmatically:

```python
# Example configuration
settings = {
    'pixelSize': 108,
    'frameLength': 100,
    'colorMap': 'viridis',
    'pointSize': 5
}
```

## Common Parameters

- `pixelSize`: Physical size of camera pixels (nm)
- `frameLength`: Time between frames (ms)
- `pointSize`: Display size for particles
- `lineWidth`: Track line thickness

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit pull request

## Troubleshooting

Common issues:
- Plugin loading errors: Check FLIKA version compatibility
- Display issues: Verify PyQt installation
- Performance: Adjust point/track display settings
- Memory: Process data in smaller batches

## Citation

When using these plugins, please cite:
```
Bertaccini et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, 
Cellular and Tissue Imaging. bioRxiv 2023.12.22.573117
```

## Contact

For issues and support:
- Submit GitHub issues
- See paper for author contact
