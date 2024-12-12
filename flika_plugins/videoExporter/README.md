# Video Exporter Plugin

A FLIKA plugin for exporting microscopy data as video, with particular focus on ROI-based exports and frame rate control. Designed for creating presentation-quality videos from PIEZO1 imaging data.

## Features

### Video Export
- Full-frame export
- ROI-specific export
- Frame rate control
- Resolution settings

### ROI Management
- Custom ROI selection
- Multiple ROI support
- Size adjustment
- Position control

### Parameters
- Pixel size calibration
- Frame length control
- Export format options
- Scale settings

## Installation

1. Copy to FLIKA plugins directory:
```bash
cp -r videoExporter ~/.FLIKA/plugins/
```

2. Dependencies:
```python
import numpy as np
from flika.window import Window
import pyqtgraph as pg
from tqdm import tqdm
```

## Usage

### Basic Operation

1. Load data in FLIKA:
```python
import flika
flika.start_flika()
window = open_file('data.tif')
```

2. Initialize plugin:
```python
from flika.plugins.videoExporter import VideoExporter
exporter = VideoExporter()
```

3. Set parameters and export:
```python
# Configure export
exporter.pixelSize.setValue(108)  # nm/pixel
exporter.frameLength.setValue(100)  # ms
```

## Key Parameters

### Spatial Settings
```python
settings = {
    'pixelSize': 108,  # nm/pixel
    'ROIsize': [512, 512],  # pixels
}
```

### Temporal Settings
```python
timing = {
    'frameLength': 100,  # ms
    'fps': 30,  # frames per second
}
```

## GUI Components

### Main Window
- Parameter controls
- ROI toggle
- Export buttons

### ROI Plot Window
- ROI selection
- Size controls
- Position adjustment

## Export Options

### Full Frame
- Complete field of view
- Original resolution
- All time points

### ROI Export
- Selected regions only
- Custom dimensions
- Specific time ranges

## Usage Examples

### Basic Export
```python
# Set parameters
exporter.pixelSize.setValue(108)
exporter.frameLength.setValue(100)

# Toggle ROI export
exporter.displayROIplot_checkbox.setChecked(True)

# Export video
exporter.ROIplot.exportVideo()
```

### ROI Selection
```python
# Enable ROI mode
exporter.toggleROIplot()

# Define ROI
exporter.ROIplot.addROI()
```

## Output Formats

- `.avi`: Standard video format
- `.mp4`: Compressed video
- Customizable:
  - Resolution
  - Frame rate
  - Compression

## Tips

1. Before Export:
   - Set correct pixel size
   - Verify frame timing
   - Check ROI positions

2. During Export:
   - Monitor progress
   - Verify file size
   - Check frame rate

3. After Export:
   - Validate video
   - Check timing
   - Verify quality

## Troubleshooting

Common issues and solutions:

1. Export Problems:
   - Check file permissions
   - Verify memory availability
   - Confirm format support

2. ROI Issues:
   - Reset ROI
   - Check boundaries
   - Verify selection

3. Timing Problems:
   - Verify frame length
   - Check frame rate
   - Adjust parameters

## Best Practices

### File Management
- Use descriptive names
- Organize by experiment
- Back up raw data

### ROI Selection
- Consider size limits
- Check boundaries
- Use appropriate margins

### Export Settings
- Match source frame rate
- Use appropriate compression
- Consider file size

## Parameters Reference

### Required Settings
```python
required = {
    'pixelSize': int,  # nm/pixel
    'frameLength': int,  # ms
}
```

### Optional Settings
```python
optional = {
    'fps': int,
    'compression': str,
    'format': str
}
```

## Contributing

To contribute:
1. Fork repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

## Citation

When using this plugin, please cite:
```
Bertaccini et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, 
Cellular and Tissue Imaging. bioRxiv 2023.12.22.573117
```

## Support

For assistance:
- Open GitHub issue
- Check FLIKA documentation
- Contact authors via paper

## Known Limitations

- Memory constraints with large datasets
- Format compatibility
- Processing speed
- ROI size limits

## Future Improvements

Planned features:
- Additional export formats
- Batch processing
- Advanced ROI tools
- Enhanced compression options

## License

MIT License - See LICENSE file for details
