# Overlay Plugin

A FLIKA plugin for overlaying and aligning microscopy channels with advanced visualization controls. Particularly useful for analyzing multi-channel data of PIEZO1 proteins and cellular structures.

## Features

### Image Overlay
- Overlay two image channels
- Independent channel control
- Adjustable opacity
- Customizable color mapping

### Image Processing
- Gamma correction
- Live preview
- Background subtraction
- Intensity scaling

### Visualization Controls
- Independent histogram adjustment
- Channel blending modes
- Opacity controls
- LUT customization

## Installation

1. Copy to FLIKA plugins directory:
```bash
cp -r overlay ~/.FLIKA/plugins/
```

2. Dependencies:
```python
# Included with FLIKA
- PyQt5
- numpy
- pyqtgraph
```

## Usage

### Basic Operation

1. Load data in FLIKA:
```python
import flika
flika.start_flika()
ch1 = open_file('channel1.tif')
ch2 = open_file('channel2.tif')
```

2. Initialize plugin:
```python
from flika.plugins.overlay import Overlay
overlay = Overlay()
```

3. Select channels and create overlay:
```python
# Through GUI:
# 1. Select CH1 (red)
# 2. Select CH2 (green)
# 3. Click 'Overlay'
```

### Key Parameters

#### Channel Selection
- CH1: Primary channel (typically red)
- CH2: Secondary channel (typically green)

#### Gamma Correction
```python
# Enable gamma correction
overlay.gammaCorrect.setChecked(True)

# Set gamma value
overlay.gamma.setValue(1.5)  # Range: 0.0-20.0
```

#### Preview Options
- Live gamma preview
- Channel visibility toggling
- Blending mode selection

## GUI Components

### Main Window
- Channel selectors
- Control buttons
- Processing options

### Histogram Controls
- Independent channel levels
- LUT selection
- Opacity adjustment

### Gamma Controls
- Gamma value slider
- Preview toggle
- Live update

## Advanced Features

### Custom LUTs
```python
# Set custom lookup table
overlay.gradientPreset = 'thermal'  # Options: thermal, flame, yellowy, etc.
```

### Blending Modes
```python
# Available modes
modes = {
    'Normal': QPainter.CompositionMode_SourceOver,
    'Screen': QPainter.CompositionMode_Screen,
    'Multiply': QPainter.CompositionMode_Multiply
}
```

### Export Options
- Save overlay as TIFF
- Export settings
- Save individual channels

## Configuration

### Default Settings
```python
settings = {
    'gamma': 1.0,
    'opacity': 0.5,
    'gradientPreset': 'grey',
    'usePreset': False
}
```

### Display Options
```python
display_settings = {
    'showHistogram': True,
    'linkChannels': False,
    'autoRange': False,
    'autoLevels': False
}
```

## Tips & Best Practices

1. Channel Preparation:
   - Match dimensions
   - Consider bit depth
   - Pre-process if needed

2. Overlay Setup:
   - Start with default gamma
   - Adjust levels first
   - Fine-tune opacity

3. Performance:
   - Close preview when not needed
   - Use appropriate zoom level
   - Consider ROI for large images

## Troubleshooting

Common issues and solutions:

1. Display Issues:
   - Check channel dimensions
   - Verify bit depth compatibility
   - Reset histogram levels

2. Performance:
   - Reduce preview updates
   - Close unused windows
   - Use ROI selection

3. Alignment:
   - Verify channel registration
   - Check scaling
   - Consider pre-alignment

## Example Workflows

### Basic Overlay
```python
# 1. Load channels
# 2. Set gamma correction
# 3. Adjust levels
# 4. Set opacity
# 5. Export result
```

### Advanced Processing
```python
# 1. Pre-process channels
# 2. Apply gamma correction
# 3. Custom LUT selection
# 4. Fine-tune blending
# 5. Export with settings
```

## Known Limitations

- Large file handling
- Memory management
- Real-time processing speed
- LUT restrictions

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

## License

MIT License - See LICENSE file for details
