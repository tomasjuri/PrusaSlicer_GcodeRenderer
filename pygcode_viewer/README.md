# pygcode-viewer

Python wrapper for PrusaSlicer's GCode visualization library.

## Features

- Load and visualize GCODE files (ASCII and binary)
- Render to PNG images with configurable camera and view settings
- Headless rendering support (no display required)
- Cross-platform: Linux, macOS, Windows
- JSON-based configuration for all visualization options

## Installation

```bash
pip install pygcode-viewer
```

### Building from source

```bash
git clone https://github.com/prusa3d/PrusaSlicer.git
cd PrusaSlicer/pygcode_viewer
pip install .
```

## Quick Start

```python
import pygcode_viewer

# Create viewer
viewer = pygcode_viewer.GCodeViewer()

# Load GCODE file
viewer.load("model.gcode")

# Set camera position
viewer.set_camera(
    pos=(150, 150, 100),
    target=(100, 100, 50)
)

# Render to PNG
viewer.render_to_file("output.png", width=1920, height=1080)
```

## Configuration

You can customize the visualization using a JSON configuration file:

```python
import pygcode_viewer

viewer = pygcode_viewer.GCodeViewer()
viewer.load("model.gcode")

# Load configuration from file
viewer.set_config("config.json")

# Or use a ViewConfig object
config = pygcode_viewer.ViewConfig()
config.view_type = "Speed"
config.layer_range = (0, 50)
config.visible_features["travels"] = True
viewer.set_config(config)

viewer.render_to_file("output.png")
```

### Configuration Schema

```json
{
  "view_type": "FeatureType",
  "layer_range": [0, null],
  "visible_features": {
    "travels": false,
    "retractions": true,
    "seams": true
  },
  "extrusion_roles": {
    "Perimeter": {"visible": true, "color": "#FFA500"},
    "ExternalPerimeter": {"visible": true, "color": "#FFFF00"},
    "InternalInfill": {"visible": true, "color": "#B22222"}
  },
  "tool_colors": ["#FF8000", "#00FF00", "#0000FF"],
  "output": {
    "width": 1920,
    "height": 1080,
    "background_color": "#FFFFFF"
  }
}
```

### View Types

- `FeatureType` - Color by extrusion role (perimeter, infill, etc.)
- `Height` - Color by layer height
- `Width` - Color by extrusion width
- `Speed` - Color by print speed
- `FanSpeed` - Color by fan speed
- `Temperature` - Color by nozzle temperature
- `VolumetricFlowRate` - Color by volumetric flow rate
- `Tool` - Color by extruder/tool
- `ColorPrint` - Color print visualization

## API Reference

### GCodeViewer

- `load(gcode_path)` - Load a GCODE file
- `set_camera(pos, target, up, fov)` - Set camera parameters
- `set_config(config)` - Set visualization configuration
- `set_layer_range(min_layer, max_layer)` - Set layer range to render
- `set_view_type(view_type)` - Set coloring mode
- `render_to_file(output_path, width, height)` - Render to PNG
- `get_layer_count()` - Get number of layers
- `get_bounding_box()` - Get model bounding box
- `get_estimated_time()` - Get estimated print time

### CameraParams

- `position` - Camera position (x, y, z)
- `target` - Look-at target (x, y, z)
- `up` - Up vector (default: (0, 0, 1))
- `fov` - Field of view in degrees

### ViewConfig

- `view_type` - Coloring mode
- `layer_range` - Layer range to render
- `visible_features` - Feature visibility dict
- `extrusion_roles` - Role visibility and colors
- `tool_colors` - Tool/extruder colors
- `width`, `height` - Output image size
- `background_color` - Background color

## License

AGPL-3.0-or-later (same as PrusaSlicer)
