#!/usr/bin/env python3
"""
Example: Basic GCode rendering with pygcode_viewer

This example demonstrates how to:
1. Load a GCODE file
2. Configure the camera
3. Set visualization options
4. Render to a PNG image
"""

import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygcode_viewer

def main():
    # Create the viewer
    viewer = pygcode_viewer.GCodeViewer()
    
    # Load a GCODE file
    gcode_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "tests", "sample.gcode"
    )
    
    if not os.path.exists(gcode_path):
        print(f"GCODE file not found: {gcode_path}")
        return 1
    
    print(f"Loading GCODE: {gcode_path}")
    viewer.load(gcode_path)
    
    # Print info about the loaded GCODE
    print(f"Layers: {viewer.get_layer_count()}")
    print(f"Bounding box: {viewer.get_bounding_box()}")
    print(f"Estimated time: {viewer.get_estimated_time():.1f} seconds")
    
    # Set camera position
    # Position the camera to look at the model from an isometric angle
    bbox = viewer.get_bounding_box()
    center_x = (bbox[0] + bbox[3]) / 2
    center_y = (bbox[1] + bbox[4]) / 2
    center_z = (bbox[2] + bbox[5]) / 2
    
    # Camera distance based on model size
    size = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    distance = size * 2
    
    viewer.set_camera(
        pos=(center_x + distance, center_y + distance, center_z + distance * 0.8),
        target=(center_x, center_y, center_z)
    )
    
    # Configure visualization
    config = pygcode_viewer.ViewConfig()
    config.view_type = "FeatureType"
    config.visible_features["travels"] = False  # Hide travel moves
    config.width = 1280
    config.height = 720
    config.background_color = "#F0F0F0"
    
    viewer.set_config(config)
    
    # Render to file
    output_path = "output.png"
    print(f"Rendering to: {output_path}")
    viewer.render_to_file(output_path)
    
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
