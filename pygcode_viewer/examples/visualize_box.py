#!/usr/bin/env python3
"""
Visualize the Shape-Box GCODE file
"""

import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygcode_viewer

def main():
    # Path to your GCODE file
    gcode_path = "/Users/tomasjurica/projects/PrusaSlicer/Shape-Box_0.4n_0.2mm_PLA_MK4IS_20m.gcode"
    
    if not os.path.exists(gcode_path):
        print(f"Error: GCODE file not found: {gcode_path}")
        return 1
    
    print(f"Loading: {gcode_path}")
    
    # Create viewer and load GCODE
    viewer = pygcode_viewer.GCodeViewer()
    viewer.load(gcode_path)
    
    # Print info
    layer_count = viewer.get_layer_count()
    bbox = viewer.get_bounding_box()
    est_time = viewer.get_estimated_time()
    
    print(f"Layers: {layer_count}")
    print(f"Bounding box: X[{bbox[0]:.1f}, {bbox[3]:.1f}] Y[{bbox[1]:.1f}, {bbox[4]:.1f}] Z[{bbox[2]:.1f}, {bbox[5]:.1f}]")
    print(f"Estimated time: {est_time/60:.1f} minutes")
    
    # Calculate camera position based on model size
    center_x = (bbox[0] + bbox[3]) / 2
    center_y = (bbox[1] + bbox[4]) / 2
    center_z = (bbox[2] + bbox[5]) / 2
    size_x = bbox[3] - bbox[0]
    size_y = bbox[4] - bbox[1]
    size_z = bbox[5] - bbox[2]
    max_size = max(size_x, size_y, size_z)
    distance = max_size * 1.5
    
    # Configure visualization with PrusaSlicer-like colors
    config = pygcode_viewer.ViewConfig()
    config.view_type = "FeatureType"
    config.visible_features["travels"] = False
    config.width = 1920
    config.height = 1080
    config.background_color = "#EAEAEA"  # Light gray like PrusaSlicer
    viewer.set_config(config)
    
    # Render isometric view (front-right)
    viewer.set_camera(
        pos=(center_x + distance * 0.8, center_y - distance * 0.8, center_z + distance * 0.6),
        target=(center_x, center_y, center_z),
        fov=45.0
    )
    output_path = "shape_box_full.png"
    print(f"\nRendering isometric view to: {output_path}")
    viewer.render_to_file(output_path)
    print(f"Saved: {output_path}")
    
    # Render top view - camera directly above with Y as up vector
    viewer.set_camera(
        pos=(center_x, center_y, center_z + distance * 2),
        target=(center_x, center_y, center_z),
        up=(0.0, 1.0, 0.0),  # Y-up for top view (can't use Z when looking down Z axis)
        fov=45.0
    )
    output_path = "shape_box_top.png"
    print(f"Rendering top view to: {output_path}")
    viewer.render_to_file(output_path)
    print(f"Saved: {output_path}")
    
    # Render front view
    viewer.set_camera(
        pos=(center_x, center_y - distance * 2, center_z),
        target=(center_x, center_y, center_z),
        fov=45.0
    )
    output_path = "shape_box_front.png"
    print(f"Rendering front view to: {output_path}")
    viewer.render_to_file(output_path)
    print(f"Saved: {output_path}")
    
    # Render with speed coloring
    viewer.set_camera(
        pos=(center_x + distance * 0.8, center_y - distance * 0.8, center_z + distance * 0.6),
        target=(center_x, center_y, center_z),
        fov=45.0
    )
    viewer.set_view_type("Speed")
    output_path = "shape_box_speed.png"
    print(f"Rendering speed view to: {output_path}")
    viewer.render_to_file(output_path)
    print(f"Saved: {output_path}")
    
    print("\nDone! Check the PNG files in the current directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
