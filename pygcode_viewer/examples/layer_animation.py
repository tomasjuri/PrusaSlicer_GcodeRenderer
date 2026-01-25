#!/usr/bin/env python3
"""
Example: Render layer-by-layer animation frames

This example demonstrates how to:
1. Load a GCODE file
2. Render each layer to a separate image
3. These images can then be combined into an animation
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
    
    layer_count = viewer.get_layer_count()
    print(f"Total layers: {layer_count}")
    
    # Create output directory
    output_dir = "animation_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up camera
    bbox = viewer.get_bounding_box()
    center_x = (bbox[0] + bbox[3]) / 2
    center_y = (bbox[1] + bbox[4]) / 2
    center_z = (bbox[2] + bbox[5]) / 2
    size = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    distance = size * 2
    
    viewer.set_camera(
        pos=(center_x + distance, center_y + distance, center_z + distance * 0.5),
        target=(center_x, center_y, center_z)
    )
    
    # Render each layer
    for layer in range(layer_count):
        viewer.set_layer_range(0, layer)
        
        output_path = os.path.join(output_dir, f"frame_{layer:04d}.png")
        print(f"Rendering layer {layer + 1}/{layer_count} -> {output_path}")
        
        viewer.render_to_file(output_path, width=640, height=480)
    
    print(f"\nFrames saved to: {output_dir}/")
    print("To create an animation, you can use ffmpeg:")
    print(f"  ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 animation.mp4")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
