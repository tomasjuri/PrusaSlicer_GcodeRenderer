#!/usr/bin/env python3
"""
PrusaSlicer-style GCODE visualization with layer cuts.

This script creates visualizations that match PrusaSlicer's appearance:
- Exact color scheme from PrusaSlicer
- Multiple layer cut views
- Various camera angles
"""

import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygcode_viewer

def main():
    # Path to GCODE file
    gcode_path = "/Users/tomasjurica/projects/PrusaSlicer/Shape-Box_0.4n_0.2mm_PLA_MK4IS_20m.gcode"
    
    if not os.path.exists(gcode_path):
        print(f"Error: GCODE file not found: {gcode_path}")
        return 1
    
    print(f"Loading: {gcode_path}")
    
    # Create viewer
    viewer = pygcode_viewer.GCodeViewer()
    viewer.load(gcode_path)
    
    # Get model info
    layer_count = viewer.get_layer_count()
    bbox = viewer.get_bounding_box()
    
    print(f"Layers: {layer_count}")
    print(f"Bounding box: X[{bbox[0]:.1f}, {bbox[3]:.1f}] Y[{bbox[1]:.1f}, {bbox[4]:.1f}] Z[{bbox[2]:.1f}, {bbox[5]:.1f}]")
    
    # Calculate camera positions
    center_x = (bbox[0] + bbox[3]) / 2
    center_y = (bbox[1] + bbox[4]) / 2
    center_z = (bbox[2] + bbox[5]) / 2
    size_x = bbox[3] - bbox[0]
    size_y = bbox[4] - bbox[1]
    size_z = bbox[5] - bbox[2]
    max_xy = max(size_x, size_y)
    distance = max_xy * 1.5
    
    # Configure with PrusaSlicer colors - 4K resolution
    config = pygcode_viewer.ViewConfig()
    config.view_type = "FeatureType"
    config.visible_features["travels"] = False
    config.width = 3840
    config.height = 2160
    # Dark gray background (almost black) - default
    config.background_color = "#1A1A1A"
    viewer.set_config(config)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.dirname(output_dir)  # Go up to pygcode_viewer dir
    
    # =========================================================================
    # 1. Full model - Isometric view (front-right-top)
    # =========================================================================
    viewer.set_camera(
        pos=(center_x + distance * 0.8, center_y - distance * 0.8, center_z + distance * 0.6),
        target=(center_x, center_y, center_z),
        fov=45.0
    )
    viewer.set_layer_range(0, layer_count - 1)  # All layers
    
    output_path = os.path.join(output_dir, "prusa_full_isometric.png")
    print(f"\nRendering full model (isometric): {output_path}")
    viewer.render_to_file(output_path)
    
    # =========================================================================
    # 2. Top view - Camera directly above, looking down
    # =========================================================================
    # For top view, position camera above and use Y as up vector
    top_distance = max_xy * 1.2
    viewer.set_camera(
        pos=(center_x, center_y, bbox[5] + top_distance),
        target=(center_x, center_y, center_z),
        up=(0.0, 1.0, 0.0),  # Y-axis as up when looking down Z
        fov=45.0
    )
    
    output_path = os.path.join(output_dir, "prusa_full_top.png")
    print(f"Rendering full model (top view): {output_path}")
    viewer.render_to_file(output_path)
    
    # =========================================================================
    # 3. Front view
    # =========================================================================
    front_distance = max_xy * 1.5
    viewer.set_camera(
        pos=(center_x, bbox[1] - front_distance, center_z),
        target=(center_x, center_y, center_z),
        up=(0.0, 0.0, 1.0),  # Z-axis as up
        fov=45.0
    )
    
    output_path = os.path.join(output_dir, "prusa_full_front.png")
    print(f"Rendering full model (front view): {output_path}")
    viewer.render_to_file(output_path)
    
    # =========================================================================
    # 4. Side view
    # =========================================================================
    viewer.set_camera(
        pos=(bbox[3] + front_distance, center_y, center_z),
        target=(center_x, center_y, center_z),
        up=(0.0, 0.0, 1.0),  # Z-axis as up
        fov=45.0
    )
    
    output_path = os.path.join(output_dir, "prusa_full_side.png")
    print(f"Rendering full model (side view): {output_path}")
    viewer.render_to_file(output_path)
    
    # =========================================================================
    # 5. Layer cuts at various heights
    # =========================================================================
    # Reset camera to isometric
    viewer.set_camera(
        pos=(center_x + distance * 0.8, center_y - distance * 0.8, center_z + distance * 0.6),
        target=(center_x, center_y, center_z),
        fov=45.0
    )
    
    # Layer cuts at 25%, 50%, 75%, and 100%
    cut_percentages = [25, 50, 75, 100]
    
    for pct in cut_percentages:
        cut_layer = int((pct / 100.0) * (layer_count - 1))
        viewer.set_layer_range(0, cut_layer)
        
        output_path = os.path.join(output_dir, f"prusa_layer_{pct}pct.png")
        print(f"Rendering layer cut at {pct}% (layer {cut_layer}/{layer_count-1}): {output_path}")
        viewer.render_to_file(output_path)
    
    # =========================================================================
    # 6. Individual layers (first, middle, last)
    # =========================================================================
    layer_indices = [
        (1, "first"),
        (layer_count // 2, "middle"),
        (layer_count - 1, "last")
    ]
    
    for layer_idx, name in layer_indices:
        viewer.set_layer_range(layer_idx, layer_idx)
        
        output_path = os.path.join(output_dir, f"prusa_single_layer_{name}.png")
        print(f"Rendering single layer ({name}, layer {layer_idx}): {output_path}")
        viewer.render_to_file(output_path)
    
    # =========================================================================
    # 7. Speed visualization
    # =========================================================================
    viewer.set_layer_range(0, layer_count - 1)  # Reset to all layers
    viewer.set_view_type("Speed")
    
    output_path = os.path.join(output_dir, "prusa_speed_view.png")
    print(f"Rendering speed view: {output_path}")
    viewer.render_to_file(output_path)
    
    # =========================================================================
    # 8. Height visualization
    # =========================================================================
    viewer.set_view_type("Height")
    
    output_path = os.path.join(output_dir, "prusa_height_view.png")
    print(f"Rendering height view: {output_path}")
    viewer.render_to_file(output_path)
    
    print(f"\nDone! Generated {8 + len(cut_percentages) + len(layer_indices)} visualizations.")
    print(f"Output directory: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
