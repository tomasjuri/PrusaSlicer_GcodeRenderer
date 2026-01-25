#!/usr/bin/env python3
"""
Simple GCODE visualization with top-down camera view and layer cuts.
Uses camera intrinsic parameters for realistic projection.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygcode_viewer


def main():
    # Paths
    gcode_path = "/Users/tomasjurica/projects/PrusaSlicer/3dbenchy_0.4n_0.2mm_PETG_MK4IS_45m.gcode"
    intrinsics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "camera_intrinsic.json")
    
    if not os.path.exists(gcode_path):
        print(f"Error: GCODE file not found: {gcode_path}")
        return 1
    
    if not os.path.exists(intrinsics_path):
        print(f"Error: Camera intrinsics not found: {intrinsics_path}")
        return 1
    
    print(f"Loading: {gcode_path}")
    
    # Create viewer and load GCODE
    viewer = pygcode_viewer.GCodeViewer()
    viewer.load(gcode_path)
    
    layer_count = viewer.get_layer_count()
    print(f"Layers: {layer_count}")
    
    # Load camera intrinsics
    intrinsics = pygcode_viewer.CameraIntrinsics.from_json_file(intrinsics_path)
    print(f"Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")
    print(f"Image size: {intrinsics.image_width}x{intrinsics.image_height}")
    
    viewer.set_intrinsics(intrinsics)
    
    # Configure visualization (use intrinsics image size)
    config = pygcode_viewer.ViewConfig()
    config.view_type = "FeatureType"
    config.visible_features["travels"] = False
    config.width = intrinsics.image_width
    config.height = intrinsics.image_height
    config.background_color = "#1A1A1A"
    viewer.set_config(config)
    
    # Get model bounding box to center camera on the model
    bbox = viewer.get_bounding_box()
    model_center_x = (bbox[0] + bbox[3]) / 2
    model_center_y = (bbox[1] + bbox[4]) / 2
    model_center_z = (bbox[2] + bbox[5]) / 2
    model_height = bbox[5] - bbox[2]
    model_size = max(bbox[3] - bbox[0], bbox[4] - bbox[1])
    
    print(f"Model center: ({model_center_x:.1f}, {model_center_y:.1f}, {model_center_z:.1f})")
    print(f"Model size: {model_size:.1f}mm, height: {model_height:.1f}mm")
    
    # Bed config (Prusa MK4: 250x210mm)
    viewer.set_bed(size=(250, 210), show_outline=True, outline_color="#4D4D4D")
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Camera: slightly angled top-down view (avoids gimbal lock issues)
    # Position camera above and slightly offset, looking at model center
    camera_distance = model_size * 1.5
    camera_height = bbox[5] + camera_distance * 0.8  # Above model
    camera_offset = camera_distance * 0.3  # Slight offset for angle
    
    viewer.set_camera(
        pos=(model_center_x + camera_offset, model_center_y - camera_offset, camera_height),
        target=(model_center_x, model_center_y, model_center_z),
        up=(0.0, 0.0, 1.0),  # Z-axis as up
        fov=45.0
    )
    
    # Render layer cuts at 25%, 50%, 75%, 100%
    for pct in [25, 50, 75, 100]:
        cut_layer = int((pct / 100.0) * (layer_count - 1))
        viewer.set_layer_range(0, cut_layer)
        
        output_path = os.path.join(output_dir, f"layer_{pct}pct.png")
        print(f"Rendering {pct}% (layer {cut_layer}): {output_path}")
        viewer.render_to_file(output_path)
    
    print(f"\nDone! Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
