#!/usr/bin/env python3
"""
Visualize G-code cut at Z=5.4mm
"""

import sys
import os
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygcode_viewer


def apply_homography(x, y, H):
    """Apply homography transformation to a 2D point."""
    pt = np.array([x, y, 1.0])
    result = H @ pt
    return result[0] / result[2], result[1] / result[2]


def warp_image(image_path, H, output_path, target_size=None):
    """Warp an image using a homography matrix."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    h, w = img.shape[:2]
    if target_size is None:
        target_size = (w, h)
    
    # Warp the image
    warped = cv2.warpPerspective(img, H, target_size)
    cv2.imwrite(output_path, warped)
    return warped


def compute_camera_offset_from_homography(H, K, z_height):
    """
    Estimate camera offset from homography using camera intrinsics.
    
    For a top-down camera, the homography between two views of a plane
    can be decomposed to find the translation.
    
    Simplified approach: use the translation component of H
    and convert from pixels to mm using focal length and height.
    """
    # Extract translation from homography (last column, normalized)
    tx_pixels = H[0, 2]
    ty_pixels = H[1, 2]
    
    # Convert pixels to mm using: mm = pixels * z_height / focal_length
    fx = K[0, 0]
    fy = K[1, 1]
    
    # Camera offset in mm
    offset_x = tx_pixels * z_height / fx
    offset_y = ty_pixels * z_height / fy
    
    return offset_x, offset_y


def main():
    gcode_path = "/Users/tomasjurica/projects/PrusaSlicer/1_Shape-Box_0.4n_0.2mm_PLA_MK4IS_20m.gcode"
    
    if not os.path.exists(gcode_path):
        print(f"Error: GCODE file not found: {gcode_path}")
        return 1
    
    print(f"Loading: {gcode_path}")
    
    # Create viewer and load GCODE
    viewer = pygcode_viewer.GCodeViewer()
    viewer.load(gcode_path)
    
    layer_count = viewer.get_layer_count()
    print(f"Total layers: {layer_count}")
    
    # Get model bounding box
    bbox = viewer.get_bounding_box()
    model_center_x = (bbox[0] + bbox[3]) / 2
    model_center_y = (bbox[1] + bbox[4]) / 2
    model_center_z = (bbox[2] + bbox[5]) / 2
    model_height = bbox[5] - bbox[2]
    model_size = max(bbox[3] - bbox[0], bbox[4] - bbox[1])
    
    print(f"Bounding box: min=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}) max=({bbox[3]:.1f}, {bbox[4]:.1f}, {bbox[5]:.1f})")
    print(f"Model center: ({model_center_x:.1f}, {model_center_y:.1f}, {model_center_z:.1f})")
    print(f"Model size: {model_size:.1f}mm, height: {model_height:.1f}mm")
    
    # Load camera intrinsics
    intrinsics_path = "/Users/tomasjurica/projects/PrusaSlicer/pygcode_viewer/camera_intrinsic.json"
    with open(intrinsics_path) as f:
        intrinsics_data = json.load(f)
    
    # Target image resolution (must match for homography to work correctly)
    target_width = 1280
    target_height = 720
    
    # Scale intrinsics from calibration resolution (1920x1080) to target (1280x720)
    calib_width, calib_height = intrinsics_data["image_size"]
    scale_x = target_width / calib_width
    scale_y = target_height / calib_height
    
    cam_matrix = intrinsics_data["camera_matrix"]
    fx = cam_matrix[0][0] * scale_x
    fy = cam_matrix[1][1] * scale_y
    cx = cam_matrix[0][2] * scale_x
    cy = cam_matrix[1][2] * scale_y
    
    print(f"Camera intrinsics (scaled to {target_width}x{target_height}):")
    print(f"  fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Set camera intrinsics
    intrinsics = pygcode_viewer.CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy,
        image_width=target_width, image_height=target_height
    )
    viewer.set_intrinsics(intrinsics)
    
    # Configure visualization
    config = pygcode_viewer.ViewConfig()
    config.view_type = "FeatureType"
    config.visible_features["travels"] = False
    config.width = target_width
    config.height = target_height
    config.background_color = "#1A1A1A"
    viewer.set_config(config)
    
    # Bed config (Prusa MK4: 250x210mm)
    viewer.set_bed(size=(250, 210), show_outline=True, outline_color="#4D4D4D")
    
    # Z=5.4mm corresponds to layer 26 (0-indexed)
    # Layer heights: 0.2, 0.4, 0.6, ... = (Z - 0.0) / 0.2 - 1 = 27-1 = 26
    # But we need to find the exact layer, so let's search for it
    target_z = 5.4
    # With 0.2mm layer height starting at Z=0.2: layer = (Z / 0.2) - 1
    # Z=5.4 -> 5.4/0.2 = 27 -> layer 26 (0-indexed)
    cut_layer = 26
    
    print(f"\nCutting at Z={target_z}mm (layer {cut_layer})")
    viewer.set_layer_range(0, cut_layer)
    
    # Output directory
    output_dir = "/Users/tomasjurica/projects/PrusaSlicer/pygcode_viewer/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Nozzle position from filename img_ZN26_Z5.4_X95_Y122_Z58.jpg
    nozzle_x = 95
    nozzle_y = 122
    nozzle_z = 58
    
    # Load homography data
    homography_path = "/Users/tomasjurica/projects/PrusaSlicer/pygcode_viewer/tests/H_cam.json"
    with open(homography_path) as f:
        H_data = json.load(f)
    
    # Get all matrices
    H = np.array(H_data["geom_info"]["Homography"])
    H1 = np.array(H_data["geom_info"]["H1"])
    H2 = np.array(H_data["geom_info"]["H2"])
    
    # ========== RENDER: Base image without transform ==========
    print(f"\n--- Rendering base image ---")
    print(f"Camera pos: ({nozzle_x}, {nozzle_y}, {nozzle_z})")
    
    viewer.set_camera(
        pos=(nozzle_x, nozzle_y, nozzle_z),
        target=(nozzle_x, nozzle_y + 0.001, 0.0),
        up=(0.0, -1.0, 0.0),
        fov=45.0
    )
    
    base_path = os.path.join(output_dir, f"gcode_Z{target_z:.1f}mm_base.png")
    print(f"Rendering to: {base_path}")
    viewer.render_to_file(base_path)
    
    # ========== COMPUTE: Camera offset from homography ==========
    print(f"\n--- Computing camera offset from homography ---")
    
    # Build camera matrix K (scaled to target resolution)
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    offset_x, offset_y = compute_camera_offset_from_homography(H, K, nozzle_z)
    print(f"Homography translation (pixels): tx={H[0,2]:.1f}, ty={H[1,2]:.1f}")
    print(f"Estimated camera offset (mm): dx={offset_x:.1f}, dy={offset_y:.1f}")
    
    # ========== WARP: Apply homography to rendered image ==========
    print(f"\n--- Warping rendered image with homography ---")
    
    warped_path = os.path.join(output_dir, f"gcode_Z{target_z:.1f}mm_warped_H.png")
    print(f"Warping with H to: {warped_path}")
    warp_image(base_path, H, warped_path, (target_width, target_height))
    
    warped_inv_path = os.path.join(output_dir, f"gcode_Z{target_z:.1f}mm_warped_H_inv.png")
    print(f"Warping with H_inv to: {warped_inv_path}")
    warp_image(base_path, np.linalg.inv(H), warped_inv_path, (target_width, target_height))
    
    # ========== RENDER: With estimated camera offset ==========
    print(f"\n--- Rendering with estimated camera offset ---")
    cam_x = nozzle_x + offset_x
    cam_y = nozzle_y + offset_y
    print(f"Camera pos with offset: ({cam_x:.1f}, {cam_y:.1f}, {nozzle_z})")
    
    viewer.set_camera(
        pos=(cam_x, cam_y, nozzle_z),
        target=(cam_x, cam_y + 0.001, 0.0),
        up=(0.0, -1.0, 0.0),
        fov=45.0
    )
    
    offset_path = os.path.join(output_dir, f"gcode_Z{target_z:.1f}mm_with_offset.png")
    print(f"Rendering to: {offset_path}")
    viewer.render_to_file(offset_path)
    
    print(f"\n========== Results ==========")
    print(f"  Base render:     {base_path}")
    print(f"  Warped (H):      {warped_path}")
    print(f"  Warped (H_inv):  {warped_inv_path}")
    print(f"  With offset:     {offset_path}")
    print(f"\n  Estimated camera offset from nozzle: dx={offset_x:.1f}mm, dy={offset_y:.1f}mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
