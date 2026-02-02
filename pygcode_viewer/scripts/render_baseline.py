#!/usr/bin/env python3
"""
Phase 1: Render baseline G-code visualization from nozzle position.

This script:
1. Loads a capture image and parses nozzle position from filename
2. Renders G-code from the nozzle position (camera pointing straight down)
3. Undistorts the real camera image using calibration data
4. Saves both outputs for manual homography point matching

Usage:
    python render_baseline.py <capture_folder> <image_filename> [--output-dir OUTPUT]

Example:
    python render_baseline.py \
        /path/to/2_Shape-Box_0.4n_0.2mm_PLA_MK4IS_20m.gcode_20260127_085607 \
        img_ZN18_Z3.8_X92_Y125_Z71.jpg
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for pygcode_viewer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygcode_viewer


def parse_capture_filename(filename: str) -> dict:
    """
    Parse capture filename to extract position information.
    
    Format: img_ZN{layer}_Z{z_height}_X{nozzle_x}_Y{nozzle_y}_Z{nozzle_z}.jpg
    
    Returns:
        dict with layer_n, z_height, nozzle_x, nozzle_y, nozzle_z
    """
    pattern = r'img_ZN(\d+)_Z([\d.]+)_X(\d+)_Y(\d+)_Z(\d+)\.jpg'
    match = re.match(pattern, filename)
    if match:
        return {
            'layer_n': int(match.group(1)),
            'z_height': float(match.group(2)),
            'nozzle_x': int(match.group(3)),
            'nozzle_y': int(match.group(4)),
            'nozzle_z': int(match.group(5))
        }
    return None


def load_camera_intrinsics(intrinsics_path: str) -> tuple:
    """
    Load camera intrinsics from JSON file.
    
    Returns:
        (K, dist_coeffs, image_size) - camera matrix, distortion coefficients, (w, h)
    """
    with open(intrinsics_path, 'r') as f:
        data = json.load(f)
    
    K = np.array(data['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(data['distortion_coefficients'], dtype=np.float64).flatten()
    image_size = tuple(data['image_size'])  # (width, height)
    
    return K, dist_coeffs, image_size


def undistort_image(image_path: str, K: np.ndarray, dist_coeffs: np.ndarray) -> tuple:
    """
    Undistort an image using camera calibration.
    
    Returns:
        (undistorted_image, new_K) - undistorted image and new camera matrix
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Get optimal new camera matrix (alpha=1 keeps all pixels)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort
    undistorted = cv2.undistort(img, K, dist_coeffs, None, new_K)
    
    return undistorted, new_K, roi


def find_gcode_file(capture_folder: str) -> str:
    """
    Find the corresponding G-code file for a capture folder.
    
    The G-code is typically in the parent folder with a matching name.
    """
    capture_path = Path(capture_folder)
    parent = capture_path.parent
    
    # Extract gcode name from folder name (e.g., "2_Shape-Box_0.4n_0.2mm_PLA_MK4IS_20m.gcode_20260127_085607")
    folder_name = capture_path.name
    
    # Try to find gcode in parent directory
    for gcode_file in parent.glob("*.gcode"):
        # Check if folder name starts with gcode filename
        if folder_name.startswith(gcode_file.stem):
            return str(gcode_file)
    
    # Also check in the capture folder itself
    for gcode_file in capture_path.glob("*.gcode"):
        return str(gcode_file)
    
    # Try exact name matching (folder name before timestamp)
    parts = folder_name.rsplit('_', 2)  # Split off timestamp (YYYYMMDD_HHMMSS)
    if len(parts) >= 2:
        gcode_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
        possible_path = parent / gcode_name
        if possible_path.exists():
            return str(possible_path)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Render baseline G-code from nozzle position and undistort camera image'
    )
    parser.add_argument('capture_folder', help='Path to the capture folder containing images')
    parser.add_argument('image_filename', help='Image filename (e.g., img_ZN18_Z3.8_X92_Y125_Z71.jpg)')
    parser.add_argument('--output-dir', '-o', 
                        default=None,
                        help='Output directory (default: pygcode_viewer/outputs/baseline)')
    parser.add_argument('--intrinsics', '-i',
                        default=None,
                        help='Path to camera intrinsics JSON (default: camera_calib/camera_intrinsic.json)')
    parser.add_argument('--gcode', '-g',
                        default=None,
                        help='Path to G-code file (auto-detected if not specified)')
    parser.add_argument('--layer-mode', '-l',
                        choices=['cumulative', 'single'],
                        default='cumulative',
                        help='Layer rendering mode: cumulative (all up to layer) or single (only current layer)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent
    capture_folder = Path(args.capture_folder)
    image_path = capture_folder / args.image_filename
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1
    
    # Parse filename
    capture_info = parse_capture_filename(args.image_filename)
    if not capture_info:
        print(f"Error: Could not parse filename: {args.image_filename}")
        print("Expected format: img_ZN{layer}_Z{z}_X{x}_Y{y}_Z{z}.jpg")
        return 1
    
    print(f"Parsed capture info:")
    print(f"  Layer: {capture_info['layer_n']} (Z={capture_info['z_height']}mm)")
    print(f"  Nozzle position: X={capture_info['nozzle_x']}, Y={capture_info['nozzle_y']}, Z={capture_info['nozzle_z']}mm")
    
    # Find intrinsics
    if args.intrinsics:
        intrinsics_path = Path(args.intrinsics)
    else:
        intrinsics_path = script_dir / 'camera_calib' / 'camera_intrinsic.json'
    
    if not intrinsics_path.exists():
        print(f"Error: Intrinsics file not found: {intrinsics_path}")
        return 1
    
    K, dist_coeffs, image_size = load_camera_intrinsics(str(intrinsics_path))
    print(f"\nCamera intrinsics loaded from: {intrinsics_path}")
    print(f"  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"  cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    print(f"  Image size: {image_size[0]}x{image_size[1]}")
    
    # Calculate expected FOV and visible area for diagnostics
    nozzle_z = capture_info['nozzle_z']
    z_height = capture_info['z_height']
    camera_to_print_dist = nozzle_z - z_height  # Distance from camera to print surface
    
    # Calculate horizontal FOV from intrinsics
    half_width = image_size[0] / 2.0
    half_fov_h_rad = np.arctan(half_width / K[0, 0])
    fov_h_deg = 2 * np.degrees(half_fov_h_rad)
    
    # Calculate visible area at print surface
    visible_width = 2 * camera_to_print_dist * np.tan(half_fov_h_rad)
    visible_height = 2 * camera_to_print_dist * np.tan(np.arctan((image_size[1] / 2.0) / K[1, 1]))
    
    # Pixels per mm at print surface
    px_per_mm = K[0, 0] / camera_to_print_dist
    
    print(f"\n  === Scale Diagnostics ===")
    print(f"  Camera-to-print distance: {camera_to_print_dist:.1f}mm (nozzle_z={nozzle_z} - z_height={z_height})")
    print(f"  Horizontal FOV: {fov_h_deg:.1f}°")
    print(f"  Visible area at print surface: {visible_width:.1f}mm x {visible_height:.1f}mm")
    print(f"  Scale at print surface: {px_per_mm:.1f} pixels/mm")
    
    # Find G-code
    if args.gcode:
        gcode_path = Path(args.gcode)
    else:
        gcode_path = find_gcode_file(str(capture_folder))
        if gcode_path:
            gcode_path = Path(gcode_path)
    
    if not gcode_path or not gcode_path.exists():
        print(f"Error: G-code file not found. Searched in: {capture_folder.parent}")
        print("Use --gcode to specify path explicitly.")
        return 1
    
    print(f"\nUsing G-code: {gcode_path}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / 'outputs' / 'baseline'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filenames
    layer_n = capture_info['layer_n']
    z_height = capture_info['z_height']
    basename = f"ZN{layer_n}_Z{z_height}"
    
    render_output = output_dir / f"render_{basename}.png"
    undistorted_output = output_dir / f"undistorted_{basename}.jpg"
    info_output = output_dir / f"info_{basename}.json"
    
    # Step 1: Undistort the real image
    print(f"\nUndistorting image...")
    undistorted_img, new_K, roi = undistort_image(str(image_path), K, dist_coeffs)
    cv2.imwrite(str(undistorted_output), undistorted_img)
    print(f"  Saved: {undistorted_output}")
    
    # Step 2: Render G-code from nozzle position
    print(f"\nRendering G-code from nozzle position...")
    
    viewer = pygcode_viewer.GCodeViewer()
    viewer.load(str(gcode_path))
    
    layer_count = viewer.get_layer_count()
    print(f"  Total layers in G-code: {layer_count}")
    
    # Set camera intrinsics for full resolution
    intrinsics = pygcode_viewer.CameraIntrinsics(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        image_width=image_size[0],
        image_height=image_size[1]
    )
    viewer.set_intrinsics(intrinsics)
    
    # Configure visualization
    config = pygcode_viewer.ViewConfig()
    config.view_type = "FeatureType"
    config.visible_features["travels"] = False
    config.width = image_size[0]
    config.height = image_size[1]
    config.background_color = "#1A1A1A"
    viewer.set_config(config)
    
    # Set bed (Prusa MK4: 250x210mm)
    viewer.set_bed(size=(250, 210), show_outline=True, outline_color="#4D4D4D")
    
    # Set layer range
    if args.layer_mode == 'cumulative':
        viewer.set_layer_range(0, layer_n)
        print(f"  Rendering layers 0-{layer_n} (cumulative)")
    else:
        viewer.set_layer_range(layer_n, layer_n)
        print(f"  Rendering layer {layer_n} only")
    
    # Set camera position at nozzle, looking straight down
    nozzle_x = capture_info['nozzle_x']
    nozzle_y = capture_info['nozzle_y']
    nozzle_z = capture_info['nozzle_z']
    
    viewer.set_camera(
        pos=(nozzle_x, nozzle_y, nozzle_z),
        target=(nozzle_x, nozzle_y, 0.0),  # Looking at bed (Z=0)
        up=(0.0, -1.0, 0.0),  # Y-axis points down in image
        fov=45.0  # Ignored when intrinsics are set
    )
    
    # Render
    viewer.render_to_file(str(render_output))
    print(f"  Saved: {render_output}")
    
    # Save metadata
    info = {
        'image_filename': args.image_filename,
        'gcode_path': str(gcode_path),
        'capture_info': capture_info,
        'camera_intrinsics': {
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2]),
            'image_size': list(image_size)
        },
        'distortion_coefficients': dist_coeffs.tolist(),
        'render_position': {
            'x': nozzle_x,
            'y': nozzle_y,
            'z': nozzle_z
        },
        'layer_mode': args.layer_mode,
        'outputs': {
            'render': str(render_output),
            'undistorted': str(undistorted_output)
        }
    }
    
    with open(info_output, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: {info_output}")
    
    print(f"\n" + "="*60)
    print("Phase 1 Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Open these two images side by side:")
    print(f"     - Render:     {render_output}")
    print(f"     - Undistorted: {undistorted_output}")
    print(f"  2. Manually identify 4+ corresponding points")
    print(f"  3. Run compute_transform.py with the point correspondences")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
