#!/usr/bin/env python3
"""
Intrinsic camera calibration script.
Calibrates individual cameras separately.

Edit the variables below and run the script.
"""

import os
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from utils import create_object_points

# ===== CONFIGURATION =====
# Set IMAGE_DIR to your capture folder (created by capture.py)
IMAGE_DIR = "/home/tomasjurica/projects/FingernailCalibration/Data/captures_baseline_95_105_20251215_131329_smallchessboard_calib"
CHESSBOARD_WIDTH = 9  # Number of inner corners horizontally
CHESSBOARD_HEIGHT = 6  # Number of inner corners vertically
SQUARE_SIZE = 12.79  # Size of one square in mm (127.9mm per 10 squares)
OUTPUT_FILE = "camera_intrinsic.json"  # Output JSON file path
VISUALIZE_CORNERS = False  # Show images with detected corners (set False for headless)
SAVE_VISUALIZATIONS = True  # Save visualization images to disk
VISUALIZATION_DIR = "corners_visualized_95_105_0251215_131329"  # Directory to save visualization images
# =========================


def find_chessboard_corners(image_path: str, chessboard_size: Tuple[int, int], 
                           square_size: float) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find chessboard corners in an image.
    
    Args:
        image_path: Path to the image
        chessboard_size: Tuple (width, height) - number of inner corners
        square_size: Size of one square in mm
    
    Returns:
        Tuple (success, object_points, image_points)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return False, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, 
        chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Create object points
        objp = create_object_points(chessboard_size, square_size)
        
        return True, objp, corners_refined
    
    return False, None, None


def calibrate_camera(image_paths: List[str], chessboard_size: Tuple[int, int], 
                    square_size: float, image_size: Optional[Tuple[int, int]] = None,
                    visualize: bool = False, save_viz: bool = False, viz_dir: str = "corners_visualized") -> Dict:
    """
    Perform intrinsic calibration for a single camera.
    
    Args:
        image_paths: List of paths to calibration images
        chessboard_size: Tuple (width, height) - number of inner corners
        square_size: Size of one square in mm
        image_size: Optional (width, height) of images. If None, will be detected from first image.
        visualize: If True, display images with detected corners
        save_viz: If True, save visualization images to disk
        viz_dir: Directory to save visualization images
    
    Returns:
        Dictionary containing calibration results
    """
    # Prepare object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    successful_images = []
    
    print(f"Processing {len(image_paths)} images...")
    
    # Detect image size from first image if not provided
    if image_size is None:
        first_img = cv2.imread(image_paths[0])
        if first_img is None:
            raise ValueError(f"Could not read first image: {image_paths[0]}")
        image_size = (first_img.shape[1], first_img.shape[0])
    
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] Processing {os.path.basename(img_path)}...", end=" ")
        
        img = cv2.imread(img_path)
        success, objp, imgp = find_chessboard_corners(img_path, chessboard_size, square_size)
        
        if success:
            objpoints.append(objp)
            imgpoints.append(imgp)
            successful_images.append(img_path)
            print("✓ Found corners")
            
            # Visualize corners
            if visualize or save_viz:
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, chessboard_size, imgp, success)
                
                if visualize:
                    cv2.imshow('Chessboard Corners', img_with_corners)
                    cv2.waitKey(500)  # Show for 500ms
                
                if save_viz:
                    vis_dir_path = Path(viz_dir)
                    vis_dir_path.mkdir(exist_ok=True)
                    vis_path = vis_dir_path / f"corners_{os.path.basename(img_path)}"
                    cv2.imwrite(str(vis_path), img_with_corners)
        else:
            print("✗ No corners found")
    
    if len(objpoints) < 3:
        raise ValueError(f"Need at least 3 successful images for calibration. Found {len(objpoints)}.")
    
    print(f"\nCalibrating camera with {len(objpoints)} successful images...")
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    print(f"\nCalibration complete!")
    print(f"Mean reprojection error: {mean_error:.3f} pixels")
    print(f"\nCamera matrix:")
    print(camera_matrix)
    print(f"\nDistortion coefficients:")
    print(dist_coeffs)
    
    return {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'image_size': image_size,
        'chessboard_size': chessboard_size,
        'square_size': square_size,
        'mean_reprojection_error': float(mean_error),
        'num_images_used': len(objpoints),
        'successful_images': successful_images
    }


if __name__ == '__main__':
    # Get all images from the folder
    image_paths = []
    for filename in sorted(os.listdir(IMAGE_DIR)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(IMAGE_DIR, filename))
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {IMAGE_DIR}")
        exit(1)
    
    print(f"Found {len(image_paths)} images in folder")
    
    # Perform calibration
    chessboard_size = (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT)
    results = calibrate_camera(image_paths, chessboard_size, SQUARE_SIZE,
                               visualize=VISUALIZE_CORNERS, 
                               save_viz=SAVE_VISUALIZATIONS,
                               viz_dir=VISUALIZATION_DIR)
    
    # Close visualization window if opened
    if VISUALIZE_CORNERS:
        cv2.destroyAllWindows()
    
    # Save results
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Also save in OpenCV format (YAML)
    yaml_path = output_path.with_suffix('.yaml')
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', np.array(results['camera_matrix']))
    fs.write('distortion_coefficients', np.array(results['distortion_coefficients']))
    fs.write('image_size', np.array(results['image_size'], dtype=np.int32))
    fs.release()
    
    print(f"OpenCV format saved to: {yaml_path}")

