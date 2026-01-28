import cv2
import numpy as np
import json
from pathlib import Path

# Load camera intrinsics from calibration file
CALIB_PATH = Path(__file__).parent.parent / "camera_calib" / "camera_intrinsic.json"

with open(CALIB_PATH, "r") as f:
    calib_data = json.load(f)

K = np.array(calib_data["camera_matrix"], dtype=np.float32)
dist_coeffs = np.array(calib_data["distortion_coefficients"], dtype=np.float32)

# 1. Defined Printer Coordinates (e.g., Bed Corners in mm)
# Format: numpy array of floats, shape (N, 3)
center_x = 125
center_y = 105
x_size = 25
y_size = 25


object_points = np.array([
    [center_x - x_size/2, center_y + y_size/2, 0],       # top-left corner
    [center_x + x_size/2, center_y + y_size/2, 0],       # top-right corner
    [center_x + x_size/2, center_y - y_size/2, 0],       # bottom-right corner
    [center_x - x_size/2, center_y - y_size/2, 0]        # bottom-left corner
], dtype=np.float32)

# 2. Corresponding Pixel Coordinates (u, v) from your image
image_points = np.array([
    [1585, 658],      # Pixel for Front-Left
    [2316, 628],      # Pixel for Front-Right
    [1591, 1374],      # Pixel for Back-Right
    [2317, 1364]       # Pixel for Back-Left
], dtype=np.float32)

# 3. Solve PnP
# This finds the rotation (rvec) and translation (tvec) of the camera
success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs)

if success:
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Camera position in world coordinates
    # The camera center in world coords is: C = -R^T * t
    camera_position = -R.T @ tvec
    
    print("=" * 50)
    print("PnP Solution Found!")
    print("=" * 50)
    print("\nRotation Vector (rvec):")
    print(rvec.flatten())
    print("\nTranslation Vector (tvec):")
    print(tvec.flatten())
    print("\nCamera Position in Printer Coordinates (mm):")
    print(f"  X: {camera_position[0, 0]:.2f} mm")
    print(f"  Y: {camera_position[1, 0]:.2f} mm")
    print(f"  Z: {camera_position[2, 0]:.2f} mm")
    print("\nRotation Matrix:")
    print(R)
else:
    print("PnP failed to find a solution!")
