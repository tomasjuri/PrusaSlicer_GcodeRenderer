import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of layers to process (ZN1 to ZN_N)
N_LAYERS = 10

# Path to the layer captures folder
FOLDER_PATH = Path("/Users/tomasjurica/projects/PrusaSlicer/data/layer_captures/2_Shape-Box_0.4n_0.2mm_PLA_MK4IS_20m.gcode_20260127_085607")

# Load camera intrinsics from calibration file
CALIB_PATH = Path(__file__).parent.parent / "camera_calib" / "camera_intrinsic.json"

with open(CALIB_PATH, "r") as f:
    calib_data = json.load(f)

K = np.array(calib_data["camera_matrix"], dtype=np.float32)
dist_coeffs = np.array(calib_data["distortion_coefficients"], dtype=np.float32)

# =============================================================================
# OBJECT POINTS (CONSTANT)
# =============================================================================
# Defined Printer Coordinates (e.g., Bed Corners in mm)
# Format: numpy array of floats, shape (N, 3)
center_x = 125
center_y = 105
x_size = 25
y_size = 25

OBJECT_POINTS = np.array([
    [center_x - x_size/2, center_y - y_size/2, 0],  # bottom-left corner
    [center_x + x_size/2, center_y - y_size/2, 0],  # bottom-right corner
    [center_x - x_size/2, center_y + y_size/2, 0],  # top-left corner
    [center_x + x_size/2, center_y + y_size/2, 0]   # top-right corner
], dtype=np.float32)

CORNER_NAMES = ["Bottom-Left", "Bottom-Right", "Top-Left", "Top-Right"]
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR


# =============================================================================
# IMAGE LOADING AND METADATA
# =============================================================================

def find_images_zn(folder_path: Path, n_layers: int) -> List[Dict]:
    """
    Find images from ZN1 to ZN_N in the given folder.
    Returns list of dicts with image_path, json_path, and layer_n.
    """
    images = []
    
    for layer_n in range(1, n_layers + 1):
        # Pattern: img_ZN{layer_n}_*.jpg
        pattern = f"img_ZN{layer_n}_*.jpg"
        matches = list(folder_path.glob(pattern))
        
        if matches:
            # Take the first match (should be only one per layer)
            img_path = matches[0]
            json_path = img_path.with_suffix('.json')
            
            images.append({
                'layer_n': layer_n,
                'image_path': img_path,
                'json_path': json_path if json_path.exists() else None
            })
        else:
            print(f"Warning: No image found for ZN{layer_n}")
    
    return images


def parse_metadata(json_path: Path) -> Optional[Dict]:
    """
    Load nozzle position from companion JSON file.
    Returns dict with 'x', 'y', 'z' or None if file doesn't exist.
    """
    if json_path is None or not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'capture_position' in data:
            return {
                'x': data['capture_position']['x'],
                'y': data['capture_position']['y'],
                'z': data['capture_position']['z'],
                'layer_n': data.get('layer_n'),
                'layer_height': data.get('layer_height')
            }
    except Exception as e:
        print(f"Error reading metadata: {e}")
    
    return None


def parse_nozzle_from_filename(filename: str) -> Optional[Dict]:
    """
    Fallback: parse nozzle position from filename.
    Pattern: img_ZN{layer}_Z{height}_X{x}_Y{y}_Z{z}.jpg
    """
    # Match pattern like X103_Y127_Z50
    match = re.search(r'X(\d+)_Y(\d+)_Z(\d+)', filename)
    if match:
        return {
            'x': float(match.group(1)),
            'y': float(match.group(2)),
            'z': float(match.group(3))
        }
    return None


# =============================================================================
# GUI FOR CORNER ANNOTATION
# =============================================================================

class CornerAnnotator:
    """Interactive GUI for annotating 4 corners on an image."""
    
    def __init__(self, image_path: Path, layer_n: int):
        self.image_path = image_path
        self.layer_n = layer_n
        self.original_image = cv2.imread(str(image_path))
        self.display_image = self.original_image.copy()
        self.points = []
        self.skipped = False
        self.quit_all = False
        self.window_name = f"Annotate ZN{layer_n} - Click corners: BL, BR, TL, TR"
        
        # Scale factor for display (images are 4608x2592)
        self.scale = 0.4
        self.display_h = int(self.original_image.shape[0] * self.scale)
        self.display_w = int(self.original_image.shape[1] * self.scale)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for corner annotation."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            # Convert display coordinates to original image coordinates
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.points.append((orig_x, orig_y))
            self.update_display()
    
    def update_display(self):
        """Redraw the display image with current annotations."""
        self.display_image = self.original_image.copy()
        
        # Draw existing points
        for i, (px, py) in enumerate(self.points):
            color = CORNER_COLORS[i]
            cv2.circle(self.display_image, (px, py), 15, color, -1)
            cv2.circle(self.display_image, (px, py), 17, (255, 255, 255), 2)
            cv2.putText(self.display_image, f"{i+1}:{CORNER_NAMES[i]}", 
                       (px + 20, py + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, color, 3)
        
        # Draw status text
        if len(self.points) < 4:
            next_corner = CORNER_NAMES[len(self.points)]
            status = f"Click {next_corner} corner ({len(self.points)+1}/4)"
        else:
            status = "All 4 corners marked. Press ENTER to confirm, R to reset"
        
        # Draw status bar at top
        cv2.rectangle(self.display_image, (0, 0), 
                     (self.display_image.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(self.display_image, status, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Draw controls hint
        controls = "R=Reset | S=Skip | Q=Quit | ENTER=Confirm"
        cv2.putText(self.display_image, controls, 
                   (self.display_image.shape[1] - 900, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    def run(self) -> Optional[np.ndarray]:
        """
        Run the annotation GUI.
        Returns: numpy array of 4 image points, or None if skipped/quit.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_w, self.display_h)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            # Resize for display
            display = cv2.resize(self.display_image, 
                               (self.display_w, self.display_h))
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            # R - Reset
            if key == ord('r') or key == ord('R'):
                self.points = []
                self.update_display()
            
            # S - Skip
            elif key == ord('s') or key == ord('S'):
                self.skipped = True
                cv2.destroyWindow(self.window_name)
                return None
            
            # Q or ESC - Quit all
            elif key == ord('q') or key == ord('Q') or key == 27:
                self.quit_all = True
                cv2.destroyWindow(self.window_name)
                return None
            
            # ENTER or SPACE - Confirm (only if 4 points)
            elif key == 13 or key == 32:  # Enter or Space
                if len(self.points) == 4:
                    cv2.destroyWindow(self.window_name)
                    return np.array(self.points, dtype=np.float32)
        
        cv2.destroyWindow(self.window_name)
        return None


def annotate_corners(image_path: Path, layer_n: int) -> tuple:
    """
    GUI to annotate 4 corners.
    Returns: (image_points or None, skipped, quit_all)
    """
    annotator = CornerAnnotator(image_path, layer_n)
    points = annotator.run()
    return points, annotator.skipped, annotator.quit_all


# =============================================================================
# PNP SOLVING
# =============================================================================

def solve_pnp_single(image_points: np.ndarray) -> Optional[Dict]:
    """
    Solve PnP for single image.
    Returns dict with camera_position, rvec, tvec, R or None if failed.
    """
    success, rvec, tvec = cv2.solvePnP(OBJECT_POINTS, image_points, K, dist_coeffs)
    
    if not success:
        return None
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Camera position in world coordinates: C = -R^T * t
    camera_position = -R.T @ tvec
    
    return {
        'camera_position': camera_position.flatten(),
        'rvec': rvec.flatten(),
        'tvec': tvec.flatten(),
        'R': R
    }


# =============================================================================
# RESULTS AGGREGATION AND PRINTING
# =============================================================================

def print_results(results: List[Dict]):
    """Print per-image and aggregated results."""
    
    if not results:
        print("\nNo images were annotated!")
        return
    
    print("\n" + "=" * 80)
    print("PER-IMAGE RESULTS")
    print("=" * 80)
    
    transformations = []
    camera_positions = []
    
    for r in results:
        layer_n = r['layer_n']
        cam = r['camera_position']
        nozzle = r['nozzle_position']
        
        camera_positions.append(cam)
        
        if nozzle:
            delta = np.array([
                cam[0] - nozzle['x'],
                cam[1] - nozzle['y'],
                cam[2] - nozzle['z']
            ])
            transformations.append(delta)
            
            print(f"\nImage ZN{layer_n}:")
            print(f"  Camera Position:  X={cam[0]:8.2f}  Y={cam[1]:8.2f}  Z={cam[2]:8.2f} mm")
            print(f"  Nozzle Position:  X={nozzle['x']:8.2f}  Y={nozzle['y']:8.2f}  Z={nozzle['z']:8.2f} mm")
            print(f"  Delta (Cam-Noz):  X={delta[0]:8.2f}  Y={delta[1]:8.2f}  Z={delta[2]:8.2f} mm")
        else:
            print(f"\nImage ZN{layer_n}:")
            print(f"  Camera Position:  X={cam[0]:8.2f}  Y={cam[1]:8.2f}  Z={cam[2]:8.2f} mm")
            print(f"  Nozzle Position:  (not available)")
    
    # Aggregated results
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    
    camera_positions = np.array(camera_positions)
    
    print(f"\nNumber of images processed: {len(results)}")
    
    # Mean camera position
    mean_cam = np.mean(camera_positions, axis=0)
    print(f"\nMean Camera Position:")
    print(f"  X={mean_cam[0]:8.2f}  Y={mean_cam[1]:8.2f}  Z={mean_cam[2]:8.2f} mm")
    
    # Std dev of camera positions
    std_cam = np.std(camera_positions, axis=0)
    print(f"\nCamera Position Std Dev:")
    print(f"  X={std_cam[0]:8.2f}  Y={std_cam[1]:8.2f}  Z={std_cam[2]:8.2f} mm")
    
    if transformations:
        transformations = np.array(transformations)
        
        # Mean transformation
        mean_trans = np.mean(transformations, axis=0)
        print(f"\nMean Transformation (Camera - Nozzle):")
        print(f"  X={mean_trans[0]:8.2f}  Y={mean_trans[1]:8.2f}  Z={mean_trans[2]:8.2f} mm")
        
        # Median transformation
        median_trans = np.median(transformations, axis=0)
        print(f"\nMedian Transformation (Camera - Nozzle):")
        print(f"  X={median_trans[0]:8.2f}  Y={median_trans[1]:8.2f}  Z={median_trans[2]:8.2f} mm")
        
        # Std dev of transformations
        std_trans = np.std(transformations, axis=0)
        print(f"\nTransformation Std Dev:")
        print(f"  X={std_trans[0]:8.2f}  Y={std_trans[1]:8.2f}  Z={std_trans[2]:8.2f} mm")
    
    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PnP Corner Annotation Tool")
    print("=" * 80)
    print(f"\nFolder: {FOLDER_PATH}")
    print(f"Processing layers: ZN1 to ZN{N_LAYERS}")
    print(f"\nObject points (constant 25x25mm box):")
    for i, name in enumerate(CORNER_NAMES):
        pt = OBJECT_POINTS[i]
        print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})")
    
    # Find images
    images = find_images_zn(FOLDER_PATH, N_LAYERS)
    
    if not images:
        print("\nNo images found!")
        return
    
    print(f"\nFound {len(images)} images to process.")
    print("\nControls:")
    print("  Left Click - Mark corner point")
    print("  R - Reset current image")
    print("  S - Skip current image")
    print("  Q/ESC - Quit")
    print("  ENTER/SPACE - Confirm and proceed")
    print("\nClick corners in order: Bottom-Left, Bottom-Right, Top-Left, Top-Right")
    print("-" * 80)
    
    results = []
    
    for img_info in images:
        layer_n = img_info['layer_n']
        image_path = img_info['image_path']
        json_path = img_info['json_path']
        
        print(f"\nProcessing ZN{layer_n}: {image_path.name}")
        
        # Get nozzle position from metadata or filename
        nozzle_pos = parse_metadata(json_path)
        if nozzle_pos is None:
            nozzle_pos = parse_nozzle_from_filename(image_path.name)
        
        # Annotate corners
        image_points, skipped, quit_all = annotate_corners(image_path, layer_n)
        
        if quit_all:
            print("\nQuitting...")
            break
        
        if skipped:
            print(f"  Skipped ZN{layer_n}")
            continue
        
        if image_points is None:
            print(f"  No points collected for ZN{layer_n}")
            continue
        
        print(f"  Annotated points:")
        for i, (px, py) in enumerate(image_points):
            print(f"    {CORNER_NAMES[i]}: ({px:.0f}, {py:.0f})")
        
        # Solve PnP
        pnp_result = solve_pnp_single(image_points)
        
        if pnp_result is None:
            print(f"  PnP failed for ZN{layer_n}")
            continue
        
        print(f"  PnP solved! Camera at: ({pnp_result['camera_position'][0]:.2f}, "
              f"{pnp_result['camera_position'][1]:.2f}, {pnp_result['camera_position'][2]:.2f})")
        
        results.append({
            'layer_n': layer_n,
            'image_path': str(image_path),
            'image_points': image_points,
            'camera_position': pnp_result['camera_position'],
            'nozzle_position': nozzle_pos,
            'rvec': pnp_result['rvec'],
            'tvec': pnp_result['tvec']
        })
    
    # Print aggregated results
    print_results(results)
    
    # Save results to JSON
    if results:
        output_path = FOLDER_PATH / "pnp_results.json"
        output_data = []
        for r in results:
            output_data.append({
                'layer_n': r['layer_n'],
                'image_path': r['image_path'],
                'image_points': r['image_points'].tolist(),
                'camera_position': r['camera_position'].tolist(),
                'nozzle_position': r['nozzle_position'],
                'rvec': r['rvec'].tolist(),
                'tvec': r['tvec'].tolist()
            })
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
