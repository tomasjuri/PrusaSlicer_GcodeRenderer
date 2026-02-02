import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of random images to process
N_FILES = 4

# Optional random seed for deterministic sampling (set to None for random)
RANDOM_SEED = 42

# Path to the layer captures folder
FOLDER_PATH = Path("/Users/tomasjurica/projects/PrusaSlicer/data/layer_captures/2_Shape-Box_0.4n_0.2mm_PLA_MK4IS_20m.gcode_20260127_085607")

# Load camera intrinsics from calibration file
CALIB_PATH = Path(__file__).parent.parent / "camera_calib" / "camera_intrinsic.json"

with open(CALIB_PATH, "r") as f:
    calib_data = json.load(f)

K = np.array(calib_data["camera_matrix"], dtype=np.float32)
dist_coeffs = np.array(calib_data["distortion_coefficients"], dtype=np.float32)

# =============================================================================
# OBJECT POINTS (CONSTANT XY, Z PER-IMAGE)
# =============================================================================
# Printer coordinate system:
#   X+ = right (looking at printer from front)
#   Y+ = back (away from you, towards back of printer)
#   Z+ = up
#
# Click order: 1,2,3,4 starting at front-left of PRINTER, going clockwise

center_x = 125  # Print center X (mm)
center_y = 105  # Print center Y (mm)
x_size = 25     # Print width X (mm)
y_size = 25     # Print depth Y (mm)

def build_object_points(z_height: float) -> np.ndarray:
    """Create object points with constant XY and per-image Z height."""
    return np.array([
        [center_x - x_size / 2, center_y - y_size / 2, z_height],  # 1: Front-Left of printer
        [center_x + x_size / 2, center_y - y_size / 2, z_height],  # 2: Front-Right of printer
        [center_x - x_size / 2, center_y + y_size / 2, z_height],  # 3: Back-Left of printer
        [center_x + x_size / 2, center_y + y_size / 2, z_height]   # 4: Back-Right of printer
    ], dtype=np.float32)

CORNER_NAMES = [
    "1: PRINTER Front-Left",
    "2: PRINTER Front-Right",
    "3: PRINTER Back-Left",
    "4: PRINTER Back-Right"
]
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR: Red, Green, Blue, Yellow


# =============================================================================
# IMAGE LOADING AND METADATA
# =============================================================================

def parse_layer_info_from_filename(filename: str) -> Optional[Dict]:
    """
    Parse layer number and printed layer height from filename.
    Example: img_ZN25_Z5.2_X89_Y141_Z74.jpg -> ZN=25, layer_height=5.2
    """
    match = re.search(r'ZN(\d+)_Z([0-9]+(?:\.[0-9]+)?)', filename)
    if not match:
        return None
    return {
        'layer_n': int(match.group(1)),
        'layer_height': float(match.group(2))
    }


def find_images_random(folder_path: Path, n_files: int, seed: Optional[int]) -> List[Dict]:
    """
    Randomly select N images from all available ZN images in the folder.
    Returns list of dicts with image_path, json_path, layer_n, layer_height.
    """
    matches = sorted(folder_path.glob("img_ZN*.jpg"))
    images = []

    for img_path in matches:
        info = parse_layer_info_from_filename(img_path.name)
        if info is None:
            print(f"Warning: Could not parse layer info from {img_path.name}")
            continue
        json_path = img_path.with_suffix('.json')
        images.append({
            'layer_n': info['layer_n'],
            'layer_height': info['layer_height'],
            'image_path': img_path,
            'json_path': json_path if json_path.exists() else None
        })

    if not images:
        return []

    if n_files >= len(images):
        print(f"Warning: Requested {n_files} images, only {len(images)} available.")
        return images

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=n_files, replace=False)
    return [images[i] for i in sorted(indices)]


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
    # Match pattern like X103_Y127_Z50 (floats allowed)
    match = re.search(r'X([0-9]+(?:\.[0-9]+)?)_Y([0-9]+(?:\.[0-9]+)?)_Z([0-9]+(?:\.[0-9]+)?)', filename)
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
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.points = []
        self.skipped = False
        self.quit_all = False
        self.window_name = f"Annotate ZN{layer_n} - Click corners: BL, BR, TL, TR"
        self.zoom_window_name = f"Zoom ZN{layer_n}"
        
        # Scale factor for display (images are 4608x2592)
        self.scale = 0.4
        self.display_h = int(self.original_image.shape[0] * self.scale)
        self.display_w = int(self.original_image.shape[1] * self.scale)

        # Zoom settings (auto-zoom around cursor)
        self.zoom_enabled = True
        self.zoom_scale = 4.0
        self.zoom_window_size = 600
        self.cursor_pos = (
            int(self.original_image.shape[1] / 2),
            int(self.original_image.shape[0] / 2)
        )
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for corner annotation."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_pos = (int(x / self.scale), int(y / self.scale))
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            # Convert display coordinates to original image coordinates
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.cursor_pos = (orig_x, orig_y)
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

        # Draw rectangle when all 4 points are set (BL, BR, TL, TR)
        if len(self.points) == 4:
            rect_order = [0, 1, 3, 2]
            rect_pts = np.array([self.points[i] for i in rect_order], dtype=np.int32)
            cv2.polylines(self.display_image, [rect_pts], True, (255, 255, 255), 3)
        
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
        controls = "R=Reset | S=Skip | Q=Quit | ENTER=Confirm | +/-=Zoom"
        cv2.putText(self.display_image, controls, 
                   (self.display_image.shape[1] - 900, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    def get_zoom_view(self) -> np.ndarray:
        """Build a zoomed view around the current cursor position."""
        cx, cy = self.cursor_pos
        half_size = max(int(self.zoom_window_size / (2 * self.zoom_scale)), 20)
        x0 = max(cx - half_size, 0)
        x1 = min(cx + half_size, self.original_image.shape[1])
        y0 = max(cy - half_size, 0)
        y1 = min(cy + half_size, self.original_image.shape[0])
        crop = self.original_image[y0:y1, x0:x1]
        if crop.size == 0:
            crop = self.original_image
        zoom_view = cv2.resize(crop, (self.zoom_window_size, self.zoom_window_size))
        # Crosshair at center
        mid = int(self.zoom_window_size / 2)
        cv2.line(zoom_view, (mid - 20, mid), (mid + 20, mid), (0, 255, 255), 2)
        cv2.line(zoom_view, (mid, mid - 20), (mid, mid + 20), (0, 255, 255), 2)
        return zoom_view
    
    def run(self) -> Optional[np.ndarray]:
        """
        Run the annotation GUI.
        Returns: numpy array of 4 image points, or None if skipped/quit.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_w, self.display_h)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        if self.zoom_enabled:
            cv2.namedWindow(self.zoom_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.zoom_window_name, self.zoom_window_size, self.zoom_window_size)
        
        self.update_display()
        
        while True:
            # Resize for display
            display = cv2.resize(self.display_image, 
                               (self.display_w, self.display_h))
            cv2.imshow(self.window_name, display)

            if self.zoom_enabled:
                zoom_view = self.get_zoom_view()
                cv2.imshow(self.zoom_window_name, zoom_view)
            
            key = cv2.waitKey(1) & 0xFF
            
            # R - Reset
            if key == ord('r') or key == ord('R'):
                self.points = []
                self.update_display()
            
            # S - Skip
            elif key == ord('s') or key == ord('S'):
                self.skipped = True
                cv2.destroyWindow(self.window_name)
                if self.zoom_enabled:
                    cv2.destroyWindow(self.zoom_window_name)
                return None
            
            # Q or ESC - Quit all
            elif key == ord('q') or key == ord('Q') or key == 27:
                self.quit_all = True
                cv2.destroyWindow(self.window_name)
                if self.zoom_enabled:
                    cv2.destroyWindow(self.zoom_window_name)
                return None

            # Zoom in/out
            elif key in (ord('+'), ord('=')):
                self.zoom_scale = min(self.zoom_scale + 1.0, 10.0)
            elif key in (ord('-'), ord('_')):
                self.zoom_scale = max(self.zoom_scale - 1.0, 1.0)
            
            # ENTER or SPACE - Confirm (only if 4 points)
            elif key == 13 or key == 32:  # Enter or Space
                if len(self.points) == 4:
                    cv2.destroyWindow(self.window_name)
                    if self.zoom_enabled:
                        cv2.destroyWindow(self.zoom_window_name)
                    return np.array(self.points, dtype=np.float32)
        
        cv2.destroyWindow(self.window_name)
        if self.zoom_enabled:
            cv2.destroyWindow(self.zoom_window_name)
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

def refine_image_points(image_points: np.ndarray, gray_image: np.ndarray) -> np.ndarray:
    """Refine 2D points to subpixel accuracy."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    corners = image_points.reshape(-1, 1, 2).astype(np.float32)
    refined = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2)


def refine_pnp(object_points: np.ndarray, image_points: np.ndarray,
               rvec: np.ndarray, tvec: np.ndarray) -> tuple:
    """Refine PnP result using Levenberg-Marquardt if available."""
    try:
        rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
            object_points, image_points, K, dist_coeffs, rvec, tvec
        )
        return rvec_ref, tvec_ref
    except Exception as e:
        print(f"  Warning: PnP refine failed ({e}); using initial solution.")
        return rvec, tvec


def compute_reprojection_error(object_points: np.ndarray, image_points: np.ndarray,
                               rvec: np.ndarray, tvec: np.ndarray) -> float:
    """Compute RMS reprojection error in pixels."""
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    projected = projected.reshape(-1, 2)
    diff = projected - image_points
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def check_homography(object_points: np.ndarray, image_points: np.ndarray) -> float:
    """
    Check if points form a valid homography (sanity check for coplanar points).
    Returns reprojection error using homography.
    """
    # Use only XY of object points (they're coplanar)
    obj_2d = object_points[:, :2].astype(np.float32)
    img_2d = image_points.astype(np.float32)
    
    H, mask = cv2.findHomography(obj_2d, img_2d, cv2.RANSAC, 5.0)
    if H is None:
        return float('inf')
    
    # Reproject using homography
    obj_h = np.hstack([obj_2d, np.ones((len(obj_2d), 1))]).T  # 3xN
    proj_h = H @ obj_h  # 3xN
    proj_2d = (proj_h[:2] / proj_h[2]).T  # Nx2
    
    diff = proj_2d - img_2d
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def visualize_reprojection(image_path: Path, image_points: np.ndarray, 
                           object_points: np.ndarray, rvec: np.ndarray, 
                           tvec: np.ndarray, layer_n: int) -> None:
    """Show clicked points vs reprojected points for debugging."""
    img = cv2.imread(str(image_path))
    
    # Reproject object points to image
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    projected = projected.reshape(-1, 2)
    
    # Draw clicked points (green circles)
    for i, (px, py) in enumerate(image_points):
        cv2.circle(img, (int(px), int(py)), 20, (0, 255, 0), 3)
        cv2.putText(img, f"Click{i+1}", (int(px)+25, int(py)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    # Draw reprojected points (red circles)
    for i, (px, py) in enumerate(projected):
        cv2.circle(img, (int(px), int(py)), 20, (0, 0, 255), 3)
        cv2.putText(img, f"Proj{i+1}", (int(px)+25, int(py)+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    # Draw lines between clicked and projected
    for (cx, cy), (px, py) in zip(image_points, projected):
        cv2.line(img, (int(cx), int(cy)), (int(px), int(py)), (255, 255, 0), 2)
    
    # Show
    window_name = f"Reprojection Check ZN{layer_n} (Green=Clicked, Red=Projected)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)
    cv2.imshow(window_name, img)
    print("  Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def solve_pnp_single(image_points: np.ndarray, object_points: np.ndarray) -> Optional[Dict]:
    """
    Solve PnP for single image using method optimized for coplanar points.
    Returns dict with camera_position, rvec, tvec, R or None if failed.
    """
    # For coplanar points, use IPPE (Infinitesimal Plane-based Pose Estimation)
    # This handles the planar degeneracy case much better than the default method
    method_used = "IPPE"
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        # Fallback to iterative method
        method_used = "ITERATIVE"
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    
    print(f"    PnP method: {method_used}")
    
    if not success:
        return None
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Camera position in world coordinates: C = -R^T * t
    camera_position = -R.T @ tvec

    # Refine pose using Levenberg-Marquardt
    rvec_ref, tvec_ref = refine_pnp(object_points, image_points, rvec, tvec)
    R_ref, _ = cv2.Rodrigues(rvec_ref)
    camera_position_ref = -R_ref.T @ tvec_ref
    reproj_error = compute_reprojection_error(object_points, image_points, rvec_ref, tvec_ref)
    
    return {
        'camera_position': camera_position.flatten(),
        'rvec': rvec.flatten(),
        'tvec': tvec.flatten(),
        'R': R,
        'camera_position_refined': camera_position_ref.flatten(),
        'rvec_refined': rvec_ref.flatten(),
        'tvec_refined': tvec_ref.flatten(),
        'reprojection_error_px': reproj_error
    }


def solve_pnp_global(results: List[Dict]) -> Optional[Dict]:
    """Solve a single camera pose from all images combined using IPPE for coplanar points."""
    if not results:
        return None

    object_points_all = []
    image_points_all = []
    for r in results:
        object_points_all.append(r['object_points'])
        image_points_all.append(r['image_points_refined'])

    object_points_all = np.vstack(object_points_all).astype(np.float32)
    image_points_all = np.vstack(image_points_all).astype(np.float32)

    if len(object_points_all) < 4:
        return None

    # Use IPPE for coplanar points (all Z values are layer heights, but XY pattern repeats)
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points_all, image_points_all, K, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        success, rvec, tvec = cv2.solvePnP(
            object_points_all, image_points_all, K, dist_coeffs
        )
    
    if not success:
        return None

    rvec_ref, tvec_ref = refine_pnp(object_points_all, image_points_all, rvec, tvec)
    R_ref, _ = cv2.Rodrigues(rvec_ref)
    camera_position_ref = -R_ref.T @ tvec_ref
    reproj_error = compute_reprojection_error(
        object_points_all, image_points_all, rvec_ref, tvec_ref
    )

    return {
        'camera_position_refined': camera_position_ref.flatten(),
        'rvec_refined': rvec_ref.flatten(),
        'tvec_refined': tvec_ref.flatten(),
        'reprojection_error_px': reproj_error
    }


# =============================================================================
# RESULTS AGGREGATION AND PRINTING
# =============================================================================

def print_results(results: List[Dict], global_result: Optional[Dict]):
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
        layer_height = r['layer_height']
        cam = r['camera_position_refined']
        nozzle = r['nozzle_position']
        
        camera_positions.append(cam)
        
        if nozzle:
            delta = np.array([
                cam[0] - nozzle['x'],
                cam[1] - nozzle['y'],
                cam[2] - nozzle['z']
            ])
            transformations.append(delta)
            
            print(f"\nImage ZN{layer_n} (layer_height={layer_height:.3f}):")
            print(f"  Camera Position (refined):  X={cam[0]:8.2f}  Y={cam[1]:8.2f}  Z={cam[2]:8.2f} mm")
            print(f"  Nozzle Position:  X={nozzle['x']:8.2f}  Y={nozzle['y']:8.2f}  Z={nozzle['z']:8.2f} mm")
            print(f"  Delta (Cam-Noz):  X={delta[0]:8.2f}  Y={delta[1]:8.2f}  Z={delta[2]:8.2f} mm")
        else:
            print(f"\nImage ZN{layer_n} (layer_height={layer_height:.3f}):")
            print(f"  Camera Position (refined):  X={cam[0]:8.2f}  Y={cam[1]:8.2f}  Z={cam[2]:8.2f} mm")
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
        
        # Copy-paste ready values
        print("\n" + "=" * 80)
        print("COPY THESE VALUES TO visualize_benchy.py:")
        print("=" * 80)
        print(f"CAMERA_OFFSET_X = {median_trans[0]:.1f}")
        print(f"CAMERA_OFFSET_Y = {median_trans[1]:.1f}")
        print(f"CAMERA_OFFSET_Z = {median_trans[2]:.1f}")
    
    print("\n" + "=" * 80)

    if global_result:
        cam = global_result['camera_position_refined']
        print("\nGLOBAL REFINEMENT (ALL IMAGES)")
        print(f"  Camera Position (refined):  X={cam[0]:8.2f}  Y={cam[1]:8.2f}  Z={cam[2]:8.2f} mm")
        print(f"  Reprojection Error (RMS):   {global_result['reprojection_error_px']:.3f} px")
        print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PnP Corner Annotation Tool")
    print("=" * 80)
    print(f"\nFolder: {FOLDER_PATH}")
    print(f"Randomly selecting {N_FILES} images")
    if RANDOM_SEED is not None:
        print(f"Random seed: {RANDOM_SEED}")
    print(f"\nObject points: constant XY (25x25mm box), per-image Z from filename")
    
    # Find images
    images = find_images_random(FOLDER_PATH, N_FILES, RANDOM_SEED)
    
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
    print("\nClick corners in order (PRINTER orientation, not image):")
    print("  1. PRINTER Front-Left  (where Y is smallest, X is smallest)")
    print("  2. PRINTER Front-Right (where Y is smallest, X is largest)")
    print("  3. PRINTER Back-Left   (where Y is largest, X is smallest)")
    print("  4. PRINTER Back-Right  (where Y is largest, X is largest)")
    print("-" * 80)
    
    results = []
    
    for img_info in images:
        layer_n = img_info['layer_n']
        layer_height = img_info['layer_height']
        image_path = img_info['image_path']
        json_path = img_info['json_path']
        
        print(f"\nProcessing ZN{layer_n} (layer_height={layer_height:.3f}): {image_path.name}")
        
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
        gray_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
        image_points_refined = refine_image_points(image_points, gray_image)
        
        print(f"  Annotated image points (refined):")
        for i, (px, py) in enumerate(image_points_refined):
            print(f"    {CORNER_NAMES[i]}: pixel ({px:.1f}, {py:.1f})")
        
        # Show corresponding object points
        object_points_preview = build_object_points(layer_height)
        print(f"  Corresponding object points (printer coords, Z={layer_height}):")
        for i, (ox, oy, oz) in enumerate(object_points_preview):
            print(f"    {CORNER_NAMES[i]}: mm ({ox:.1f}, {oy:.1f}, {oz:.1f})")
        
        # Solve PnP with per-image object points (Z = layer height)
        object_points = build_object_points(layer_height)
        
        # Sanity check with homography first
        h_error = check_homography(object_points, image_points_refined)
        print(f"  Homography reprojection error: {h_error:.1f} px")
        if h_error > 50:
            print(f"  WARNING: High homography error suggests wrong point correspondence!")
        
        pnp_result = solve_pnp_single(image_points_refined, object_points)
        
        if pnp_result is None:
            print(f"  PnP failed for ZN{layer_n}")
            continue
        
        print(f"  PnP refined! Camera at: ({pnp_result['camera_position_refined'][0]:.2f}, "
              f"{pnp_result['camera_position_refined'][1]:.2f}, {pnp_result['camera_position_refined'][2]:.2f})")
        print(f"  Reprojection Error (RMS): {pnp_result['reprojection_error_px']:.3f} px")
        
        # Show reprojection visualization if error is high
        if pnp_result['reprojection_error_px'] > 10.0:
            print(f"  WARNING: High reprojection error! Showing visualization...")
            visualize_reprojection(image_path, image_points_refined, object_points,
                                  pnp_result['rvec_refined'], pnp_result['tvec_refined'], layer_n)
        
        results.append({
            'layer_n': layer_n,
            'layer_height': layer_height,
            'image_path': str(image_path),
            'image_points': image_points,
            'image_points_refined': image_points_refined,
            'object_points': object_points,
            'camera_position': pnp_result['camera_position'],
            'camera_position_refined': pnp_result['camera_position_refined'],
            'nozzle_position': nozzle_pos,
            'rvec': pnp_result['rvec'],
            'tvec': pnp_result['tvec'],
            'rvec_refined': pnp_result['rvec_refined'],
            'tvec_refined': pnp_result['tvec_refined'],
            'reprojection_error_px': pnp_result['reprojection_error_px']
        })
    
    # Global refinement across images
    global_result = solve_pnp_global(results)
    
    # Print aggregated results
    print_results(results, global_result)

    print("\nPrecision improvement ideas:")
    print("  - Add more annotated points per image (not just 4 corners).")
    print("  - Use subpixel corner refinement (cv2.cornerSubPix).")
    print("  - Use solvePnPRefineLM or solvePnPGeneric for refinement.")
    print("  - Recalibrate camera with more images + better coverage.")
    print("  - Use bundle adjustment across multiple images.")
    
    # Save results to JSON
    if results:
        output_path = FOLDER_PATH / "pnp_results.json"
        
        # Convert global_result numpy arrays to lists
        global_data = None
        if global_result:
            global_data = {
                'camera_position_refined': global_result['camera_position_refined'].tolist(),
                'rvec_refined': global_result['rvec_refined'].tolist(),
                'tvec_refined': global_result['tvec_refined'].tolist(),
                'reprojection_error_px': global_result['reprojection_error_px']
            }
        
        output_data = {
            'images': [],
            'global': global_data
        }
        for r in results:
            output_data['images'].append({
                'layer_n': r['layer_n'],
                'layer_height': r['layer_height'],
                'image_path': r['image_path'],
                'image_points': r['image_points'].tolist(),
                'image_points_refined': r['image_points_refined'].tolist(),
                'object_points': r['object_points'].tolist(),
                'camera_position': r['camera_position'].tolist(),
                'camera_position_refined': r['camera_position_refined'].tolist(),
                'nozzle_position': r['nozzle_position'],
                'rvec': r['rvec'].tolist(),
                'tvec': r['tvec'].tolist(),
                'rvec_refined': r['rvec_refined'].tolist(),
                'tvec_refined': r['tvec_refined'].tolist(),
                'reprojection_error_px': r['reprojection_error_px']
            })
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
