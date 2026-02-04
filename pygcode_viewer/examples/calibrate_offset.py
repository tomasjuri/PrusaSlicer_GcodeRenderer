#!/usr/bin/env python3
"""
Interactive camera offset calibration tool.

Use arrow keys to adjust camera extrinsic offset until the overlay aligns.
Browse multiple images in a folder to verify calibration.

Controls:
    Position:
        Arrow Left/Right or A/D: Adjust X offset (+/- 1mm)
        Arrow Up/Down or W/X:    Adjust Y offset (+/- 1mm)
        +/-:                     Adjust Z offset (+/- 1mm)
    
    Rotation:
        Z/C:                     Roll (rotate in image plane) +/- 0.5 deg
        T/G:                     Pitch (tilt forward/back) +/- 0.5 deg
        F/H:                     Yaw (pan left/right) +/- 0.5 deg
    
    Navigation:
        N or Space:              Next image
        P or Backspace:          Previous image
        R:                       Reset to initial values
        S:                       Save/print current values
        Q/ESC:                   Quit

Usage:
    python calibrate_offset.py [folder_or_image_path]
"""

import sys
import json
import re
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import pygcode_viewer

import cv2
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
CALIB_PATH = SCRIPT_DIR / "camera_calib" / "camera_intrinsic.json"
DEFAULT_FOLDER = Path("/Users/tomasjurica/Downloads/layer_captures/3boxesSeamRandom_0.4n_0.25mm_PLA_MK4IS_1h12m.gcode_20260204_110630")

# Initial offset values (from visualize_benchy.py)
INITIAL_OFFSET_X = 31.0
INITIAL_OFFSET_Y = -26.0
INITIAL_OFFSET_Z = 8.0

# Initial rotation values (degrees)
INITIAL_ROT_X = 0.0   # Pitch - tilt forward/backward
INITIAL_ROT_Y = 0.0   # Yaw - pan left/right  
INITIAL_ROT_Z = -2.0   # Roll - rotation in image plane

# Display scale (images are 4608x2592, scale down for display)
DISPLAY_SCALE = 0.35


def compute_camera_orientation(rot_x: float, rot_y: float, rot_z: float) -> tuple:
    """Compute rotated camera target offset and up vector.
    
    The camera initially looks straight down (-Z direction) with Y as up.
    Rotations are applied in order: Z (roll), X (pitch), Y (yaw).
    
    Args:
        rot_x: Pitch angle in degrees (tilt forward/backward around X axis)
        rot_y: Yaw angle in degrees (pan left/right around Y axis)
        rot_z: Roll angle in degrees (rotate in image plane around Z axis)
    
    Returns:
        Tuple of (look_direction, up_vector) as numpy arrays
    """
    # Convert to radians
    rx = np.radians(rot_x)
    ry = np.radians(rot_y)
    rz = np.radians(rot_z)
    
    # Initial vectors: camera looks down (-Z), up is +Y
    look_dir = np.array([0.0, 0.0, -1.0])
    up_vec = np.array([0.0, 1.0, 0.0])
    
    # Rotation matrix around X axis (pitch)
    def rot_x_matrix(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    # Rotation matrix around Y axis (yaw)
    def rot_y_matrix(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    # Rotation matrix around Z axis (roll)
    def rot_z_matrix(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    # Apply rotations in order: Z (roll), X (pitch), Y (yaw)
    # This order makes intuitive sense for a downward-looking camera
    combined = rot_y_matrix(ry) @ rot_x_matrix(rx) @ rot_z_matrix(rz)
    
    look_dir = combined @ look_dir
    up_vec = combined @ up_vec
    
    # Normalize
    look_dir = look_dir / np.linalg.norm(look_dir)
    up_vec = up_vec / np.linalg.norm(up_vec)
    
    return look_dir, up_vec


def find_images(path: Path) -> list:
    """Find all capture images in a folder, or return single image as list."""
    path = Path(path).resolve()
    
    if path.is_file():
        return [path]
    
    if path.is_dir():
        # Find all img_ZN*.jpg files
        images = sorted(path.glob("img_ZN*.jpg"))
        if not images:
            # Try any jpg
            images = sorted(path.glob("*.jpg"))
        return images
    
    return []


def parse_capture(image_path: Path) -> dict:
    """Extract metadata from capture image and its companion JSON."""
    image_path = Path(image_path).resolve()
    json_path = image_path.with_suffix('.json')
    
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        pos = data.get('capture_position', {})
        meta = {
            'layer_n': data['layer_n'],
            'layer_height': data['layer_height'],
            'nozzle_x': pos.get('x', 0),
            'nozzle_y': pos.get('y', 0),
            'nozzle_z': pos.get('z', 0),
            'gcode_name': data.get('gcode_path'),
        }
    else:
        match = re.search(r'ZN(\d+)_Z([\d.]+)_X([\d.]+)_Y([\d.]+)_Z([\d.]+)', image_path.name)
        if not match:
            raise ValueError(f"Cannot parse: {image_path.name}")
        meta = {
            'layer_n': int(match.group(1)),
            'layer_height': float(match.group(2)),
            'nozzle_x': float(match.group(3)),
            'nozzle_y': float(match.group(4)),
            'nozzle_z': float(match.group(5)),
            'gcode_name': None,
        }
    
    dir_name = image_path.parent.name
    gcode_match = re.match(r'(.+\.gcode)_\d{8}_\d{6}$', dir_name)
    gcode_name = meta['gcode_name'] or (gcode_match.group(1) if gcode_match else None)
    gcode_path = image_path.parent.parent / gcode_name if gcode_name else None
    
    if not gcode_path or not gcode_path.exists():
        raise FileNotFoundError(f"GCode not found for: {image_path}")
    
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2] if img is not None else (2592, 4608)
    
    meta.update({
        'image_path': image_path,
        'gcode_path': gcode_path,
        'width': w,
        'height': h,
    })
    return meta


def render_with_offset(meta: dict, offset_x: float, offset_y: float, offset_z: float,
                       rot_x: float, rot_y: float, rot_z: float,
                       viewer: pygcode_viewer.GCodeViewer, temp_path: Path) -> np.ndarray:
    """Render GCode with given offset and rotation, return as numpy array.
    
    Args:
        meta: Image metadata dict with nozzle position
        offset_x, offset_y, offset_z: Camera position offsets in mm
        rot_x, rot_y, rot_z: Camera rotation angles in degrees (pitch, yaw, roll)
        viewer: GCodeViewer instance
        temp_path: Path to save temporary render
        
    Returns:
        Rendered image as numpy array
    """
    camera_x = meta['nozzle_x'] + offset_x
    camera_y = meta['nozzle_y'] + offset_y
    camera_z = meta['nozzle_z'] + offset_z
    
    # Compute rotated camera orientation
    look_dir, up_vec = compute_camera_orientation(rot_x, rot_y, rot_z)
    
    # Target is camera position + look direction (scaled to reach Z=0 plane approximately)
    # For a camera at height camera_z looking mostly down, scale to reach bed
    target_dist = camera_z if abs(look_dir[2]) > 0.1 else 100.0
    target_x = camera_x + look_dir[0] * target_dist
    target_y = camera_y + look_dir[1] * target_dist
    target_z = camera_z + look_dir[2] * target_dist
    
    viewer.set_camera(
        pos=(camera_x, camera_y, camera_z),
        target=(target_x, target_y, target_z),
        up=(float(up_vec[0]), float(up_vec[1]), float(up_vec[2])),
    )
    
    viewer.render_to_file(str(temp_path), require_intrinsics=True)
    render = cv2.imread(str(temp_path))
    
    return render


def load_image_and_viewer(image_path: Path, viewer_cache: dict) -> tuple:
    """Load image, parse metadata, and get/create viewer. Returns (meta, real_image, viewer)."""
    meta = parse_capture(image_path)
    real_image = cv2.imread(str(meta['image_path']))
    
    if real_image is None:
        raise ValueError(f"Could not load image: {meta['image_path']}")
    
    gcode_path = str(meta['gcode_path'])
    
    # Reuse viewer if same gcode, otherwise create new
    if viewer_cache.get('gcode_path') != gcode_path:
        print(f"Loading GCode: {meta['gcode_path'].name}")
        viewer = pygcode_viewer.GCodeViewer()
        viewer.load(gcode_path)
        viewer.set_intrinsics(str(CALIB_PATH))
        viewer.set_near_far(1.0, 500.0)
        
        config = pygcode_viewer.ViewConfig()
        config.background_color = "#000000"
        config.width = meta['width']
        config.height = meta['height']
        viewer.set_config(config)
        viewer.set_bed(size=(250, 210), show_outline=True, outline_color="#333333")
        
        viewer_cache['gcode_path'] = gcode_path
        viewer_cache['viewer'] = viewer
    else:
        viewer = viewer_cache['viewer']
    
    # Update layer range for this image
    viewer.set_layer_range(0, meta['layer_n'])
    
    return meta, real_image, viewer


def main():
    # Parse arguments
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FOLDER
    
    # Find all images
    images = find_images(input_path)
    if not images:
        print(f"No images found in: {input_path}")
        return 1
    
    print(f"Found {len(images)} image(s) in: {input_path}")
    
    # Current state
    current_idx = 0
    offset_x = INITIAL_OFFSET_X
    offset_y = INITIAL_OFFSET_Y
    offset_z = INITIAL_OFFSET_Z
    rot_x = INITIAL_ROT_X  # Pitch
    rot_y = INITIAL_ROT_Y  # Yaw
    rot_z = INITIAL_ROT_Z  # Roll
    
    # Viewer cache (reuse if same gcode)
    viewer_cache = {}
    
    # Create temp file for renders
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / "render.png"
    
    # Create window
    window_name = "Camera Offset Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("\nControls:")
    print("  Position: WASD/Arrows=X/Y  +/-=Z  (1mm steps)")
    print("  Rotation: Z/C=Roll  T/G=Pitch  F/H=Yaw  (0.5° steps)")
    print("  Nav: N/Space=Next  P/Back=Prev  R=Reset  S=Save  Q=Quit")
    print("-" * 60)
    
    needs_reload = True
    needs_render = True
    meta = None
    real_image = None
    viewer = None
    render = None
    display_w = 1600
    display_h = 900
    
    while True:
        # Load new image if needed
        if needs_reload:
            image_path = images[current_idx]
            print(f"\nLoading [{current_idx + 1}/{len(images)}]: {image_path.name}")
            
            try:
                meta, real_image, viewer = load_image_and_viewer(image_path, viewer_cache)
                print(f"  Layer: {meta['layer_n']}, Z: {meta['layer_height']}mm")
                print(f"  Nozzle: X={meta['nozzle_x']:.1f}, Y={meta['nozzle_y']:.1f}, Z={meta['nozzle_z']:.1f}")
                
                display_w = int(meta['width'] * DISPLAY_SCALE)
                display_h = int(meta['height'] * DISPLAY_SCALE)
                cv2.resizeWindow(window_name, display_w, display_h)
                
                needs_reload = False
                needs_render = True
            except Exception as e:
                print(f"  Error: {e}")
                # Skip to next image
                if len(images) > 1:
                    current_idx = (current_idx + 1) % len(images)
                    continue
                else:
                    return 1
        
        # Render if needed
        if needs_render and meta is not None:
            print(f"  Rendering... Pos=({offset_x:+.1f},{offset_y:+.1f},{offset_z:+.1f}) Rot=({rot_x:+.1f},{rot_y:+.1f},{rot_z:+.1f})", end="\r")
            render = render_with_offset(meta, offset_x, offset_y, offset_z, rot_x, rot_y, rot_z, viewer, temp_path)
            if render is None:
                print("  Error: Render failed")
                continue
            
            # Resize render to match real if needed
            if real_image.shape[:2] != render.shape[:2]:
                render = cv2.resize(render, (real_image.shape[1], real_image.shape[0]))
            
            needs_render = False
        
        if render is None or real_image is None:
            continue
        
        # Create overlay
        overlay = cv2.addWeighted(real_image, 0.5, render, 0.5, 0)
        
        # Draw image index
        idx_text = f"[{current_idx + 1}/{len(images)}] Layer {meta['layer_n']}"
        cv2.putText(overlay, idx_text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)
        cv2.putText(overlay, idx_text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 3)
        
        # Draw position offset info
        pos_text = f"Pos: X:{offset_x:+.1f}  Y:{offset_y:+.1f}  Z:{offset_z:+.1f}"
        cv2.putText(overlay, pos_text, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)
        cv2.putText(overlay, pos_text, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 2)
        
        # Draw rotation info
        rot_text = f"Rot: P:{rot_x:+.1f}  Y:{rot_y:+.1f}  R:{rot_z:+.1f}"
        cv2.putText(overlay, rot_text, (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)
        cv2.putText(overlay, rot_text, (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 200, 0), 2)
        
        # Draw help
        help_text = "Pos:WASD/+/-  Rot:ZC/TG/FH  Nav:NP  R:Reset  S:Save  Q:Quit"
        cv2.putText(overlay, help_text, (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        cv2.putText(overlay, help_text, (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
        
        # Show scaled
        display = cv2.resize(overlay, (display_w, display_h))
        cv2.imshow(window_name, display)
        
        # Wait for key
        key = cv2.waitKey(100) & 0xFF
        step = 1.0      # Position step in mm
        rot_step = 0.5  # Rotation step in degrees
        
        if key == 255:  # No key
            continue
        
        # Quit
        if key == ord('q') or key == ord('Q') or key == 27:
            print("\nQuitting...")
            break
        
        # Next image
        elif key == ord('n') or key == ord('N') or key == ord(' '):
            if len(images) > 1:
                current_idx = (current_idx + 1) % len(images)
                needs_reload = True
        
        # Previous image
        elif key == ord('p') or key == ord('P') or key == 8:  # 8 = backspace
            if len(images) > 1:
                current_idx = (current_idx - 1) % len(images)
                needs_reload = True
        
        # Reset
        elif key == ord('r') or key == ord('R'):
            offset_x = INITIAL_OFFSET_X
            offset_y = INITIAL_OFFSET_Y
            offset_z = INITIAL_OFFSET_Z
            rot_x = INITIAL_ROT_X
            rot_y = INITIAL_ROT_Y
            rot_z = INITIAL_ROT_Z
            needs_render = True
            print(f"\nReset to: Pos=({offset_x:+.1f},{offset_y:+.1f},{offset_z:+.1f}) Rot=({rot_x:+.1f},{rot_y:+.1f},{rot_z:+.1f})")
        
        # Save
        elif key == ord('s') or key == ord('S'):
            print(f"\n" + "=" * 60)
            print("Camera Calibration Values:")
            print("-" * 60)
            print("# Position offsets (mm)")
            print(f"CAMERA_OFFSET_X = {offset_x:.1f}")
            print(f"CAMERA_OFFSET_Y = {offset_y:.1f}")
            print(f"CAMERA_OFFSET_Z = {offset_z:.1f}")
            print()
            print("# Rotation offsets (degrees)")
            print(f"CAMERA_ROT_X = {rot_x:.1f}  # Pitch")
            print(f"CAMERA_ROT_Y = {rot_y:.1f}  # Yaw")
            print(f"CAMERA_ROT_Z = {rot_z:.1f}  # Roll")
            print("=" * 60)
        
        # Z up
        elif key == ord('+') or key == ord('='):
            offset_z += 1.0
            needs_render = True
        
        # Z down
        elif key == ord('-') or key == ord('_'):
            offset_z -= 1.0
            needs_render = True
        
        # X/Y with arrows (macOS)
        elif key == 0:  # Arrow up
            offset_y -= step
            needs_render = True
        elif key == 1:  # Arrow down
            offset_y += step
            needs_render = True
        elif key == 2:  # Arrow left
            offset_x -= step
            needs_render = True
        elif key == 3:  # Arrow right
            offset_x += step
            needs_render = True
        
        # X/Y with arrows (Linux/Windows)
        elif key == 81:  # Left
            offset_x -= step
            needs_render = True
        elif key == 82:  # Up
            offset_y -= step
            needs_render = True
        elif key == 83:  # Right
            offset_x += step
            needs_render = True
        elif key == 84:  # Down
            offset_y += step
            needs_render = True
        
        # WASD for position
        elif key == ord('a') or key == ord('A'):
            offset_x -= step
            needs_render = True
        elif key == ord('d') or key == ord('D'):
            offset_x += step
            needs_render = True
        elif key == ord('w') or key == ord('W'):
            offset_y -= step
            needs_render = True
        elif key == ord('x') or key == ord('X'):
            offset_y += step
            needs_render = True
        
        # Rotation controls: Z/C for Roll (rotation around Z axis - in image plane)
        elif key == ord('z'):
            rot_z -= rot_step
            needs_render = True
        elif key == ord('c'):
            rot_z += rot_step
            needs_render = True
        
        # T/G for Pitch (rotation around X axis - tilt forward/back)
        elif key == ord('t'):
            rot_x -= rot_step
            needs_render = True
        elif key == ord('g'):
            rot_x += rot_step
            needs_render = True
        
        # F/H for Yaw (rotation around Y axis - pan left/right)
        elif key == ord('f'):
            rot_y -= rot_step
            needs_render = True
        elif key == ord('h'):
            rot_y += rot_step
            needs_render = True
    
    cv2.destroyAllWindows()
    
    # Clean up temp file
    try:
        temp_path.unlink()
        Path(temp_dir).rmdir()
    except:
        pass
    
    print("\nFinal values:")
    print(f"  # Position offsets (mm)")
    print(f"  CAMERA_OFFSET_X = {offset_x:.1f}")
    print(f"  CAMERA_OFFSET_Y = {offset_y:.1f}")
    print(f"  CAMERA_OFFSET_Z = {offset_z:.1f}")
    print(f"  # Rotation offsets (degrees)")
    print(f"  CAMERA_ROT_X = {rot_x:.1f}  # Pitch")
    print(f"  CAMERA_ROT_Y = {rot_y:.1f}  # Yaw")
    print(f"  CAMERA_ROT_Z = {rot_z:.1f}  # Roll")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
