#!/usr/bin/env python3
"""
Visualize GCode at a specific layer based on a camera capture.

Automatically extracts from the capture file:
- Layer number and height
- Nozzle position  
- Image resolution

Usage:
    python visualize_benchy.py [image_path]
"""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import pygcode_viewer

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
CALIB_PATH = SCRIPT_DIR / "camera_calib" / "camera_intrinsic.json"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_IMAGE = SCRIPT_DIR.parent / "data" / "layer_captures" / \
    "4_3dbenchy_0.4n_0.2mm_PLA_MK4IS_45m.gcode_20260127_085607" / \
    "img_ZN41_Z8.4_X74_Y113_Z72.jpg"

# Camera extrinsic offset from nozzle (mm)
# Camera position = Nozzle position + offset
CAMERA_OFFSET_X = 30.0   # Camera is 3cm in +X from nozzle
CAMERA_OFFSET_Y = -10.0  # Camera is 1cm in -Y from nozzle
CAMERA_OFFSET_Z = 5.0    # Camera is 0.5cm above nozzle


def parse_capture(image_path: Path) -> dict:
    """Extract metadata from capture image and its companion JSON."""
    image_path = Path(image_path).resolve()
    json_path = image_path.with_suffix('.json')
    
    # Load from JSON if available
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
        # Parse from filename: img_ZN{layer}_Z{height}_X{x}_Y{y}_Z{z}.jpg
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
    
    # Find GCode file
    dir_name = image_path.parent.name
    gcode_match = re.match(r'(.+\.gcode)_\d{8}_\d{6}$', dir_name)
    gcode_name = meta['gcode_name'] or (gcode_match.group(1) if gcode_match else None)
    gcode_path = image_path.parent.parent / gcode_name if gcode_name else None
    
    if not gcode_path or not gcode_path.exists():
        raise FileNotFoundError(f"GCode not found for: {image_path}")
    
    # Get image dimensions
    if HAS_OPENCV:
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2] if img is not None else (2592, 4608)
    else:
        w, h = 4608, 2592
    
    meta.update({
        'image_path': image_path,
        'gcode_path': gcode_path,
        'width': w,
        'height': h,
    })
    return meta


def render_layer(meta: dict) -> Path:
    """Render GCode at the specified layer."""
    viewer = pygcode_viewer.GCodeViewer()
    viewer.load(str(meta['gcode_path']))
    viewer.set_intrinsics(str(CALIB_PATH))
    
    # Compute camera position from nozzle + extrinsic offset
    camera_x = meta['nozzle_x'] + CAMERA_OFFSET_X
    camera_y = meta['nozzle_y'] + CAMERA_OFFSET_Y
    camera_z = meta['nozzle_z'] + CAMERA_OFFSET_Z
    
    # Camera looking down at the bed
    # Up vector (0, 1, 0) means Y points up in the image (rotated 180° around Z vs (0,-1,0))
    viewer.set_camera(
        pos=(camera_x, camera_y, camera_z),
        target=(camera_x, camera_y + 0.001, 0.0),
        up=(0.0, 1.0, 0.0),
    )
    viewer.set_near_far(1.0, 500.0)
    
    # Store camera position in meta for printing
    meta['camera_x'] = camera_x
    meta['camera_y'] = camera_y
    meta['camera_z'] = camera_z
    
    # Configure view
    config = pygcode_viewer.ViewConfig()
    config.background_color = "#000000"
    config.width = meta['width']
    config.height = meta['height']
    viewer.set_config(config)
    viewer.set_bed(size=(250, 210), show_outline=True, outline_color="#333333")
    
    # Cut at the layer number (Z height from file: layer_height mm)
    # Must be after set_config to avoid being overwritten
    viewer.set_layer_range(0, meta['layer_n'])
    
    # Render
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"render_ZN{meta['layer_n']}_Z{meta['layer_height']}.png"
    viewer.render_to_file(str(output_path), require_intrinsics=True)
    
    return output_path


def create_overlay(meta: dict, render_path: Path) -> dict:
    """Create overlay and comparison images."""
    if not HAS_OPENCV:
        return {}
    
    real = cv2.imread(str(meta['image_path']))
    render = cv2.imread(str(render_path))
    if real is None or render is None:
        return {}
    
    # Resize render to match real if needed
    if real.shape[:2] != render.shape[:2]:
        render = cv2.resize(render, (real.shape[1], real.shape[0]))
    
    # Overlay (50% blend)
    overlay = cv2.addWeighted(real, 0.5, render, 0.5, 0)
    cv2.putText(overlay, f"Layer {meta['layer_n']} (Z={meta['layer_height']}mm)", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    overlay_path = OUTPUT_DIR / f"overlay_ZN{meta['layer_n']}_Z{meta['layer_height']}.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    
    # Side-by-side comparison
    scale = min(1920 / real.shape[1], 1.0) / 2
    small_size = (int(real.shape[1] * scale), int(real.shape[0] * scale))
    comparison = np.hstack([cv2.resize(real, small_size), cv2.resize(render, small_size)])
    cv2.putText(comparison, "Real", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(comparison, "Render", (small_size[0] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    comparison_path = OUTPUT_DIR / f"comparison_ZN{meta['layer_n']}_Z{meta['layer_height']}.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    
    return {'overlay': overlay_path, 'comparison': comparison_path}


def main():
    # Get image path
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    
    print(f"Image: {image_path.name}")
    
    # Parse metadata
    meta = parse_capture(image_path)
    print(f"Layer: {meta['layer_n']}")
    print(f"Z cut height: {meta['layer_height']} mm")
    print(f"Nozzle: X={meta['nozzle_x']:.1f}, Y={meta['nozzle_y']:.1f}, Z={meta['nozzle_z']:.1f}")
    print(f"Resolution: {meta['width']}x{meta['height']}")
    
    # Render
    print("\nRendering...")
    render_path = render_layer(meta)
    print(f"  Camera: X={meta['camera_x']:.1f}, Y={meta['camera_y']:.1f}, Z={meta['camera_z']:.1f}")
    print(f"  (offset: X={CAMERA_OFFSET_X:+.1f}, Y={CAMERA_OFFSET_Y:+.1f}, Z={CAMERA_OFFSET_Z:+.1f})")
    print(f"  {render_path}")
    
    # Overlay
    if HAS_OPENCV:
        print("\nCreating overlay...")
        for name, path in create_overlay(meta, render_path).items():
            print(f"  {path}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
