#!/usr/bin/env python3
"""
Batch process layer captures with GCode visualization.

Processes all layer capture folders and generates:
- Source image (copy with _source suffix)
- Render image (GCode visualization with _render suffix)
- Overlay image (50% blend with _overlay suffix)

Usage:
    python batch_visualize.py [--input-dir DIR] [--output-dir DIR] [--limit N] [--dry-run]
"""

import sys
import json
import re
import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import pygcode_viewer

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available. Overlay images will not be generated.")

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
CALIB_PATH = SCRIPT_DIR / "camera_calib" / "camera_intrinsic.json"
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "data" / "layer_captures"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "batch"

# Camera extrinsic offset from nozzle (mm)
CAMERA_OFFSET_X = 29.0
CAMERA_OFFSET_Y = -24.0
CAMERA_OFFSET_Z = 8.0


def discover_capture_folders(input_dir: Path) -> List[Path]:
    """Find all capture folders (directories with timestamp suffix)."""
    folders = []
    for item in sorted(input_dir.iterdir()):
        if item.is_dir() and re.match(r'.+\.gcode_\d{8}_\d{6}$', item.name):
            folders.append(item)
    return folders


def find_gcode_for_folder(folder: Path) -> Optional[Path]:
    """Find the GCode file associated with a capture folder."""
    match = re.match(r'(.+\.gcode)_\d{8}_\d{6}$', folder.name)
    if not match:
        return None
    gcode_name = match.group(1)
    gcode_path = folder.parent / gcode_name
    return gcode_path if gcode_path.exists() else None


def find_images_in_folder(folder: Path) -> List[Path]:
    """Find all capture images in a folder."""
    return sorted(folder.glob("img_ZN*.jpg"))


def parse_image_metadata(image_path: Path, gcode_path: Path) -> Dict:
    """Extract metadata from capture image and its companion JSON."""
    json_path = image_path.with_suffix('.json')
    meta = None
    
    # Try to load from JSON if available and non-empty
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            with open(json_path) as f:
                data = json.load(f)
            pos = data.get('capture_position', {})
            meta = {
                'layer_n': data['layer_n'],
                'layer_height': data['layer_height'],
                'nozzle_x': pos.get('x', 0),
                'nozzle_y': pos.get('y', 0),
                'nozzle_z': pos.get('z', 0),
            }
        except (json.JSONDecodeError, KeyError):
            meta = None  # Fall back to filename parsing
    
    # Fall back to parsing from filename
    if meta is None:
        match = re.search(r'ZN(\d+)_Z([\d.]+)_X([\d.]+)_Y([\d.]+)_Z([\d.]+)', image_path.name)
        if not match:
            raise ValueError(f"Cannot parse: {image_path.name}")
        meta = {
            'layer_n': int(match.group(1)),
            'layer_height': float(match.group(2)),
            'nozzle_x': float(match.group(3)),
            'nozzle_y': float(match.group(4)),
            'nozzle_z': float(match.group(5)),
        }
    
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


def create_viewer_for_gcode(gcode_path: Path) -> pygcode_viewer.GCodeViewer:
    """Create and configure a GCode viewer for the given file."""
    viewer = pygcode_viewer.GCodeViewer()
    viewer.load(str(gcode_path))
    viewer.set_intrinsics(str(CALIB_PATH))
    return viewer


def render_layer(viewer: pygcode_viewer.GCodeViewer, meta: Dict, output_path: Path) -> bool:
    """Render GCode at the specified layer to output_path."""
    try:
        # Compute camera position from nozzle + extrinsic offset
        camera_x = meta['nozzle_x'] + CAMERA_OFFSET_X
        camera_y = meta['nozzle_y'] + CAMERA_OFFSET_Y
        camera_z = meta['nozzle_z'] + CAMERA_OFFSET_Z
        
        # Camera looking down at the bed
        viewer.set_camera(
            pos=(camera_x, camera_y, camera_z),
            target=(camera_x, camera_y + 0.001, 0.0),
            up=(0.0, 1.0, 0.0),
        )
        viewer.set_near_far(1.0, 500.0)
        
        # Configure view
        config = pygcode_viewer.ViewConfig()
        config.background_color = "#000000"
        config.width = meta['width']
        config.height = meta['height']
        viewer.set_config(config)
        viewer.set_bed(size=(250, 210), show_outline=True, outline_color="#333333")
        
        # Cut at the layer number
        viewer.set_layer_range(0, meta['layer_n'])
        
        # Render
        output_path.parent.mkdir(parents=True, exist_ok=True)
        viewer.render_to_file(str(output_path), require_intrinsics=True)
        return True
    except Exception as e:
        print(f"    Error rendering: {e}")
        return False


def create_overlay(source_path: Path, render_path: Path, overlay_path: Path, meta: Dict) -> bool:
    """Create overlay image (50% blend of source and render)."""
    if not HAS_OPENCV:
        return False
    
    try:
        real = cv2.imread(str(source_path))
        render = cv2.imread(str(render_path))
        if real is None or render is None:
            return False
        
        # Resize render to match real if needed
        if real.shape[:2] != render.shape[:2]:
            render = cv2.resize(render, (real.shape[1], real.shape[0]))
        
        # Overlay (50% blend)
        overlay = cv2.addWeighted(real, 0.5, render, 0.5, 0)
        cv2.putText(overlay, f"Layer {meta['layer_n']} (Z={meta['layer_height']}mm)", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay)
        return True
    except Exception as e:
        print(f"    Error creating overlay: {e}")
        return False


def get_output_paths(image_path: Path, output_folder: Path) -> Tuple[Path, Path, Path]:
    """Generate output paths with appropriate suffixes."""
    stem = image_path.stem
    source_path = output_folder / f"{stem}_source.jpg"
    render_path = output_folder / f"{stem}_render.png"
    overlay_path = output_folder / f"{stem}_overlay.jpg"
    return source_path, render_path, overlay_path


def process_folder(folder: Path, output_base: Path, limit: Optional[int] = None, 
                   dry_run: bool = False) -> Dict:
    """Process all images in a capture folder."""
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    
    # Find GCode file
    gcode_path = find_gcode_for_folder(folder)
    if not gcode_path:
        print(f"  Skipping: GCode not found for {folder.name}")
        return stats
    
    # Find images
    images = find_images_in_folder(folder)
    if limit:
        images = images[:limit]
    
    if not images:
        print(f"  Skipping: No images found in {folder.name}")
        return stats
    
    # Create output folder (mirrors input structure)
    output_folder = output_base / folder.name
    
    if dry_run:
        print(f"  Would process {len(images)} images -> {output_folder}")
        stats['processed'] = len(images)
        return stats
    
    # Load GCode once for this folder
    print(f"  Loading GCode: {gcode_path.name}")
    try:
        viewer = create_viewer_for_gcode(gcode_path)
    except Exception as e:
        print(f"  Error loading GCode: {e}")
        stats['errors'] = len(images)
        return stats
    
    # Process each image
    for i, image_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {image_path.name}")
        
        try:
            # Parse metadata
            meta = parse_image_metadata(image_path, gcode_path)
            
            # Get output paths
            source_path, render_path, overlay_path = get_output_paths(image_path, output_folder)
            
            # Copy source image
            output_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, source_path)
            
            # Render GCode layer
            if not render_layer(viewer, meta, render_path):
                stats['errors'] += 1
                continue
            
            # Create overlay
            if HAS_OPENCV:
                create_overlay(source_path, render_path, overlay_path, meta)
            
            stats['processed'] += 1
            
        except Exception as e:
            print(f"    Error: {e}")
            stats['errors'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch process layer captures with GCode visualization")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help=f"Input directory (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--folders", nargs="*", type=str,
                        help="Process only these folders (by name)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit images per folder (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without processing")
    args = parser.parse_args()
    
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    # Discover capture folders
    folders = discover_capture_folders(args.input_dir)
    if args.folders:
        folders = [f for f in folders if f.name in args.folders]
    
    print(f"Found {len(folders)} capture folder(s)")
    
    if not folders:
        print("No folders to process")
        return 0
    
    # Process each folder
    total_stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] {folder.name}")
        stats = process_folder(folder, args.output_dir, args.limit, args.dry_run)
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {total_stats['processed']}")
    print(f"  Skipped:   {total_stats['skipped']}")
    print(f"  Errors:    {total_stats['errors']}")
    print(f"  Output:    {args.output_dir}")
    
    return 0 if total_stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
