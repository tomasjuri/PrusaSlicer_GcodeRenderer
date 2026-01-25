#!/usr/bin/env python3
"""
Example: Custom color configuration

This example demonstrates how to:
1. Customize extrusion role colors
2. Use different view types
3. Configure feature visibility via JSON
"""

import os
import sys
import json

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygcode_viewer


def render_with_config(viewer, config_name, config_dict, output_name):
    """Render with a specific configuration."""
    print(f"Rendering with '{config_name}' configuration...")
    
    config = pygcode_viewer.ViewConfig.from_dict(config_dict)
    viewer.set_config(config)
    viewer.render_to_file(output_name, width=800, height=600)
    print(f"  Saved: {output_name}")


def main():
    # Create the viewer
    viewer = pygcode_viewer.GCodeViewer()
    
    # Load a GCODE file
    gcode_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "tests", "sample.gcode"
    )
    
    if not os.path.exists(gcode_path):
        print(f"GCODE file not found: {gcode_path}")
        return 1
    
    print(f"Loading GCODE: {gcode_path}")
    viewer.load(gcode_path)
    
    # Set up camera
    bbox = viewer.get_bounding_box()
    center_x = (bbox[0] + bbox[3]) / 2
    center_y = (bbox[1] + bbox[4]) / 2
    center_z = (bbox[2] + bbox[5]) / 2
    size = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    distance = size * 2
    
    viewer.set_camera(
        pos=(center_x + distance, center_y + distance, center_z + distance * 0.5),
        target=(center_x, center_y, center_z)
    )
    
    # Configuration 1: Default (feature type colors)
    render_with_config(viewer, "Default", {
        "view_type": "FeatureType",
        "output": {"background_color": "#FFFFFF"}
    }, "output_default.png")
    
    # Configuration 2: Speed view (cool-warm gradient)
    render_with_config(viewer, "Speed", {
        "view_type": "Speed",
        "output": {"background_color": "#FFFFFF"}
    }, "output_speed.png")
    
    # Configuration 3: Height view
    render_with_config(viewer, "Height", {
        "view_type": "Height",
        "output": {"background_color": "#FFFFFF"}
    }, "output_height.png")
    
    # Configuration 4: Custom bright colors
    render_with_config(viewer, "Bright Colors", {
        "view_type": "FeatureType",
        "extrusion_roles": {
            "Perimeter": {"visible": True, "color": "#FF6B6B"},
            "ExternalPerimeter": {"visible": True, "color": "#4ECDC4"},
            "InternalInfill": {"visible": True, "color": "#45B7D1"},
            "SolidInfill": {"visible": True, "color": "#96CEB4"},
            "TopSolidInfill": {"visible": True, "color": "#FFEAA7"},
            "Skirt": {"visible": True, "color": "#DDA0DD"},
            "SupportMaterial": {"visible": True, "color": "#98D8C8"},
        },
        "output": {"background_color": "#2C3E50"}
    }, "output_bright.png")
    
    # Configuration 5: Dark mode
    render_with_config(viewer, "Dark Mode", {
        "view_type": "FeatureType",
        "visible_features": {
            "travels": False,
            "retractions": False,
        },
        "extrusion_roles": {
            "Perimeter": {"visible": True, "color": "#FF9500"},
            "ExternalPerimeter": {"visible": True, "color": "#FFCC00"},
            "InternalInfill": {"visible": True, "color": "#00CED1"},
            "SolidInfill": {"visible": True, "color": "#32CD32"},
            "TopSolidInfill": {"visible": True, "color": "#FF69B4"},
        },
        "output": {"background_color": "#1A1A2E"}
    }, "output_dark.png")
    
    # Configuration 6: Infill only
    render_with_config(viewer, "Infill Only", {
        "view_type": "FeatureType",
        "extrusion_roles": {
            "Perimeter": {"visible": False},
            "ExternalPerimeter": {"visible": False},
            "Skirt": {"visible": False},
            "InternalInfill": {"visible": True, "color": "#FF0000"},
            "SolidInfill": {"visible": True, "color": "#00FF00"},
            "TopSolidInfill": {"visible": True, "color": "#0000FF"},
            "BridgeInfill": {"visible": True, "color": "#FFFF00"},
        },
        "output": {"background_color": "#FFFFFF"}
    }, "output_infill.png")
    
    print("\nAll renders complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
