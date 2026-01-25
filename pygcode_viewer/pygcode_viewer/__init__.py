"""
pygcode_viewer - Python wrapper for PrusaSlicer's GCode visualization

This package provides a Python interface to render GCODE files using
the libvgcode library from PrusaSlicer.

Example usage:
    import pygcode_viewer
    
    viewer = pygcode_viewer.GCodeViewer()
    viewer.load("model.gcode")
    viewer.set_camera(pos=(100, 100, 100), target=(0, 0, 50))
    viewer.render_to_file("output.png", width=1920, height=1080)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["GCodeViewer", "CameraParams", "ViewConfig"]

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Import the C++ extension module
_core = None
_EXTENSION_AVAILABLE = False

try:
    from . import _pygcode_viewer as _core
    _EXTENSION_AVAILABLE = True
except ImportError:
    # Extension not built - allow using Python-only features
    pass


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    
    position: Tuple[float, float, float] = (100.0, 100.0, 100.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 50.0)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 45.0
    
    def to_dict(self) -> dict:
        return {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fov": self.fov,
        }


@dataclass
class ViewConfig:
    """Configuration for GCode visualization."""
    
    # View type: "FeatureType", "Height", "Width", "Speed", "Tool", etc.
    view_type: str = "FeatureType"
    
    # Layer range [min, max], None means full range
    layer_range: Optional[Tuple[int, Optional[int]]] = None
    
    # Feature visibility
    visible_features: Dict[str, bool] = field(default_factory=lambda: {
        "travels": False,
        "wipes": False,
        "retractions": True,
        "unretractions": True,
        "seams": True,
        "tool_changes": True,
        "color_changes": True,
        "pause_prints": True,
        "custom_gcodes": True,
    })
    
    # Extrusion role visibility and colors (exact PrusaSlicer defaults from libvgcode)
    extrusion_roles: Dict[str, Dict[str, Union[bool, str]]] = field(default_factory=lambda: {
        "None": {"visible": True, "color": "#E6B3B3"},               # rgb(230, 179, 179)
        "Perimeter": {"visible": True, "color": "#FFE64D"},          # rgb(255, 230, 77)
        "ExternalPerimeter": {"visible": True, "color": "#FF7D38"},  # rgb(255, 125, 56)
        "OverhangPerimeter": {"visible": True, "color": "#1F1FFF"},  # rgb(31, 31, 255)
        "InternalInfill": {"visible": True, "color": "#B03029"},     # rgb(176, 48, 41)
        "SolidInfill": {"visible": True, "color": "#9654CC"},        # rgb(150, 84, 204)
        "TopSolidInfill": {"visible": True, "color": "#F04040"},     # rgb(240, 64, 64)
        "Ironing": {"visible": True, "color": "#FF8C69"},            # rgb(255, 140, 105)
        "BridgeInfill": {"visible": True, "color": "#4D80BA"},       # rgb(77, 128, 186)
        "GapFill": {"visible": True, "color": "#FFFFFF"},            # rgb(255, 255, 255)
        "Skirt": {"visible": True, "color": "#00876E"},              # rgb(0, 135, 110)
        "SupportMaterial": {"visible": True, "color": "#00FF00"},    # rgb(0, 255, 0)
        "SupportMaterialInterface": {"visible": True, "color": "#008000"},  # rgb(0, 128, 0)
        "WipeTower": {"visible": True, "color": "#B3E3AB"},          # rgb(179, 227, 171)
        "Custom": {"visible": True, "color": "#5ED194"},             # rgb(94, 209, 148)
    })
    
    # Tool colors (for multi-extruder)
    tool_colors: List[str] = field(default_factory=lambda: [
        "#FF8000", "#00FF00", "#0000FF", "#FF00FF", "#00FFFF"
    ])
    
    # Output settings
    width: int = 1920
    height: int = 1080
    background_color: str = "#FFFFFF"
    
    def to_dict(self) -> dict:
        return {
            "view_type": self.view_type,
            "layer_range": list(self.layer_range) if self.layer_range else None,
            "visible_features": self.visible_features,
            "extrusion_roles": self.extrusion_roles,
            "tool_colors": self.tool_colors,
            "output": {
                "width": self.width,
                "height": self.height,
                "background_color": self.background_color,
            }
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ViewConfig":
        config = cls()
        if "view_type" in data:
            config.view_type = data["view_type"]
        if "layer_range" in data and data["layer_range"]:
            config.layer_range = tuple(data["layer_range"])
        if "visible_features" in data:
            config.visible_features.update(data["visible_features"])
        if "extrusion_roles" in data:
            config.extrusion_roles.update(data["extrusion_roles"])
        if "tool_colors" in data:
            config.tool_colors = data["tool_colors"]
        if "output" in data:
            output = data["output"]
            if "width" in output:
                config.width = output["width"]
            if "height" in output:
                config.height = output["height"]
            if "background_color" in output:
                config.background_color = output["background_color"]
        return config
    
    @classmethod
    def from_json(cls, json_str: str) -> "ViewConfig":
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ViewConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class GCodeViewer:
    """
    GCode visualization viewer.
    
    Loads GCODE files and renders them to PNG images or displays
    them in an interactive window.
    """
    
    def __init__(self):
        """Initialize the GCode viewer."""
        if not _EXTENSION_AVAILABLE:
            raise ImportError(
                "pygcode_viewer C++ extension is not available. "
                "Please build the package first with: pip install ."
            )
        self._viewer = _core.GCodeViewerCore()
        self._camera = CameraParams()
        self._config = ViewConfig()
        self._loaded = False
    
    def load(self, gcode_path: Union[str, Path]) -> None:
        """
        Load a GCODE file.
        
        Args:
            gcode_path: Path to the GCODE file (ASCII or binary)
        """
        path = Path(gcode_path)
        if not path.exists():
            raise FileNotFoundError(f"GCODE file not found: {path}")
        
        self._viewer.load(str(path))
        self._loaded = True
        
        # Auto-center camera on model
        bbox = self._viewer.get_bounding_box()
        center = (
            (bbox[0] + bbox[3]) / 2,
            (bbox[1] + bbox[4]) / 2,
            (bbox[2] + bbox[5]) / 2,
        )
        self._camera.target = center
    
    def set_camera(
        self,
        pos: Optional[Tuple[float, float, float]] = None,
        target: Optional[Tuple[float, float, float]] = None,
        up: Optional[Tuple[float, float, float]] = None,
        fov: Optional[float] = None,
    ) -> None:
        """
        Set camera parameters.
        
        Args:
            pos: Camera position (x, y, z)
            target: Point to look at (x, y, z)
            up: Up vector (x, y, z), default is (0, 0, 1)
            fov: Field of view in degrees
        """
        if pos is not None:
            self._camera.position = pos
        if target is not None:
            self._camera.target = target
        if up is not None:
            self._camera.up = up
        if fov is not None:
            self._camera.fov = fov
    
    def set_config(self, config: Union[ViewConfig, dict, str, Path]) -> None:
        """
        Set visualization configuration.
        
        Args:
            config: ViewConfig object, dict, JSON string, or path to JSON file
        """
        if isinstance(config, ViewConfig):
            self._config = config
        elif isinstance(config, dict):
            self._config = ViewConfig.from_dict(config)
        elif isinstance(config, (str, Path)):
            path = Path(config)
            if path.exists():
                self._config = ViewConfig.from_file(path)
            else:
                # Assume it's a JSON string
                self._config = ViewConfig.from_json(str(config))
    
    def set_layer_range(
        self,
        min_layer: int = 0,
        max_layer: Optional[int] = None
    ) -> None:
        """
        Set the layer range to render.
        
        Args:
            min_layer: First layer to render (0-indexed)
            max_layer: Last layer to render (None = all layers)
        """
        self._config.layer_range = (min_layer, max_layer)
        # Also set on the C++ viewer immediately if loaded
        if self._loaded:
            if max_layer is None:
                max_layer = self.get_layer_count() - 1
            self._viewer.set_layer_range(min_layer, max_layer)
    
    def set_view_type(self, view_type: str) -> None:
        """
        Set the view type for coloring.
        
        Args:
            view_type: One of "FeatureType", "Height", "Width", "Speed", 
                      "FanSpeed", "Temperature", "VolumetricFlowRate", 
                      "Tool", "ColorPrint"
        """
        self._config.view_type = view_type
        # Also set on the C++ viewer immediately if loaded
        if self._loaded:
            self._viewer.set_view_type(view_type)
    
    def render_to_file(
        self,
        output_path: Union[str, Path],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """
        Render the GCODE to a PNG file.
        
        Args:
            output_path: Output PNG file path
            width: Image width (uses config if not specified)
            height: Image height (uses config if not specified)
        """
        if not self._loaded:
            raise RuntimeError("No GCODE file loaded. Call load() first.")
        
        w = width or self._config.width
        h = height or self._config.height
        
        self._viewer.render_to_file(
            str(output_path),
            w,
            h,
            self._camera.to_dict(),
            self._config.to_dict(),
        )
    
    def get_layer_count(self) -> int:
        """Get the number of layers in the loaded GCODE."""
        if not self._loaded:
            return 0
        return self._viewer.get_layer_count()
    
    def get_bounding_box(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get the bounding box of the GCODE.
        
        Returns:
            Tuple of (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        if not self._loaded:
            return (0, 0, 0, 0, 0, 0)
        return self._viewer.get_bounding_box()
    
    def get_estimated_time(self) -> float:
        """Get estimated print time in seconds."""
        if not self._loaded:
            return 0.0
        return self._viewer.get_estimated_time()
