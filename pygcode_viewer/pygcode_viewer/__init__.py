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
__all__ = ["GCodeViewer", "CameraParams", "CameraIntrinsics", "ViewConfig", "BedConfig"]

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
class BedConfig:
    """Configuration for bed outline rendering."""
    
    # Bed size in mm (Prusa MK4 default: 250x210)
    size_x: float = 250.0
    size_y: float = 210.0
    
    # Origin offset (where 0,0 is relative to bed corner)
    origin_x: float = 0.0
    origin_y: float = 0.0
    
    # Outline color (RGBA hex or tuple)
    outline_color: str = "#4D4D4D"  # Light gray (slightly lighter than dark background)
    
    # Grid settings
    show_outline: bool = True
    show_grid: bool = False
    grid_color: str = "#333333"
    grid_spacing: float = 10.0
    
    # Line width
    line_width: float = 2.0
    
    def _parse_color(self, color: str) -> Tuple[float, float, float, float]:
        """Parse hex color to RGBA tuple (0-1 range)."""
        color = color.lstrip('#')
        if len(color) == 6:
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
            return (r, g, b, 1.0)
        elif len(color) == 8:
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
            a = int(color[6:8], 16) / 255.0
            return (r, g, b, a)
        return (0.3, 0.3, 0.3, 1.0)
    
    def to_core_config(self):
        """Convert to C++ BedConfig object."""
        if not _EXTENSION_AVAILABLE:
            return None
        config = _core.BedConfig()
        config.size_x = self.size_x
        config.size_y = self.size_y
        config.origin_x = self.origin_x
        config.origin_y = self.origin_y
        config.outline_color = list(self._parse_color(self.outline_color))
        config.grid_color = list(self._parse_color(self.grid_color))
        config.show_outline = self.show_outline
        config.show_grid = self.show_grid
        config.grid_spacing = self.grid_spacing
        config.line_width = self.line_width
        return config


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic matrix parameters from calibration.
    
    These parameters match the output format of OpenCV's camera calibration,
    allowing you to render with a realistic camera projection that matches
    a physical camera.
    
    The intrinsic matrix has the form:
        [fx,  0, cx]
        [ 0, fy, cy]
        [ 0,  0,  1]
    
    Where:
        - fx, fy: Focal lengths in pixels
        - cx, cy: Principal point (optical center) in pixels
    """
    
    fx: float = 0.0  # Focal length x (pixels)
    fy: float = 0.0  # Focal length y (pixels)
    cx: float = 0.0  # Principal point x (pixels)
    cy: float = 0.0  # Principal point y (pixels)
    image_width: int = 0   # Original calibration image width
    image_height: int = 0  # Original calibration image height
    
    @property
    def is_valid(self) -> bool:
        """Check if intrinsics are valid (have non-zero focal lengths)."""
        return self.fx > 0 and self.fy > 0 and self.image_width > 0 and self.image_height > 0
    
    @classmethod
    def from_matrix(cls, matrix: List[List[float]], image_size: Tuple[int, int]) -> "CameraIntrinsics":
        """
        Create from 3x3 camera matrix (as returned by OpenCV calibration).
        
        Args:
            matrix: 3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            image_size: (width, height) of the calibration images
            
        Returns:
            CameraIntrinsics object
        """
        return cls(
            fx=matrix[0][0],
            fy=matrix[1][1],
            cx=matrix[0][2],
            cy=matrix[1][2],
            image_width=image_size[0],
            image_height=image_size[1],
        )
    
    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "CameraIntrinsics":
        """
        Load camera intrinsics from a JSON calibration file.
        
        Expected JSON format (OpenCV calibration output):
        {
            "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "image_size": [width, height]
        }
        
        Args:
            path: Path to the JSON calibration file
            
        Returns:
            CameraIntrinsics object
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        matrix = data["camera_matrix"]
        image_size = data["image_size"]
        
        return cls.from_matrix(matrix, tuple(image_size))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for passing to C++ bindings."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "image_size": [self.image_width, self.image_height],
            "camera_matrix": [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
        }


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    
    position: Tuple[float, float, float] = (100.0, 100.0, 100.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 50.0)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 45.0
    intrinsics: Optional[CameraIntrinsics] = None
    
    def to_dict(self) -> dict:
        result = {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fov": self.fov,
        }
        if self.intrinsics is not None and self.intrinsics.is_valid:
            result["intrinsics"] = self.intrinsics.to_dict()
        return result


@dataclass
class ViewConfig:
    """Configuration for GCode visualization."""
    
    # View type: "FeatureType", "Height", "Width", "Speed", "Tool", etc.
    view_type: str = "FeatureType"
    
    # Layer range [min, max], None means full range
    layer_range: Optional[Tuple[int, Optional[int]]] = None
    
    # Feature visibility - only extrusions visible by default
    visible_features: Dict[str, bool] = field(default_factory=lambda: {
        "travels": False,
        "wipes": False,
        "retractions": False,
        "unretractions": False,
        "seams": False,
        "tool_changes": False,
        "color_changes": False,
        "pause_prints": False,
        "custom_gcodes": False,
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
    background_color: str = "#1A1A1A"  # Dark gray (almost black)
    
    # Bed display
    show_bed: bool = True
    bed_color: str = "#4D4D4D"  # Slightly lighter than background
    
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
            },
            "bed": {
                "show": self.show_bed,
                "color": self.bed_color,
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
        if "bed" in data:
            bed = data["bed"]
            if "show" in bed:
                config.show_bed = bed["show"]
            if "color" in bed:
                config.bed_color = bed["color"]
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
        self._bed_config = BedConfig()
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
        
        # Apply bed config
        core_config = self._bed_config.to_core_config()
        if core_config:
            self._viewer.set_bed_config(core_config)
    
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
    
    def set_intrinsics(
        self,
        intrinsics: Optional[Union[CameraIntrinsics, str, Path, dict]] = None,
    ) -> None:
        """
        Set camera intrinsic matrix for realistic camera projection.
        
        When intrinsics are set, the projection matrix will match a physical
        camera with the same calibration parameters. This is useful for
        overlaying rendered GCode on real camera images or for accurate
        camera simulation.
        
        Args:
            intrinsics: One of:
                - CameraIntrinsics object
                - Path to JSON calibration file
                - Dict with fx, fy, cx, cy, image_size keys
                - None to disable intrinsics (use FOV-based projection)
        
        Example:
            # From calibration file
            viewer.set_intrinsics("camera_intrinsic.json")
            
            # From dict
            viewer.set_intrinsics({
                "fx": 766.4, "fy": 768.7,
                "cx": 940.5, "cy": 597.3,
                "image_size": [1920, 1080]
            })
            
            # From CameraIntrinsics object
            intrinsics = CameraIntrinsics.from_json_file("calibration.json")
            viewer.set_intrinsics(intrinsics)
        """
        if intrinsics is None:
            self._camera.intrinsics = None
        elif isinstance(intrinsics, CameraIntrinsics):
            self._camera.intrinsics = intrinsics
        elif isinstance(intrinsics, (str, Path)):
            path = Path(intrinsics)
            if path.exists():
                self._camera.intrinsics = CameraIntrinsics.from_json_file(path)
            else:
                raise FileNotFoundError(f"Intrinsics file not found: {path}")
        elif isinstance(intrinsics, dict):
            if "camera_matrix" in intrinsics:
                # OpenCV format
                self._camera.intrinsics = CameraIntrinsics.from_matrix(
                    intrinsics["camera_matrix"],
                    tuple(intrinsics.get("image_size", [1920, 1080]))
                )
            else:
                # Direct parameters
                self._camera.intrinsics = CameraIntrinsics(
                    fx=intrinsics.get("fx", 0),
                    fy=intrinsics.get("fy", 0),
                    cx=intrinsics.get("cx", 0),
                    cy=intrinsics.get("cy", 0),
                    image_width=intrinsics.get("image_size", [1920, 1080])[0],
                    image_height=intrinsics.get("image_size", [1920, 1080])[1],
                )
    
    def clear_intrinsics(self) -> None:
        """Remove camera intrinsics and use FOV-based projection."""
        self._camera.intrinsics = None
    
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
        
        # Apply bed settings from config
        self._bed_config.show_outline = self._config.show_bed
        self._bed_config.outline_color = self._config.bed_color
        if self._loaded:
            core_config = self._bed_config.to_core_config()
            if core_config:
                self._viewer.set_bed_config(core_config)
    
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
    
    def set_bed_config(self, config: BedConfig) -> None:
        """
        Set bed outline configuration.
        
        Args:
            config: BedConfig object with bed settings
        """
        self._bed_config = config
        if self._loaded:
            core_config = config.to_core_config()
            if core_config:
                self._viewer.set_bed_config(core_config)
    
    def set_bed(
        self,
        size: Optional[Tuple[float, float]] = None,
        show_outline: bool = True,
        outline_color: Optional[str] = None,
        show_grid: bool = False,
        grid_color: Optional[str] = None,
    ) -> None:
        """
        Configure the bed outline.
        
        Args:
            size: Bed size (x, y) in mm. Default is Prusa MK4 (250, 210)
            show_outline: Whether to show bed outline
            outline_color: Outline color as hex string (e.g., "#4D4D4D")
            show_grid: Whether to show grid lines
            grid_color: Grid color as hex string
        """
        if size is not None:
            self._bed_config.size_x = size[0]
            self._bed_config.size_y = size[1]
        self._bed_config.show_outline = show_outline
        if outline_color is not None:
            self._bed_config.outline_color = outline_color
        self._bed_config.show_grid = show_grid
        if grid_color is not None:
            self._bed_config.grid_color = grid_color
        
        if self._loaded:
            core_config = self._bed_config.to_core_config()
            if core_config:
                self._viewer.set_bed_config(core_config)
