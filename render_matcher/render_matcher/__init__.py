"""Render Matcher - CNN-based render vs reality image matching."""

from .config import Config, load_config
from .model import SiameseNetwork
from .dataset import RenderMatcherDataset
from .trainer import Trainer

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "SiameseNetwork", "RenderMatcherDataset", "Trainer"]
