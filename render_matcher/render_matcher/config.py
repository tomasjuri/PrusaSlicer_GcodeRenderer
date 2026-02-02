"""Configuration management for Render Matcher training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "resnet18"
    pretrained: bool = True
    embedding_dim: int = 256
    freeze_backbone: bool = False


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_root: str = "../pygcode_viewer/outputs/batch"
    patch_size: int = 224
    num_workers: int = 4
    min_negative_offset: int = 448
    negative_ratio: float = 0.5
    train_split: float = 0.8
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0001
    threshold: float = 0.5
    patience: int = 10
    save_every: int = 10


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    enabled: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: int = 15
    brightness: float = 0.2
    contrast: float = 0.2
    blur: bool = False
    blur_limit: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration."""
    sample_grid_size: int = 10
    log_every_n_steps: int = 100
    save_sample_grid: bool = True
    run_dir: str = "runs"


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Convert dicts to dataclasses if needed."""
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.augmentations, dict):
            self.augmentations = AugmentationConfig(**self.augmentations)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        
    Returns:
        Config object with loaded settings.
    """
    if config_path is None:
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    return Config(
        model=data.get("model", {}),
        data=data.get("data", {}),
        training=data.get("training", {}),
        augmentations=data.get("augmentations", {}),
        logging=data.get("logging", {}),
    )


def save_config(config: Config, path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save.
        path: Output file path.
    """
    from dataclasses import asdict
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
