"""Dataset for render-reality image pair matching."""

import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config, DataConfig, AugmentationConfig
from .augmentations import PairedTransform


class ImagePair:
    """Container for a source/render image pair."""
    
    def __init__(self, source_path: Path, render_path: Path):
        """Initialize image pair.
        
        Args:
            source_path: Path to source (reality) image.
            render_path: Path to render image.
        """
        self.source_path = source_path
        self.render_path = render_path
        self._source: Optional[np.ndarray] = None
        self._render: Optional[np.ndarray] = None
        self._valid_mask: Optional[np.ndarray] = None
    
    @property
    def source(self) -> np.ndarray:
        """Load and cache source image."""
        if self._source is None:
            self._source = cv2.imread(str(self.source_path))
            self._source = cv2.cvtColor(self._source, cv2.COLOR_BGR2RGB)
        return self._source
    
    @property
    def render(self) -> np.ndarray:
        """Load and cache render image."""
        if self._render is None:
            self._render = cv2.imread(str(self.render_path))
            self._render = cv2.cvtColor(self._render, cv2.COLOR_BGR2RGB)
        return self._render
    
    @property
    def valid_mask(self) -> np.ndarray:
        """Get mask of non-black pixels in render image."""
        if self._valid_mask is None:
            render = self.render
            # Non-black means at least one channel > threshold
            self._valid_mask = np.any(render > 10, axis=2).astype(np.uint8)
        return self._valid_mask
    
    def clear_cache(self):
        """Clear cached images to free memory."""
        self._source = None
        self._render = None
        self._valid_mask = None


def find_image_pairs(data_root: str) -> List[ImagePair]:
    """Find all source/render image pairs in the data directory.
    
    Expected structure:
        data_root/
        ├── session1/
        │   ├── img_ZN100_..._source.jpg
        │   ├── img_ZN100_..._render.png
        │   └── ...
        └── session2/
            └── ...
    
    Args:
        data_root: Root directory containing image pairs.
        
    Returns:
        List of ImagePair objects.
    """
    data_root = Path(data_root)
    pairs = []
    
    # Find all source images
    for source_path in data_root.rglob("*_source.jpg"):
        # Find corresponding render
        render_path = source_path.with_name(
            source_path.name.replace("_source.jpg", "_render.png")
        )
        if render_path.exists():
            pairs.append(ImagePair(source_path, render_path))
    
    return pairs


def find_valid_patch_location(
    valid_mask: np.ndarray,
    patch_size: int,
    exclude_locations: Optional[List[Tuple[int, int]]] = None,
    min_distance: int = 0,
    max_attempts: int = 100,
) -> Optional[Tuple[int, int]]:
    """Find a random location for a patch in non-black regions.
    
    Args:
        valid_mask: Binary mask where 1 indicates valid (non-black) regions.
        patch_size: Size of the patch to extract.
        exclude_locations: List of (x, y) locations to avoid.
        min_distance: Minimum distance from excluded locations.
        max_attempts: Maximum number of random attempts.
        
    Returns:
        Tuple of (x, y) top-left corner, or None if no valid location found.
    """
    h, w = valid_mask.shape
    
    # Create a mask of valid patch locations
    # A location is valid if the entire patch area has valid pixels
    if patch_size >= min(h, w):
        return None
    
    # Erode mask to find locations where full patch fits in valid region
    kernel_size = patch_size // 4  # Use smaller kernel for efficiency
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(valid_mask, kernel, iterations=1)
    
    # Find valid coordinates
    valid_y, valid_x = np.where(eroded > 0)
    if len(valid_x) == 0:
        return None
    
    # Filter to ensure patch fits within image bounds
    margin = patch_size
    valid_indices = (
        (valid_x >= 0) & (valid_x < w - margin) &
        (valid_y >= 0) & (valid_y < h - margin)
    )
    valid_x = valid_x[valid_indices]
    valid_y = valid_y[valid_indices]
    
    if len(valid_x) == 0:
        return None
    
    # Try random locations
    for _ in range(max_attempts):
        idx = random.randint(0, len(valid_x) - 1)
        x, y = int(valid_x[idx]), int(valid_y[idx])
        
        # Check distance from excluded locations
        if exclude_locations and min_distance > 0:
            too_close = False
            for ex_x, ex_y in exclude_locations:
                dist = np.sqrt((x - ex_x) ** 2 + (y - ex_y) ** 2)
                if dist < min_distance:
                    too_close = True
                    break
            if too_close:
                continue
        
        return (x, y)
    
    return None


class RenderMatcherDataset(Dataset):
    """Dataset for render-reality matching training."""
    
    def __init__(
        self,
        image_pairs: List[ImagePair],
        config: DataConfig,
        aug_config: AugmentationConfig,
        is_train: bool = True,
    ):
        """Initialize dataset.
        
        Args:
            image_pairs: List of ImagePair objects.
            config: Data configuration.
            aug_config: Augmentation configuration.
            is_train: Whether this is for training.
        """
        self.image_pairs = image_pairs
        self.config = config
        self.is_train = is_train
        self.transform = PairedTransform(aug_config, config.patch_size, is_train)
        
        # Calculate number of samples per epoch
        # Each image pair can generate multiple patches
        self.samples_per_pair = 4  # Generate 4 samples per image pair per epoch
    
    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return len(self.image_pairs) * self.samples_per_pair
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample.
        
        Returns:
            Dictionary with:
                - source: Source patch tensor (3, H, W)
                - render: Render patch tensor (3, H, W)
                - label: Target label (1.0 for match, 0.0 for non-match)
                - is_positive: Boolean indicating if this is a positive pair
        """
        # Get image pair
        pair_idx = idx % len(self.image_pairs)
        pair = self.image_pairs[pair_idx]
        
        # Decide if this should be a positive or negative sample
        is_positive = random.random() > self.config.negative_ratio
        
        if is_positive:
            # Positive: same location in source and render
            return self._get_positive_sample(pair)
        else:
            # Negative: different image or different location
            return self._get_negative_sample(pair, pair_idx)
    
    def _extract_patch(
        self,
        image: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        """Extract a patch from an image.
        
        Args:
            image: Source image (H, W, C).
            x: Top-left x coordinate.
            y: Top-left y coordinate.
            
        Returns:
            Patch of shape (patch_size, patch_size, C).
        """
        patch_size = self.config.patch_size
        return image[y:y + patch_size, x:x + patch_size].copy()
    
    def _get_positive_sample(self, pair: ImagePair) -> Dict[str, Any]:
        """Get a positive (matching) sample.
        
        Args:
            pair: Image pair to sample from.
            
        Returns:
            Sample dictionary.
        """
        # Find valid patch location
        location = find_valid_patch_location(
            pair.valid_mask,
            self.config.patch_size,
        )
        
        if location is None:
            # Fallback to random location if no valid region found
            h, w = pair.render.shape[:2]
            x = random.randint(0, w - self.config.patch_size - 1)
            y = random.randint(0, h - self.config.patch_size - 1)
            location = (x, y)
        
        x, y = location
        
        # Extract patches from same location
        source_patch = self._extract_patch(pair.source, x, y)
        render_patch = self._extract_patch(pair.render, x, y)
        
        # Apply transforms
        source_tensor, render_tensor = self.transform(source_patch, render_patch)
        
        return {
            "source": source_tensor,
            "render": render_tensor,
            "label": torch.tensor(1.0, dtype=torch.float32),
            "is_positive": True,
        }
    
    def _get_negative_sample(
        self, pair: ImagePair, pair_idx: int
    ) -> Dict[str, Any]:
        """Get a negative (non-matching) sample.
        
        Args:
            pair: Primary image pair.
            pair_idx: Index of the primary pair.
            
        Returns:
            Sample dictionary.
        """
        # Randomly choose between different-image and different-location negative
        use_different_image = random.random() < 0.5 and len(self.image_pairs) > 1
        
        if use_different_image:
            # Get render from a different image
            other_idx = pair_idx
            while other_idx == pair_idx:
                other_idx = random.randint(0, len(self.image_pairs) - 1)
            other_pair = self.image_pairs[other_idx]
            
            # Get locations from each image
            source_loc = find_valid_patch_location(
                pair.valid_mask, self.config.patch_size
            )
            render_loc = find_valid_patch_location(
                other_pair.valid_mask, self.config.patch_size
            )
            
            if source_loc is None or render_loc is None:
                # Fallback to positive if can't find valid locations
                return self._get_positive_sample(pair)
            
            source_patch = self._extract_patch(pair.source, *source_loc)
            render_patch = self._extract_patch(other_pair.render, *render_loc)
            
        else:
            # Different location in same image
            source_loc = find_valid_patch_location(
                pair.valid_mask, self.config.patch_size
            )
            
            if source_loc is None:
                return self._get_positive_sample(pair)
            
            # Find a different location with minimum distance
            render_loc = find_valid_patch_location(
                pair.valid_mask,
                self.config.patch_size,
                exclude_locations=[source_loc],
                min_distance=self.config.min_negative_offset,
            )
            
            if render_loc is None:
                # Can't find distant location, use different image instead
                if len(self.image_pairs) > 1:
                    return self._get_negative_sample(pair, pair_idx)
                else:
                    # Only one image, fallback to positive
                    return self._get_positive_sample(pair)
            
            source_patch = self._extract_patch(pair.source, *source_loc)
            render_patch = self._extract_patch(pair.render, *render_loc)
        
        # Apply transforms
        source_tensor, render_tensor = self.transform(source_patch, render_patch)
        
        return {
            "source": source_tensor,
            "render": render_tensor,
            "label": torch.tensor(0.0, dtype=torch.float32),
            "is_positive": False,
        }
    
    def get_raw_sample(self, idx: int) -> Dict[str, Any]:
        """Get a raw sample without transforms (for visualization).
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with raw numpy arrays.
        """
        pair_idx = idx % len(self.image_pairs)
        pair = self.image_pairs[pair_idx]
        
        location = find_valid_patch_location(
            pair.valid_mask, self.config.patch_size
        )
        
        if location is None:
            h, w = pair.render.shape[:2]
            x = random.randint(0, max(0, w - self.config.patch_size - 1))
            y = random.randint(0, max(0, h - self.config.patch_size - 1))
            location = (x, y)
        
        x, y = location
        source_patch = self._extract_patch(pair.source, x, y)
        render_patch = self._extract_patch(pair.render, x, y)
        
        return {
            "source": source_patch,
            "render": render_patch,
            "source_path": str(pair.source_path),
            "render_path": str(pair.render_path),
            "location": location,
        }


def create_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, List[ImagePair]]:
    """Create train and validation dataloaders.
    
    Args:
        config: Full configuration.
        
    Returns:
        Tuple of (train_loader, val_loader, all_pairs).
    """
    # Find all image pairs
    pairs = find_image_pairs(config.data.data_root)
    
    if len(pairs) == 0:
        raise ValueError(f"No image pairs found in {config.data.data_root}")
    
    print(f"Found {len(pairs)} image pairs")
    
    # Split into train/val
    random.seed(config.data.seed)
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * config.data.train_split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:] if split_idx < len(pairs) else pairs[-1:]
    
    print(f"Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs")
    
    # Create datasets
    train_dataset = RenderMatcherDataset(
        train_pairs, config.data, config.augmentations, is_train=True
    )
    val_dataset = RenderMatcherDataset(
        val_pairs, config.data, config.augmentations, is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, pairs
