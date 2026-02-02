"""Augmentation transforms for Render Matcher training."""

from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from .config import AugmentationConfig


def get_train_transforms(
    config: AugmentationConfig,
    patch_size: int,
) -> A.Compose:
    """Get training augmentation transforms.
    
    Args:
        config: Augmentation configuration.
        patch_size: Size of input patches.
        
    Returns:
        Albumentations Compose object with transforms.
    """
    transforms_list = []
    
    if config.enabled:
        if config.horizontal_flip:
            transforms_list.append(A.HorizontalFlip(p=0.5))
        
        if config.vertical_flip:
            transforms_list.append(A.VerticalFlip(p=0.5))
        
        if config.rotation > 0:
            transforms_list.append(
                A.Rotate(limit=config.rotation, p=0.5, border_mode=0)
            )
        
        if config.brightness > 0 or config.contrast > 0:
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=config.brightness,
                    contrast_limit=config.contrast,
                    p=0.5,
                )
            )
        
        if config.blur:
            transforms_list.append(
                A.GaussianBlur(blur_limit=(3, config.blur_limit), p=0.3)
            )
    
    # Always normalize and convert to tensor
    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def get_val_transforms(patch_size: int) -> A.Compose:
    """Get validation transforms (no augmentation).
    
    Args:
        patch_size: Size of input patches.
        
    Returns:
        Albumentations Compose object with transforms.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def apply_paired_transform(
    source: np.ndarray,
    render: np.ndarray,
    transform: A.Compose,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply identical transform to both source and render images.
    
    This ensures spatial augmentations (flip, rotation) are applied
    identically to both images in the pair.
    
    Args:
        source: Source (reality) image as numpy array.
        render: Render image as numpy array.
        transform: Albumentations transform to apply.
        
    Returns:
        Tuple of (transformed_source, transformed_render).
    """
    # Stack images and use ReplayCompose behavior
    # We use additional_targets to apply same transform to both
    replay_transform = A.ReplayCompose(
        transform.transforms,
        additional_targets={"render": "image"}
    )
    
    result = replay_transform(image=source, render=render)
    
    return result["image"], result["render"]


class PairedTransform:
    """Wrapper to apply same transforms to source and render images."""
    
    def __init__(self, config: AugmentationConfig, patch_size: int, is_train: bool = True):
        """Initialize paired transform.
        
        Args:
            config: Augmentation configuration.
            patch_size: Size of input patches.
            is_train: Whether this is for training (with augmentations).
        """
        self.is_train = is_train
        self.patch_size = patch_size
        
        # Build transform list
        transforms_list = []
        
        if is_train and config.enabled:
            if config.horizontal_flip:
                transforms_list.append(A.HorizontalFlip(p=0.5))
            
            if config.vertical_flip:
                transforms_list.append(A.VerticalFlip(p=0.5))
            
            if config.rotation > 0:
                transforms_list.append(
                    A.Rotate(limit=config.rotation, p=0.5, border_mode=0)
                )
            
            if config.brightness > 0 or config.contrast > 0:
                transforms_list.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=config.brightness,
                        contrast_limit=config.contrast,
                        p=0.5,
                    )
                )
            
            if config.blur:
                transforms_list.append(
                    A.GaussianBlur(blur_limit=(3, config.blur_limit), p=0.3)
                )
        
        # Always normalize and convert to tensor
        transforms_list.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.transform = A.ReplayCompose(
            transforms_list,
            additional_targets={"render": "image"}
        )
    
    def __call__(
        self, source: np.ndarray, render: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transform to source and render images.
        
        Args:
            source: Source (reality) image as numpy array (H, W, C).
            render: Render image as numpy array (H, W, C).
            
        Returns:
            Tuple of (transformed_source, transformed_render) as tensors.
        """
        result = self.transform(image=source, render=render)
        return result["image"], result["render"]
