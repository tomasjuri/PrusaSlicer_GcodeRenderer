"""Visualization utilities for Render Matcher training."""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .dataset import RenderMatcherDataset, ImagePair


def create_sample_grid(
    dataset: RenderMatcherDataset,
    grid_size: int = 10,
    num_pairs: Optional[int] = None,
) -> np.ndarray:
    """Create a grid visualization of sample image pairs.
    
    Creates a grid where each cell shows source|render side-by-side.
    
    Args:
        dataset: Dataset to sample from.
        grid_size: Size of the grid (grid_size x grid_size cells).
        num_pairs: Number of pairs to show. If None, uses grid_size^2 / 2.
        
    Returns:
        Grid image as numpy array (H, W, 3).
    """
    if num_pairs is None:
        num_pairs = (grid_size * grid_size) // 2
    
    patch_size = dataset.config.patch_size
    
    # Each cell shows source|render, so cell width = 2 * patch_size
    cell_width = patch_size * 2
    cell_height = patch_size
    
    # Calculate grid dimensions
    # We show num_pairs pairs, each taking one cell
    # Arrange in grid_size columns
    cols = grid_size
    rows = (num_pairs + cols - 1) // cols
    
    # Create empty grid
    grid_h = rows * cell_height
    grid_w = cols * cell_width
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Fill grid with samples
    for i in range(min(num_pairs, len(dataset.image_pairs))):
        row = i // cols
        col = i % cols
        
        # Get raw sample (without augmentation)
        sample = dataset.get_raw_sample(i)
        source = sample["source"]
        render = sample["render"]
        
        # Ensure correct size
        if source.shape[0] != patch_size or source.shape[1] != patch_size:
            import cv2
            source = cv2.resize(source, (patch_size, patch_size))
            render = cv2.resize(render, (patch_size, patch_size))
        
        # Create side-by-side cell
        cell = np.hstack([source, render])
        
        # Add separator line between source and render
        cell[:, patch_size-1:patch_size+1, :] = [128, 128, 128]
        
        # Place in grid
        y_start = row * cell_height
        x_start = col * cell_width
        grid[y_start:y_start + cell_height, x_start:x_start + cell_width] = cell
    
    # Add grid lines
    for i in range(1, rows):
        grid[i * cell_height - 1:i * cell_height + 1, :, :] = [64, 64, 64]
    for i in range(1, cols):
        grid[:, i * cell_width - 1:i * cell_width + 1, :] = [64, 64, 64]
    
    return grid


def create_prediction_grid(
    sources: torch.Tensor,
    renders: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 16,
) -> np.ndarray:
    """Create a grid visualization of predictions.
    
    Shows source|render pairs with true label and predicted score.
    
    Args:
        sources: Source images tensor (N, 3, H, W).
        renders: Render images tensor (N, 3, H, W).
        labels: True labels tensor (N,).
        predictions: Predicted scores tensor (N,).
        num_samples: Number of samples to show.
        
    Returns:
        Grid image as numpy array (H, W, 3).
    """
    import cv2
    
    n = min(num_samples, sources.shape[0])
    patch_size = sources.shape[2]
    
    # 4 columns, dynamic rows
    cols = 4
    rows = (n + cols - 1) // cols
    
    cell_width = patch_size * 2 + 60  # Extra space for text
    cell_height = patch_size + 30     # Extra space for text
    
    grid = np.ones((rows * cell_height, cols * cell_width, 3), dtype=np.uint8) * 255
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Denormalize
        source = sources[i].cpu() * std + mean
        render = renders[i].cpu() * std + mean
        
        # Convert to numpy (H, W, C)
        source = (source.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        render = (render.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Clip values
        source = np.clip(source, 0, 255)
        render = np.clip(render, 0, 255)
        
        # Create cell
        cell = np.hstack([source, render])
        
        # Get label and prediction
        label = labels[i].item()
        pred = predictions[i].item()
        
        # Color based on correctness
        correct = (pred >= 0.5) == (label >= 0.5)
        color = (0, 200, 0) if correct else (200, 0, 0)
        
        # Place image in grid
        y_start = row * cell_height
        x_start = col * cell_width
        grid[y_start:y_start + patch_size, x_start:x_start + patch_size * 2] = cell
        
        # Add text
        text = f"L:{label:.0f} P:{pred:.2f}"
        cv2.putText(
            grid, text,
            (x_start + 5, y_start + patch_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    
    return grid


def log_sample_grid(
    writer: SummaryWriter,
    dataset: RenderMatcherDataset,
    grid_size: int = 10,
    save_path: Optional[str] = None,
    step: int = 0,
) -> None:
    """Log sample grid to TensorBoard and optionally save to disk.
    
    Args:
        writer: TensorBoard SummaryWriter.
        dataset: Dataset to sample from.
        grid_size: Size of the grid.
        save_path: Optional path to save the grid image.
        step: Global step for TensorBoard.
    """
    import cv2
    
    grid = create_sample_grid(dataset, grid_size)
    
    # Log to TensorBoard (expects CHW format)
    grid_tensor = torch.from_numpy(grid).permute(2, 0, 1)
    writer.add_image("samples/grid", grid_tensor, step)
    
    # Save to disk if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert RGB to BGR for cv2
        cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Saved sample grid to {save_path}")


def log_predictions(
    writer: SummaryWriter,
    sources: torch.Tensor,
    renders: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    step: int,
    tag: str = "predictions",
    num_samples: int = 16,
) -> None:
    """Log prediction visualization to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter.
        sources: Source images tensor.
        renders: Render images tensor.
        labels: True labels tensor.
        predictions: Predicted scores tensor.
        step: Global step.
        tag: Tag prefix for TensorBoard.
        num_samples: Number of samples to show.
    """
    grid = create_prediction_grid(sources, renders, labels, predictions, num_samples)
    grid_tensor = torch.from_numpy(grid).permute(2, 0, 1)
    writer.add_image(f"{tag}/examples", grid_tensor, step)
