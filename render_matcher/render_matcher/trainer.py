"""Training loop for Render Matcher."""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .config import Config, load_config
from .model import SiameseNetwork, count_parameters
from .dataset import create_dataloaders
from .visualization import log_sample_grid, log_predictions


class Trainer:
    """Trainer for Siamese network."""
    
    def __init__(
        self,
        config: Config,
        device: Optional[str] = None,
    ):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = SiameseNetwork(config.model).to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,} trainable")
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.all_pairs = create_dataloaders(config)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.logging.run_dir) / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        
        # Save config
        from .config import save_config
        save_config(config, self.run_dir / "config.yaml")
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            source = batch["source"].to(self.device)
            render = batch["render"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(source, render).squeeze()
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            
            # Log to TensorBoard
            if self.global_step % self.config.logging.log_every_n_steps == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
            
            self.global_step += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self._compute_metrics(all_labels, all_preds, "train")
        metrics["loss"] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate the model.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Tuple of (metrics_dict, sources, renders, labels, predictions).
        """
        self.model.eval()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        # Store some samples for visualization
        sample_sources = []
        sample_renders = []
        sample_labels = []
        sample_preds = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for batch in pbar:
            source = batch["source"].to(self.device)
            render = batch["render"].to(self.device)
            labels = batch["label"].to(self.device)
            
            predictions = self.model(source, render).squeeze()
            loss = self.criterion(predictions, labels)
            
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            
            # Store samples for visualization
            if len(sample_sources) < 16:
                n_needed = 16 - len(sample_sources)
                sample_sources.append(source[:n_needed])
                sample_renders.append(render[:n_needed])
                sample_labels.append(labels[:n_needed])
                sample_preds.append(predictions[:n_needed])
            
            pbar.set_postfix({"loss": loss.item()})
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._compute_metrics(all_labels, all_preds, "val")
        metrics["loss"] = avg_loss
        
        # Concatenate samples
        sources = torch.cat(sample_sources, dim=0)[:16]
        renders = torch.cat(sample_renders, dim=0)[:16]
        labels = torch.cat(sample_labels, dim=0)[:16]
        preds = torch.cat(sample_preds, dim=0)[:16]
        
        return metrics, sources, renders, labels, preds
    
    def _compute_metrics(
        self,
        labels: list,
        predictions: list,
        prefix: str,
    ) -> Dict[str, float]:
        """Compute classification metrics.
        
        Args:
            labels: True labels.
            predictions: Predicted scores.
            prefix: Metric name prefix.
            
        Returns:
            Dictionary of metrics.
        """
        import numpy as np
        
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        # Threshold predictions
        threshold = self.config.training.threshold
        pred_binary = (predictions >= threshold).astype(int)
        label_binary = (labels >= threshold).astype(int)
        
        metrics = {
            f"accuracy": accuracy_score(label_binary, pred_binary),
            f"precision": precision_score(label_binary, pred_binary, zero_division=0),
            f"recall": recall_score(label_binary, pred_binary, zero_division=0),
            f"f1": f1_score(label_binary, pred_binary, zero_division=0),
        }
        
        return metrics
    
    def train(self) -> None:
        """Run full training loop."""
        print(f"\nStarting training for {self.config.training.epochs} epochs")
        print(f"Logging to: {self.run_dir}")
        
        # Log sample grid at start
        print("\nCreating sample grid...")
        train_dataset = self.train_loader.dataset
        grid_path = self.run_dir / "samples_grid.png" if self.config.logging.save_sample_grid else None
        log_sample_grid(
            self.writer,
            train_dataset,
            self.config.logging.sample_grid_size,
            grid_path,
            step=0,
        )
        
        for epoch in range(self.config.training.epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics, sources, renders, labels, preds = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_metrics["loss"])
            
            # Log metrics
            for name, value in train_metrics.items():
                self.writer.add_scalar(f"train/{name}", value, epoch)
            for name, value in val_metrics.items():
                self.writer.add_scalar(f"val/{name}", value, epoch)
            
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            
            # Log predictions visualization
            log_predictions(
                self.writer,
                sources, renders, labels, preds,
                epoch,
                tag="val",
                num_samples=16,
            )
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Save checkpoints
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  New best model saved!")
            else:
                self.epochs_without_improvement += 1
            
            if (epoch + 1) % self.config.training.save_every == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if self.config.training.patience > 0:
                if self.epochs_without_improvement >= self.config.training.patience:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    break
        
        # Save final model
        self._save_checkpoint(epoch, is_best=False, filename="final.pt")
        
        self.writer.close()
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"Logs saved to: {self.run_dir}")
    
    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model so far.
            filename: Optional custom filename.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
        }
        
        if filename:
            path = self.run_dir / filename
        elif is_best:
            path = self.run_dir / "best.pt"
        else:
            path = self.run_dir / f"checkpoint_epoch{epoch+1}.pt"
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> "Trainer":
        """Load trainer from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            device: Device to load model on.
            
        Returns:
            Trainer instance with loaded state.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        trainer = cls(checkpoint["config"], device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.best_val_loss = checkpoint["best_val_loss"]
        trainer.global_step = checkpoint["global_step"]
        
        return trainer


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Render Matcher CNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    if args.resume:
        trainer = Trainer.load_checkpoint(args.resume, args.device)
    else:
        config = load_config(args.config)
        trainer = Trainer(config, args.device)
    
    trainer.train()


if __name__ == "__main__":
    main()
