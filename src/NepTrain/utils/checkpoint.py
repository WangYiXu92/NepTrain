"""
Checkpoint Manager for NepTrain Training Pipeline

This module provides a CheckpointManager class for saving and loading model checkpoints,
including model weights, optimizer states, scheduler states, and training metadata.
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Supports saving and loading:
    - Model weights
    - Optimizer states
    - Scheduler states
    - Training metadata (epoch, step, metrics, etc.)

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        overwrite: Whether to overwrite existing checkpoints
        save_best_only: Whether to only save best checkpoints
        monitor: Metric to monitor for best checkpoint
        mode: One of 'min', 'max', 'auto' for monitor comparison
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_to_keep: int = 5,
        overwrite: bool = False,
        save_best_only: bool = False,
        monitor: str = "loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_to_keep = max_to_keep
        self.overwrite = overwrite
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        # Track best metric value
        self.best_value: Optional[float] = None

        # Validate mode
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def _is_better(self, current: float, best: Optional[float]) -> bool:
        """Check if current value is better than best value."""
        if best is None:
            return True
        if self.mode == "min":
            return current < best
        else:
            return current > best

    def _get_checkpoint_path(self, epoch: int, step: int) -> Path:
        """Generate checkpoint file path."""
        return self.checkpoint_dir / f"checkpoint_ep{epoch}_step{step}.pt"

    def _get_metadata_path(self, epoch: int, step: int) -> Path:
        """Generate metadata file path."""
        return self.checkpoint_dir / f"checkpoint_ep{epoch}_step{step}.json"

    def _save_metadata(
        self,
        metadata: Dict[str, Any],
        epoch: int,
        step: int,
    ) -> Path:
        """Save metadata to JSON file."""
        metadata_path = self._get_metadata_path(epoch, step)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata_path

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch number
            step: Current step number
            metrics: Metrics to save
            extra: Any extra data to save

        Returns:
            Path to saved checkpoint
        """
        if epoch is None or step is None:
            warnings.warn(
                "epoch and step are not specified, using default values",
                UserWarning,
            )
            epoch = epoch if epoch is not None else -1
            step = step if step is not None else -1

        checkpoint_path = self._get_checkpoint_path(epoch, step)

        # Check if checkpoint already exists
        if checkpoint_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint_path}"
            )

        # Build checkpoint dictionary
        checkpoint: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        if extra is not None:
            checkpoint["extra"] = extra

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Save metadata
        metadata = {
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "timestamp": str(Path(checkpoint_path).stat().st_mtime),
        }
        self._save_metadata(metadata, epoch, step)

        # Clean up old checkpoints
        self._cleanup_checkpoints()

        # Update best value
        if self.save_best_only and metrics is not None:
            current_value = metrics.get(self.monitor)
            if current_value is not None:
                if self._is_better(current_value, self.best_value):
                    self.best_value = current_value
                    self._save_best_metadata(current_value, epoch, step)

        return checkpoint_path

    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state dict into
            optimizer: Optimizer to load state dict into
            scheduler: Scheduler to load state dict into
            map_location: Map location for loading
            strict: Whether to strictly enforce that the keys match

        Returns:
            Dictionary of loaded data
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load model state dict
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Load optimizer state dict
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state dict
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Return extra data
        result = {
            "epoch": checkpoint.get("epoch"),
            "step": checkpoint.get("step"),
            "metrics": checkpoint.get("metrics", {}),
            "extra": checkpoint.get("extra", {}),
        }

        return result

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load best checkpoint.

        Args:
            model: Model to load state dict into
            optimizer: Optimizer to load state dict into
            scheduler: Scheduler to load state dict into
            map_location: Map location for loading
            strict: Whether to strictly enforce that the keys match

        Returns:
            Dictionary of loaded data
        """
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")

        return self.load(best_path, model, optimizer, scheduler, map_location, strict)

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        checkpoint_files = list(
            self.checkpoint_dir.glob("checkpoint_ep*_step*.pt")
        )
        if len(checkpoint_files) <= self.max_to_keep:
            return

        # Sort by modification time (newest first)
        checkpoint_files.sort(
            key=lambda x: x.stat().st_mtime, reverse=True
        )

        # Remove old checkpoints
        for checkpoint_file in checkpoint_files[self.max_to_keep:]:
            metadata_file = self.checkpoint_dir / (
                checkpoint_file.stem + ".json"
            )
            if metadata_file.exists():
                metadata_file.unlink()
            checkpoint_file.unlink()

    def _save_best_metadata(
        self, value: float, epoch: int, step: int
    ) -> None:
        """Save information about best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        current_path = self._get_checkpoint_path(epoch, step)

        # Copy current checkpoint as best
        if best_path.exists():
            best_path.unlink()
        best_path.symlink_to(current_path.absolute())

        # Save best metadata
        best_metadata_path = self.checkpoint_dir / "best.json"
        metadata = {
            "value": value,
            "epoch": epoch,
            "step": step,
            "timestamp": str(best_path.stat().st_mtime),
        }
        with open(best_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint information dicts with keys:
            - path: Path to checkpoint file
            - epoch: Epoch number
            - step: Step number
            - metrics: Metrics at checkpoint
        """
        checkpoints = []
        for checkpoint_file in sorted(
            self.checkpoint_dir.glob("checkpoint_ep*_step*.pt")
        ):
            metadata_file = self.checkpoint_dir / (
                checkpoint_file.stem + ".json"
            )
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

            checkpoints.append(
                {
                    "path": checkpoint_file,
                    "epoch": metadata.get("epoch"),
                    "step": metadata.get("step"),
                    "metrics": metadata.get("metrics", {}),
                }
            )

        return checkpoints

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None if not found
        """
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return best_path.resolve()
        return None

    def get_last_checkpoint(self) -> Optional[Path]:
        """
        Get path to most recent checkpoint.

        Returns:
            Path to last checkpoint or None if not found
        """
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[-1]["path"]
        return None


def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """
    Convenience function to save a checkpoint.

    Args:
        model: Model to save
        path: Path to save checkpoint
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch number
        step: Current step number
        metrics: Metrics to save

    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(Path(path).parent)
    return manager.save(
        model, optimizer, scheduler, epoch, step, metrics
    )


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to load a checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load state dict into
        optimizer: Optimizer to load state dict into
        scheduler: Scheduler to load state dict into
        map_location: Map location for loading
        strict: Whether to strictly enforce that the keys match

    Returns:
        Dictionary of loaded data
    """
    manager = CheckpointManager(Path(path).parent)
    return manager.load(path, model, optimizer, scheduler, map_location, strict)
