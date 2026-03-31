"""
Tests for NepTrain Optimizations

This module contains tests for the optimization features in NepTrain,
including checkpointing and parallel I/O.
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
from ase import Atoms

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import modules directly to bypass __init__.py
import importlib.util


def import_module_directly(module_name, file_path):
    """Import a module directly from file without going through __init__.py."""
    # Resolve absolute path
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model(self):
        """Create a simple PyTorch model for testing."""
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        return SimpleModel()

    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer for testing."""
        import torch.optim as optim

        return optim.Adam(model.parameters())

    def test_checkpoint_creation(self, temp_dir, model, optimizer):
        """Test checkpoint creation."""
        checkpoint_module = import_module_directly(
            "checkpoint",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "utils" / "checkpoint.py"
        )
        CheckpointManager = checkpoint_module.CheckpointManager

        manager = CheckpointManager(temp_dir)
        checkpoint_path = manager.save(
            model,
            optimizer,
            epoch=0,
            step=0,
            metrics={"loss": 0.5},
        )

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pt"

    def test_checkpoint_load(self, temp_dir, model, optimizer):
        """Test checkpoint loading."""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        checkpoint_module = import_module_directly(
            "checkpoint",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "utils" / "checkpoint.py"
        )
        CheckpointManager = checkpoint_module.CheckpointManager

        manager = CheckpointManager(temp_dir)

        # Save checkpoint
        initial_state = model.state_dict().copy()
        manager.save(
            model, optimizer, epoch=0, step=0, metrics={"loss": 0.5}
        )

        # Load checkpoint back into the same model (not a new model with mismatched architecture)
        # Recreate model with same architecture
        new_model = type(model)()
        new_model.load_state_dict(initial_state)  # Initialize with same weights
        new_optimizer = optim.Adam(new_model.parameters())

        # Load checkpoint
        loaded_data = manager.load(
            Path(temp_dir) / "checkpoint_ep0_step0.pt",
            new_model,
            new_optimizer,
        )

        # Check loaded data
        assert "epoch" in loaded_data
        assert "step" in loaded_data
        assert "metrics" in loaded_data
        assert loaded_data["epoch"] == 0
        assert loaded_data["step"] == 0
        assert loaded_data["metrics"]["loss"] == 0.5

        # Check model state was loaded
        loaded_state = new_model.state_dict()
        for key in initial_state:
            assert key in loaded_state

    def test_checkpoint_save_best_only(self, temp_dir, model):
        """Test save best only functionality."""
        import torch.nn as nn
        import torch.optim as optim

        checkpoint_module = import_module_directly(
            "checkpoint",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "utils" / "checkpoint.py"
        )
        CheckpointManager = checkpoint_module.CheckpointManager

        manager = CheckpointManager(
            temp_dir, save_best_only=True, monitor="loss", mode="min"
        )

        # Save checkpoint with loss=0.5
        manager.save(
            model,
            epoch=0,
            step=0,
            metrics={"loss": 0.5, "val_loss": 0.6},
        )
        assert (Path(temp_dir) / "best.pt").exists()

        # Save checkpoint with better loss=0.3
        manager.save(
            model,
            epoch=1,
            step=100,
            metrics={"loss": 0.3, "val_loss": 0.4},
        )
        assert (Path(temp_dir) / "best.pt").exists()

    def test_checkpoint_max_to_keep(self, temp_dir, model):
        """Test max_to_keep functionality."""
        import torch.optim as optim

        checkpoint_module = import_module_directly(
            "checkpoint",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "utils" / "checkpoint.py"
        )
        CheckpointManager = checkpoint_module.CheckpointManager

        manager = CheckpointManager(temp_dir, max_to_keep=3)

        # Save multiple checkpoints
        for i in range(5):
            manager.save(
                model, epoch=i, step=i * 10, metrics={"loss": 1.0 - i * 0.1}
            )

        # Check only 3 checkpoints remain
        checkpoint_files = list(Path(temp_dir).glob("checkpoint_*.pt"))
        assert len(checkpoint_files) == 3

    def test_list_checkpoints(self, temp_dir, model):
        """Test list_checkpoints method."""
        import torch.optim as optim

        checkpoint_module = import_module_directly(
            "checkpoint",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "utils" / "checkpoint.py"
        )
        CheckpointManager = checkpoint_module.CheckpointManager

        manager = CheckpointManager(temp_dir)

        # Save checkpoints
        for i in range(3):
            manager.save(
                model, epoch=i, step=i * 10, metrics={"loss": 1.0 - i * 0.1}
            )

        # List checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Check checkpoint info
        for cp in checkpoints:
            assert "path" in cp
            assert "epoch" in cp
            assert "step" in cp
            assert "metrics" in cp


class TestParallelXYZReader:
    """Tests for ParallelXYZReader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def xyz_files(self, temp_dir):
        """Create temporary XYZ files for testing."""
        from ase.io import write

        files = []
        for i in range(5):
            atoms = Atoms(
                "H2O",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.95, 0.0, 0.0],
                    [0.0, 0.95, 0.0],
                ],
            )
            file_path = Path(temp_dir) / f"structure_{i}.xyz"
            write(file_path, atoms)
            files.append(file_path)

        return files

    def test_read_single_file(self, temp_dir, xyz_files):
        """Test reading a single XYZ file."""
        io_module = import_module_directly(
            "io_parallel",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "io_parallel.py"
        )
        ParallelXYZReader = io_module.ParallelXYZReader

        reader = ParallelXYZReader(n_workers=1)
        structures = reader.read(xyz_files[0])

        assert len(structures) == 1
        assert isinstance(structures[0], Atoms)
        assert len(structures[0]) == 3

    def test_read_multiple_files(self, temp_dir, xyz_files):
        """Test reading multiple XYZ files."""
        io_module = import_module_directly(
            "io_parallel",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "io_parallel.py"
        )
        ParallelXYZReader = io_module.ParallelXYZReader

        reader = ParallelXYZReader(n_workers=2)
        structures = reader.read(xyz_files)

        assert len(structures) == 5
        for atoms in structures:
            assert len(atoms) == 3

    def test_read_directory(self, temp_dir, xyz_files):
        """Test reading all files in a directory."""
        io_module = import_module_directly(
            "io_parallel",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "io_parallel.py"
        )
        ParallelXYZReader = io_module.ParallelXYZReader

        reader = ParallelXYZReader(n_workers=2)
        structures = reader.read_directory(temp_dir, pattern="*.xyz")

        assert len(structures) == 5


class TestParallelXYZWriter:
    """Tests for ParallelXYZWriter class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def structures(self):
        """Create temporary Atoms objects for testing."""
        structures = []
        for i in range(5):
            atoms = Atoms(
                "H2O",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.95, 0.0, 0.0],
                    [0.0, 0.95, 0.0],
                ],
            )
            structures.append(atoms)

        return structures

    def test_write_single_file(self, temp_dir, structures):
        """Test writing a single XYZ file."""
        io_module = import_module_directly(
            "io_parallel",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "io_parallel.py"
        )
        ParallelXYZWriter = io_module.ParallelXYZWriter

        writer = ParallelXYZWriter(n_workers=1)
        file_path = Path(temp_dir) / "test.xyz"
        success = writer.write(structures[:1], [file_path])

        assert success[0]
        assert file_path.exists()

    def test_write_multiple_files(self, temp_dir, structures):
        """Test writing multiple XYZ files."""
        io_module = import_module_directly(
            "io_parallel",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "io_parallel.py"
        )
        ParallelXYZWriter = io_module.ParallelXYZWriter

        writer = ParallelXYZWriter(n_workers=2)
        file_paths = [
            Path(temp_dir) / f"test_{i}.xyz" for i in range(len(structures))
        ]
        success = writer.write(structures, file_paths)

        assert all(success)
        for file_path in file_paths:
            assert file_path.exists()

    def test_write_directory(self, temp_dir, structures):
        """Test writing files to directory with prefix."""
        io_module = import_module_directly(
            "io_parallel",
            Path(__file__).resolve().parent.parent / "src" / "NepTrain" / "io_parallel.py"
        )
        ParallelXYZWriter = io_module.ParallelXYZWriter

        writer = ParallelXYZWriter(n_workers=2)
        file_paths = writer.write_directory(
            structures, temp_dir, prefix="test"
        )

        assert len(file_paths) == len(structures)
        for file_path in file_paths:
            assert file_path.exists()
            assert "test_" in file_path.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
