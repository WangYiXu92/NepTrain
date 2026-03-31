"""
Parallel XYZ File I/O for NepTrain

This module provides ParallelXYZReader and ParallelXYZWriter classes for
efficient reading and writing of XYZ files with parallel processing support.
"""

from __future__ import annotations

import gzip
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import read, write


class ParallelXYZReader:
    """
    Parallel XYZ file reader for large datasets.

    Uses multiprocessing to read multiple XYZ files in parallel.

    Args:
        n_workers: Number of worker processes
        chunk_size: Number of files per chunk
        verbose: Whether to show progress
    """

    def __init__(
        self,
        n_workers: int = 4,
        chunk_size: int = 100,
        verbose: bool = False,
    ):
        self.n_workers = n_workers
        self.chunk_size = chunk_size
        self.verbose = verbose

    def _read_single_file(
        self, file_path: Path, file_format: Optional[str] = None
    ) -> Optional[List[Atoms]]:
        """Read a single XYZ file."""
        try:
            if file_format == "xyz":
                atoms_list = read(file_path, index=":")
            elif file_format == "xyz.gz":
                with gzip.open(file_path, "rt") as f:
                    atoms_list = read(f, index=":")
            else:
                atoms_list = read(file_path, index=":")

            # Ensure it's a list
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]

            return atoms_list
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Failed to read {file_path}: {e}")
            return None

    def read(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        file_format: Optional[str] = None,
    ) -> List[Atoms]:
        """
        Read XYZ files in parallel.

        Args:
            file_paths: Single file path or list of file paths
            file_format: File format (xyz, xyz.gz, or None for auto-detect)

        Returns:
            List of Atoms objects
        """
        # Normalize file paths
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        file_paths = [Path(p) for p in file_paths]

        # Auto-detect format if not specified
        if file_format is None:
            file_format = self._detect_format(file_paths[0])

        # Read files in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(
                    self._read_single_file, file_path, file_format
                ): file_path
                for file_path in file_paths
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.extend(result)

        if self.verbose:
            print(f"Successfully read {len(results)} structures")

        return results

    def read_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.xyz",
        recursive: bool = False,
    ) -> List[Atoms]:
        """
        Read all XYZ files in a directory.

        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search recursively

        Returns:
            List of Atoms objects
        """
        directory = Path(directory)

        if recursive:
            file_paths = list(directory.rglob(pattern))
        else:
            file_paths = list(directory.glob(pattern))

        if self.verbose:
            print(f"Found {len(file_paths)} files matching {pattern}")

        return self.read(file_paths)

    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".gz":
            return "xyz.gz"
        elif suffix == ".xyz":
            return "xyz"
        else:
            return "xyz"


class ParallelXYZWriter:
    """
    Parallel XYZ file writer.

    Writes multiple XYZ files in parallel.

    Args:
        n_workers: Number of worker processes
        chunk_size: Number of files per chunk
        verbose: Whether to show progress
        compressor: Compression level (0-9) or None for no compression
    """

    def __init__(
        self,
        n_workers: int = 4,
        chunk_size: int = 100,
        verbose: bool = False,
        compressor: Optional[int] = None,
    ):
        self.n_workers = n_workers
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.compressor = compressor

    def _write_single_file(
        self,
        atoms: Atoms,
        file_path: Path,
        format: str = "xyz",
    ) -> bool:
        """Write a single Atoms object to XYZ file."""
        try:
            if self.compressor:
                # Compress with gzip
                with gzip.open(file_path, "wt") as f:
                    write(f, atoms, format=format)
            else:
                # Write normal file
                write(file_path, atoms, format=format)
            return True
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Failed to write {file_path}: {e}")
            return False

    def write(
        self,
        structures: List[Atoms],
        file_paths: Union[str, Path, List[Union[str, Path]]],
    ) -> List[bool]:
        """
        Write structures to XYZ files in parallel.

        Args:
            structures: List of Atoms objects
            file_paths: Single file path or list of file paths

        Returns:
            List of success flags
        """
        if len(structures) != len(file_paths):
            raise ValueError(
                f"Number of structures ({len(structures)}) "
                f"does not match number of file paths ({len(file_paths)})"
            )

        # Normalize file paths
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        file_paths = [Path(p) for p in file_paths]

        # Create parent directories
        for file_path in file_paths:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write files in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(
                    self._write_single_file, structure, file_path
                ): file_path
                for structure, file_path in zip(structures, file_paths)
            }

            for future in as_completed(futures):
                results.append(future.result())

        if self.verbose:
            success_count = sum(results)
            print(
                f"Successfully wrote {success_count}/{len(results)} files"
            )

        return results

    def write_directory(
        self,
        structures: List[Atoms],
        directory: Union[str, Path],
        prefix: str = "structure",
        start_index: int = 0,
    ) -> List[Path]:
        """
        Write structures to XYZ files in a directory.

        Args:
            structures: List of Atoms objects
            directory: Directory to write files
            prefix: Filename prefix
            start_index: Starting index for filenames

        Returns:
            List of written file paths
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        file_paths = [
            directory / f"{prefix}_{i + start_index}.xyz"
            for i in range(len(structures))
        ]

        success = self.write(structures, file_paths)

        if not all(success):
            warnings.warn("Some files failed to write")

        return [p for p, s in zip(file_paths, success) if s]

    def write_parametric(
        self,
        structures: List[Atoms],
        directory: Union[str, Path],
        prefix: str = "structure",
        start_index: int = 0,
        step: int = 1,
    ) -> List[Path]:
        """
        Write structures with parametric naming.

        Args:
            structures: List of Atoms objects
            directory: Directory to write files
            prefix: Filename prefix
            start_index: Starting index
            step: Step between indices

        Returns:
            List of written file paths
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        file_paths = [
            directory / f"{prefix}_{i * step + start_index}.xyz"
            for i in range(len(structures))
        ]

        success = self.write(structures, file_paths)

        if not all(success):
            warnings.warn("Some files failed to write")

        return [p for p, s in zip(file_paths, success) if s]


def read_xyz_parallel(
    file_paths: Union[str, Path, List[Union[str, Path]]],
    n_workers: int = 4,
    file_format: Optional[str] = None,
    verbose: bool = False,
) -> List[Atoms]:
    """
    Convenience function for parallel XYZ reading.

    Args:
        file_paths: Single file path or list of file paths
        n_workers: Number of worker processes
        file_format: File format (xyz, xyz.gz, or None for auto-detect)
        verbose: Whether to show progress

    Returns:
        List of Atoms objects
    """
    reader = ParallelXYZReader(
        n_workers=n_workers,
        chunk_size=100,
        verbose=verbose,
    )
    return reader.read(file_paths, file_format)


def write_xyz_parallel(
    structures: List[Atoms],
    file_paths: Union[str, Path, List[Union[str, Path]]],
    n_workers: int = 4,
    verbose: bool = False,
) -> List[bool]:
    """
    Convenience function for parallel XYZ writing.

    Args:
        structures: List of Atoms objects
        file_paths: Single file path or list of file paths
        n_workers: Number of worker processes
        verbose: Whether to show progress

    Returns:
        List of success flags
    """
    writer = ParallelXYZWriter(
        n_workers=n_workers,
        chunk_size=100,
        verbose=verbose,
    )
    return writer.write(structures, file_paths)
