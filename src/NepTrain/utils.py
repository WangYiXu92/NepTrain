"""
Common utilities for NepTrain.

This module provides shared utilities across the package.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def ensure_directory(path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory
        
    Returns:
        Path: Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_file_exists(path, must_exist=True):
    """
    Check if a file exists and optionally raise an error.
    
    Args:
        path: Path to file
        must_exist: If True, raise exception when file doesn't exist
        
    Returns:
        bool: True if file exists (when must_exist=False)
        
    Raises:
        FileNotFoundError: When must_exist=True and file doesn't exist
    """
    path = Path(path)
    exists = path.exists()
    
    if must_exist and not exists:
        raise FileNotFoundError(f"File not found: {path}")
    
    return exists


def format_size(bytes_size):
    """
    Format byte size to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        str: Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_cpu_count():
    """
    Get the number of available CPU cores.
    
    Returns:
        int: Number of CPU cores
    """
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def get_config_path():
    """
    Get the path to the configuration file.
    
    The config file is searched in the following order:
    1. Current working directory
    2. User's home directory
    3. Package directory
    
    Returns:
        str: Path to the configuration file
    """
    config_name = "config.ini"
    
    # Check current working directory
    cwd_config = os.path.join(os.getcwd(), config_name)
    if os.path.exists(cwd_config):
        return cwd_config
    
    # Check user home directory
    home_config = os.path.expanduser(f"~/{config_name}")
    if os.path.exists(home_config):
        return home_config
    
    # Default to package directory
    return cwd_config


# ============================================================================
# Logging Utility Functions
# ============================================================================


def _get_logger() -> logging.Logger:
    """Get the default logger for NepTrain.
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger("NepTrain")


def print_msg(*args, level: int = logging.INFO) -> None:
    """
    Print a message using the logging system.
    
    Args:
        *args: Message components to join
        level: Logging level (default: INFO)
    """
    logger = _get_logger()
    message = " ".join(str(arg) for arg in args)
    logger.log(level, message)


def print_success(message: str) -> None:
    """
    Print a success message.
    
    Args:
        message: Success message to print
    """
    _get_logger().info(f"✅ {message}")


def print_warning(message: str) -> None:
    """
    Print a warning message.
    
    Args:
        message: Warning message to print
    """
    _get_logger().warning(f"⚠️ {message}")


def print_tip(message: str) -> None:
    """
    Print a tip message.
    
    Args:
        message: Tip message to print
    """
    _get_logger().info(f"💡 {message}")


def print_error(message: str) -> None:
    """
    Print an error message.
    
    Args:
        message: Error message to print
    """
    _get_logger().error(f"❌ {message}")


def set_log_level(level: int) -> None:
    """
    Set the global logging level.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logging.getLogger("NepTrain").setLevel(level)


def iter_path_to_atoms(patterns, show_progress=True, **kwargs):
    """
    Decorator to process multiple files and yield atoms objects.
    
    Args:
        patterns: List of file patterns to match
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments (e.g., description for rich.progress)
        
    Returns:
        Decorator function
    """
    from functools import wraps
    try:
        from rich.progress import Progress
    except ImportError:
        Progress = None
    
    def decorator(func):
        @wraps(func)
        def wrapper(path, *args, **kwargs):
            import os
            from ase.io import read as ase_read
            
            if os.path.isdir(path):
                import glob
                files = []
                for pattern in patterns:
                    files.extend(glob.glob(os.path.join(path, pattern)))
                files = list(set(files))  # Remove duplicates
            else:
                files = [path]
            
            for file_path in files:
                if not os.path.exists(file_path):
                    continue
                    
                atoms = ase_read(file_path)
                
                # Call the function with atoms
                result = func(atoms, *args, **kwargs)
                yield result
                
        return wrapper
    return decorator
