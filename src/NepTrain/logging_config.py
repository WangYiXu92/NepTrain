"""
Logging configuration for NepTrain.

This module sets up customized logging with both console and file output.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_level=logging.INFO,
    log_file=None,
    log_dir=None,
    module_name=None
):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Specific log file path (optional)
        log_dir: Directory for log files (default: current directory)
        module_name: Name of the module using logging
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(module_name or __name__)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_logger(module_name=None):
    """
    Get a logger with default configuration.
    
    Args:
        module_name: Name of the module using logging
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logging(module_name=module_name)


# Convenience function for modules to use
def get_logger(name):
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
