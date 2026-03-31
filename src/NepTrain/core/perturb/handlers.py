"""
Defect Handler Registry System

This module provides a clean, extensible architecture for perturbation generation.
Each defect type is handled by a dedicated handler class that encapsulates:
- Dimension calculation (for Sobol sampling)
- Parameter parsing and validation
- Perturbation application
- Metadata generation
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
from ase import Atoms


class DefectHandler(ABC):
    """Base class for all defect perturbation handlers"""
    
    @abstractmethod
    def get_dims(self, atoms: Atoms, **params) -> int:
        """
        Calculate number of Sobol dimensions needed for this defect.
        
        Args:
            atoms: Input structure (may be modified by previous defects)
            **params: Defect-specific parameters
            
        Returns:
            Number of dimensions required
        """
        pass
    
    @abstractmethod
    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        """
        Apply the perturbation to the structure.
        
        Args:
            atoms: Input structure
            samples: Sobol samples for this defect (length = get_dims())
            **params: Defect-specific parameters
            
        Returns:
            Perturbed structure
        """
        pass
    
    def get_metadata(self, **params) -> dict:
        """
        Generate metadata for annotation (optional).
        
        Args:
            **params: Parameters used during application
            
        Returns:
            Metadata dictionary
        """
        return {}


def _is_range(val) -> bool:
    """Check if a value represents a range"""
    if isinstance(val, str) and ',' in val:
        return True
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return True
    return False


def _parse_range(val, type_func=float) -> Tuple[Any, Any, bool]:
    """
    Parse a value which could be a scalar or a range.
    Returns (min, max, is_range).
    """
    if isinstance(val, str) and ',' in val:
        parts = val.split(',')
        return type_func(parts[0]), type_func(parts[1]), True
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return type_func(val[0]), type_func(val[1]), True
    
    # Handle non-numeric strings ('csl', 'random', etc)
    try:
        f_val = type_func(val)
        return f_val, f_val, False
    except (ValueError, TypeError):
        return val, val, False


def _interpolate(sample: float, min_val: float, max_val: float) -> float:
    """Interpolate a [0,1] sample to [min_val, max_val]"""
    return min_val + sample * (max_val - min_val)


def _discrete_sample(sample: float, min_val: int, max_val: int) -> int:
    """Sample a discrete integer from [min_val, max_val] using a [0,1] sample"""
    n_opts = max_val - min_val + 1
    int_off = int(sample * n_opts)
    if int_off == n_opts:
        int_off -= 1
    return min_val + int_off


# Registry of all defect handlers
DEFECT_HANDLERS = {}

# Execution order for defects (topology-modifying first)
DEFECT_ORDER = [
    'surface',
    'grain_boundary', 
    'dislocation',
    'twinning',
    'stacking_fault',
    'amorphous',
    'magnetic',
    'rotation',
    'vacancy',
    'shuffle',
    'volume'
]


def register_handler(name: str):
    """Decorator to register a handler in the global registry"""
    def decorator(handler_class):
        DEFECT_HANDLERS[name] = handler_class()
        return handler_class
    return decorator
