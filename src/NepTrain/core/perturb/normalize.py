#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Normalization utilities for perturbation parameters."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np


def normalize_int_list(val: Any, default_if_random: Optional[List[int]] = None) -> Union[List[int], Any]:
    """Normalize a value to a list of integers.
    
    Handles 'random', comma-separated strings, and pass-through.
    """
    if val == 'random':
        return default_if_random
    if isinstance(val, str):
        return [int(x) for x in val.split(',')]
    return val


def normalize_float_list(val: Any, default_if_random: Optional[List[float]] = None) -> Union[List[float], Any]:
    """Normalize a value to a list of floats.
    
    Handles 'random', comma-separated strings, and pass-through.
    """
    if val == 'random':
        return default_if_random
    if isinstance(val, str):
        return [float(x) for x in val.split(',')]
    return val


def normalize_axis_index(val: Union[int, str, List[float], Tuple[float, ...], np.ndarray]) -> int:
    """
    Normalize an axis input (index or vector) to an integer index (0, 1, 2).
    
    Examples:
        '0,0,1' -> 2
        'x' -> 0
        [1, 0, 0] -> 0
    """
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, str):
        val_lower = val.lower().strip()
        if val_lower in ['x', '0']:
            return 0
        if val_lower in ['y', '1']:
            return 1
        if val_lower in ['z', '2']:
            return 2
        try:
            parts = [float(x) for x in val.split(',')]
            return int(np.argmax(np.abs(parts)))
        except Exception:
            return val  # Maybe it's 'random' or invalid
    elif isinstance(val, (list, tuple, np.ndarray)):
        return int(np.argmax(np.abs(val)))
    return val


def normalize_vector_norm(val: Union[str, List[float], Tuple[float, ...], np.ndarray]) -> float:
    """
    Normalize a vector input to its norm (float).
    
    Examples:
        [1, 0, 0] -> 1.0
        '1,0,0' -> 1.0
    """
    if isinstance(val, str):
        try:
            vec = [float(x) for x in val.split(',')]
            return float(np.linalg.norm(vec))
        except Exception:
            return val
    elif isinstance(val, (list, tuple, np.ndarray)):
        return float(np.linalg.norm(val))
    return val


def parse_range(val: Any, type_func: type = float) -> Union[Tuple[float, float, bool], Tuple[Any, Any, bool]]:
    """
    Parse a value which could be a scalar or a range string/list.
    
    Returns:
        (min, max, is_range) tuple.
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
