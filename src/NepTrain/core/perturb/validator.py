"""
Input validation utilities for perturbation module.

This module provides comprehensive input validation for all perturbation functions,
including type checking, value range validation, and required parameter checks.
"""

import numpy as np
from ase import Atoms
from typing import List, Dict, Any, Optional, Union
from NepTrain.exceptions import ValidationError


def validate_atoms(atoms: Any, param_name: str = "atoms") -> Atoms:
    """
    Validate that input is an ASE Atoms object.
    
    Args:
        atoms: Input to validate
        param_name: Name of the parameter (for error messages)
        
    Returns:
        Atoms: Validated Atoms object
        
    Raises:
        TypeError: If input is not an Atoms object
    """
    if not isinstance(atoms, Atoms):
        raise ValidationError(
            f"{param_name} must be an ASE Atoms object, "
            f"got {type(atoms).__name__}"
        )
    return atoms


def validate_positive_integer(value: Any, param_name: str, 
                               min_value: int = 1) -> int:
    """
    Validate that input is a positive integer.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        min_value: Minimum allowed value
        
    Returns:
        int: Validated integer value
        
    Raises:
        TypeError: If value is not an integer
        ValidationError: If value is not positive or below minimum
    """
    if not isinstance(value, (int, np.integer)):
        raise ValidationError(
            f"{param_name} must be an integer, got {type(value).__name__}"
        )
    
    int_value = int(value)
    if int_value < min_value:
        raise ValidationError(
            f"{param_name} must be >= {min_value}, got {int_value}"
        )
    
    return int_value


def validate_positive_float(value: Any, param_name: str,
                            min_value: float = 0.0, 
                            max_value: Optional[float] = None) -> float:
    """
    Validate that input is a positive float.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value (optional)
        
    Returns:
        float: Validated float value
        
    Raises:
        TypeError: If value is not a number
        ValidationError: If value is not positive or outside range
    """
    if not isinstance(value, (int, float, np.number)):
        raise ValidationError(
            f"{param_name} must be a number, got {type(value).__name__}"
        )
    
    float_value = float(value)
    if float_value < min_value:
        raise ValidationError(
            f"{param_name} must be >= {min_value}, got {float_value}"
        )
    
    if max_value is not None and float_value > max_value:
        raise ValidationError(
            f"{param_name} must be <= {max_value}, got {float_value}"
        )
    
    return float_value


def validate_string(value: Any, param_name: str, 
                    allowed_values: Optional[List[str]] = None) -> str:
    """
    Validate that input is a string with optional allowed values.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        allowed_values: List of allowed string values (optional)
        
    Returns:
        str: Validated string
        
    Raises:
        TypeError: If value is not a string
        ValidationError: If value is not in allowed values
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{param_name} must be a string, got {type(value).__name__}"
        )
    
    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(
            f"{param_name} must be one of {allowed_values}, got '{value}'"
        )
    
    return value


def validate_list_of_atoms(value: Any, param_name: str) -> List[Atoms]:
    """
    Validate that input is a list of Atoms objects.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        
    Returns:
        list[Atoms]: Validated list of Atoms objects
        
    Raises:
        TypeError: If value is not a list or contains non-Atoms elements
    """
    if not isinstance(value, list):
        raise ValidationError(
            f"{param_name} must be a list, got {type(value).__name__}"
        )
    
    for i, item in enumerate(value):
        if not isinstance(item, Atoms):
            raise ValidationError(
                f"{param_name}[{i}] must be an ASE Atoms object, "
                f"got {type(item).__name__}"
            )
    
    return value


def validate_ratio(value: Any, param_name: str) -> float:
    """
    Validate that input is a ratio (0 <= value <= 1).
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        
    Returns:
        float: Validated ratio value
        
    Raises:
        ValidationError: If value is not in [0, 1]
    """
    float_value = validate_positive_float(value, param_name, 
                                           min_value=0.0, max_value=1.0)
    return float_value


def validate_bool(value: Any, param_name: str) -> bool:
    """
    Validate that input is a boolean.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        
    Returns:
        bool: Validated boolean value
        
    Raises:
        TypeError: If value is not a boolean
    """
    if not isinstance(value, bool):
        raise ValidationError(
            f"{param_name} must be a boolean, got {type(value).__name__}"
        )
    return value


def validate_indices(value: Any, param_name: str, max_index: Optional[int] = None) -> List[int]:
    """
    Validate atom indices.
    
    Args:
        value: Value to validate (int, list of ints, or string)
        param_name: Name of the parameter (for error messages)
        max_index: Maximum allowed index (optional, for length checking)
        
    Returns:
        list[int]: Validated list of indices
        
    Raises:
        ValidationError: If indices are invalid
    """
    if isinstance(value, str):
        # Parse string like "1,2,3" or "0:10"
        if ':' in value:
            try:
                parts = value.split(':')
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else max_index
                step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
                value = list(range(start, end, step))
            except (ValueError, IndexError) as e:
                raise ValidationError(
                    f"{param_name} invalid format: '{value}'"
                ) from e
        else:
            try:
                value = [int(x.strip()) for x in value.split(',')]
            except ValueError as e:
                raise ValidationError(
                    f"{param_name} must be comma-separated integers, got '{value}'"
                ) from e
    elif isinstance(value, int):
        value = [value]
    elif not isinstance(value, (list, tuple)):
        raise ValidationError(
            f"{param_name} must be int, list of ints, or string, "
            f"got {type(value).__name__}"
        )
    
    # Validate indices are positive integers
    for i, idx in enumerate(value):
        if not isinstance(idx, (int, np.integer)):
            raise ValidationError(
                f"{param_name}[{i}] must be an integer, "
                f"got {type(idx).__name__}"
            )
        if idx < 0:
            raise ValidationError(
                f"{param_name}[{i}] must be >= 0, got {idx}"
            )
        if max_index is not None and idx >= max_index:
            raise ValidationError(
                f"{param_name}[{i}] must be < {max_index}, got {idx}"
            )
    
    return list(value)


def validate_vector(value: Any, param_name: str, 
                    expected_length: Optional[int] = None) -> List[float]:
    """
    Validate vector input.
    
    Args:
        value: Value to validate (list, tuple, string, or numpy array)
        param_name: Name of the parameter (for error messages)
        expected_length: Expected vector length (optional)
        
    Returns:
        list[float]: Validated vector
        
    Raises:
        ValidationError: If vector is invalid
    """
    if isinstance(value, str):
        try:
            value = [float(x.strip()) for x in value.split(',')]
        except ValueError as e:
            raise ValidationError(
                f"{param_name} must be comma-separated numbers, got '{value}'"
            ) from e
    elif isinstance(value, (list, tuple)):
        value = list(value)
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    else:
        raise ValidationError(
            f"{param_name} must be list, tuple, string, or numpy array, "
            f"got {type(value).__name__}"
        )
    
    # Validate all elements are numbers
    for i, elem in enumerate(value):
        if not isinstance(elem, (int, float, np.number)):
            raise ValidationError(
                f"{param_name}[{i}] must be a number, got {type(elem).__name__}"
            )
        value[i] = float(elem)
    
    # Validate length if specified
    if expected_length is not None and len(value) != expected_length:
        raise ValidationError(
            f"{param_name} must have length {expected_length}, "
            f"got {len(value)}"
        )
    
    return value


def validate_dict(value: Any, param_name: str,
                  required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate dictionary input.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        required_keys: List of required keys (optional)
        
    Returns:
        dict: Validated dictionary
        
    Raises:
        TypeError: If value is not a dict
        ValidationError: If required keys are missing
    """
    if not isinstance(value, dict):
        raise ValidationError(
            f"{param_name} must be a dict, got {type(value).__name__}"
        )
    
    if required_keys is not None:
        missing_keys = set(required_keys) - set(value.keys())
        if missing_keys:
            raise ValidationError(
                f"{param_name} missing required keys: {missing_keys}"
            )
    
    return value


def validate_sampler(value: Any, param_name: str = "sampler") -> str:
    """
    Validate sampler type.
    
    Args:
        value: Sampler type
        param_name: Name of the parameter (for error messages)
        
    Returns:
        str: Validated sampler type
        
    Raises:
        ValidationError: If sampler type is invalid
    """
    valid_samplers = ['random', 'sobol']
    return validate_string(value, param_name, allowed_values=valid_samplers)


def validate_perturbation_type(value: Any, param_name: str = "perturbation_type") -> str:
    """
    Validate perturbation type.
    
    Args:
        value: Perturbation type
        param_name: Name of the parameter (for error messages)
        
    Returns:
        str: Validated perturbation type
        
    Raises:
        ValidationError: If perturbation type is invalid
    """
    valid_types = [
        'interstitial', 'twinning', 'vacancy', 'magnetic',
        'surface', 'grain_boundary', 'dislocation', 
        'stacking_fault', 'amorphous', 'rotation', 'shuffle', 'volume'
    ]
    return validate_string(value, param_name, allowed_values=valid_types)


def validate_magnetic_mode(value: Any, param_name: str = "mag_mode") -> Optional[str]:
    """
    Validate magnetic perturbation mode.
    
    Args:
        value: Magnetic mode
        param_name: Name of the parameter (for error messages)
        
    Returns:
        str or None: Validated magnetic mode (None if no magnetism)
        
    Raises:
        ValidationError: If magnetic mode is invalid
    """
    if value is None:
        return None
    
    valid_modes = [
        'flip', 'rotate', 'noise', 'fixed', 'adapt', 
        'flip_rotate', 'all'
    ]
    return validate_string(value, param_name, allowed_values=valid_modes)
