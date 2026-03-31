"""
Custom exception classes for NepTrain.

This module defines a hierarchy of custom exceptions that provide
more specific error handling and debugging information.

All exceptions inherit from NepTrainError base class for consistent 
error handling across the package.
"""

from typing import Dict, Any, Optional


class NepTrainError(Exception):
    """Base exception for all NepTrain-related errors."""
    
    def __init__(self, message: str, *args, **kwargs):
        """Initialize NepTrainError.
        
        Args:
            message: Error message
            *args: Additional arguments for Exception
            **kwargs: Additional keyword arguments
        """
        super().__init__(message, *args)
        self.message = message
        self.details = kwargs.get('details', {})


class ConfigurationError(NepTrainError):
    """Raised when there is an issue with configuration."""
    pass


class ValidationError(NepTrainError):
    """Raised when input validation fails."""
    pass


class FileOperationError(NepTrainError):
    """Raised when file operations fail."""
    
    def __init__(self, message: str, path: str = None, *args, **kwargs):
        """Initialize FileOperationError.
        
        Args:
            message: Error message
            path: File path that caused the error
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        if path:
            message = f"[{path}] {message}"
        super().__init__(message, *args, **kwargs)
        self.path = path


class CalculationError(NepTrainError):
    """Raised when calculation fails."""
    pass


class ParallelizationError(NepTrainError):
    """Raised when parallelization-related errors occur."""
    pass


class ConvergenceError(NepTrainError):
    """Raised when calculation does not converge."""
    pass


class ResourceError(NepTrainError):
    """Raised when resource limits are exceeded."""
    pass


# ============================================================================
# GPUMD-Specific Exceptions
# ============================================================================


class GPUMDExecutionError(CalculationError):
    """Raised when GPUMD calculation fails."""
    
    def __init__(self, message: str, return_code: int = None, 
                 error_file: str = None, *args, **kwargs):
        """Initialize GPUMDExecutionError.
        
        Args:
            message: Error message
            return_code: Return code from GPUMD process
            error_file: Path to error output file
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        full_msg = message
        if return_code is not None:
            full_msg += f" (return_code={return_code})"
        if error_file:
            full_msg += f" (error_file={error_file})"
        super().__init__(full_msg, *args, **kwargs)
        self.return_code = return_code
        self.error_file = error_file


class GPUMDOutputError(CalculationError):
    """Raised when GPUMD output file validation fails."""
    
    def __init__(self, message: str, expected_file: str = None, 
                 *args, **kwargs):
        """Initialize GPUMDOutputError.
        
        Args:
            message: Error message
            expected_file: Expected output file path
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        if expected_file:
            message = f"[{expected_file}] {message}"
        super().__init__(message, *args, **kwargs)
        self.expected_file = expected_file


# ============================================================================
# NEP-Specific Exceptions
# ============================================================================


class NEPParsingError(CalculationError):
    """Raised when NEP file parsing fails."""
    pass


class NEPExecutionError(CalculationError):
    """Raised when NEP calculation fails."""
    pass


# ============================================================================
# File Format Exceptions
# ============================================================================


class FileFormatError(ValidationError):
    """Raised when file format validation fails."""
    
    def __init__(self, message: str, file_path: str = None, 
                 expected_format: str = None, *args, **kwargs):
        """Initialize FileFormatError.
        
        Args:
            message: Error message
            file_path: Path to the file
            expected_format: Expected file format
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        full_msg = message
        if file_path:
            full_msg = f"[{file_path}] {full_msg}"
        if expected_format:
            full_msg += f" (expected: {expected_format})"
        super().__init__(full_msg, *args, **kwargs)
        self.file_path = file_path
        self.expected_format = expected_format


# ============================================================================
# Concentration/Simulation Parameters Exceptions
# ============================================================================


class ConcentrationError(ValidationError):
    """Raised when concentration parameters are invalid."""
    
    def __init__(self, message: str, concentration: Dict[str, float] = None,
                 *args, **kwargs):
        """Initialize ConcentrationError.
        
        Args:
            message: Error message
            concentration: Dictionary of element concentrations
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        if concentration:
            message += f" (concentrations: {concentration})"
        super().__init__(message, *args, **kwargs)
        self.concentration = concentration


class TemperatureError(ValidationError):
    """Raised when temperature parameters are invalid."""
    
    def __init__(self, message: str, temperature: float = None,
                 *args, **kwargs):
        """Initialize TemperatureError.
        
        Args:
            message: Error message
            temperature: Invalid temperature value
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        if temperature is not None:
            message += f" (temperature={temperature}K)"
        super().__init__(message, *args, **kwargs)
        self.temperature = temperature


# ============================================================================
# Input Validation Exceptions
# ============================================================================


class EmptyInputError(ValidationError):
    """Raised when input is empty or None."""
    
    def __init__(self, message: str, input_name: str = None, *args, **kwargs):
        """Initialize EmptyInputError.
        
        Args:
            message: Error message
            input_name: Name of the empty input
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        if input_name:
            message = f"[{input_name}] {message}"
        super().__init__(message, *args, **kwargs)
        self.input_name = input_name


class ParameterRangeError(ValidationError):
    """Raised when parameter value is outside allowed range."""
    
    def __init__(self, message: str, parameter: str = None, 
                 value: Any = None, min_val: float = None, 
                 max_val: float = None, *args, **kwargs):
        """Initialize ParameterRangeError.
        
        Args:
            message: Error message
            parameter: Name of the parameter
            value: Invalid value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        full_msg = message
        if parameter:
            full_msg = f"[{parameter}] {full_msg}"
        if value is not None:
            full_msg += f" (value={value})"
        if min_val is not None or max_val is not None:
            range_str = f"[{min_val}, {max_val}]" if min_val is not None and max_val is not None else f">={min_val}" if min_val is not None else f"<={max_val}"
            full_msg += f" (allowed: {range_str})"
        super().__init__(full_msg, *args, **kwargs)
        self.parameter = parameter
        self.value = value
        self.min_val = min_val
        self.max_val = max_val


# ============================================================================
# Concurrent Execution Exceptions
# ============================================================================


class ConcurrentAccessError(ParallelizationError):
    """Raised when concurrent access to shared resources fails."""
    
    def __init__(self, message: str, resource: str = None, 
                 *args, **kwargs):
        """Initialize ConcurrentAccessError.
        
        Args:
            message: Error message
            resource: Name of the resource
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        if resource:
            message = f"[{resource}] {message}"
        super().__init__(message, *args, **kwargs)
        self.resource = resource


# ============================================================================
# Temp File Management Exceptions
# ============================================================================


class TempFileError(FileOperationError):
    """Raised when temporary file operations fail."""
    
    def __init__(self, message: str, temp_file: str = None, 
                 *args, **kwargs):
        """Initialize TempFileError.
        
        Args:
            message: Error message
            temp_file: Path to temporary file
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(message, temp_file, *args, **kwargs)
        self.temp_file = temp_file
