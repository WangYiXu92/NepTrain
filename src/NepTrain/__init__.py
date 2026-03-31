#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 16:22
# @Author  : 兵
# @email    : 1747193328@qq.com
import configparser
import logging
import os
import shutil
from typing import Dict, Any

from watchdog.observers import Observer

from NepTrain import utils
from NepTrain.exceptions import (
    NepTrainError,
    ConfigurationError,
    ValidationError,
    FileOperationError,
    CalculationError,
    ParallelizationError,
    ConvergenceError,
    ResourceError,
    # GPUMD-specific exceptions
    GPUMDExecutionError,
    GPUMDOutputError,
    NEPParsingError,
    NEPExecutionError,
    FileFormatError,
    ConcentrationError,
    TemperatureError,
    EmptyInputError,
    ParameterRangeError,
    ConcurrentAccessError,
    TempFileError,
    # Perturbation-specific exceptions
    PerturbationError,
    DimensionMismatchError,
    AtomicOverlapError,
    TopologyError,
    MemoryLimitError,
)
from NepTrain.logging_config import setup_logging, get_logger, get_default_logger

# Set up package-level logging
logger = get_logger(__name__)

# Configure logging for package
setup_logging(log_level=logging.INFO)


from importlib.metadata import version

__version__ = version("NepTrain")
config_path = utils.get_config_path()

module_path = os.path.dirname(__file__)

if not os.path.exists(config_path)  :
    shutil.copy(os.path.join(module_path,"config.ini"), config_path)

Config = configparser.RawConfigParser()
Config.read(config_path,encoding="utf8")


__all__ = [
    'NepTrainError',
    'ConfigurationError',
    'ValidationError',
    'FileOperationError',
    'CalculationError',
    'ParallelizationError',
    'ConvergenceError',
    'ResourceError',
    # GPUMD-specific exceptions
    'GPUMDExecutionError',
    'GPUMDOutputError',
    'NEPParsingError',
    'NEPExecutionError',
    'FileFormatError',
    'ConcentrationError',
    'TemperatureError',
    'EmptyInputError',
    'ParameterRangeError',
    'ConcurrentAccessError',
    'TempFileError',
    # Perturbation-specific exceptions
    'PerturbationError',
    'DimensionMismatchError',
    'AtomicOverlapError',
    'TopologyError',
    'MemoryLimitError',
]


#
# if platform.is_linux():
#     # wsl测试默认的有问题  强行切换到poll机制
#     observer = PollingObserver()
# else:
#     observer = Observer()


observer = Observer()
