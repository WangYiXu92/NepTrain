"""Configuration validation for NepTrain using Pydantic.

This module provides Pydantic-based validation for NepTrain YAML configuration files,
ensuring all required fields are present and have correct types.

Usage:
    from NepTrain.validation import TrainConfig, validate_config
    
    # Load and validate config
    config_dict = YAML().load(config_path)
    config = validate_config(config_dict)
    
    # Access validated config
    print(config.gpumd.temperature_every_step)
"""

from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.networks import AnyUrl
import yaml


class GLobalConfig(BaseModel):
    """Global configuration for NepTrain."""
    work_path: str = Field(..., description="Working directory for calculations")
    current_job: str = Field(..., description="Current job type: nep, gpumd, dft, select, pred")
    generation: int = Field(1, ge=1, description="Generation number")
    dft_job: int = Field(1, ge=1, description="Number of DFT jobs")
    
    class Config:
        extra = 'allow'  # Allow additional fields for flexibility


class GPUMDConfig(BaseModel):
    """Configuration for GPUMD calculations."""
    model_path: str = Field(..., description="Path to the NEP potential file")
    run_in_path: str = Field("./run.in", description="Path to run.in template")
    step_times: list[int] = Field(..., description="List of step times in ps for each generation")
    temperature_every_step: list[float] = Field(..., description="Temperatures for each generation")
    max_steps: int = Field(100000, ge=1, description="Maximum number of MD steps")
    time_step: float = Field(1.0, ge=0.1, description="Time step in fs")
    
    @field_validator('step_times')
    @classmethod
    def validate_step_times(cls, v):
        if not v:
            raise ValueError("step_times cannot be empty")
        if any(t <= 0 for t in v):
            raise ValueError("All step times must be positive")
        return v
    
    @field_validator('temperature_every_step')
    @classmethod
    def validate_temperatures(cls, v):
        if not v:
            raise ValueError("temperature_every_step cannot be empty")
        if any(t < 0 for t in v):
            raise ValueError("All temperatures must be non-negative")
        return v


class NEPConfig(BaseModel):
    """Configuration for NEP training."""
    nep_in_path: str = Field(..., description="Path to nep.in template")
    test: str = Field(..., description="Path to test data")
    nep_restart: bool = Field(False, description="Whether to restart from previous training")
    nep_restart_step: int = Field(0, ge=0, description="Step to restart from")
    
    class Config:
        extra = 'allow'


class SelectConfig(BaseModel):
    """Configuration for structure selection."""
    max_selected: int = Field(100, ge=1, description="Maximum number of structures to select")
    min_distance: float = Field(2.0, ge=0.1, description="Minimum interatomic distance")
    filter: Optional[float] = Field(None, ge=0.0, le=1.0, description="Filter threshold")
    
    class Config:
        extra = 'allow'


class DFTConfig(BaseModel):
    """Configuration for DFT calculations."""
    software: str = Field("vasp", description="DFT software: vasp, abacus, qe")
    incar_path: str = Field("auto", description="Path to INCAR or 'auto' for default")
    cpu_core: int = Field(8, ge=1, description="Number of CPU cores")
    kpoints_use_gamma: bool = Field(False, description="Use gamma-centered k-points")
    use_k_stype: str = Field("kpoints", description="K-points specification: kpoints or kspacing")
    kpoints: Optional[list[int]] = Field(None, description="K-point mesh [kx, ky, kz]")
    kspacing: Optional[float] = Field(None, ge=0.01, description="K-point spacing in 1/A")
    
    @model_validator(mode='after')
    def check_kpoints_config(self):
        if self.use_k_stype == "kpoints":
            if not self.kpoints:
                raise ValueError("kpoints must be specified when use_k_stype='kpoints'")
        elif self.use_k_stype == "kspacing":
            if not self.kspacing:
                raise ValueError("kspacing must be specified when use_k_stype='kspacing'")
        return self


class MachineConfig(BaseModel):
    """Configuration for HPC machine."""
    context_type: str = Field("local", description="Machine context type")
    local_root: str = Field("./", description="Local root directory")
    remote_root: str = Field("./", description="Remote root directory")
    batch_type: str = Field("slurm", description="Batch system type")


class ResourcesConfig(BaseModel):
    """Configuration for compute resources."""
    number_nodes: int = Field(1, ge=1, description="Number of nodes")
    cpu_per_node: int = Field(24, ge=1, description="CPUs per node")
    gpu_per_node: int = Field(0, ge=0, description="GPUs per node")
    memory_per_node: str = Field("128GB", description="Memory per node")


class TrainConfig(BaseModel):
    """Complete NepTrain configuration."""
    # Global settings
    work_path: str = Field(..., description="Working directory")
    current_job: str = Field(..., description="Current job type")
    generation: int = Field(1, ge=1)
    dft_job: int = Field(1, ge=1)
    init_train_xyz: str = Field(..., description="Initial training data")
    init_nep_txt: Optional[str] = Field(None, description="Initial NEP potential (optional)")
    
    # Job configurations
    gpumd: GPUMDConfig = Field(..., description="GPUMD configuration")
    nep: NEPConfig = Field(..., description="NEP training configuration")
    select: SelectConfig = Field(..., description="Structure selection configuration")
    dft: DFTConfig = Field(..., description="DFT calculation configuration")
    
    # Machine configurations (optional)
    gpumd_machine: Optional[MachineConfig] = Field(None, description="GPUMD machine config")
    gpumd_resources: Optional[ResourcesConfig] = Field(None, description="GPUMD resources config")
    nep_machine: Optional[MachineConfig] = Field(None, description="NEP machine config")
    nep_resources: Optional[ResourcesConfig] = Field(None, description="NEP resources config")
    select_machine: Optional[MachineConfig] = Field(None, description="Select machine config")
    select_resources: Optional[ResourcesConfig] = Field(None, description="Select resources config")
    dft_machine: Optional[MachineConfig] = Field(None, description="DFT machine config")
    dft_resources: Optional[ResourcesConfig] = Field(None, description="DFT resources config")
    
    # Validation
    @field_validator('current_job')
    @classmethod
    def validate_current_job(cls, v):
        valid_jobs = ['nep', 'gpumd', 'dft', 'select', 'pred']
        if v not in valid_jobs:
            raise ValueError(f"current_job must be one of {valid_jobs}")
        return v
    
    @model_validator(mode='after')
    def check_restart_conditions(self):
        """Check restart-specific requirements."""
        if self.current_job == 'gpumd' and not self.init_nep_txt:
            raise ValueError("Restarting as gpumd requires init_nep_txt to be set")
        return self


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    file_path = Path(config_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {file_path}: {e}")


def validate_config(config_dict: dict) -> TrainConfig:
    """
    Validate configuration dictionary using Pydantic.
    
    Args:
        config_dict: Configuration dictionary from YAML
        
    Returns:
        Validated TrainConfig object
        
    Raises:
        ValueError: If validation fails
        TypeError: If types don't match
    """
    try:
        return TrainConfig(**config_dict)
    except Exception as e:
        # Provide clear error messages
        error_msg = f"Configuration validation failed:\n{e}"
        raise ValueError(error_msg) from e


def validate_config_file(config_path: str) -> TrainConfig:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated TrainConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If validation fails
    """
    config_dict = load_config(config_path)
    return validate_config(config_dict)
