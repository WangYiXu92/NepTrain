#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PerturbConfig: Pydantic-based configuration model for perturb()."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms


# Minimal field descriptor (no pydantic dependency required)
class _Field:
    """Simple field descriptor with default and validator."""
    __slots__ = ('default', 'validator', 'name')

    def __init__(self, default=None, validator=None):
        self.default = default
        self.validator = validator
        self.name = None


class PerturbConfig:
    """
    Configuration container for perturbation parameters.
    
    Can be constructed from keyword arguments matching the old ``perturb()`` signature.
    Provides attribute access and serialization helpers.
    """

    # Generation
    num: int = 20
    cell_pert_fraction: float = 0.03
    min_distance: float = 0.1
    sampler: str = 'random'
    scramble: bool = True
    skip_normal: bool = False
    seed: Optional[int] = None
    state_file: Optional[str] = None
    resume: bool = False

    # Topology: Surface
    surface: bool = False
    surface_indices: Any = '1,1,1'
    surface_vacuum: Any = 10.0
    surface_layers: Any = 3

    # Topology: Grain Boundary
    gb: bool = False
    gb_axis: Any = '0,0,1'
    gb_angle: Any = 36.87
    gb_dist: Any = 0.0
    gb_overlap_dist: Any = 1.2
    gb_delete_overlap: bool = True

    # Topology: Dislocation
    dislocation: bool = False
    dislocation_type: str = 'edge'
    dislocation_axis: Any = '0,0,1'
    dislocation_burgers: Any = '1,0,0'

    # Topology: Twinning
    twinning: bool = False
    twinning_indices: Any = (1, 1, 2)
    twinning_z: Any = 0.5
    twinning_min_dist: float = 1.5

    # Topology: Stacking Fault
    stacking_fault: bool = False
    sf_normal: Any = '1,1,1'
    sf_shift: Any = '0,0,0'
    sf_height: Any = 0.5
    sf_min_dist: float = 1.5

    # Topology: Amorphous
    amorphous: bool = False
    amorphous_min_dist: float = 1.5
    amorphous_rattle: float = 0.1
    amorphous_steps: int = 100

    # Magnetic
    mag_mode: Optional[str] = None
    mag_noise: float = 0.0
    mag_kwargs: Optional[dict] = None

    # Rotation
    rotate_formula: Optional[str] = None

    # Vacancy
    vac_elements: Optional[Any] = None
    vac_num: int = 0

    # Antisite
    antisite: bool = False
    antisite_pairs: Optional[Any] = None   # 'Fe,Al' or [('Fe','Al')]
    antisite_num: int = 1
    antisite_mode: str = 'symmetry_aware'
    antisite_symprec: float = 1e-2

    # Shuffle
    shuffle_elements: Optional[Any] = None
    shuffle_method: str = 'fisher_yates'

    # Rigid body
    rigid: bool = False
    rigid_method: str = 'auto'
    rigid_list: Optional[Any] = None
    rigid_mode: str = 'inter'
    rigid_composition: Optional[Any] = None

    # Cell rotation
    rotate_cell: bool = False

    # Symmetry-preserving strain
    sym_strain: bool = False
    sym_strain_fraction: float = 0.03
    sym_strain_crystal_system: Optional[str] = None  # None = auto-detect
    sym_strain_symprec: float = 1e-2

    # Volume
    vol_pert_fraction: float = 0.0

    # Filtering & validation
    filter_bonds: bool = False
    validate_structure: bool = True
    validate_coefficient: Optional[float] = None
    similarity_threshold: float = 0.999
    debug_plot: bool = False

    def __init__(self, **kwargs):
        # Set defaults from class attributes, then override with kwargs
        _defaults = self._get_defaults()
        _defaults.update(kwargs)
        for k, v in _defaults.items():
            setattr(self, k, v)

    @classmethod
    def _get_defaults(cls) -> dict:
        return {
            'num': 20, 'cell_pert_fraction': 0.03, 'min_distance': 0.1,
            'sampler': 'random', 'scramble': True, 'skip_normal': False,
            'seed': None, 'state_file': None, 'resume': False,
            'surface': False, 'surface_indices': '1,1,1',
            'surface_vacuum': 10.0, 'surface_layers': 3,
            'gb': False, 'gb_axis': '0,0,1', 'gb_angle': 36.87,
            'gb_dist': 0.0, 'gb_overlap_dist': 1.2, 'gb_delete_overlap': True,
            'dislocation': False, 'dislocation_type': 'edge',
            'dislocation_axis': '0,0,1', 'dislocation_burgers': '1,0,0',
            'twinning': False, 'twinning_indices': (1, 1, 2),
            'twinning_z': 0.5, 'twinning_min_dist': 1.5,
            'stacking_fault': False, 'sf_normal': '1,1,1',
            'sf_shift': '0,0,0', 'sf_height': 0.5, 'sf_min_dist': 1.5,
            'amorphous': False, 'amorphous_min_dist': 1.5,
            'amorphous_rattle': 0.1, 'amorphous_steps': 100,
            'mag_mode': None, 'mag_noise': 0.0, 'mag_kwargs': None,
            'rotate_formula': None,
            'vac_elements': None, 'vac_num': 0,
            'antisite': False, 'antisite_pairs': None, 'antisite_num': 1,
            'antisite_mode': 'symmetry_aware', 'antisite_symprec': 1e-2,
            'shuffle_elements': None, 'shuffle_method': 'fisher_yates',
            'rigid': False, 'rigid_method': 'auto', 'rigid_list': None,
            'rigid_mode': 'inter', 'rigid_composition': None,
            'rotate_cell': False, 'vol_pert_fraction': 0.0,
            'sym_strain': False, 'sym_strain_fraction': 0.03,
            'sym_strain_crystal_system': None, 'sym_strain_symprec': 1e-2,
            'filter_bonds': False, 'validate_structure': True,
            'validate_coefficient': None, 'similarity_threshold': 0.999,
            'debug_plot': False,
        }

    def to_dict(self) -> dict:
        """Serialize non-None config values to a dict."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> 'PerturbConfig':
        """Create a config from a dict, ignoring unknown keys."""
        known = cls._get_defaults()
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
