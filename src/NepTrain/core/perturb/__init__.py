#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/13 19:20
# @Author  : 兵
# @email    : 1747193328@qq.com
from .config import PerturbConfig
from .normalize import normalize_int_list, normalize_float_list, normalize_axis_index, normalize_vector_norm, parse_range
from .dimensions import calculate_sobol_dimensions
from .run import run_perturb
from .symmetry_strain import generate_symmetry_preserving_strain, detect_crystal_system_spglib, get_independent_strain_count
from .rotate import rotate_fragments_by_formula
from .shuffle import shuffle_element_positions
from .stacking_fault import generate_stacking_fault
from .antisite import generate_antisite_defects, get_equivalent_sites
from .compatibility import validate_compatibility
from .validation import (
    get_bond_lengths,
    get_min_bond_length,
    get_coordination_numbers,
    check_atomic_overlap,
    get_structure_summary,
    filter_valid_structures
)
from .interstitial import InterstitialGenerator, BatchGenerator, generate_interstitial
from .twinning import TwinningGenerator, generate_twinning, generate_twinning_boundary, get_twinning_plane_info
from .crystal_detector import CrystalDetector
from .validator import (
    validate_atoms,
    validate_positive_integer,
    validate_positive_float,
    validate_string,
    validate_ratio,
    validate_bool,
    validate_indices,
    validate_vector,
    validate_sampler,
    validate_perturbation_type,
    validate_magnetic_mode,
    validate_list_of_atoms,
    validate_dict
)

__all__ = [
    'InterstitialGenerator',
    'TwinningGenerator',
    'BatchGenerator',
    'TwinningBatchGenerator',
    'CrystalDetector',
    'generate_interstitial',
    'generate_twinning',
    'generate_twinning_boundary',
    'get_twinning_plane_info',
    'run_perturb',
    'rotate_fragments_by_formula',
    'shuffle_element_positions',
    'generate_stacking_fault',
    'generate_antisite_defects',
    'get_equivalent_sites',
    'get_bond_lengths',
    'get_min_bond_length',
    'get_coordination_numbers',
    'check_atomic_overlap',
    'get_structure_summary',
    'filter_valid_structures',
    # Validator
    'validate_atoms',
    'validate_positive_integer',
    'validate_positive_float',
    'validate_string',
    'validate_ratio',
    'validate_bool',
    'validate_indices',
    'validate_vector',
    'validate_sampler',
    'validate_perturbation_type',
    'validate_magnetic_mode',
    'validate_list_of_atoms',
    'validate_dict',
    # Config & utilities
    'PerturbConfig',
    'normalize_int_list',
    'normalize_float_list',
    'normalize_axis_index',
    'normalize_vector_norm',
    'parse_range',
    'calculate_sobol_dimensions',
    # Symmetry-preserving strain
    'generate_symmetry_preserving_strain',
    'detect_crystal_system_spglib',
    'get_independent_strain_count',
    # Compatibility
    'validate_compatibility',
]
