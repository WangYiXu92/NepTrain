#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sobol dimension calculation for perturbation sampling."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms

from .normalize import (
    normalize_int_list, normalize_float_list, normalize_axis_index,
    normalize_vector_norm, parse_range,
)
from .magnetic import get_magmom_config, get_magnetic_perturbation_dims
from .rotate import get_molecules, parse_formula_dict
from .surface import generate_surface
from .grain_boundary import generate_grain_boundary
from .dislocation import generate_dislocation
from .twinning import generate_twinning
from .stacking_fault import generate_stacking_fault
from .amorphous import generate_amorphous
from .crystal import create_oriented_supercell, get_burgers_vector
from .sampler import SobolSampler
from .rigid import RigidBodyManager, parse_rigid_list_string

logger = logging.getLogger(__name__)


def calculate_sobol_dimensions(
    atoms: Atoms,
    *,
    # Topology flags
    surface: bool = False,
    surface_indices: Any = '1,1,1',
    surface_vacuum: Any = 10.0,
    surface_layers: Any = 3,
    gb: bool = False,
    gb_axis: Any = '0,0,1',
    gb_angle: Any = 36.87,
    gb_dist: Any = 0.0,
    gb_overlap_dist: Any = 1.2,
    gb_delete_overlap: bool = True,
    dislocation: bool = False,
    dislocation_type: str = 'edge',
    dislocation_axis: Any = '0,0,1',
    dislocation_burgers: Any = '1,0,0',
    twinning: bool = False,
    twinning_indices: Any = (1, 1, 2),
    twinning_z: Any = 0.5,
    twinning_min_dist: float = 1.5,
    stacking_fault: bool = False,
    sf_normal: Any = '1,1,1',
    sf_shift: Any = '0,0,0',
    sf_height: Any = 0.5,
    sf_min_dist: float = 1.5,
    amorphous: bool = False,
    amorphous_min_dist: float = 1.5,
    amorphous_rattle: float = 0.1,
    amorphous_steps: int = 100,
    # Other perturbation params
    mag_mode: Optional[str] = None,
    mag_noise: float = 0.0,
    mag_kwargs: Optional[dict] = None,
    mag_config: Optional[Any] = None,
    rotate_formula: Optional[str] = None,
    vac_elements: Optional[Any] = None,
    vac_num: int = 0,
    shuffle_elements: Optional[Any] = None,
    shuffle_method: str = 'fisher_yates',
    rigid: bool = False,
    rigid_method: str = 'auto',
    rigid_list: Optional[Any] = None,
    rigid_mode: str = 'inter',
    rigid_composition: Optional[Any] = None,
    vol_pert_fraction: float = 0.0,
    cell_pert_fraction: float = 0.03,
    # kwargs
    **kwargs,
) -> Dict[str, Any]:
    """
    Calculate Sobol dimensions and build topology-modified dummy atoms.
    
    Returns a dict with:
        - d_cell, d_disp, d_mag, d_rot, d_vac, d_shuf, d_amorphous,
          d_dislocation, d_gb, d_twinning, d_sf, d_surface, d_vol, total_d
        - dummy_atoms: topology-modified atoms for downstream use
        - rigid_manager: if rigid mode enabled
        - Normalized range tuples: surf_vac_*, surf_lay_*, gb_angle_*, gb_dist_*, gb_overlap_*
        - normalized axis/index values
    """
    # Normalize inputs (mirrors run.py logic)
    if surface:
        surface_indices = normalize_int_list(surface_indices)
        surf_vac_min, surf_vac_max, surf_vac_is_range = parse_range(surface_vacuum, float)
        surf_lay_min, surf_lay_max, surf_lay_is_range = parse_range(surface_layers, int)
    else:
        surf_vac_min = surf_vac_max = surf_vac_is_range = None
        surf_lay_min = surf_lay_max = surf_lay_is_range = None

    if gb:
        gb_axis = normalize_float_list(gb_axis, default_if_random='random')
        gb_angle_min, gb_angle_max, gb_angle_is_range = parse_range(gb_angle, float)
        gb_dist_min, gb_dist_max, gb_dist_is_range = parse_range(gb_dist, float)
        gb_overlap_min, gb_overlap_max, gb_overlap_is_range = parse_range(gb_overlap_dist, float)
    else:
        gb_angle_min = gb_angle_max = gb_angle_is_range = None
        gb_dist_min = gb_dist_max = gb_dist_is_range = None
        gb_overlap_min = gb_overlap_max = gb_overlap_is_range = None

    if dislocation:
        dislocation_axis = normalize_axis_index(dislocation_axis)
        dislocation_burgers = normalize_vector_norm(dislocation_burgers)

    if twinning:
        twinning_indices = normalize_int_list(twinning_indices)

    if stacking_fault:
        sf_normal = normalize_float_list(sf_normal)
        sf_shift = normalize_float_list(sf_shift, default_if_random='random')

    # Build dummy atoms with topology applied
    dummy_atoms = atoms.copy()

    if surface:
        dummy_atoms = generate_surface(dummy_atoms, indices=surface_indices, vacuum=surf_vac_min, layers=surf_lay_min)
    if gb:
        dummy_angle = 36.87
        dummy_axis = [0, 0, 1]
        if gb_axis == 'random':
            dummy_axis = [0, 0, 1]
        elif isinstance(gb_axis, (list, tuple, np.ndarray)):
            dummy_axis = gb_axis
        elif isinstance(gb_axis, str):
            try:
                dummy_axis = [float(x) for x in gb_axis.split(',')]
            except Exception:
                pass
        if gb_angle == 'random':
            dummy_angle = 36.87
        elif gb_angle_is_range:
            dummy_angle = (gb_angle_min + gb_angle_max) / 2
        elif isinstance(gb_angle, (int, float)):
            dummy_angle = gb_angle
        else:
            dummy_angle = 36.87
        dummy_atoms = generate_grain_boundary(dummy_atoms, axis=dummy_axis, angle_deg=dummy_angle, min_dist=gb_overlap_min, delete_overlap=gb_delete_overlap)
    if dislocation:
        d_type_dummy = 'edge' if dislocation_type == 'random' else dislocation_type
        dummy_atoms = generate_dislocation(dummy_atoms, type=d_type_dummy, axis=dislocation_axis, burgers=dislocation_burgers)
    if twinning:
        dummy_tz = 0.5 if twinning_z == 'random' else twinning_z
        dummy_atoms = generate_twinning(dummy_atoms, miller_indices=twinning_indices, z_frac=dummy_tz, min_dist=twinning_min_dist)
    if stacking_fault:
        dummy_s = [0, 0, 0] if sf_shift == 'random' else sf_shift
        dummy_h = 0.5 if sf_height == 'random' else sf_height
        dummy_atoms = generate_stacking_fault(dummy_atoms, plane_normal=sf_normal, shift_vector=dummy_s, plane_height_frac=dummy_h)
    if amorphous:
        dummy_atoms = generate_amorphous(dummy_atoms, min_dist=amorphous_min_dist, rattle_strength=amorphous_rattle, max_steps=amorphous_steps)

    # Calculate dimensions
    d_cell = 9

    # Rigid body manager
    rigid_manager = None
    if rigid:
        rigid_manager = RigidBodyManager(atoms)
        if rigid_method == 'auto':
            rigid_manager.detect_auto()
        elif rigid_method == 'composition' and rigid_composition:
            r_comp_list = [x.strip() for x in rigid_composition.split(',')] if isinstance(rigid_composition, str) else rigid_composition
            rigid_manager.detect_by_composition(r_comp_list)
        elif rigid_method == 'manual' and rigid_list:
            parsed_list = parse_rigid_list_string(rigid_list) if isinstance(rigid_list, str) else rigid_list
            rigid_manager.set_manual(parsed_list)

    if rigid and rigid_manager:
        bodies, loose = rigid_manager.get_bodies()
        if rigid_mode == 'intra':
            n_atoms_in_bodies = sum(len(indices) for indices in bodies.values())
            d_disp = 3 * n_atoms_in_bodies
        else:
            d_disp = 3 * len(loose) + 6 * len(bodies)
    else:
        d_disp = 3 * len(atoms)

    # Magnetic dims
    d_mag = 0
    if mag_mode:
        mag_kw = (mag_kwargs or {}).copy()
        mag_kw['noise'] = mag_noise
        d_mag = get_magnetic_perturbation_dims(dummy_atoms, mode=mag_mode, mag_config=mag_config, **mag_kw)

    # Rotation dims
    d_rot = 0
    if rotate_formula:
        mols = get_molecules(dummy_atoms)
        target_counts = parse_formula_dict(rotate_formula)
        n_frags = 0
        for indices in mols:
            fragment_syms = [dummy_atoms[i].symbol for i in indices]
            fragment_counts = {}
            for s in fragment_syms:
                fragment_counts[s] = fragment_counts.get(s, 0) + 1
            if fragment_counts == target_counts:
                n_frags += 1
        d_rot = 3 * n_frags

    # Vacancy dims
    d_vac = vac_num if (vac_elements and vac_num > 0) else 0

    # Shuffle dims
    d_shuf = 0
    if shuffle_elements:
        count = 0
        if isinstance(shuffle_elements, (list, tuple)):
            if all(isinstance(x, (int, np.integer)) for x in shuffle_elements):
                count = len(shuffle_elements)
            elif all(isinstance(x, str) for x in shuffle_elements):
                count = sum(1 for a in dummy_atoms if a.symbol in shuffle_elements)
            else:
                count = len(dummy_atoms)
        elif isinstance(shuffle_elements, str):
            if ':' in shuffle_elements:
                try:
                    parts = [int(x) if x.strip() else None for x in shuffle_elements.split(':')]
                    count = len(range(*slice(*parts).indices(len(dummy_atoms))))
                except Exception:
                    count = len(dummy_atoms)
            else:
                count = sum(1 for a in dummy_atoms if a.symbol == shuffle_elements)
        elif isinstance(shuffle_elements, slice):
            count = len(range(*shuffle_elements.indices(len(dummy_atoms))))
        else:
            count = len(dummy_atoms)
        d_shuf = count

    # Amorphous dims
    d_amorphous = 1 if amorphous else 0

    # Dislocation dims
    d_dislocation = 0
    if dislocation:
        d_dislocation = 2
        if dislocation_type == 'random':
            d_dislocation += 1

    # GB dims
    d_gb = 0
    d_gb_axis_dim = 0
    d_gb_angle_dim = 0
    if gb:
        d_gb = 2  # translation_x, translation_y
        if gb_axis == 'random':
            d_gb += 1
            d_gb_axis_dim = 1
        if gb_angle == 'random' or (gb_angle_is_range if gb_angle_is_range else False):
            d_gb += 1
            d_gb_angle_dim = 1
        elif gb_angle == 'csl':
            d_gb += 1
            d_gb_angle_dim = 1
        if gb_dist_is_range:
            d_gb += 1
        if gb_overlap_is_range:
            d_gb += 1

    # Twinning dims
    d_twinning = 0
    if twinning:
        d_twinning = 2
        if twinning_z == 'random':
            d_twinning += 1

    # Stacking fault dims
    d_sf = 0
    if stacking_fault:
        if sf_shift == 'random':
            d_sf += 2
        if sf_height == 'random':
            d_sf += 1

    # Surface dims
    d_surface = 0
    if surface:
        if surf_vac_is_range:
            d_surface += 1
        if surf_lay_is_range:
            d_surface += 1

    d_vol = 1 if vol_pert_fraction > 0 else 0

    total_d = d_cell + d_disp + d_mag + d_rot + d_vac + d_shuf + d_amorphous + d_dislocation + d_gb + d_twinning + d_sf + d_surface + d_vol

    return {
        'd_cell': d_cell, 'd_disp': d_disp, 'd_mag': d_mag, 'd_rot': d_rot,
        'd_vac': d_vac, 'd_shuf': d_shuf, 'd_amorphous': d_amorphous,
        'd_dislocation': d_dislocation, 'd_gb': d_gb, 'd_twinning': d_twinning,
        'd_sf': d_sf, 'd_surface': d_surface, 'd_vol': d_vol, 'total_d': total_d,
        'dummy_atoms': dummy_atoms,
        'rigid_manager': rigid_manager,
        # Normalized values
        'surface_indices': surface_indices,
        'surf_vac_min': surf_vac_min, 'surf_vac_max': surf_vac_max, 'surf_vac_is_range': surf_vac_is_range,
        'surf_lay_min': surf_lay_min, 'surf_lay_max': surf_lay_max, 'surf_lay_is_range': surf_lay_is_range,
        'gb_axis': gb_axis,
        'gb_angle_min': gb_angle_min, 'gb_angle_max': gb_angle_max, 'gb_angle_is_range': gb_angle_is_range,
        'gb_dist_min': gb_dist_min, 'gb_dist_max': gb_dist_max, 'gb_dist_is_range': gb_dist_is_range,
        'gb_overlap_min': gb_overlap_min, 'gb_overlap_max': gb_overlap_max, 'gb_overlap_is_range': gb_overlap_is_range,
        'd_gb_axis_dim': d_gb_axis_dim, 'd_gb_angle_dim': d_gb_angle_dim,
        'dislocation_axis': dislocation_axis,
        'dislocation_burgers': dislocation_burgers,
        'twinning_indices': twinning_indices,
        'sf_normal': sf_normal, 'sf_shift': sf_shift,
    }
