#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 18:12
# @Author  : 兵
# @email    : 1747193328@qq.com
import logging
import os
import sys
import json
from typing import List, Dict, Any, Optional, Generator, Union, Tuple

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.io import read as ase_read
from rich.progress import Progress

from NepTrain import utils
from NepTrain.exceptions import FileOperationError, ValidationError, PerturbationError
from NepTrain.logging_config import get_logger
from NepTrain.core.select.filter import adjust_reasonable, get_mini_distance_info
from NepTrain.core.select.select import filter_by_bonds, compute_min_bond_lengths
from .magnetic import apply_magnetic_perturbation, get_magmom_config, get_magnetic_perturbation_dims
from .rotate import rotate_fragments_by_formula, get_molecules, parse_formula_dict
from .vacancy import generate_vacancies, _filter_vacancies_for_export
from .shuffle import shuffle_element_positions, _parse_element_range, _filter_fixed_indices
from .surface import generate_surface
from .grain_boundary import generate_grain_boundary
# from .csl_core import get_csl_data, generate_integer_axes
from .dislocation import generate_dislocation
from .twinning import generate_twinning
from .stacking_fault import generate_stacking_fault
from .amorphous import generate_amorphous
from .crystal import create_oriented_supercell, get_burgers_vector
from .sampler import SobolSampler, RandomSampler
from .rigid import generate_rigid_perturbed_structure, RigidBodyManager, parse_rigid_list_string
from .plot import plot_comparison
from .validator import (
    validate_atoms, validate_positive_integer, validate_positive_float,
    validate_string, validate_ratio, validate_bool, validate_indices,
    validate_vector, validate_sampler, validate_perturbation_type,
    validate_magnetic_mode
)
# from ._hiphive import generate_mc_rattled_structures

# Set up module logger
logger = get_logger(__name__)

class SimilarityFilter:
    """Simple fingerprint filter based on Bond Length Histograms."""
    def __init__(self, threshold=0.98, n_bins=50, r_max=6.0):
        self.threshold = threshold
        self.n_bins = n_bins
        self.r_max = r_max
        self.history = []

    def get_fingerprint(self, atoms):
        from ase.neighborlist import NeighborList
        nl = NeighborList([self.r_max/2]*len(atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(atoms)
        dists = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            if len(indices) > 0:
                pos = atoms.positions
                cell = atoms.cell
                for j, offset in zip(indices, offsets):
                    diff = pos[j] + np.dot(offset, cell) - pos[i]
                    dists.append(np.linalg.norm(diff))
        if not dists: return np.zeros(self.n_bins)
        hist, _ = np.histogram(dists, bins=self.n_bins, range=(0, self.r_max), density=True)
        return hist

    def is_redundant(self, atoms):
        if self.threshold >= 1.0: return False
        fp = self.get_fingerprint(atoms)
        for h_fp in self.history:
            # Correlation coefficient
            corr = np.corrcoef(fp, h_fp)[0, 1]
            if corr > self.threshold:
                return True
        self.history.append(fp)
        return False


def _load_state(state_file: str) -> Optional[Dict[str, Any]]:
    """Load state from JSON file."""
    if not os.path.exists(state_file):
        logger.warning(f"State file not found: {state_file}")
        return None
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load state file {state_file}: {e}")
        return None

def _save_state(state_file: str, num_generated: int, total_d: int, seed: Optional[int] = None) -> None:
    """Save state to JSON file."""
    state = {
        'num_generated': num_generated,
        'total_d': total_d,
        'seed': seed
    }
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)
        logger.debug(f"State saved to {state_file}")
    except Exception as e:
        logger.error(f"Failed to save state file {state_file}: {e}")
        raise FileOperationError(f"Failed to save state file {state_file}: {e}")


def perturb_position(prim: Atoms, min_distance: float, rng_values: Optional[np.ndarray] = None) -> Atoms:

    atoms = prim.copy()
    # 获取原子位置
    positions = atoms.get_positions()
    
    if rng_values is not None:
        n_needed = positions.size
        if len(rng_values) < n_needed:
             raise ValueError(f"Not enough random values for position perturbation. Needed {n_needed}, got {len(rng_values)}.")
        
        # Scale uniform [0, 1] to [-min_distance, min_distance]
        vals = rng_values[:n_needed].reshape(positions.shape)
        deltas = vals * (2 * min_distance) - min_distance
        perturbed_positions = positions + deltas
    else:
        # 添加随机微扰
        perturbed_positions = positions + np.random.uniform(
            low=-min_distance,
            high=min_distance,
            size=positions.shape
        )

    # 更新结构的原子位置
    atoms.set_positions(perturbed_positions)
    return atoms

def generate_deformed_structure(prim: Atoms, strain_lim: List[float], min_distance: float) -> Atoms:
    """
    Generate a deformed structure by applying a general deformation tensor.
    """
    atoms = prim.copy()
    cell = atoms.get_cell()
    
    # Generate random deformation tensor components within strain_lim
    # strain_lim is expected to be [min, max]
    strains = np.random.uniform(strain_lim[0], strain_lim[1], (3, 3))
    
    # Deformation gradient F = I + epsilon
    deformation = np.eye(3) + strains
    
    # Apply deformation: new_cell = cell @ deformation (if vectors are rows)
    # or deformation @ cell?
    # Standard: v_new = F @ v_old (column vectors)
    # ASE uses row vectors: V_new = V_old @ F.T
    # So new_cell = cell @ deformation.T
    # Let's assume strains corresponds to F-I directly.
    # If we want F applied to basis vectors.
    new_cell = np.dot(cell, deformation.T)
    
    atoms.set_cell(new_cell, scale_atoms=True)
    atoms = perturb_position(atoms, min_distance)
    return atoms

def generate_strained_structure(prim: Atoms, strain_lim: List[float], min_distance: float, 
                                strain_tensor: Optional[np.ndarray] = None, 
                                rotation: Optional[Union[tuple, np.ndarray]] = None,
                                rng_values: Optional[np.ndarray] = None) -> Atoms:
    atoms = prim.copy()
    cell = atoms.get_cell()
    
    strain_mat = np.zeros((3, 3))
    
    if strain_tensor is None:
        # Default to random diagonal strain if no tensor provided
        strains = np.random.uniform(*strain_lim, (3, ))
        np.fill_diagonal(strain_mat, strains)
    else:
        strain_tensor = np.array(strain_tensor)
        if strain_tensor.shape == (3,):
             np.fill_diagonal(strain_mat, strain_tensor)
        elif strain_tensor.shape == (6,):
             # xx, yy, zz, yz, xz, xy
             strain_mat[0, 0] = strain_tensor[0]
             strain_mat[1, 1] = strain_tensor[1]
             strain_mat[2, 2] = strain_tensor[2]
             
             strain_mat[1, 2] = strain_tensor[3]
             strain_mat[2, 1] = strain_tensor[3]
             
             strain_mat[0, 2] = strain_tensor[4]
             strain_mat[2, 0] = strain_tensor[4]
             
             strain_mat[0, 1] = strain_tensor[5]
             strain_mat[1, 0] = strain_tensor[5]
        elif strain_tensor.shape == (3, 3):
             strain_mat = strain_tensor
        else:
             raise ValueError(f"Invalid strain tensor shape: {strain_tensor.shape}")

    # Apply strain: F = I + epsilon
    deformation = np.eye(3) + strain_mat
    
    # new_cell = cell @ deformation.T (ASE row vectors)
    new_cell = np.dot(cell, deformation.T)
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # Apply global rotation if provided
    if rotation is not None:
        # rotation can be a matrix (3x3) or (axis, angle) tuple
        if isinstance(rotation, tuple) and len(rotation) == 2:
            axis, angle = rotation
            # Rotate atoms positions and cell
            # ASE's rotate method handles both if rotate_cell=True
            atoms.rotate(angle, axis, rotate_cell=True)
        elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
             # Apply rotation matrix
             # cell_new = cell @ R^T (ASE row vectors)
             # pos_new = pos @ R^T
             # ASE set_cell/set_positions or use rotate with matrix?
             # atoms.rotate doesn't take matrix directly easily without conversion to Euler or axis/angle.
             # Easier to do manual:
             new_pos = atoms.get_positions() @ rotation.T
             new_cell = atoms.get_cell() @ rotation.T
             atoms.set_cell(new_cell)
             atoms.set_positions(new_pos)
             
    atoms = perturb_position(atoms, min_distance, rng_values=rng_values)

    return atoms

# Imported from normalize module for backward compat (also available as private names)
from .normalize import (
    normalize_int_list as _normalize_int_list,
    normalize_float_list as _normalize_float_list,
    normalize_axis_index as _normalize_axis_index,
    normalize_vector_norm as _normalize_vector_norm,
    parse_range as _parse_range,
)
from .config import PerturbConfig
from .dimensions import calculate_sobol_dimensions

def perturb(atoms: Atoms,
            num=20,
            cell_pert_fraction=0.03,
            min_distance=0.1,
            mag_mode=None,
            mag_noise=0.0,
            mag_kwargs=None,
            rotate_formula=None,
            vac_elements=None, vac_num=0,
            shuffle_elements=None, shuffle_method='fisher_yates',
            surface=False, surface_indices='1,1,1', surface_vacuum=10.0, surface_layers=3,
            gb=False, gb_axis='0,0,1', gb_angle=36.87, gb_dist=0.0, gb_overlap_dist=1.2, gb_delete_overlap=True,
            dislocation=False, dislocation_type='edge', dislocation_axis='0,0,1', dislocation_burgers='1,0,0',
            twinning=False, twinning_indices=(1,1,2), twinning_z=0.5, twinning_min_dist=1.5,
            stacking_fault=False, sf_normal='1,1,1', sf_shift='0,0,0', sf_height=0.5, sf_min_dist=1.5,
            amorphous=False, amorphous_min_dist=1.5, amorphous_rattle=0.1, amorphous_steps=100,
            scramble=True,
            sampler='random',
            skip_normal=False,
            filter_bonds=False,
            state_file=None,
            resume=False,
            rigid=False,
            rigid_method='auto',
            rigid_list=None,
            rigid_mode='inter',
            rigid_composition=None,
            rotate_cell=False,
            debug_plot=False,
            validate_structure=True,
            validate_coefficient=None,
            vol_pert_fraction=0.0,
            similarity_threshold=0.999,
            **kwargs):
    """
    Generate perturbed structures.
    Merges functionality from legacy perturb and modern generator.
    """
    
    # Check if Sobol sampler is used
    use_sobol = (sampler == 'sobol')
    
    # Pre-load mag_config to avoid redundant parsing
    mag_config = None
    if mag_mode:
        magnetic_elements = kwargs.get('magnetic_elements', ['Fe', 'Co', 'Ni', 'Mn', 'Gd', 'Cr'])
        mag_config = get_magmom_config(atoms, magnetic_elements)
    
    # Initialize Sobol variables
    sobol_mag_samples = None
    sobol_rot_samples = None
    sobol_vac_samples = None
    sobol_shuf_samples = None
    sobol_dislocation_samples = None
    scaled_samples = None
    
    rigid_manager = None

    # Normalize Inputs
    if surface:
        surface_indices = _normalize_int_list(surface_indices)
        surf_vac_min, surf_vac_max, surf_vac_is_range = _parse_range(surface_vacuum, float)
        surf_lay_min, surf_lay_max, surf_lay_is_range = _parse_range(surface_layers, int)
    if gb:
        gb_axis = _normalize_float_list(gb_axis, default_if_random='random')
        gb_angle_min, gb_angle_max, gb_angle_is_range = _parse_range(gb_angle, float)
        gb_dist_min, gb_dist_max, gb_dist_is_range = _parse_range(gb_dist, float)
        gb_overlap_min, gb_overlap_max, gb_overlap_is_range = _parse_range(gb_overlap_dist, float)
    if dislocation:
        dislocation_axis = _normalize_axis_index(dislocation_axis)
        dislocation_burgers_input = dislocation_burgers
        dislocation_burgers = _normalize_vector_norm(dislocation_burgers)
    if twinning:
        twinning_indices = _normalize_int_list(twinning_indices)
    if stacking_fault:
        sf_normal = _normalize_float_list(sf_normal)
        sf_shift = _normalize_float_list(sf_shift, default_if_random='random')

    if use_sobol:
        # Delegate dimension calculation to dimensions module
        dims = calculate_sobol_dimensions(
            atoms,
            surface=surface, surface_indices=surface_indices,
            surface_vacuum=surface_vacuum, surface_layers=surface_layers,
            gb=gb, gb_axis=gb_axis, gb_angle=gb_angle,
            gb_dist=gb_dist, gb_overlap_dist=gb_overlap_dist,
            gb_delete_overlap=gb_delete_overlap,
            dislocation=dislocation, dislocation_type=dislocation_type,
            dislocation_axis=dislocation_axis, dislocation_burgers=dislocation_burgers,
            twinning=twinning, twinning_indices=twinning_indices,
            twinning_z=twinning_z, twinning_min_dist=twinning_min_dist,
            stacking_fault=stacking_fault, sf_normal=sf_normal,
            sf_shift=sf_shift, sf_height=sf_height, sf_min_dist=sf_min_dist,
            amorphous=amorphous, amorphous_min_dist=amorphous_min_dist,
            amorphous_rattle=amorphous_rattle, amorphous_steps=amorphous_steps,
            mag_mode=mag_mode, mag_noise=mag_noise,
            mag_kwargs=mag_kwargs, mag_config=mag_config,
            rotate_formula=rotate_formula,
            vac_elements=vac_elements, vac_num=vac_num,
            shuffle_elements=shuffle_elements, shuffle_method=shuffle_method,
            rigid=rigid, rigid_method=rigid_method, rigid_list=rigid_list,
            rigid_mode=rigid_mode, rigid_composition=rigid_composition,
            vol_pert_fraction=vol_pert_fraction,
            cell_pert_fraction=cell_pert_fraction,
        )
        # Unpack dimension info
        d_cell = dims['d_cell']
        d_disp = dims['d_disp']
        d_mag = dims['d_mag']
        d_rot = dims['d_rot']
        d_vac = dims['d_vac']
        d_shuf = dims['d_shuf']
        d_amorphous = dims['d_amorphous']
        d_dislocation = dims['d_dislocation']
        d_gb = dims['d_gb']
        d_twinning = dims['d_twinning']
        d_sf = dims['d_sf']
        d_surface = dims['d_surface']
        d_vol = dims['d_vol']
        total_d = dims['total_d']
        d_gb_axis_dim = dims['d_gb_axis_dim']
        d_gb_angle_dim = dims['d_gb_angle_dim']
        rigid_manager = dims['rigid_manager']

        # Unpack normalized values used in the generation loop
        surface_indices = dims['surface_indices']
        surf_vac_min = dims['surf_vac_min']
        surf_vac_max = dims['surf_vac_max']
        surf_vac_is_range = dims['surf_vac_is_range']
        surf_lay_min = dims['surf_lay_min']
        surf_lay_max = dims['surf_lay_max']
        surf_lay_is_range = dims['surf_lay_is_range']
        gb_axis = dims['gb_axis']
        gb_angle_min = dims['gb_angle_min']
        gb_angle_max = dims['gb_angle_max']
        gb_angle_is_range = dims['gb_angle_is_range']
        gb_dist_min = dims['gb_dist_min']
        gb_dist_max = dims['gb_dist_max']
        gb_dist_is_range = dims['gb_dist_is_range']
        gb_overlap_min = dims['gb_overlap_min']
        gb_overlap_max = dims['gb_overlap_max']
        gb_overlap_is_range = dims['gb_overlap_is_range']
        dislocation_axis = dims['dislocation_axis']
        dislocation_burgers = dims['dislocation_burgers']
        twinning_indices = dims['twinning_indices']
        sf_normal = dims['sf_normal']
        sf_shift = dims['sf_shift']

    # Pre-calculate base bond lengths if filtering
    base_bond = None
    if filter_bonds:
        base_bond = compute_min_bond_lengths(atoms)
    
    # Batch processing for Sobol generation
    BATCH_SIZE = 100
    
    with Progress(transient=False) as progress:
        task_id = progress.add_task("[cyan]Perturbing...", total=num)
        
        for start_batch in range(0, num, BATCH_SIZE):
            end_batch = min(start_batch + BATCH_SIZE, num)
        batch_size = end_batch - start_batch
        
        sobol_batch_disp = None
        sobol_batch_mag = None
        sobol_batch_rot = None
        sobol_batch_vac = None
        sobol_batch_shuf = None
        sobol_batch_amorphous = None
        sobol_batch_dislocation = None
        sobol_batch_gb = None
        sobol_batch_twinning = None
        sobol_batch_sf = None
        sobol_batch_surface = None
        sobol_batch_vol = None
        scaled_batch = None
        
        if use_sobol:
            raw_samples = s_sampler.random(n=batch_size)
            
            # Scale Cell samples
            cell_raw = raw_samples[:, :9]
            if cell_pert_fraction > 1e-9:
                l_bounds = [-cell_pert_fraction] * 9
                u_bounds = [cell_pert_fraction] * 9
                scaled_batch = s_sampler.scale(cell_raw, l_bounds=l_bounds, u_bounds=u_bounds)
            else:
                scaled_batch = np.zeros_like(cell_raw)
            
            # Disp samples
            sobol_batch_disp = raw_samples[:, 9:9+d_disp]
            
            current_dim = 9 + d_disp
            
            # Mag samples
            if d_mag > 0:
                sobol_batch_mag = raw_samples[:, current_dim:current_dim+d_mag]
                current_dim += d_mag
                
            # Rot samples
            if d_rot > 0:
                sobol_batch_rot = raw_samples[:, current_dim:current_dim+d_rot]
                current_dim += d_rot
            
            # Vac samples
            if d_vac > 0:
                sobol_batch_vac = raw_samples[:, current_dim:current_dim+d_vac]
                current_dim += d_vac
            
            # Shuf samples
            if d_shuf > 0:
                sobol_batch_shuf = raw_samples[:, current_dim:current_dim+d_shuf]
                current_dim += d_shuf

            # Amorphous samples
            if d_amorphous > 0:
                sobol_batch_amorphous = raw_samples[:, current_dim:current_dim+d_amorphous]
                current_dim += d_amorphous

            # Dislocation samples
            if d_dislocation > 0:
                sobol_batch_dislocation = raw_samples[:, current_dim:current_dim+d_dislocation]
                current_dim += d_dislocation

            # GB samples
            if d_gb > 0:
                sobol_batch_gb = raw_samples[:, current_dim:current_dim+d_gb]
                current_dim += d_gb

            # Twinning samples
            if d_twinning > 0:
                sobol_batch_twinning = raw_samples[:, current_dim:current_dim+d_twinning]
                current_dim += d_twinning

            # Stacking Fault samples
            if d_sf > 0:
                sobol_batch_sf = raw_samples[:, current_dim:current_dim+d_sf]
                current_dim += d_sf

            # Surface samples
            if d_surface > 0:
                sobol_batch_surface = raw_samples[:, current_dim:current_dim+d_surface]
                current_dim += d_surface

            # Volume samples
            if d_vol > 0:
                sobol_batch_vol = raw_samples[:, current_dim:current_dim+d_vol]
                current_dim += d_vol

        for i_local in range(batch_size):
            i_global = start_batch + i_local
            consumed_d = 0

            
            # For progress tracking, we might want to manually update or just print
            # Using rich track inside a loop is tricky if not the main iterator.
            # But let's assume we just iterate.
            
            struct = atoms.copy()
            
            # Initialize topology_modified flag
            topology_modified = False
            
            # 2. Topology (Surface, GB, etc)
            if surface:
                 topology_modified = True
                 curr_vac = surf_vac_min
                 curr_lay = surf_lay_min
                 
                 if use_sobol and sobol_batch_surface is not None:
                     consumed_d += d_surface
                     idx = 0
                     raw_surf = sobol_batch_surface[i_local]
                     
                     if surf_vac_is_range:
                         val = raw_surf[idx]
                         curr_vac = surf_vac_min + val * (surf_vac_max - surf_vac_min)
                         idx += 1
                         
                     if surf_lay_is_range:
                         val = raw_surf[idx]
                         n_opts = surf_lay_max - surf_lay_min + 1
                         int_off = int(val * n_opts)
                         if int_off == n_opts: int_off -= 1
                         curr_lay = surf_lay_min + int_off
                         idx += 1
                 
                 elif surf_vac_is_range or surf_lay_is_range:
                     # Random sampler
                     if surf_vac_is_range:
                         curr_vac = np.random.uniform(surf_vac_min, surf_vac_max)
                     if surf_lay_is_range:
                         curr_lay = np.random.randint(surf_lay_min, surf_lay_max + 1)

                 struct = generate_surface(struct, indices=surface_indices, vacuum=curr_vac, layers=curr_lay)
                 struct.info['perturb_annotation'] = {
                     'type': 'surface',
                     'metadata': {
                         'indices': surface_indices,
                         'vacuum': curr_vac,
                         'layers': curr_lay
                     }
                 }

            if gb:
                 topology_modified = True
                 curr_angle = gb_angle
                 curr_sigma = None
                 curr_trans = None
                 
                 # 1. Determine Axis
                 gb_axis_vec = gb_axis
                 # If random, we will pick later. But if not random, parse it.
                 if gb_axis != 'random':
                     if isinstance(gb_axis, str):
                         if ',' in gb_axis:
                             try:
                                gb_axis_vec = [float(x) for x in gb_axis.split(',')]
                             except:
                                pass
                     elif isinstance(gb_axis, (list, tuple, np.ndarray)):
                         try:
                            gb_axis_vec = [float(x) for x in gb_axis]
                         except:
                            pass
                 else:
                     # Default placeholder if random but not using sobol (handled later)
                     gb_axis_vec = [0,0,1]

                 if use_sobol and sobol_batch_gb is not None:
                     consumed_d += d_gb
                     raw_gb = sobol_batch_gb[i_local]
                     
                     # Indices in raw_gb:
                     # 0, 1: Translation (always)
                     # 2: Axis (if d_gb_axis_dim == 1)
                     # 2+d_gb_axis_dim: Angle (if d_gb_angle_dim == 1)
                     
                     # 1. Translation
                     t_frac_x = raw_gb[0]
                     t_frac_y = raw_gb[1]
                     # Convert to Cartesian later or pass fractional
                     curr_trans_frac = (t_frac_x, t_frac_y)
                     
                     # 2. Axis
                     if d_gb_axis_dim == 1:
                         axis_u = raw_gb[2]
                         all_axes = generate_integer_axes(max_index=3)
                         a_idx = int(axis_u * len(all_axes))
                         if a_idx == len(all_axes): a_idx -= 1
                         gb_axis_vec = all_axes[a_idx]
                     
                     # 3. Angle / Sigma
                     angle_ptr = 2 + d_gb_axis_dim
                     
                     if d_gb_angle_dim == 1:
                         angle_u = raw_gb[angle_ptr]
                         
                         if gb_angle == 'random':
                             # Map [0, 1] to [15, 90] degrees
                             curr_angle = 15.0 + angle_u * (90.0 - 15.0)
                         elif gb_angle_is_range:
                             curr_angle = gb_angle_min + angle_u * (gb_angle_max - gb_angle_min)
                         elif gb_angle == 'csl':
                             idx_val = angle_u
                             csl_data = get_csl_data(gb_axis_vec)
                             if csl_data:
                                 c_idx = int(idx_val * len(csl_data))
                                 if c_idx == len(csl_data): c_idx -= 1
                                 curr_angle = csl_data[c_idx]['angle']
                                 curr_sigma = csl_data[c_idx]['sigma']
                             else:
                                 curr_angle = 36.87
                     
                     # 4. Dist / Overlap
                     ptr = 2 + d_gb_axis_dim + d_gb_angle_dim
                     
                     curr_gb_dist = gb_dist_min
                     curr_gb_overlap = gb_overlap_min
                     
                     if gb_dist_is_range:
                         curr_gb_dist = gb_dist_min + raw_gb[ptr] * (gb_dist_max - gb_dist_min)
                         ptr += 1
                     if gb_overlap_is_range:
                         curr_gb_overlap = gb_overlap_min + raw_gb[ptr] * (gb_overlap_max - gb_overlap_min)
                         ptr += 1

                 else: 
                     # Non-sobol random sampling
                     # 1. Axis
                     if gb_axis == 'random':
                         all_axes = generate_integer_axes(max_index=3)
                         gb_axis_vec = all_axes[np.random.randint(len(all_axes))]
                     
                     # 2. Angle
                     if gb_angle == 'random':
                          curr_angle = np.random.uniform(15.0, 90.0)
                     elif gb_angle_is_range:
                          curr_angle = np.random.uniform(gb_angle_min, gb_angle_max)
                     elif gb_angle == 'csl':
                          csl_data = get_csl_data(gb_axis_vec)
                          if csl_data:
                              c_idx = np.random.randint(len(csl_data))
                              curr_angle = csl_data[c_idx]['angle']
                              curr_sigma = csl_data[c_idx]['sigma']
                          else:
                              curr_angle = 36.87
                     elif isinstance(curr_angle, str):
                          pass                 

                     # 3. Dist / Overlap
                     curr_gb_dist = gb_dist_min
                     curr_gb_overlap = gb_overlap_min
                     
                     if gb_dist_is_range:
                         curr_gb_dist = np.random.uniform(gb_dist_min, gb_dist_max)
                     if gb_overlap_is_range:
                         curr_gb_overlap = np.random.uniform(gb_overlap_min, gb_overlap_max)
                         
                     # 4. Translation (Random if not provided)
                     # Handled below
                     curr_trans_frac = tuple(np.random.uniform(0, 1, 2))

                 # Dispatch based on method
                 curr_trans = None
                 
                 # Check kwargs first (explicit config overrides random unless sobol used)
                 # Wait, if sobol used, we already set curr_trans_frac.
                 # If non-sobol, we set random curr_trans_frac above.
                 # But if user provided explicit translation in kwargs, we should respect it if not sampling?
                 # But 'random' implies sampling.
                 
                 if 'translation' in kwargs:
                     curr_trans = kwargs['translation']
                     curr_trans_frac = None # Override fractional
                 if 'translation_frac' in kwargs:
                     curr_trans_frac = kwargs['translation_frac']

                 # Sobol overrides everything if active
                 if use_sobol and sobol_batch_gb is not None:
                      # We set curr_trans_frac above
                      curr_trans = None 
                 
                 # Call unified generate_grain_boundary
                 try:
                    struct = generate_grain_boundary(struct, axis=gb_axis_vec, angle_deg=curr_angle, min_dist=curr_gb_overlap, vacuum=curr_gb_dist, translation=curr_trans, translation_frac=curr_trans_frac, sigma=curr_sigma)
                 except Exception as e:
                    if "Sigma" in str(e):
                         # Fallback or re-raise
                         logger.warning(f"CSL generation failed for {curr_angle} (Sigma {curr_sigma}), trying simple rotation. Error: {e}")
                         struct = generate_grain_boundary(struct, axis=gb_axis_vec, angle_deg=curr_angle, min_dist=curr_gb_overlap, vacuum=curr_gb_dist, translation=curr_trans, translation_frac=curr_trans_frac, sigma=None)
                    else:
                         raise
                 
                 # Annotation (Unified)
                 if 'perturb_annotation' not in struct.info:
                      struct.info['perturb_annotation'] = {}
                 struct.info['perturb_annotation']['type'] = 'grain_boundary'
                 if curr_sigma is not None:
                      struct.info['perturb_annotation']['sigma'] = curr_sigma
            if dislocation:
                 topology_modified = True
                 curr_type = dislocation_type
                 curr_center = None
                 
                 # Use the already normalized axis index
                 d_axis_idx = dislocation_axis if isinstance(dislocation_axis, (int, np.integer)) else 2
                 
                 if use_sobol and sobol_batch_dislocation is not None:
                     consumed_d += d_dislocation
                     
                     raw_d = sobol_batch_dislocation[i_local]
                     # center dims are first 2
                     c_frac = raw_d[:2]
                     
                     # Calculate center based on axis
                     cell_diag = struct.cell.lengths()
                     
                     if d_axis_idx == 0: # X axis, center in YZ
                         curr_center = [cell_diag[0]/2, c_frac[0]*cell_diag[1], c_frac[1]*cell_diag[2]]
                     elif d_axis_idx == 1: # Y axis, center in XZ
                         curr_center = [c_frac[0]*cell_diag[0], cell_diag[1]/2, c_frac[1]*cell_diag[2]]
                     else: # Z axis (2), center in XY
                         curr_center = [c_frac[0]*cell_diag[0], c_frac[1]*cell_diag[1], cell_diag[2]/2]
                     
                     if dislocation_type == 'random':
                         # type dim is index 2
                         if raw_d[2] < 0.5:
                             curr_type = 'edge'
                         else:
                             curr_type = 'screw'
                 
                 # If random sampler, ensure curr_center is set (default to center of cell)
                 if curr_center is None:
                    curr_center = struct.get_cell().sum(axis=0) / 2
                    
                 struct = generate_dislocation(struct, type=curr_type, axis=d_axis_idx, burgers=dislocation_burgers, center=curr_center)

                 struct.info['perturb_annotation'] = {
                     'type': 'dislocation',
                     'metadata': {
                         'dislocation_type': curr_type,
                         'axis': dislocation_axis,
                         'burgers': dislocation_burgers_input,
                         'center': curr_center
                     }
                 }
            if twinning:
                 topology_modified = True
                 curr_trans_frac = None
                 curr_trans = None
                 
                 # Check kwargs first
                 if 'translation' in kwargs:
                     curr_trans = kwargs['translation']
                 if 'translation_frac' in kwargs:
                     curr_trans_frac = kwargs['translation_frac']

                 if use_sobol and sobol_batch_twinning is not None:
                     consumed_d += d_twinning
                     raw_tw = sobol_batch_twinning[i_local]
                     # First 2 dims are translation
                     curr_trans_frac = [raw_tw[0], raw_tw[1]]
                     curr_trans = None
                 elif curr_trans is None and curr_trans_frac is None:
                     # Random translation if nothing specified, to match GB behavior
                     curr_trans_frac = tuple(np.random.uniform(0, 1, 2))

                 # Apply Twinning
                 # z_frac is ignored in new implementation which builds a symmetric slab
                 struct = generate_twinning(struct, miller_indices=twinning_indices, min_dist=twinning_min_dist, translation=curr_trans, translation_frac=curr_trans_frac)
                 
                 twin_normal_cart = [0.0, 0.0, 1.0]
                 # d is roughly half cell height for symmetric slab
                 twin_d = 0.5 * struct.cell[2, 2]

                 # Merge with existing annotation if present
                 if 'perturb_annotation' in struct.info:
                     ann = struct.info['perturb_annotation']
                     if 'metadata' not in ann: ann['metadata'] = {}
                     ann['metadata'].update({
                         'indices': twinning_indices,
                         'translation_frac': curr_trans_frac,
                         'normal_cart': twin_normal_cart,
                         'plane_d': twin_d
                     })
                 else:
                     struct.info['perturb_annotation'] = {
                         'type': 'twinning',
                         'metadata': {
                             'indices': twinning_indices,
                             'translation_frac': curr_trans_frac,
                             'normal_cart': twin_normal_cart, # For plotting
                             'plane_d': twin_d # For plotting
                         }
                     }
            if stacking_fault:
                 topology_modified = True
                 curr_sf_s = sf_shift
                 curr_sf_h = sf_height
                 curr_trans_frac = None
                 
                 if use_sobol and sobol_batch_sf is not None:
                     consumed_d += d_sf
                     raw_sf = sobol_batch_sf[i_local]
                     idx = 0
                     
                     if sf_shift == 'random':
                         curr_trans_frac = [raw_sf[idx], raw_sf[idx+1]]
                         curr_sf_s = [0.0, 0.0, 0.0]
                         idx += 2
                     else:
                         # Use explicit shift
                         pass
                         
                     if sf_height == 'random':
                         # Map [0, 1] to [0.1, 0.9]
                         curr_sf_h = 0.1 + raw_sf[idx] * 0.8
                
                 else:
                     # Random Sampler Logic
                     if sf_shift == 'random':
                         curr_trans_frac = np.random.uniform(0, 1, 2)
                         curr_sf_s = [0.0, 0.0, 0.0]
                         
                     if sf_height == 'random':
                         curr_sf_h = np.random.uniform(0.1, 0.9)
                     
                 struct = generate_stacking_fault(struct, plane_normal=sf_normal, shift_vector=curr_sf_s, plane_height_frac=curr_sf_h, translation_frac=curr_trans_frac, min_dist=sf_min_dist)
                 
                 # Retrieve actual shift vector (including extra_shift from translation_frac)
                 # stored by generate_stacking_fault in 'shift_vector' or 'metadata'
                 actual_shift = curr_sf_s
                 if 'perturb_annotation' in struct.info:
                     ann = struct.info['perturb_annotation']
                     if 'shift_vector' in ann:
                         actual_shift = ann['shift_vector']
                     elif 'metadata' in ann and 'input_shift' in ann['metadata']:
                         # generate_stacking_fault stores 'shift_vector' at top level
                         pass

                # Calculate Cartesian height for plotting
                 # Logic matches generate_stacking_fault: proj range
                 # But we need the normal first
                 # Handle Miller indices if needed
                 sf_normal_cart = np.array(sf_normal, dtype=float)
                 if all(isinstance(x, (int, np.integer)) for x in sf_normal):
                     reciprocal_cell = struct.cell.reciprocal()
                     sf_normal_cart = np.dot(sf_normal, reciprocal_cell)
                 
                 sf_norm = np.linalg.norm(sf_normal_cart)
                 if sf_norm > 1e-8:
                     sf_normal_cart /= sf_norm
                 
                 # Estimate plane constant d (height)
                 # We need atom positions to find range
                 positions = struct.get_positions()
                 projections = np.dot(positions, sf_normal_cart)
                 min_proj, max_proj = np.min(projections), np.max(projections)
                 plane_d = min_proj + curr_sf_h * (max_proj - min_proj)

                 struct.info['perturb_annotation'] = {
                     'type': 'stacking_fault',
                     'metadata': {
                         'normal': sf_normal, # Keep original for reference
                         'normal_cart': sf_normal_cart, # Add Cartesian for plotting
                         'shift': actual_shift,
                         'height': curr_sf_h, # Fractional
                         'plane_d': plane_d, # Cartesian plane constant for plotting
                         'translation_frac': curr_trans_frac
                     }
                 }

            if amorphous:
                 topology_modified = True
                 rng_amor = None
                 if use_sobol and d_amorphous > 0:
                     val = sobol_batch_amorphous[i_local][0]
                     consumed_d += d_amorphous
                     seed_loc = int(val * (2**32 - 1))
                     rng_loc = np.random.default_rng(seed_loc)
                     rng_amor = rng_loc.random(size=3 * len(struct))
                 
                 struct = generate_amorphous(struct, min_dist=amorphous_min_dist, rattle_strength=amorphous_rattle, max_steps=amorphous_steps, rng_values=rng_amor)
                 
            # 1. Cell Perturbation (Delayed)
            rng_disp = None
            if use_sobol:
                consumed_d += 9 # Cell (always consumed)
                strain_tensor = scaled_batch[i_local, :3]
                
                # Generate Global Rotation if requested using unused cell dims (3-5)
                rotation = None
                if rotate_cell:
                    # Use dims 3, 4, 5 from raw_samples (unscaled [0, 1])
                    u = raw_samples[i_local, 3]
                    v = raw_samples[i_local, 4]
                    
                    # Map to Sphere
                    z = 2 * u - 1
                    r = np.sqrt(max(0, 1 - z*z))
                    theta_ang = 2 * np.pi * v
                    axis = np.array([r * np.cos(theta_ang), r * np.sin(theta_ang), z])
                    angle = raw_samples[i_local, 5] * 360.0
                    rotation = (axis, angle)
                
                # Consume displacement dims to keep Sobol sync, but only use if topology not modified
                _rng_disp_sobol = sobol_batch_disp[i_local]
                consumed_d += d_disp
                
                if not topology_modified:
                    rng_disp = _rng_disp_sobol
                else:
                    rng_disp = None # Fallback to random thermal noise if topology changed size
                
                if rigid:
                    if rigid_manager:
                         # Warning: Rigid manager IDs match INITIAL atoms.
                         # If topology modified, IDs are invalid!
                         if not topology_modified:
                            struct.set_array('rigid_id', rigid_manager.ids)
                         else:
                            # Cannot use rigid model on modified topology easily?
                            # Fallback to non-rigid strain or try to re-detect?
                            # For GB/Dislocation, rigid body assumption might break anyway.
                            # We'll skip setting rigid_id and let it fail or default?
                            pass
                    
                    # If topology modified, rigid mode might fail if it relies on 'rigid_id'.
                    # We'll assume user knows what they are doing or fallback to simple strain if rigid fails?
                    # generate_rigid_perturbed_structure checks for rigid_id.
                    if topology_modified:
                        # Fallback to standard strain?
                        struct = generate_strained_structure(struct, [-cell_pert_fraction, cell_pert_fraction], min_distance, strain_tensor=strain_tensor, rotation=rotation, rng_values=rng_disp)
                    else:
                        struct = generate_rigid_perturbed_structure(struct, 
                                                                    mode=rigid_mode,
                                                                    strain_lim=[-cell_pert_fraction, cell_pert_fraction],
                                                                    min_distance=min_distance,
                                                                    strain_tensor=strain_tensor,
                                                                    rng_values=rng_disp)
                else:
                    struct = generate_strained_structure(struct, [-cell_pert_fraction, cell_pert_fraction], min_distance, strain_tensor=strain_tensor, rotation=rotation, rng_values=rng_disp)
            else:
                rotation = None
                if rotate_cell:
                     # Random rotation
                     axis = np.random.normal(size=3)
                     axis /= np.linalg.norm(axis)
                     angle = np.random.uniform(0, 360)
                     rotation = (axis, angle)

                if not skip_normal:
                     if rigid:
                        if rigid_manager and not topology_modified:
                             struct.set_array('rigid_id', rigid_manager.ids)
                        
                        if topology_modified:
                            struct = generate_strained_structure(struct, [-cell_pert_fraction, cell_pert_fraction], min_distance, rotation=rotation)
                        else:
                            struct = generate_rigid_perturbed_structure(struct, 
                                                                        mode=rigid_mode,
                                                                        strain_lim=[-cell_pert_fraction, cell_pert_fraction],
                                                                        min_distance=min_distance,
                                                                        rng_values=None)
                     else:
                        struct = generate_strained_structure(struct, [-cell_pert_fraction, cell_pert_fraction], min_distance, rotation=rotation)
                 
            # 3. Magnetic Perturbation
            if mag_mode:
                # Backup existing annotation if it's topological
                prior_annotation = struct.info.get('perturb_annotation')
                if prior_annotation:
                    prior_annotation = prior_annotation.copy()

                rng_mag = None
                if use_sobol and d_mag > 0:
                    rng_mag = sobol_batch_mag[i_local]
                    consumed_d += d_mag
                    
                m_kwargs = mag_kwargs if mag_kwargs else {}
                try:
                    struct = apply_magnetic_perturbation(struct, mode=mag_mode, mag_config=mag_config, noise=mag_noise, rng_values=rng_mag, **m_kwargs)
                except ValueError as e:
                    if "Not enough random values" in str(e):
                        raise ValueError(f"Sobol dimension mismatch in Magnetic Perturbation: {e}. This may happen if atom count changed due to topology defects.") from e
                    raise
                
                # Restore annotation if it was topological
                if prior_annotation and prior_annotation.get('type') in ['surface', 'grain_boundary', 'dislocation', 'twinning', 'stacking_fault', 'amorphous']:
                     # Add magnetic info to metadata for completeness
                     if 'metadata' not in prior_annotation:
                         prior_annotation['metadata'] = {}
                     prior_annotation['metadata']['magnetic_mode'] = mag_mode
                     
                     struct.info['perturb_annotation'] = prior_annotation

                
            # 4. Rotation
            if rotate_formula:
                rng_rot = None
                if use_sobol and d_rot > 0:
                    val = sobol_batch_rot[i_local][0]
                    consumed_d += d_rot
                    seed_loc = int(val * (2**32 - 1))
                    rng_loc = np.random.default_rng(seed_loc)
                    rng_rot = rng_loc.random(size=10 * len(struct))
                
                try:
                    struct = rotate_fragments_by_formula(struct, rotate_formula, rng_values=rng_rot)
                except ValueError as e:
                    if "Not enough random values" in str(e):
                         raise ValueError(f"Sobol dimension mismatch in Rotation: {e}.") from e
                    raise
            
            # 5. Vacancy
            if vac_elements and vac_num > 0:
                rng_vac = None
                if use_sobol and d_vac > 0:
                    rng_vac = sobol_batch_vac[i_local]
                    consumed_d += d_vac
                
                try:
                    struct, vac_meta = generate_vacancies(struct, vac_elements, vac_num, mode='random', rng_values=rng_vac)
                    struct.info['perturb_annotation'] = {'type': 'vacancy', 'metadata': vac_meta}
                except ValueError as e:
                    if "Not enough random values" in str(e):
                        raise ValueError(f"Sobol dimension mismatch in Vacancy Generation: {e}. This may happen if atom count changed due to topology defects.") from e
                    raise
                
            # 6. Shuffle
            if shuffle_elements:
                rng_shuf = None
                if use_sobol and d_shuf > 0:
                    rng_shuf = sobol_batch_shuf[i_local]
                    consumed_d += d_shuf
                
                try:
                    struct, shuf_meta = shuffle_element_positions(struct, shuffle_elements, shuffle_method=shuffle_method, rng_values=rng_shuf)
                    struct.info['perturb_annotation'] = {'type': 'shuffle', 'metadata': shuf_meta}
                except ValueError as e:
                     if "Not enough random values" in str(e):
                         raise ValueError(f"Sobol dimension mismatch in Shuffle: {e}. This may happen if atom count changed due to topology defects.") from e
                     raise

            # 7. Volume Scaling
            if vol_pert_fraction > 0:
                if use_sobol:
                    v_scale = 1.0 + (sobol_batch_vol[i_local][0] - 0.5) * 2 * vol_pert_fraction
                    consumed_d += 1
                else:
                    v_scale = 1.0 + (np.random.uniform(0, 1) - 0.5) * 2 * vol_pert_fraction
                
                struct.set_cell(struct.get_cell() * v_scale, scale_atoms=True)
                if 'perturb_annotation' in struct.info:
                    if 'metadata' not in struct.info['perturb_annotation']:
                        struct.info['perturb_annotation']['metadata'] = {}
                    struct.info['perturb_annotation']['metadata']['vol_scale'] = v_scale

            if use_sobol:
                assert consumed_d == total_d, f"Strict dimension checking failed! Consumed {consumed_d} but allocated {total_d} dimensions."

            # Construct Config_type
            config_types = []
            if not skip_normal:
                 config_types.append(f"strain_{cell_pert_fraction}")
            
            def _fmt(val):
                if isinstance(val, (list, tuple, np.ndarray)):
                    # Format floats as integers if they are integers
                    parts = []
                    for x in val:
                        try:
                            f = float(x)
                            if f.is_integer():
                                parts.append(str(int(f)))
                            else:
                                parts.append(str(f))
                        except:
                            parts.append(str(x))
                    return ",".join(parts)
                return str(val)

            if surface:
                 config_types.append(f"surf({_fmt(surface_indices)})")
            if gb:
                 config_types.append(f"gb{_fmt(gb_axis)}_{curr_angle:.1f}")
            if dislocation:
                 config_types.append(f"disloc_{curr_type}")
            if twinning:
                 config_types.append(f"twin_{_fmt(twinning_indices)}")
            if stacking_fault:
                 config_types.append(f"sf_{_fmt(sf_normal)}")
            if amorphous:
                 config_types.append("amorphous")
            if mag_mode:
                 config_types.append(f"mag_{mag_mode}")
            if vac_elements and vac_num > 0:
                 config_types.append(f"vac_{vac_num}")
            if vol_pert_fraction > 0:
                 config_types.append(f"vol_{vol_pert_fraction}")
            
            if config_types:
                struct.info['Config_type'] = "+".join(config_types)

            # Filter Vacancies (remove X atoms)
            if vac_elements and vac_num > 0:
                 struct = _filter_vacancies_for_export(struct)

            # Isotropic Volume Scaling (Super-Coverage Feature #2)
            if vol_pert_fraction > 0:
                if use_sobol and sobol_batch_vol is not None:
                    v_val = sobol_batch_vol[i_local][0] # Use the single dimension for volume
                    # Scale v_val [0, 1] to [-vol_pert_fraction, vol_pert_fraction]
                    # V' = V * (1 + delta) where delta is in [-f, f]
                    delta = (v_val * 2 - 1) * vol_pert_fraction
                    vol_scale_factor = 1.0 + delta
                    length_scale = vol_scale_factor**(1/3)
                    struct.set_cell(struct.cell * length_scale, scale_atoms=True)
                else:
                    # Random sampler
                    delta = np.random.uniform(-vol_pert_fraction, vol_pert_fraction)
                    vol_scale_factor = 1.0 + delta
                    struct.set_cell(struct.cell * (vol_scale_factor**(1/3)), scale_atoms=True)

            # Similarity Filtering (Super-Coverage Feature #3)
            # We initialize the filter once per perturb call
            if 'sim_filter' not in locals():
                sim_filter = SimilarityFilter(threshold=similarity_threshold)
            
            if similarity_threshold < 1.0:
                if sim_filter.is_redundant(struct):
                    continue

            # Annotation layering (Super-Coverage Feature #4)
            # Ensure history of perturbations is kept if needed
            # ... (the generators already add annotations to struct.info)
            
            # Check bond lengths
            if validate_structure:
                # Determine coefficient
                eff_coeff = validate_coefficient
                if eff_coeff is None:
                    # Default to 0.7 usually, but 0.4 for major geometric defects that might be unrelaxed
                    if gb or dislocation or twinning or stacking_fault or amorphous:
                        eff_coeff = 0.3
                    else:
                        eff_coeff = 0.7
                
                if not adjust_reasonable(struct, coefficient=eff_coeff):
                    if debug_plot:
                        logger.debug(f"Structure {i_global} rejected by adjust_reasonable (coeff={eff_coeff})")
                        try:
                            dist_info = get_mini_distance_info(struct)
                            logger.debug(f"Min distances: {dist_info}")
                        except Exception as e:
                            logger.error(f"Could not get distance info: {e}")
                    continue
            
            # Filter by bonds ratio if requested
            if filter_bonds and base_bond:
                 current_bond = compute_min_bond_lengths(struct)
                 condition = [base_bond.get(key,0)*0.6 > a_b for key,a_b in current_bond.items()]
                 if any(condition):
                     continue
            
            if debug_plot:
                plot_comparison(atoms, struct, f"debug_perturb_{i_global}.png")

            yield struct
            progress.advance(task_id)

            if use_sobol and state_file:
                current_count = start_index + i_global + 1
                _save_state(state_file, current_count, total_d, seed=seed)

def run_perturb(input_file: Union[str, Any],
                num: int = 20,
                cell_pert_fraction: float = 0.03,
                min_distance: float = 0.1,
                output_file: Optional[str] = 'perturb.xyz',
                append: bool = False,
                **kwargs) -> Generator[Atoms, None, None]:
    """
    Main entry point for perturbation.
    Supports both legacy argparse.Namespace input and direct arguments.
    """
    
    # 1. Handle Legacy Namespace Input
    if hasattr(input_file, 'model_path') or hasattr(input_file, 'input_file'):
        args = input_file
        model_path = getattr(args, 'model_path', getattr(args, 'input_file', None))
        
        # Legacy parameters mapping
        # Note: args.filter might not exist in current CLI, default to False
        filter_bonds = getattr(args, 'filter', False)
        c_pert = getattr(args, 'cell_pert_fraction', cell_pert_fraction)
        m_dist = getattr(args, 'min_distance', min_distance)
        n_num = getattr(args, 'num', num)
        out_path = getattr(args, 'out_file_path', getattr(args, 'output_file', output_file))
        is_append = getattr(args, 'append', append)
        
        # Recursively call run_perturb with unpacked args to share file handling logic
        # But run_perturb expects input_file as path.
        yield from run_perturb(model_path,
                               num=n_num,
                               cell_pert_fraction=c_pert,
                               min_distance=m_dist,
                               output_file=out_path,
                               append=is_append,
                               filter_bonds=filter_bonds,
                               **kwargs)
        return

    # 2. File Handling Logic
    if os.path.isdir(input_file):
        files = [os.path.join(input_file, f) for f in os.listdir(input_file) if f.endswith(('.xyz', '.cif', '.POSCAR', '.vasp'))]
    else:
        files = [input_file]
        
    # Handle output file cleanup if not appending
    if output_file and os.path.exists(output_file) and not append:
        try:
            os.remove(output_file)
            logger.debug(f"Removed existing output file: {output_file}")
        except OSError as e:
            logger.warning(f"Could not remove {output_file}: {e}")

    # 3. Process Files
    for file_path in files:
        try:
            atoms = ase_read(file_path)
        except Exception as e:
            import traceback
            logger.error(f"Failed to read file {file_path}: {e}")
            traceback.print_exc()
            continue

    # 3. Process Files
        # Note: atom_pert_distance in kwargs should be mapped to min_distance if present?
        # The function signature uses min_distance.
        # If atom_pert_distance is passed in kwargs (from CLI wrapper), we should handle it.
        if 'atom_pert_distance' in kwargs:
            min_distance = kwargs.pop('atom_pert_distance')
            
        generator = perturb(atoms,
                            num=num,
                            cell_pert_fraction=cell_pert_fraction,
                            min_distance=min_distance,
                            **kwargs)
                            
        for struct in generator:
            if output_file:
                ase_write(output_file, struct, append=True)
            yield struct


# ============================================================================
# Validation and Safety Wrappers (P0 Level Improvements)
# ============================================================================


def _validate_perturb_args(
    atoms: Atoms,
    num: int,
    cell_pert_fraction: float,
    min_distance: float,
    mag_mode: Optional[str],
    rotate_formula: Optional[str],
    vac_elements: Optional[List[str]],
    vac_num: int,
    shuffle_elements: Optional[Any],
    shuffle_method: str,
    surface: bool,
    gb: bool,
    dislocation: bool,
    twinning: bool,
    stacking_fault: bool,
    amorphous: bool,
    sampler: str,
    rigid: bool,
    vol_pert_fraction: float,
    similarity_threshold: float,
    **kwargs
):
    """
    Validate all perturbation arguments before processing.
    
    This function performs comprehensive input validation to catch errors
    early and provide clear error messages.
    
    Args:
        atoms: Input structure
        num: Number of structures to generate
        cell_pert_fraction: Cell perturbation fraction
        min_distance: Minimum distance threshold
        mag_mode: Magnetic perturbation mode
        rotate_formula: Molecular rotation specification
        vac_elements: Vacancy element list
        vac_num: Number of vacancies
        shuffle_elements: Element shuffle specification
        shuffle_method: Shuffle algorithm
        surface: Enable surface generation
        gb: Enable grain boundary generation
        dislocation: Enable dislocation generation
        twinning: Enable twinning generation
        stacking_fault: Enable stacking fault generation
        amorphous: Enable amorphous generation
        sampler: Sampler type ('random' or 'sobol')
        rigid: Enable rigid body perturbation
        vol_pert_fraction: Volume perturbation fraction
        similarity_threshold: Similarity filtering threshold
        **kwargs: Additional keyword arguments
        
    Raises:
        ValidationError: If any argument fails validation
    """
    # Validate required atoms
    validate_atoms(atoms, "atoms")
    
    # Validate num
    if num <= 0:
        raise ValidationError(f"num must be positive, got {num}")
    
    # Validate cell_pert_fraction
    validate_ratio(cell_pert_fraction, "cell_pert_fraction")
    
    # Validate min_distance
    validate_positive_float(min_distance, "min_distance", min_value=0.0)
    
    # Validate mag_mode
    if mag_mode is not None:
        validate_magnetic_mode(mag_mode, "mag_mode")
    
    # Validate rotate_formula
    if rotate_formula is not None:
        if not isinstance(rotate_formula, str):
            raise ValidationError(
                f"rotate_formula must be a string, got {type(rotate_formula).__name__}"
            )
    
    # Validate vac_elements and vac_num
    if vac_elements is not None:
        if not isinstance(vac_elements, (list, tuple, str)):
            raise ValidationError(
                f"vac_elements must be a list, tuple, or string, "
                f"got {type(vac_elements).__name__}"
            )
        if isinstance(vac_elements, str):
            vac_elements = [vac_elements]  # Convert string to list
        for elem in vac_elements:
            if not isinstance(elem, str):
                raise ValidationError(
                    f"vac_elements must contain strings, got {type(elem).__name__}"
                )
        validate_positive_integer(vac_num, "vac_num")
    
    # Validate shuffle_method
    valid_shuffle_methods = ['fisher_yates', 'random']
    if shuffle_method not in valid_shuffle_methods:
        raise ValidationError(
            f"shuffle_method must be one of {valid_shuffle_methods}, "
            f"got '{shuffle_method}'"
        )
    
    # Validate sampler
    validate_sampler(sampler, "sampler")
    
    # Validate rigid parameters
    if rigid:
        if not isinstance(rigid, bool):
            raise ValidationError(
                f"rigid must be a boolean, got {type(rigid).__name__}"
            )
        if vol_pert_fraction > 0:
            logger.warning(
                "Volume perturbation with rigid bodies may cause issues. "
                "Consider setting vol_pert_fraction=0"
            )
    
    # Validate similarity_threshold
    validate_ratio(similarity_threshold, "similarity_threshold")
    
    # Validate that at least one perturbation type is enabled
    perturbation_count = sum([
        surface, gb, dislocation, twinning, stacking_fault, amorphous,
        mag_mode is not None, vac_elements is not None, 
        shuffle_elements is not None, rotate_formula is not None,
        rigid, vol_pert_fraction > 0
    ])
    if perturbation_count == 0:
        logger.warning(
            "No perturbation type enabled. Use skip_normal=True if intentional."
        )


def _safe_generate_surface(atoms: Atoms, **kwargs) -> Atoms:
    """
    Safely generate surface with error handling.
    
    Args:
        atoms: Input structure
        **kwargs: Surface generation parameters
        
    Returns:
        Atoms: Surface structure
        
    Raises:
        PerturbationError: If surface generation fails
    """
    try:
        return generate_surface(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Surface generation failed: {e}",
            operation="surface",
            suggestion="Check surface indices and vacuum parameters"
        ) from e


def _safe_generate_grain_boundary(atoms: Atoms, **kwargs) -> Atoms:
    """
    Safely generate grain boundary with error handling.
    
    Args:
        atoms: Input structure
        **kwargs: GB generation parameters
        
    Returns:
        Atoms: GB structure
        
    Raises:
        PerturbationError: If GB generation fails
    """
    try:
        return generate_grain_boundary(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Grain boundary generation failed: {e}",
            operation="grain_boundary",
            suggestion="Check axis, angle, and overlap distance parameters"
        ) from e


def _safe_generate_dislocation(atoms: Atoms, **kwargs) -> Atoms:
    """
    Safely generate dislocation with error handling.
    
    Args:
        atoms: Input structure
        **kwargs: Dislocation generation parameters
        
    Returns:
        Atoms: Dislocation structure
        
    Raises:
        PerturbationError: If dislocation generation fails
    """
    try:
        return generate_dislocation(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Dislocation generation failed: {e}",
            operation="dislocation",
            suggestion="Check axis, burgers vector, and center parameters"
        ) from e


def _safe_generate_twinning(atoms: Atoms, **kwargs) -> Atoms:
    """
    Safely generate twinning with error handling.
    
    Args:
        atoms: Input structure
        **kwargs: Twinning generation parameters
        
    Returns:
        Atoms: Twinning structure
        
    Raises:
        PerturbationError: If twinning generation fails
    """
    try:
        return generate_twinning(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Twinning generation failed: {e}",
            operation="twinning",
            suggestion="Check Miller indices and minimum distance parameters"
        ) from e


def _safe_generate_stacking_fault(atoms: Atoms, **kwargs) -> Atoms:
    """
    Safely generate stacking fault with error handling.
    
    Args:
        atoms: Input structure
        **kwargs: Stacking fault generation parameters
        
    Returns:
        Atoms: Stacking fault structure
        
    Raises:
        PerturbationError: If stacking fault generation fails
    """
    try:
        return generate_stacking_fault(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Stacking fault generation failed: {e}",
            operation="stacking_fault",
            suggestion="Check plane normal and shift vector parameters"
        ) from e


def _safe_generate_amorphous(atoms: Atoms, **kwargs) -> Atoms:
    """
    Safely generate amorphous structure with error handling.
    
    Args:
        atoms: Input structure
        **kwargs: Amorphous generation parameters
        
    Returns:
        Atoms: Amorphous structure
        
    Raises:
        PerturbationError: If amorphous generation fails
    """
    try:
        return generate_amorphous(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Amorphous generation failed: {e}",
            operation="amorphous",
            suggestion="Check minimum distance and rattle strength parameters"
        ) from e


def _safe_apply_magnetic_perturbation(atoms: Atoms, **kwargs) -> Atoms:
    """
    Safely apply magnetic perturbation with error handling.
    
    Args:
        atoms: Input structure
        **kwargs: Magnetic perturbation parameters
        
    Returns:
        Atoms: Magnetically perturbed structure
        
    Raises:
        PerturbationError: If magnetic perturbation fails
    """
    try:
        return apply_magnetic_perturbation(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Magnetic perturbation failed: {e}",
            operation="magnetic",
            suggestion="Check magnetic mode and configuration parameters"
        ) from e


if __name__ == "__main__":
    pass
