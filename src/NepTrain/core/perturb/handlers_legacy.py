"""
Concrete implementations of defect handlers
"""

import numpy as np
from ase import Atoms
from .handlers import (
    DefectHandler, register_handler, 
    _is_range, _parse_range, _interpolate, _discrete_sample
)
from .surface import generate_surface
from .grain_boundary import generate_grain_boundary
from .csl_core import get_rotation_matrix, find_csl_basis
from .dislocation import generate_dislocation
from .twinning import generate_twinning
from .stacking_fault import generate_stacking_fault
from .amorphous import generate_amorphous
from .magnetic import apply_magnetic_perturbation, get_magmom_config
from .rotate import rotate_fragments_by_formula
from .vacancy import generate_vacancies
from .shuffle import shuffle_element_positions


# Helper function for CSL data (compatibility wrapper)
def get_csl_data(sigma, axis, angle_deg, limit=5):
    """Wrapper to maintain compatibility with existing code."""
    M, M_prime = find_csl_basis(sigma, axis, angle_deg, limit=limit)
    return {'M': M, 'M_prime': M_prime} if M is not None else None


@register_handler('surface')
class SurfaceHandler(DefectHandler):
    """Handler for surface slab generation"""
    
    def get_dims(self, atoms: Atoms, **params) -> int:
        dims = 0
        surface_vacuum = params.get('surface_vacuum', 10.0)
        surface_layers = params.get('surface_layers', 3)
        
        if _is_range(surface_vacuum):
            dims += 1
        if _is_range(surface_layers):
            dims += 1
        
        return dims
    
    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        surface_indices = params.get('surface_indices', '1,1,1')
        surface_vacuum = params.get('surface_vacuum', 10.0)
        surface_layers = params.get('surface_layers', 3)
        
        # Parse indices
        if isinstance(surface_indices, str):
            surface_indices = tuple(int(x) for x in surface_indices.split(','))
        
        # Parse vacuum
        idx = 0
        vac_min, vac_max, vac_is_range = _parse_range(surface_vacuum, float)
        curr_vac = vac_min
        if vac_is_range:
            curr_vac = _interpolate(samples[idx], vac_min, vac_max)
            idx += 1
        
        # Parse layers
        lay_min, lay_max, lay_is_range = _parse_range(surface_layers, int)
        curr_lay = lay_min
        if lay_is_range:
            curr_lay = _discrete_sample(samples[idx], lay_min, lay_max)
            idx += 1
        
        result = generate_surface(atoms, indices=surface_indices, vacuum=curr_vac, layers=curr_lay)
        
        # Add annotation
        result.info['perturb_annotation'] = {
            'type': 'surface',
            'metadata': {
                'indices': surface_indices,
                'vacuum': curr_vac,
                'layers': curr_lay
            }
        }
        
        return result


@register_handler('grain_boundary')
class GrainBoundaryHandler(DefectHandler):
    """Handler for grain boundary generation"""
    
    def get_dims(self, atoms: Atoms, **params) -> int:
        dims = 2  # Translation (x, y)
        
        gb_angle = params.get('gb_angle', 36.87)
        gb_dist = params.get('gb_dist', 0.0)
        gb_overlap_dist = params.get('gb_overlap_dist', 1.2)
        
        # Angle dimension
        if gb_angle == 'random' or _is_range(gb_angle) or gb_angle == 'csl':
            dims += 1
        
        # Distance dimensions
        if _is_range(gb_dist):
            dims += 1
        if _is_range(gb_overlap_dist):
            dims += 1
        
        return dims
    
    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        gb_axis = params.get('gb_axis', '0,0,1')
        gb_angle = params.get('gb_angle', 36.87)
        gb_dist = params.get('gb_dist', 0.0)
        gb_overlap_dist = params.get('gb_overlap_dist', 1.2)
        gb_delete_overlap = params.get('gb_delete_overlap', True)
        
        # Parse axis
        gb_axis_vec = gb_axis
        if isinstance(gb_axis, str):
            try:
                gb_axis_vec = [float(x) for x in gb_axis.split(',')]
            except:
                gb_axis_vec = [0, 0, 1]
        
        # Parse translation (Fractional from Sobol)
        t_frac_x = samples[0]
        t_frac_y = samples[1]
        idx = 2
        
        # Parse angle
        angle_min, angle_max, angle_is_range = _parse_range(gb_angle, float)
        curr_angle = angle_min
        curr_sigma = None
        
        if gb_angle == 'random':
            curr_angle = 15.0 + samples[idx] * 75.0  # [15, 90]
            idx += 1
        elif angle_is_range:
            curr_angle = _interpolate(samples[idx], angle_min, angle_max)
            idx += 1
        elif gb_angle == 'csl':
            csl_data = get_csl_data(gb_axis_vec)
            if csl_data:
                c_idx = int(samples[idx] * len(csl_data))
                if c_idx == len(csl_data):
                    c_idx -= 1
                curr_angle = csl_data[c_idx]['angle']
                curr_sigma = csl_data[c_idx]['sigma']
            else:
                curr_angle = 36.87
                curr_sigma = 5
            idx += 1
        
        # Parse distances
        dist_min, dist_max, dist_is_range = _parse_range(gb_dist, float)
        curr_gb_dist = dist_min
        if dist_is_range:
            curr_gb_dist = _interpolate(samples[idx], dist_min, dist_max)
            idx += 1
        
        overlap_min, overlap_max, overlap_is_range = _parse_range(gb_overlap_dist, float)
        curr_gb_overlap = overlap_min
        if overlap_is_range:
            curr_gb_overlap = _interpolate(samples[idx], overlap_min, overlap_max)
            idx += 1
        
        # Apply GB generation
        # Map to correct parameter names
        # element is the first positional argument (symbol string, not Atoms object)
        # Extract unique element symbol from atoms
        symbols = atoms.get_chemical_symbols()
        unique_symbols = list(set(symbols))
        if len(unique_symbols) == 1:
            element_symbol = unique_symbols[0]
        else:
            element_symbol = symbols[0]  # Use first symbol if multiple elements
        
        result = generate_grain_boundary(
            element=element_symbol,
            axis=gb_axis_vec, 
            angle=curr_angle,
            sigma=curr_sigma if curr_sigma else 5,
            vacuum=curr_gb_dist, 
            delete_overlap=gb_delete_overlap,
            tol=curr_gb_overlap
        )
        
        # Ensure annotation
        if 'perturb_annotation' not in result.info:
            result.info['perturb_annotation'] = {}
        
        result.info['perturb_annotation']['type'] = 'grain_boundary'
        if curr_sigma is not None:
             result.info['perturb_annotation']['sigma'] = curr_sigma
        
        return result


@register_handler('dislocation')
class DislocationHandler(DefectHandler):
    """Handler for dislocation generation"""
    
    def get_dims(self, atoms: Atoms, **params) -> int:
        dims = 2  # Center (x, y) in plane perpendicular to axis
        
        dislocation_type = params.get('dislocation_type', 'edge')
        if dislocation_type == 'random':
            dims += 1  # Type selection
        
        return dims
    
    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        dislocation_type = params.get('dislocation_type', 'edge')
        dislocation_axis = params.get('dislocation_axis', '0,0,1')
        dislocation_burgers = params.get('dislocation_burgers', '1,0,0')
        
        # Normalize axis to integer
        if isinstance(dislocation_axis, str):
            if ',' in dislocation_axis:
                parts = [float(x) for x in dislocation_axis.split(',')]
                d_axis_idx = int(np.argmax(np.abs(parts)))
            elif dislocation_axis.lower() in ['x', '0']:
                d_axis_idx = 0
            elif dislocation_axis.lower() in ['y', '1']:
                d_axis_idx = 1
            elif dislocation_axis.lower() in ['z', '2']:
                d_axis_idx = 2
            else:
                d_axis_idx = 2
        elif isinstance(dislocation_axis, (int, np.integer)):
            d_axis_idx = int(dislocation_axis)
        else:
            d_axis_idx = int(np.argmax(np.abs(dislocation_axis)))
        
        # Normalize burgers to magnitude and vector for storage
        burgers_to_store = dislocation_burgers
        
        if isinstance(dislocation_burgers, str):
            burgers_vec = [float(x) for x in dislocation_burgers.split(',')]
            burgers_mag = float(np.linalg.norm(burgers_vec))
            burgers_to_store = burgers_vec
        elif isinstance(dislocation_burgers, (list, tuple, np.ndarray)):
            burgers_mag = float(np.linalg.norm(dislocation_burgers))
            burgers_to_store = list(dislocation_burgers)
        else:
            burgers_mag = float(dislocation_burgers)
            burgers_to_store = float(dislocation_burgers)
        
        # Parse type
        curr_type = dislocation_type
        idx = 0
        c_frac = samples[idx:idx+2]
        idx += 2
        
        if dislocation_type == 'random':
            curr_type = 'edge' if samples[idx] < 0.5 else 'screw'
            idx += 1
        
        # Calculate center
        cell_diag = atoms.cell.lengths()
        if d_axis_idx == 0:  # X axis, center in YZ
            curr_center = [cell_diag[0]/2, c_frac[0]*cell_diag[1], c_frac[1]*cell_diag[2]]
        elif d_axis_idx == 1:  # Y axis, center in XZ
            curr_center = [c_frac[0]*cell_diag[0], cell_diag[1]/2, c_frac[1]*cell_diag[2]]
        else:  # Z axis, center in XY
            curr_center = [c_frac[0]*cell_diag[0], c_frac[1]*cell_diag[1], cell_diag[2]/2]
        
        result = generate_dislocation(
            atoms, type=curr_type, axis=d_axis_idx, 
            burgers=burgers_mag, center=curr_center
        )
        
        # Add annotation with keys expected by plot.py
        # axis must be integer (0,1,2)
        # sub_type must be 'edge' or 'screw'
        result.info['perturb_annotation'] = {
            'type': 'dislocation',
            'metadata': {
                'sub_type': curr_type,
                'axis': d_axis_idx,
                'burgers': burgers_to_store, # Pass vector or scalar for direction
                'center': curr_center
            }
        }
        
        return result


@register_handler('volume')
class VolumeHandler(DefectHandler):
    """Handler for isotropic volume scaling"""
    
    def get_dims(self, atoms: Atoms, **params) -> int:
        vol_pert_fraction = params.get('vol_pert_fraction', 0.0)
        return 1 if vol_pert_fraction > 0 else 0
    
    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        vol_pert_fraction = params.get('vol_pert_fraction', 0.0)
        
        if vol_pert_fraction <= 0:
            return atoms
        
        # Scale: V' = V * (1 + delta), delta in [-f, f]
        delta = (samples[0] * 2 - 1) * vol_pert_fraction
        vol_scale_factor = 1.0 + delta
        length_scale = vol_scale_factor ** (1/3)
        
        result = atoms.copy()
        result.set_cell(result.cell * length_scale, scale_atoms=True)
        
        # Add to metadata if annotation exists
        if 'perturb_annotation' in result.info:
            if 'metadata' not in result.info['perturb_annotation']:
                result.info['perturb_annotation']['metadata'] = {}
            result.info['perturb_annotation']['metadata']['vol_scale'] = vol_scale_factor
        
        return result


# === Merged from handler_impl2.py ===

class TwinningHandler(DefectHandler):
    """Handler for twin boundary generation"""

    def get_dims(self, atoms: Atoms, **params) -> int:
        dims = 2  # Translation (x, y)

        twinning_z = params.get('twinning_z', 0.5)
        if twinning_z == 'random':
            dims += 1

        return dims

    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        twinning_indices = params.get('twinning_indices', (1,1,2))
        twinning_z = params.get('twinning_z', 0.5)
        twinning_min_dist = params.get('twinning_min_dist', 1.5)

        # Parse indices
        if isinstance(twinning_indices, str):
            twinning_indices = tuple(int(x) for x in twinning_indices.split(','))

        # Parse translation
        t_frac_x = samples[0]
        t_frac_y = samples[1]
        idx = 2

        # Parse z position
        curr_z = twinning_z
        if twinning_z == 'random':
            curr_z = 0.3 + samples[idx] * 0.4  # [0.3, 0.7]
            idx += 1

        # Use twinning_indices to determine the twin plane for generate_twinning_boundary
        # twinning_indices is passed directly to generate_twinning_boundary as twin_plane

        n_duplicates = int(2 + curr_z * 4)  # 2-6 duplicates based on z_frac
        result = generate_twinning_boundary(
            structure=atoms,
            n_duplicates=n_duplicates,
            twin_plane=str(twinning_indices)
        )

        # Add annotation
        result.info['perturb_annotation'] = {
            'type': 'twinning',
            'metadata': {
                'indices': twinning_indices,
                'n_duplicates': int(2 + curr_z * 4)
            }
        }

        return result


@register_handler('stacking_fault')
class StackingFaultHandler(DefectHandler):
    """Handler for stacking fault generation"""

    def get_dims(self, atoms: Atoms, **params) -> int:
        dims = 0

        sf_shift = params.get('sf_shift', '0,0,0')
        sf_height = params.get('sf_height', 0.5)

        if sf_shift == 'random':
            dims += 2  # Shift (x, y)

        if sf_height == 'random':
            dims += 1  # Height (z)

        return dims

    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        sf_normal = params.get('sf_normal', '1,1,1')
        sf_shift = params.get('sf_shift', '0,0,0')
        sf_height = params.get('sf_height', 0.5)
        sf_min_dist = params.get('sf_min_dist', 1.5)

        # Parse normal
        if isinstance(sf_normal, str):
            sf_normal = [float(x) for x in sf_normal.split(',')]

        # Parse shift
        idx = 0
        if sf_shift == 'random':
            shift_x = samples[idx]
            shift_y = samples[idx+1]
            curr_shift = [shift_x, shift_y, 0.0]
            idx += 2
        else:
            if isinstance(sf_shift, str):
                curr_shift = [float(x) for x in sf_shift.split(',')]
            else:
                curr_shift = sf_shift

        # Parse height
        curr_height = sf_height
        if sf_height == 'random':
            curr_height = 0.3 + samples[idx] * 0.4  # [0.3, 0.7]
            idx += 1

        result = generate_sf_func(
            atoms, sf_normal,
            shift_vector=curr_shift,
            plane_height_frac=curr_height, min_dist=sf_min_dist
        )

        # Add annotation
        result.info['perturb_annotation'] = {
            'type': 'stacking_fault',
            'metadata': {
                'normal': sf_normal,
                'shift': curr_shift,
                'height': curr_height,
                'z_frac': curr_height
            }
        }

        return result


@register_handler('amorphous')
class AmorphousHandler(DefectHandler):
    """Handler for amorphization"""

    def get_dims(self, atoms: Atoms, **params) -> int:
        return 1  # Random seed

    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        amorphous_min_dist = params.get('amorphous_min_dist', 1.8)

        # Use sample as seed
        seed = int(samples[0] * 1e9)

        result = generate_amorphous(atoms, min_dist=amorphous_min_dist, seed=seed)

        # Add annotation
        result.info['perturb_annotation'] = {
            'type': 'amorphous',
            'metadata': {
                'min_dist': amorphous_min_dist,
                'seed': seed
            }
        }

        return result


@register_handler('magnetic')
class MagneticHandler(DefectHandler):
    """Handler for magnetic moment perturbations"""

    def get_dims(self, atoms: Atoms, **params) -> int:
        mag_mode = params.get('mag_mode', None)
        if not mag_mode:
            return 0

        mag_config = params.get('_mag_config', None)
        if mag_config is None:
            mag_config = get_magmom_config()

        return get_magnetic_perturbation_dims(atoms, mode=mag_mode, mag_config=mag_config)

    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        mag_mode = params.get('mag_mode', None)
        mag_noise = params.get('mag_noise', 0.0)
        mag_kwargs = params.get('mag_kwargs', None)

        if not mag_mode:
            return atoms

        mag_config = params.get('_mag_config', None)
        if mag_config is None:
            mag_config = get_magmom_config()

        result = apply_magnetic_perturbation(
            atoms, mode=mag_mode, noise=mag_noise,
            rng_values=samples, mag_config=mag_config,
            **(mag_kwargs or {})
        )

        return result


@register_handler('rotation')
class RotationHandler(DefectHandler):
    """Handler for fragment rotation"""

    def get_dims(self, atoms: Atoms, **params) -> int:
        rotate_formula = params.get('rotate_formula', None)
        if not rotate_formula:
            return 0

        # Each fragment needs 3 rotation angles
        # Count fragments based on formula
        # For simplicity, assume 1 fragment = 3 dims
        # TODO: Parse formula to count fragments
        return 3

    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        rotate_formula = params.get('rotate_formula', None)

        if not rotate_formula:
            return atoms

        # Convert samples to rotation angles
        # Each sample in [0,1] -> angle in [0, 2π]
        angles = samples * 2 * np.pi

        result = rotate_fragments_by_formula(atoms, formula=rotate_formula, rng_values=angles)

        return result


@register_handler('vacancy')
class VacancyHandler(DefectHandler):
    """Handler for vacancy generation"""

    def get_dims(self, atoms: Atoms, **params) -> int:
        vac_num = params.get('vac_num', 0)
        if vac_num <= 0:
            return 0

        # Need vac_num samples to select which atoms to remove
        return vac_num

    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        vac_elements = params.get('vac_elements', None)
        vac_num = params.get('vac_num', 0)

        if vac_num <= 0:
            return atoms

        atoms_with_x, vac_metadata = generate_vacancies(
            atoms, elements=vac_elements, num=vac_num, rng_values=samples
        )

        # Filter out 'X' atoms to get actual vacancies
        result = _filter_vacancies_for_export(atoms_with_x)

        # Add annotation with vacancy positions from metadata
        result.info['perturb_annotation'] = {
            'type': 'vacancy',
            'metadata': vac_metadata # Dictionary of {index: {position, original_symbol}}
        }

        return result


@register_handler('shuffle')
class ShuffleHandler(DefectHandler):
    """Handler for element shuffling"""

    def get_dims(self, atoms: Atoms, **params) -> int:
        shuffle_elements = params.get('shuffle_elements', None)
        if not shuffle_elements:
            return 0

        # Count how many atoms will be shuffled
        if isinstance(shuffle_elements, str):
            # Could be element names or slice notation
            if ',' in shuffle_elements:
                # Element names
                elem_list = shuffle_elements.split(',')
                count = sum(1 for s in atoms.get_chemical_symbols() if s in elem_list)
            else:
                # Single element
                count = sum(1 for s in atoms.get_chemical_symbols() if s == shuffle_elements)
        elif isinstance(shuffle_elements, slice):
            count = len(range(*shuffle_elements.indices(len(atoms))))
        else:
            count = len(atoms)

        return count

    def apply(self, atoms: Atoms, samples: np.ndarray, **params) -> Atoms:
        shuffle_elements = params.get('shuffle_elements', None)
        shuffle_method = params.get('shuffle_method', 'fisher_yates')

        if not shuffle_elements:
            return atoms

        result = shuffle_element_positions(
            atoms, elements=shuffle_elements, method=shuffle_method, rng_values=samples
        )

        # Add annotation
        result.info['perturb_annotation'] = {
            'type': 'shuffle',
            'metadata': {
                'elements': shuffle_elements,
                'method': shuffle_method
            }
        }

        return result
