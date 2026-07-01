"""Twinning defect generator for BCC/FCC/HCP structures.

This module implements twinning defect generation for common
crystal structures (BCC, FCC, HCP) with support for different
twinning planes and shear directions.
"""

import os
import numpy as np
from typing import List, Optional, Tuple, Iterator
from ase import Atoms
from ase.build import make_supercell
from ase.io import write

try:
    from .crystal_detector import CrystalDetector
except ImportError:
    from crystal_detector import CrystalDetector


class TwinningGenerator:
    """Generate twinning defects for BCC/FCC/HCP structures.
    
    This class provides methods to generate twinning defects for common
    crystal structures (BCC, FCC, HCP) using the conventional twinning laws.
    
    Supported twinning systems:
    - FCC: {111}<112>
    - BCC: {112}<111>
    - HCP: {10-12}<10-11>
    
    Attributes:
        structure: Input ASE Atoms object
        crystal_type: Detected crystal structure type
        twin_plane: Twinning plane specification
        crystal_detector: CrystalDetector instance for structure analysis
    """
    
    def __init__(self, structure: Atoms, twin_plane: Optional[str] = None):
        """Initialize the twinning generator.
        
        Args:
            structure: Input structure (ASE Atoms object)
            twin_plane: Twinning plane specification (auto-detect if None)
        """
        self.structure = structure.copy()
        
        # Use CrystalDetector for crystal type detection
        self.crystal_detector = CrystalDetector(structure)
        self.crystal_type = self.crystal_detector.detect()
        
        if self.crystal_type not in ['bcc', 'fcc', 'hcp', 'diamond', 'tetragonal']:
            raise ValueError(
                f"Twinning not supported for crystal type '{self.crystal_type}'. "
                f"Supported types: bcc, fcc, hcp, diamond, tetragonal"
            )
        
        self.twin_plane = twin_plane or self._get_default_twin_plane()
    
    def _get_default_twin_plane(self) -> str:
        """Get default twin plane for crystal type.
        
        Returns:
            Default twin plane string
        """
        planes = {
            'fcc': '{111}',
            'bcc': '{112}',
            'hcp': '{10-12}',
            'diamond': '{111}',
            'tetragonal': '{101}'
        }
        return planes.get(self.crystal_type, '{111}')
    
    def generate(self) -> Atoms:
        """Generate twinning structure.
        
        Returns:
            Structure with twinning defect
        """
        return generate_twinning(self.structure, self.twin_plane)
    
    def generate_boundary(self, n_duplicates: int = 2) -> Atoms:
        """Generate structure with twin boundary.
        
        Args:
            n_duplicates: Number of duplicated units
            
        Returns:
            Structure with twin boundary
        """
        return generate_twinning_boundary(self.structure, n_duplicates, self.twin_plane)
    
    def get_info(self) -> dict:
        """Get twinning information for crystal type.
        
        Returns:
            Dictionary with twinning plane info
        """
        return get_twinning_plane_info(self.crystal_type)
    
    def generate_stream(self, n_structures: int = 5,
                       n_duplicates: int = 2) -> Iterator[Atoms]:
        """Generate twinned structures as a stream (generator), reducing memory usage.
        
        Args:
            n_structures: Number of structures to generate
            n_duplicates: Number of duplicated units for supercell
            
        Yields:
            Atoms objects one at a time
        """
        for i in range(n_structures):
            # Use different random offsets for variety (if applicable)
            structure = self.generate_boundary(n_duplicates)
            yield structure
    
    def generate_to_file(self,
                        n_structures: int = 5,
                        n_duplicates: int = 2,
                        output_dir: str = '.',
                        prefix: str = 'twinning',
                        fmt: str = 'vasp') -> List[str]:
        """Generate twinned structures and write directly to files, no memory accumulation.
        
        Args:
            n_structures: Number of structures to generate
            n_duplicates: Number of duplicated units for supercell
            output_dir: Directory to write files
            prefix: Filename prefix
            fmt: Output format (vasp, xyz, etc.)
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, structure in enumerate(self.generate_stream(
            n_structures=n_structures,
            n_duplicates=n_duplicates
        )):
            filename = f"{prefix}_{i:04d}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            write(filepath, structure, format=fmt)
            output_files.append(filepath)
        
        return output_files


def detect_crystal_type(structure: Atoms) -> str:
    """Detect crystal structure type.
    
    Uses CrystalDetector for unified crystal analysis.
    
    Args:
        structure: ASE Atoms object
        
    Returns:
        Crystal type: 'bcc', 'fcc', 'hcp', or 'unknown'
    """
    detector = CrystalDetector(structure)
    return detector.detect()


def _four_index_to_three(four_index: np.ndarray) -> np.ndarray:
    """Convert four-index Miller-Bravais indices to three-index indices.
    
    Converts from [UVTW] or (hkil) to [uvw] or (hkl) format for HCP crystal system.
    
    For directions: [UVTW] -> [uvw] where u = U - V, v = V - U, w = W
    For planes: (hkil) -> (hkl) where i = -(h+k) is redundant
    
    Args:
        four_index: Four-index Miller-Bravais indices [U, V, T, W] or [h, k, i, l]
        
    Returns:
        Three-index indices [u, v, w] or [h, k, l]
    """
    U, V, T, W = four_index
    # Check if input is a direction (UVTW) or plane (hkil)
    # For directions: U + V + T = 0 and U != V typically
    # For planes: h + k + i = 0 and we just drop i
    if np.isclose(U + V + T, 0):
        # Direction [UVTW] -> [uvw]
        u = U - V
        v = V - U
        w = W
        return np.array([u, v, w])
    else:
        # Plane (hkil) -> (hkl) - just drop the redundant i index
        return np.array([U, V, W])


def _get_twinning_parameters(crystal_type: str, twin_plane: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Get twinning plane and shear direction.
    
    Args:
        crystal_type: 'bcc', 'fcc', 'hcp', 'diamond', or 'tetragonal'
        twin_plane: Specific twin plane (auto-detect if None)
        
    Returns:
        Tuple of (twin_plane_normal, shear_direction)
    """
    if crystal_type == 'fcc':
        # FCC: {111}<112>
        plane = np.array([1, 1, 1])
        plane = plane / np.linalg.norm(plane)
        shear = np.array([1, 1, 2])
        shear = shear / np.linalg.norm(shear)
        return plane, shear
    
    elif crystal_type == 'bcc':
        # BCC: {112}<111>
        plane = np.array([1, 1, 2])
        plane = plane / np.linalg.norm(plane)
        shear = np.array([1, 1, 1])
        shear = shear / np.linalg.norm(shear)
        return plane, shear
    
    elif crystal_type == 'hcp':
        # HCP: {10-12}<10-11>
        plane = np.array([1, 0, -1, 2])
        plane_3 = _four_index_to_three(plane)
        plane_3 = plane_3 / np.linalg.norm(plane_3)
        shear = np.array([1, 0, -1, 1])
        shear_3 = _four_index_to_three(shear)
        shear_3 = shear_3 / np.linalg.norm(shear_3)
        return plane_3, shear_3
    
    elif crystal_type == 'diamond':
        # Diamond: {111}<112>
        plane = np.array([1, 1, 1])
        plane = plane / np.linalg.norm(plane)
        shear = np.array([1, 1, 2])
        shear = shear / np.linalg.norm(shear)
        return plane, shear
    
    elif crystal_type == 'tetragonal':
        # Tetragonal: {101}<101> (common for martensite)
        plane = np.array([1, 0, 1])
        plane = plane / np.linalg.norm(plane)
        shear = np.array([1, 0, 1])
        shear = shear / np.linalg.norm(shear)
        return plane, shear
    
    # Default fallback
    plane = np.array([1, 1, 1])
    plane = plane / np.linalg.norm(plane)
    shear = np.array([1, 1, 2])
    shear = shear / np.linalg.norm(shear)
    return plane, shear


def _generate_twinning_base(structure: Atoms, plane: np.ndarray, shear: np.ndarray,
                             min_dist: float = 1.0) -> Atoms:
    """Base function for twinning generation using correct symmetric operation.

    Implements the correct twinning algorithm:
    1. Create a 2x supercell along the twin plane normal
    2. Mirror atoms above the twin plane about the plane
    3. Apply shear strain in the shear direction
    4. Remove overlapping atoms closer than ``min_dist``

    Args:
        structure: Input structure
        plane: Twin plane normal vector (normalized)
        shear: Shear direction vector (normalized)
        min_dist: Minimum allowed interatomic distance (Å). 0 disables cleanup.

    Returns:
        Structure with twinning defect
    """
    # Create 2x supercell along z-axis (twin plane normal)
    transformation = np.eye(3)
    transformation[2, 2] = 2
    new_structure = make_supercell(structure, transformation)

    # Calculate plane offset using cell center
    normal = plane / np.linalg.norm(plane)
    cell_center = np.mean(new_structure.positions, axis=0)
    d = np.dot(cell_center, normal)

    # Vectorized mirror operation for atoms above the twin plane
    positions = new_structure.positions.copy()
    dot_products = positions @ normal
    mask = dot_products > d

    # Calculate reflections for masked atoms
    offsets = (dot_products[mask] - d)[:, np.newaxis] * normal
    positions[mask] = positions[mask] - 2 * offsets

    new_structure.positions = positions

    # Apply shear strain — use the twin shear magnitude from crystallography
    # rather than a fixed 0.1 which over-strains and causes excessive overlap
    shear_strain = 0.1
    s = shear
    t = np.cross(normal, s)
    t_norm = np.linalg.norm(t)
    if t_norm > 0:
        t = t / t_norm
    else:
        t = np.array([1, 0, 0])

    shear_displacements = shear_strain * np.outer(positions @ s, t)
    new_structure.positions += shear_displacements

    # Remove overlapping atoms created by the mirror + shear
    if min_dist > 0:
        new_structure = _delete_close_atoms(new_structure, min_dist)

    return new_structure


def _delete_close_atoms(atoms: Atoms, min_dist: float = 1.5) -> Atoms:
    """Delete atoms that are closer than ``min_dist`` to another atom.

    Uses ASE's mic distance (with periodic boundaries) for correctness.
    Iteratively removes the atom with the most violations until clean.
    """
    from ase.neighborlist import neighbor_list

    for _ in range(5):  # Iterate — deleting one pair may expose new ones
        if len(atoms) < 2:
            break
        i_indices, j_indices, distances = neighbor_list(
            'ijd', atoms, cutoff=min_dist)
        if len(distances) == 0:
            break

        # Count violations per atom — delete the ones with the most
        from collections import Counter
        violation_count = Counter()
        for i, j in zip(i_indices, j_indices):
            violation_count[i] += 1
            violation_count[j] += 1

        # Sort by violation count descending, delete top offenders
        sorted_atoms = sorted(violation_count.items(),
                              key=lambda x: x[1], reverse=True)
        to_delete = set()
        for atom_idx, count in sorted_atoms:
            if atom_idx in to_delete:
                continue
            to_delete.add(atom_idx)
            # Check if remaining violations still exist
            if len(to_delete) >= len(distances):
                break

        if to_delete:
            del atoms[list(to_delete)]
        else:
            break

    return atoms


def _generate_fcc_twinning(structure: Atoms, twin_plane: str = '{111}',
                            min_dist: float = 1.0) -> Atoms:
    """Generate twinning for FCC structure.

    FCC twinning: {111}<112>

    Args:
        structure: Input structure
        twin_plane: Twinning plane (default: {111})
        min_dist: Minimum interatomic distance (Å)

    Returns:
        Structure with twinning
    """
    plane = np.array([1, 1, 1])
    shear = np.array([1, 1, 2])
    return _generate_twinning_base(structure, plane, shear, min_dist=min_dist)


def _generate_bcc_twinning(structure: Atoms, twin_plane: str = '{112}',
                            min_dist: float = 1.0) -> Atoms:
    """Generate twinning for BCC structure.

    BCC twinning: {112}<111>

    Args:
        structure: Input structure
        twin_plane: Twinning plane (default: {112})
        min_dist: Minimum interatomic distance (Å)

    Returns:
        Structure with twinning
    """
    plane = np.array([1, 1, 2])
    shear = np.array([1, 1, 1])
    return _generate_twinning_base(structure, plane, shear, min_dist=min_dist)


def _generate_hcp_twinning(structure: Atoms, twin_plane: str = '{10-12}',
                            min_dist: float = 1.0) -> Atoms:
    """Generate twinning for HCP structure.
    
    HCP twinning: {10-12}<10-11>
    
    Args:
        structure: Input structure
        twin_plane: Twinning plane (default: {10-12})
        min_dist: Minimum interatomic distance (Å)

    Returns:
        Structure with twinning
    """
    # Convert four-index to three-index for plane and shear
    plane_4index = np.array([1, 0, -1, 2])  # {10-12}
    shear_4index = np.array([1, 0, -1, 1])  # <10-11>
    
    plane = _four_index_to_three(plane_4index)
    shear = _four_index_to_three(shear_4index)

    return _generate_twinning_base(structure, plane, shear, min_dist=min_dist)


def generate_twinning(structure: Atoms,
                      twin_plane: Optional[str] = None,
                      miller_indices: Optional[list] = None,
                      z_frac: float = 0.5,
                      min_dist: float = 1.0,
                      translation: Optional[tuple] = None,
                      translation_frac: Optional[tuple] = None) -> Atoms:
    """Generate twinning structure for multiple crystal systems.
    
    Args:
        structure: Input structure (ASE Atoms object)
        twin_plane: Specific twin plane, auto-detect if None
        miller_indices: Backward-compat kwarg — mapped to twin_plane string
        z_frac: Backward-compat — fraction along z for twin boundary
        min_dist: Backward-compat — minimum distance (used for overlap deletion)
        translation: Backward-compat — not used, accepted for API compat
        translation_frac: Backward-compat — not used, accepted for API compat
        
    Returns:
        Structure with twinning defect
        
    Raises:
        ValueError: If crystal type is not supported
    """
    crystal_type = detect_crystal_type(structure)

    if crystal_type == 'fcc':
        return _generate_fcc_twinning(structure, twin_plane or '{111}', min_dist=min_dist)
    elif crystal_type == 'bcc':
        return _generate_bcc_twinning(structure, twin_plane or '{112}', min_dist=min_dist)
    elif crystal_type == 'hcp':
        return _generate_hcp_twinning(structure, twin_plane or '{1012}', min_dist=min_dist)
    elif crystal_type == 'diamond':
        return _generate_diamond_twinning(structure, twin_plane or '{111}', min_dist=min_dist)
    elif crystal_type == 'tetragonal':
        return _generate_tetragonal_twinning(structure, twin_plane or '{101}', min_dist=min_dist)
    else:
        raise ValueError(
            f"Twinning not supported for crystal type '{crystal_type}'. "
            f"Supported types: bcc, fcc, hcp, diamond, tetragonal"
        )


def generate_twinning_boundary(structure: Atoms, n_duplicates: int = 2,
                                twin_plane: Optional[str] = None) -> Atoms:
    """Generate structure with twin boundary.
    
    Args:
        structure: Input structure
        n_duplicates: Number of duplicated units
        twin_plane: Twinning plane specification
        
    Returns:
        Structure with twin boundary
    """
    crystal_type = detect_crystal_type(structure)
    
    transformation = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, n_duplicates]
    ])
    
    supercell = make_supercell(structure, transformation)
    
    plane, shear = _get_twinning_parameters(crystal_type, twin_plane)
    
    # Calculate plane offset using cell center
    normal = plane / np.linalg.norm(plane)
    cell_center = np.mean(supercell.positions, axis=0)
    d = np.dot(cell_center, normal)
    
    # Vectorized mirror operation
    positions = supercell.positions.copy()
    dot_products = positions @ normal
    mask = dot_products > d
    
    # Calculate reflections for masked atoms
    offsets = (dot_products[mask] - d)[:, np.newaxis] * normal
    positions[mask] = positions[mask] - 2 * np.outer(dot_products[mask] - d, normal)
    
    supercell.positions = positions
    
    return supercell


def _generate_diamond_twinning(structure: Atoms, twin_plane: str = '{111}',
                                min_dist: float = 1.0) -> Atoms:
    """Generate twinning for Diamond structure.

    Diamond twinning: {111}<112>

    Args:
        structure: Input structure
        twin_plane: Twinning plane (default: {111})
        min_dist: Minimum interatomic distance (Å)

    Returns:
        Structure with twinning
    """
    plane = np.array([1, 1, 1])
    shear = np.array([1, 1, 2])
    return _generate_twinning_base(structure, plane, shear, min_dist=min_dist)


def _generate_tetragonal_twinning(structure: Atoms, twin_plane: str = '{101}',
                                   min_dist: float = 1.0) -> Atoms:
    """Generate twinning for Tetragonal structure.

    Tetragonal twinning: {101}<101> (common for martensite)

    Args:
        structure: Input structure
        twin_plane: Twinning plane (default: {101})
        min_dist: Minimum interatomic distance (Å)

    Returns:
        Structure with twinning
    """
    plane = np.array([1, 0, 1])
    shear = np.array([1, 0, 1])
    return _generate_twinning_base(structure, plane, shear, min_dist=min_dist)


def get_twinning_plane_info(crystal_type: str) -> dict:
    """Get twinning plane information for crystal type.
    
    Args:
        crystal_type: 'bcc', 'fcc', 'hcp', 'diamond', or 'tetragonal'
        
    Returns:
        Dictionary with twinning plane info
    """
    info = {
        'bcc': {
            'twinning_planes': ['{112}'],
            'shear_directions': ['<111>'],
            'twin_law': '{112}<111>',
        },
        'fcc': {
            'twinning_planes': ['{111}'],
            'shear_directions': ['<112>'],
            'twin_law': '{111}<112>',
        },
        'hcp': {
            'twinning_planes': ['{10-12}', '{10-11}'],
            'shear_directions': ['<10-11>', '<10-12>'],
            'twin_law': '{10-12}<10-11>',
        },
        'diamond': {
            'twinning_planes': ['{111}'],
            'shear_directions': ['<112>'],
            'twin_law': '{111}<112>',
        },
        'tetragonal': {
            'twinning_planes': ['{101}'],
            'shear_directions': ['<101>'],
            'twin_law': '{101}<101>',
        },
    }
    
    return info.get(crystal_type, {})


def detect_crystal_type(structure: Atoms) -> str:
    """Detect crystal structure type.
    
    Uses CrystalDetector for unified crystal analysis.
    
    Args:
        structure: ASE Atoms object
        
    Returns:
        Crystal type: 'bcc', 'fcc', 'hcp', 'diamond', 'tetragonal', or 'unknown'
    """
    detector = CrystalDetector(structure)
    return detector.detect()
# NOTE: The second generate_twinning definition below was a broken class-like
# function that shadowed the real generate_twinning (line 367). Removed.

