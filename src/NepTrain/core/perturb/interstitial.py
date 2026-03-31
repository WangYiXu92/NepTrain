"""Interstitial defect generator for BCC/FCC/HCP structures.

This module implements interstitial defect generation for common
crystal structures (BCC, FCC, HCP) with support for both tetrahedral
and octahedral interstitial sites.
"""

import os
import numpy as np
from typing import List, Literal, Optional, Tuple, Iterator
from ase import Atoms
from ase.spacegroup import get_spacegroup
from ase.io import write

try:
    from .crystal_detector import CrystalDetector
except ImportError:
    from crystal_detector import CrystalDetector


class InterstitialGenerator:
    """Generate interstitial defects for BCC/FCC/HCP/Diamond/Tetragonal structures.
    
    This class provides methods to identify interstitial sites and
    generate structures with interstitial atoms for common crystal
    structures (BCC, FCC, HCP, Diamond, Tetragonal).
    
    Supported site types:
    - Tetrahedral sites
    - Octahedral sites
    - Hexagonal sites (Diamond only)
    
    Supported crystal structures:
    - BCC (Body-Centered Cubic)
    - FCC (Face-Centered Cubic)
    - HCP (Hexagonal Close-Packed)
    - Diamond (Diamond cubic)
    - Tetragonal (Tetragonal)
    
    Attributes:
        structure: Input ASE Atoms object
        interstitial_element: Element to add as interstitial
        crystal_type: Detected crystal structure type
        crystal_detector: CrystalDetector instance for structure analysis
    """
    
    def __init__(self, structure: Atoms, interstitial_element: str):
        """Initialize the interstitial generator.
        
        Args:
            structure: Input structure (ASE Atoms object)
            interstitial_element: Element symbol for interstitial atoms
            
        Raises:
            ValueError: If crystal structure is not supported
        """
        self.structure = structure.copy()
        self.interstitial_element = interstitial_element
        
        # Use CrystalDetector for crystal type detection
        self.crystal_detector = CrystalDetector(structure)
        self.crystal_type = self.crystal_detector.detect()
        
        if self.crystal_type not in ['bcc', 'fcc', 'hcp', 'diamond', 'tetragonal']:
            raise ValueError(
                f"Crystal structure {self.crystal_type} not supported. "
                f"Supported types: bcc, fcc, hcp, diamond, tetragonal"
            )
        
        # Get lattice parameters from detector
        a, b, c = self.crystal_detector.get_lattice_parameters()
        angles = structure.cell.angles()
        
        self._lattice_params = {
            'a': a,
            'b': b,
            'c': c,
            'alpha': angles[0],
            'beta': angles[1],
            'gamma': angles[2],
        }
    
    def _get_lattice_parameters(self) -> dict:
        """Get lattice parameters from structure.
        
        Returns:
            Dictionary with lattice parameters
        """
        return {
            'a': self.structure.cell.lengths()[0],
            'b': self.structure.cell.lengths()[1],
            'c': self.structure.cell.lengths()[2],
            'alpha': self.structure.cell.angles()[0],
            'beta': self.structure.cell.angles()[1],
            'gamma': self.structure.cell.angles()[2],
        }
    
    def get_interstitial_sites(self, site_type: str = 'tetrahedral') -> np.ndarray:
        """Get interstitial site positions.
        
        Args:
            site_type: Type of interstitial site ('tetrahedral', 'octahedral', or 'hexagonal' for diamond)
            
        Returns:
            Array of shape (n_sites, 3) with fractional coordinates
            
        Raises:
            ValueError: If site_type is not supported
        """
        if site_type not in ['tetrahedral', 'octahedral', 'hexagonal']:
            raise ValueError(
                f"Site type '{site_type}' not supported. "
                f"Supported types: tetrahedral, octahedral, hexagonal (diamond)"
            )
        
        if self.crystal_type == 'bcc':
            return self._get_bcc_sites(site_type)
        elif self.crystal_type == 'fcc':
            return self._get_fcc_sites(site_type)
        elif self.crystal_type == 'hcp':
            return self._get_hcp_sites(site_type)
        elif self.crystal_type == 'diamond':
            return self._get_diamond_sites(site_type)
        elif self.crystal_type == 'tetragonal':
            return self._get_tetragonal_sites(site_type)
        else:
            raise ValueError(f"Unsupported crystal type: {self.crystal_type}")
    
    def _get_bcc_sites(self, site_type: str) -> np.ndarray:
        """Get interstitial sites for BCC structure.
        
        BCC unit cell:
        - Atoms at (0, 0, 0) and (1/2, 1/2, 1/2)
        
        Interstitial sites:
        - Tetrahedral: (1/4, 1/2, 0) and equivalent positions
        - Octahedral: Corrected to 6 sites (removed duplicate (1/2,1/2,0))
        
        Args:
            site_type: 'tetrahedral' or 'octahedral'
            
        Returns:
            Array of fractional coordinates
        """
        if site_type == 'tetrahedral':
            # 12 tetrahedral sites per unit cell
            sites = [
                (1/4, 1/2, 0), (3/4, 1/2, 0),
                (1/4, 0, 1/2), (3/4, 0, 1/2),
                (0, 1/4, 1/2), (0, 3/4, 1/2),
                (1/2, 1/4, 0), (1/2, 3/4, 0),
                (1/2, 0, 1/4), (1/2, 0, 3/4),
                (0, 1/2, 1/4), (0, 1/2, 3/4),
            ]
            return np.array(sites)
        
        else:  # octahedral
            # 6 octahedral sites per unit cell (corrected)
            # Edge centers: (1/2,0,0), (0,1/2,0), (0,0,1/2)
            # Face centers: (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)
            sites = [
                (1/2, 0, 0), (0, 1/2, 0), (0, 0, 1/2),  # Edge centers
                (1/2, 1/2, 0), (1/2, 0, 1/2), (0, 1/2, 1/2),  # Face centers
            ]
            return np.array(sites)
    
    def _get_fcc_sites(self, site_type: str) -> np.ndarray:
        """Get interstitial sites for FCC structure.
        
        FCC unit cell:
        - Atoms at (0, 0, 0), (1/2, 1/2, 0), (1/2, 0, 1/2), (0, 1/2, 1/2)
        
        Interstitial sites:
        - Tetrahedral: (1/4, 1/4, 1/4) and equivalent positions
        - Octahedral: (1/2, 1/2, 1/2) and edge center positions
        
        Args:
            site_type: 'tetrahedral' or 'octahedral'
            
        Returns:
            Array of fractional coordinates
        """
        if site_type == 'tetrahedral':
            # 8 tetrahedral sites per unit cell
            sites = []
            for i in [1/4, 3/4]:
                for j in [1/4, 3/4]:
                    for k in [1/4, 3/4]:
                        sites.append((i, j, k))
            return np.array(sites)
        
        else:  # octahedral
            # 4 octahedral sites per unit cell
            sites = [
                (1/2, 1/2, 1/2),  # Body center
                (1/2, 1/2, 0), (1/2, 0, 1/2), (0, 1/2, 1/2),  # Face centers
            ]
            return np.array(sites)
    
    def _get_hcp_sites(self, site_type: str) -> np.ndarray:
        """Get interstitial sites for HCP structure.
        
        HCP unit cell:
        - Atoms at (0, 0, 0) and (2/3, 1/3, 1/2)
        
        Interstitial sites:
        - Tetrahedral: (1/3, 2/3, 1/4) and equivalent positions
        - Octahedral: (0, 0, 1/4) and equivalent positions
        
        Args:
            site_type: 'tetrahedral' or 'octahedral'
            
        Returns:
            Array of fractional coordinates
        """
        if site_type == 'tetrahedral':
            # 6 tetrahedral sites per unit cell
            sites = [
                (1/3, 2/3, 1/4), (2/3, 1/3, 3/4),
                (1/3, 2/3, 3/4), (2/3, 1/3, 1/4),
                (1/2, 0, 1/4), (0, 1/2, 3/4),
            ]
            return np.array(sites)
        
        else:  # octahedral
            # 2 octahedral sites per unit cell
            sites = [
                (0, 0, 1/4), (0, 0, 3/4),
                (2/3, 1/3, 1/4), (2/3, 1/3, 3/4),
            ]
            return np.array(sites)
    
    def _get_diamond_sites(self, site_type: str) -> np.ndarray:
        """Get interstitial sites for Diamond structure.
        
        Diamond structure:
        - Atoms at (0, 0, 0), (1/2, 1/2, 0), (1/2, 0, 1/2), (0, 1/2, 1/2),
                  (1/4, 1/4, 1/4), (3/4, 3/4, 1/4), (3/4, 1/4, 3/4), (1/4, 3/4, 3/4)
        - Space group Fd-3m (227)
        - 8 atoms per unit cell
        
        Interstitial sites:
        - Tetrahedral: At diamond positions but offset by (1/2, 1/2, 1/2)
        - Hexagonal: At bond midpoints
        
        Args:
            site_type: 'tetrahedral' or 'hexagonal'
            
        Returns:
            Array of fractional coordinates
        """
        if site_type == 'tetrahedral':
            # Tetrahedral sites (offset from atomic positions by (1/2, 1/2, 1/2))
            sites = [
                (1/2, 1/2, 1/2), (0, 0, 1/2), (0, 1/2, 0), (1/2, 0, 0),
                (3/4, 3/4, 3/4), (1/4, 1/4, 3/4), (1/4, 3/4, 1/4), (3/4, 1/4, 3/4),
            ]
            return np.array(sites)
        
        elif site_type == 'hexagonal':
            # Hexagonal sites (bond midpoints in diamond structure)
            sites = [
                (1/8, 1/8, 1/8), (3/8, 3/8, 1/8), (3/8, 1/8, 3/8), (1/8, 3/8, 3/8),
                (5/8, 5/8, 1/8), (7/8, 7/8, 1/8), (7/8, 5/8, 3/8), (5/8, 7/8, 3/8),
                (5/8, 1/8, 5/8), (7/8, 3/8, 5/8), (1/8, 5/8, 5/8), (3/8, 7/8, 5/8),
                (1/8, 3/8, 7/8), (3/8, 1/8, 7/8), (5/8, 7/8, 7/8), (7/8, 5/8, 7/8),
            ]
            return np.array(sites)
        
        else:
            raise ValueError(f"Unsupported site type for diamond: {site_type}")
    
    def _get_tetragonal_sites(self, site_type: str) -> np.ndarray:
        """Get interstitial sites for Tetragonal structure.
        
        Tetragonal structure (similar to FCC but c/a ≠ 1):
        - a = b ≠ c
        - α = β = γ = 90°
        
        Interstitial sites:
        - Tetrahedral: Similar to FCC but adjusted for c/a ratio
        - Octahedral: At standard positions
        
        Args:
            site_type: 'tetrahedral' or 'octahedral'
            
        Returns:
            Array of fractional coordinates
        """
        if site_type == 'tetrahedral':
            # Tetrahedral sites (similar to FCC, adjusted for c/a)
            sites = [
                (1/2, 0, 1/4), (1/2, 0, 3/4),
                (0, 1/2, 1/4), (0, 1/2, 3/4),
                (1/2, 1/2, 0), (1/2, 1/2, 1/2),
            ]
            return np.array(sites)
        
        elif site_type == 'octahedral':
            sites = [
                (1/2, 1/2, 0), (1/2, 1/2, 1/2),
                (0, 0, 1/4), (0, 0, 3/4),
                (1/2, 0, 0), (1/2, 0, 1/2),
                (0, 1/2, 0), (0, 1/2, 1/2),
            ]
            return np.array(sites)
        
        else:
            raise ValueError(f"Unsupported site type for tetragonal: {site_type}")
    
    def generate(self, n_interstitials: int = 1,
                 site_type: str = 'tetrahedral',
                 selection_method: str = 'sequential') -> Atoms:
        """Generate structure with interstitials.
        
        Args:
            n_interstitials: Number of interstitial atoms to add
            site_type: Type of interstitial site to use
            selection_method: How to select sites ('sequential' or 'random')
            
        Returns:
            New Atoms object with interstitial atoms added
            
        Raises:
            ValueError: If selection_method is not supported
            ValueError: If n_interstitials > available sites
        """
        sites = self.get_interstitial_sites(site_type)
        
        if len(sites) == 0:
            raise ValueError(f"No {site_type} sites found for {self.crystal_type}")
        
        # Check available sites
        if selection_method == 'sequential' and n_interstitials > len(sites):
            raise ValueError(
                f"Requested {n_interstitials} interstitials but only {len(sites)} "
                f"available sites of type '{site_type}' for {self.crystal_type}"
            )
        
        # Select sites
        if selection_method == 'sequential':
            selected_sites = sites[:min(n_interstitials, len(sites))]
        elif selection_method == 'random':
            if n_interstitials > len(sites):
                # Repeat sites if needed
                indices = np.random.choice(
                    len(sites), size=n_interstitials, replace=True
                )
                selected_sites = sites[indices]
            else:
                indices = np.random.choice(len(sites), size=n_interstitials, replace=False)
                selected_sites = sites[indices]
        else:
            raise ValueError(
                f"Selection method '{selection_method}' not supported. "
                f"Supported: sequential, random"
            )
        
        # Create new structure
        new_structure = self.structure.copy()
        
        # Vectorized conversion: fractional to cartesian coordinates
        # Optimization: matrix multiplication instead of loop
        selected_sites_array = np.asarray(selected_sites)
        cart_positions = selected_sites_array @ self.structure.cell
        
        for cart_pos in cart_positions:
            new_atom = Atoms(self.interstitial_element, positions=[cart_pos])
            new_structure += new_atom
        
        return new_structure
    
    def generate_multiple(self, n_structures: int = 5,
                          n_interstitials: int = 1,
                          site_type: str = 'tetrahedral') -> List[Atoms]:
        """Generate multiple structures with random interstitial placements.
        
        Args:
            n_structures: Number of structures to generate
            n_interstitials: Number of interstitials per structure
            site_type: Type of interstitial site
            
        Returns:
            List of Atoms objects
        """
        structures = []
        for i in range(n_structures):
            # Set random seed for this structure
            np.random.seed(42 + i)
            
            structure = self.generate(
                n_interstitials=n_interstitials,
                site_type=site_type,
                selection_method='random'
            )
            structures.append(structure)
        
        return structures

    def generate_stream(self, 
                       n_structures: int = 5,
                       n_interstitials: int = 1,
                       site_type: str = 'tetrahedral',
                       random_seed_offset: int = 42) -> Iterator[Atoms]:
        """Generate structures as a stream (generator), reducing memory usage.
        
        Args:
            n_structures: Number of structures to generate
            n_interstitials: Number of interstitials per structure
            site_type: Type of interstitial site
            random_seed_offset: Base seed for random generation
            
        Yields:
            Atoms objects one at a time
        """
        for i in range(n_structures):
            # Set random seed for this structure
            np.random.seed(random_seed_offset + i)
            
            structure = self.generate(
                n_interstitials=n_interstitials,
                site_type=site_type,
                selection_method='random'
            )
            yield structure

    def generate_to_file(self,
                        n_structures: int = 5,
                        n_interstitials: int = 1,
                        site_type: str = 'tetrahedral',
                        output_dir: str = '.',
                        prefix: str = 'interstitial',
                        fmt: str = 'vasp',
                        random_seed_offset: int = 42) -> List[str]:
        """Generate structures and write directly to files, no memory accumulation.
        
        Args:
            n_structures: Number of structures to generate
            n_interstitials: Number of interstitials per structure
            site_type: Type of interstitial site
            output_dir: Directory to write files
            prefix: Filename prefix
            fmt: Output format (vasp, xyz, etc.)
            random_seed_offset: Base seed for random generation
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, structure in enumerate(self.generate_stream(
            n_structures=n_structures,
            n_interstitials=n_interstitials,
            site_type=site_type,
            random_seed_offset=random_seed_offset
        )):
            filename = f"{prefix}_{i:04d}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            write(filepath, structure, format=fmt)
            output_files.append(filepath)
        
        return output_files


def generate_interstitial(structure: Atoms, element: str,
                          n_interstitials: int = 1,
                          site_type: str = 'tetrahedral') -> Atoms:
    """Convenience function to add interstitial atoms.
    
    Args:
        structure: Input structure
        element: Element symbol for interstitial
        n_interstitials: Number of interstitials to add
        site_type: Type of interstitial site
        
    Returns:
        Structure with interstitial atoms
    """
    generator = InterstitialGenerator(structure, element)
    return generator.generate(n_interstitials, site_type)


class BatchGenerator:
    """Batch structure generator supporting multiple perturbation types.
    
    This class implements the iterator protocol to support streaming
    generation of multiple perturbed structures from a list of base
    structures. Supports various perturbation types including interstitial,
    twinning, and other perturbation modules.
    
    Attributes:
        base_structures: List of base ASE Atoms structures
        params: Dictionary of perturbation parameters
    """
    
    def __init__(self, 
                 base_structures: List[Atoms],
                 perturbation_params: dict):
        """Initialize the batch generator.
        
        Args:
            base_structures: List of base structures to perturb
            perturbation_params: Parameters dict with 'type' and type-specific args
            
        Example:
            >>> base_structures = [bulk('Fe', 'bcc'), bulk('Cu', 'fcc')]
            >>> params = {'type': 'interstitial', 'element': 'C', 'n': 1}
            >>> batch = BatchGenerator(base_structures, params)
            >>> for structure in batch:
            ...     print(len(structure))
        """
        self.base_structures = base_structures
        self.params = perturbation_params
        
        # Validate perturbation type
        ptype = perturbation_params.get('type')
        if ptype not in ['interstitial', 'twinning']:
            raise ValueError(
                f"Perturbation type '{ptype}' not supported. "
                f"Supported types: interstitial, twinning"
            )
    
    def __iter__(self) -> Iterator[Atoms]:
        """Return iterator over perturbed structures.
        
        Yields:
            Perturbed Atoms objects
        """
        for base in self.base_structures:
            yield self._generate_perturbation(base)
    
    def __len__(self) -> int:
        """Return number of base structures.
        
        Returns:
            Number of structures that will be generated
        """
        return len(self.base_structures)
    
    def __getitem__(self, index: int) -> Atoms:
        """Get perturbed structure by index.
        
        Args:
            index: Structure index
            
        Returns:
            Perturbed Atoms object
        """
        return self._generate_perturbation(self.base_structures[index])
    
    def _generate_perturbation(self, base: Atoms) -> Atoms:
        """Generate perturbation for a single base structure.
        
        Args:
            base: Base structure
            
        Returns:
            Perturbed structure
        """
        ptype = self.params.get('type')
        
        if ptype == 'interstitial':
            element = self.params.get('element')
            n_interstitials = self.params.get('n', 1)
            site_type = self.params.get('site_type', 'tetrahedral')
            
            if element is None:
                raise ValueError("'element' required for interstitial perturbation")
            
            gen = InterstitialGenerator(base, element)
            return gen.generate(n_interstitials, site_type)
            
        elif ptype == 'twinning':
            twin_plane = self.params.get('twin_plane')
            gen = TwinningGenerator(base, twin_plane)
            return gen.generate()
            
        else:
            raise ValueError(
                f"Perturbation type '{ptype}' not supported. "
                f"Supported types: interstitial, twinning"
            )
    
    def to_list(self) -> List[Atoms]:
        """Generate all structures and return as list.
        
        Returns:
            List of perturbed Atoms objects
        """
        return list(self)
    
    def to_file(self, 
                output_dir: str,
                prefix: str = 'batch',
                fmt: str = 'vasp') -> List[str]:
        """Generate all structures and write to files.
        
        Args:
            output_dir: Output directory
            prefix: Filename prefix
            fmt: Output format
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, structure in enumerate(self):
            filename = f"{prefix}_{i:04d}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            write(filepath, structure, format=fmt)
            output_files.append(filepath)
        
        return output_files
