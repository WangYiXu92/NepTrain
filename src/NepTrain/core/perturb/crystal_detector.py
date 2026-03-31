"""通用晶体结构检测器。

This module provides a unified crystal structure detector that can be used
across multiple perturbation modules (interstitial, twinning, etc.).
"""

import numpy as np
from typing import Tuple, Optional
from ase import Atoms
from ase.neighborlist import NeighborList


class CrystalDetector:
    """通用晶体结构检测器。
    
    This class provides a unified interface for detecting crystal structure types
    (BCC, FCC, HCP) using coordination number analysis and lattice geometry.
    
    Attributes:
        structure: Input ASE Atoms object
        crystal_type: Detected crystal type ('bcc', 'fcc', 'hcp', 'unknown')
        lattice_params: Lattice parameters dictionary
    """
    
    def __init__(self, structure: Atoms):
        """Initialize the crystal detector.
        
        Args:
            structure: Input structure (ASE Atoms object)
        """
        self.structure = structure.copy()
        self._crystal_type: Optional[str] = None
        self._lattice_params: Optional[Tuple[float, float, float]] = None
        self._coordination_numbers: Optional[np.ndarray] = None
    
    def detect(self) -> str:
        """Detect crystal structure type.
        
        Uses coordination number analysis and lattice geometry to determine
        if the structure is BCC, FCC, or HCP.
        
        Returns:
            Crystal type: 'bcc', 'fcc', 'hcp', or 'unknown'
        """
        if self._crystal_type is None:
            self._crystal_type = self._detect_impl()
        return self._crystal_type
    
    def _detect_impl(self) -> str:
        """Implement crystal detection logic.
        
        Returns:
            Crystal type: 'bcc', 'fcc', 'hcp', 'diamond', 'tetragonal', or 'unknown'
        """
        structure = self.structure.copy()
        cell = structure.cell
        
        # Get lattice parameters
        lengths = cell.lengths()
        angles = cell.angles()
        
        a, b, c = lengths
        alpha, beta, gamma = angles
        
        # Calculate normalized lengths
        min_len = min(lengths)
        if min_len <= 0:
            return 'unknown'
            
        a_norm = a / min_len
        b_norm = b / min_len
        c_norm = c / min_len
        
        # Tolerance for comparison
        tol = 0.1
        
        # Check for Diamond structure first (if 8 atoms per unit cell)
        if len(self.structure) == 8:
            if self._is_diamond():
                return 'diamond'
        
        # Check for Tetragonal structure (a=b, c different, all angles=90)
        if self._is_tetragonal():
            return 'tetragonal'
        
        # HCP detection: hexagonal symmetry (a=b, gamma=120)
        if (abs(a_norm - b_norm) < tol and 
            abs(gamma - 120) < tol):
            # Check c/a ratio
            if c_norm > 1.0 and abs(c_norm - 1.633) < 0.3:
                return 'hcp'
            elif c_norm > 1.0:  # Approximate HCP
                return 'hcp'
        
        # Cubic structures: a=b=c, all angles=90
        if (abs(a_norm - b_norm) < tol and 
            abs(b_norm - c_norm) < tol and
            all(abs(angle - 90) < tol for angle in angles)):
            # Use coordination number analysis for cubic structures
            return self._detect_cubic_by_coordination(structure)
        
        # Default using coordination number
        return self._detect_cubic_by_coordination(structure)
    
    def _detect_cubic_by_coordination(self, structure: Atoms) -> str:
        """Detect cubic crystal type using coordination number analysis.
        
        Args:
            structure: Input structure
            
        Returns:
            Crystal type: 'bcc', 'fcc', 'hcp', or 'unknown'
        """
        n_atoms = len(structure)
        
        # Calculate lattice-based cutoff (use first nearest neighbor distance)
        lengths = structure.cell.lengths()
        min_len = min(lengths) if min(lengths) > 0 else 4.0
        # Use slightly smaller cutoff to avoid counting too many neighbors
        cutoff = 1.4 * min_len
        
        nl = NeighborList(cutoffs=[cutoff] * n_atoms, self_interaction=False)
        nl.update(structure)
        
        coord_numbers = []
        for i in range(n_atoms):
            indices, _ = nl.get_neighbors(i)
            coord_numbers.append(len(indices))
        
        self._coordination_numbers = np.array(coord_numbers)
        avg_coord = np.mean(coord_numbers)
        
        # Tolerance for coordination number
        tol = 1.5
        
        # BCC: ~8 neighbors
        # FCC/HCP: ~12 neighbors
        if abs(avg_coord - 8) < tol:
            return 'bcc'
        elif abs(avg_coord - 12) < tol:
            # Check for hexagonal symmetry
            angles = structure.cell.angles()
            tol_deg = 5.0
            if abs(angles[2] - 120) < tol_deg:
                return 'hcp'
            else:
                return 'fcc'
        
        # Default to BCC for unknown
        return 'bcc'
    
    def get_lattice_parameters(self) -> Tuple[float, float, float]:
        """Get lattice parameters a, b, c.
        
        Returns:
            Tuple of (a, b, c) lattice parameters in Angstrom
        """
        if self._lattice_params is None:
            lengths = self.structure.cell.lengths()
            self._lattice_params = (
                lengths[0], lengths[1], lengths[2]
            )
        return self._lattice_params
    
    def get_coordination_numbers(self) -> np.ndarray:
        """Get coordination numbers for each atom.
        
        Returns:
            Array of coordination numbers
        """
        if self._coordination_numbers is None:
            # Calculate on demand
            self._detect_impl()
        return self._coordination_numbers
    
    def is_cubic(self) -> bool:
        """Check if structure is cubic (a=b=c, angles=90).
        
        Returns:
            True if cubic structure
        """
        lengths = self.structure.cell.lengths()
        angles = self.structure.cell.angles()
        
        tol = 0.1
        return (abs(lengths[0] - lengths[1]) < tol and 
                abs(lengths[1] - lengths[2]) < tol and
                all(abs(angle - 90) < tol for angle in angles))
    
    def is_hexagonal(self) -> bool:
        """Check if structure is hexagonal (a=b, gamma=120).
        
        Returns:
            True if hexagonal structure
        """
        lengths = self.structure.cell.lengths()
        angles = self.structure.cell.angles()
        
        tol = 0.1
        return (abs(lengths[0] - lengths[1]) < tol and 
                abs(angles[2] - 120) < tol)
    
    def _is_diamond(self) -> bool:
        """Detect diamond structure.
        
        Diamond structure features:
        - Space group Fd-3m (227)
        - 8 atoms per unit cell
        - Each atom has coordination number 4
        
        Returns:
            True if diamond structure
        """
        try:
            from ase.spacegroup import get_spacegroup
            spg = get_spacegroup(self.structure)
            if spg.no == 227:  # Fd-3m
                return True
            
            # Fallback: check coordination number
            if len(self.structure) == 8:
                cn = self.get_coordination_numbers()
                return np.all(cn == 4)
            return False
        except:
            return False
    
    def _is_tetragonal(self) -> bool:
        """Detect tetragonal structure.
        
        Tetragonal structure features:
        - a = b ≠ c
        - α = β = γ = 90°
        
        Returns:
            True if tetragonal structure
        """
        cell = self.structure.cell
        a, b, c = np.linalg.norm(cell, axis=1)
        
        # Calculate angles
        angles = [
            np.arccos(np.dot(cell[1], cell[2]) / (b * c)) * 180 / np.pi,
            np.arccos(np.dot(cell[0], cell[2]) / (a * c)) * 180 / np.pi,
            np.arccos(np.dot(cell[0], cell[1]) / (a * b)) * 180 / np.pi,
        ]
        
        # a ≈ b, c different, all angles ≈ 90°
        return (abs(a - b) / a < 0.1 and 
                abs(c - a) / a > 0.1 and 
                all(abs(angle - 90) < 5 for angle in angles))


def detect_crystal_type(structure: Atoms) -> str:
    """Convenience function to detect crystal type.
    
    Args:
        structure: ASE Atoms object
        
    Returns:
        Crystal type: 'bcc', 'fcc', 'hcp', or 'unknown'
    """
    detector = CrystalDetector(structure)
    return detector.detect()
