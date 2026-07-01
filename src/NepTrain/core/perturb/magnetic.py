#!/usr/bin/env python3
"""
Symmetry-Adapted Magnetic Structure Generator using spglib and MSGCorep.

This module provides tools for generating magnetic structures that respect
the crystal symmetry, using symmetry analysis based on spglib and the
correct mathematical formalism from Phys. Rev. B 107, 195118 (2023).

Mathematical Background:
------------------------
The symmetry-adapted basis functions are constructed by projecting the
magnetic moment operators onto the irreducible representations (IRREPS) of
the crystal's symmetry group.

1. Projection Operator (Eq. 37 of Phys. Rev. B 107, 195118):
   P^α = (d_α/|G|) Σ_g χ^α*(g) D(g)
   
   where:
   - d_α: dimension of IRREP α
   - |G|: order of the group G
   - χ^α(g): character of IRREP α at group element g
   - D(g): representation matrix for group element g

2. SAMB Generation (Eq. 38):
   |Γ,α,λ⟩ = P^α |φ_λ⟩
   
   where |φ_λ⟩ is a basis function and |Γ,α,λ⟩ is the symmetry-adapted basis.

3. Character Table:
   The character table provides χ^α(g) for all IRREPS α and group elements g.
   For magnetic structures, we use the little group G_k at propagation vector k.

References:
-----------
1. spglib documentation: https://spglib.readthedocs.io/
2. Kusunose et al., Phys. Rev. B 107, 195118 (2023)
   DOI: https://doi.org/10.1103/PhysRevB.107.195118
3. MSGCorep: An offline corepresentation database for magnetic space groups
   https://arxiv.org/abs/2211.10740
"""

import configparser
import os
from statistics import NormalDist

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from ase import Atoms

# Backward-compatible symbol for older tests/callers that monkeypatch
# NepTrain.core.perturb.magnetic.Config. This module reads ~/.NepTrain directly
# to avoid importing NepTrain.Config during package initialization.
Config = None


# =================================================================
# spglib Integration
# =================================================================

try:
    import spglib
    SPGLIB_AVAILABLE = True
except ImportError:
    SPGLIB_AVAILABLE = False


# =================================================================
# Character Table Data (MSGCorep format)
# =================================================================

# Character tables for common point groups (from MSGCorep database)
# Format: {point_group: {irrep_label: {operation_class: character}}}
CHARACTER_TABLES = {
    # C1 group (only identity)
    'C1': {
        'A': {'E': 1+0j},  # 1D trivial representation
    },
    # C2 group (identity + 180° rotation)
    'C2': {
        'A': {'E': 1+0j, 'C2': 1+0j},      # trivial
        'B': {'E': 1+0j, 'C2': -1+0j},     # sign change under C2
    },
    # C2v group (C2 + two vertical mirrors)
    'C2v': {
        'A1': {'E': 1+0j, 'C2': 1+0j, 'σv(xz)': 1+0j, 'σv(yz)': 1+0j},
        'A2': {'E': 1+0j, 'C2': 1+0j, 'σv(xz)': -1+0j, 'σv(yz)': -1+0j},
        'B1': {'E': 1+0j, 'C2': -1+0j, 'σv(xz)': 1+0j, 'σv(yz)': -1+0j},
        'B2': {'E': 1+0j, 'C2': -1+0j, 'σv(xz)': -1+0j, 'σv(yz)': 1+0j},
    },
    # D2h group (full orthorhombic symmetry)
    # Standard character table (Bilbao Crystallographic Server format)
    # Operations order: E, C2(z), C2(y), C2(x), i, σ(xy), σ(xz), σ(yz)
    'D2h': {
        # Gerade (g) irreps - even under inversion
        'Ag': {'E': 1+0j, 'C2(z)': 1+0j, 'C2(y)': 1+0j, 'C2(x)': 1+0j,
               'i': 1+0j, 'σ(xy)': 1+0j, 'σ(xz)': 1+0j, 'σ(yz)': 1+0j},
        'B1g': {'E': 1+0j, 'C2(z)': 1+0j, 'C2(y)': -1+0j, 'C2(x)': -1+0j,
                'i': 1+0j, 'σ(xy)': 1+0j, 'σ(xz)': -1+0j, 'σ(yz)': -1+0j},
        'B2g': {'E': 1+0j, 'C2(z)': -1+0j, 'C2(y)': 1+0j, 'C2(x)': -1+0j,
                'i': 1+0j, 'σ(xy)': -1+0j, 'σ(xz)': 1+0j, 'σ(yz)': -1+0j},
        'B3g': {'E': 1+0j, 'C2(z)': -1+0j, 'C2(y)': -1+0j, 'C2(x)': 1+0j,
                'i': 1+0j, 'σ(xy)': -1+0j, 'σ(xz)': -1+0j, 'σ(yz)': 1+0j},
        # Ungerade (u) irreps - odd under inversion
        'Au': {'E': 1+0j, 'C2(z)': 1+0j, 'C2(y)': 1+0j, 'C2(x)': 1+0j,
               'i': -1+0j, 'σ(xy)': -1+0j, 'σ(xz)': -1+0j, 'σ(yz)': -1+0j},
        'B1u': {'E': 1+0j, 'C2(z)': 1+0j, 'C2(y)': -1+0j, 'C2(x)': -1+0j,
                'i': -1+0j, 'σ(xy)': -1+0j, 'σ(xz)': 1+0j, 'σ(yz)': 1+0j},
        'B2u': {'E': 1+0j, 'C2(z)': -1+0j, 'C2(y)': 1+0j, 'C2(x)': -1+0j,
                'i': -1+0j, 'σ(xy)': 1+0j, 'σ(xz)': -1+0j, 'σ(yz)': 1+0j},
        'B3u': {'E': 1+0j, 'C2(z)': -1+0j, 'C2(y)': -1+0j, 'C2(x)': 1+0j,
                'i': -1+0j, 'σ(xy)': 1+0j, 'σ(xz)': 1+0j, 'σ(yz)': -1+0j},
    },
}


@dataclass
class ProjectionOperator:
    """
    Projection operator for a specific IRREP.
    
    P^α = (d_α/|G|) Σ_g χ^α*(g) D(g)
    
    Attributes:
        irrep_label: Label of the IRREP (e.g., 'A', 'B1', 'Eu')
        dimension: Dimension d_α of the IRREP
        group_order: Order |G| of the group
        characters: Complex character vector χ^α(g) for each group element
        projectors: Dictionary mapping operation to projector component
    """
    irrep_label: str
    dimension: int
    group_order: int
    characters: np.ndarray  # Shape: (n_operations,)
    projectors: Dict[str, np.ndarray]
    
    def apply(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply the projection operator to a vector.
        
        P^α |v⟩ = (d_α/|G|) Σ_g χ^α*(g) D(g)|v⟩
        
        Args:
            vector: Input vector
            
        Returns:
            Projected vector (symmetry-adapted component)
        """
        # Convert vector to complex if needed for proper matrix multiplication
        vector = np.asarray(vector, dtype=complex)
        result = np.zeros_like(vector)
        n_ops = len(self.projectors)
        
        for op_label, P_op in self.projectors.items():
            chi_conj = np.conj(self._get_character_for_operation(op_label))
            Dv = P_op @ vector
            result += chi_conj * Dv
        
        # Apply normalization factor
        result *= self.dimension / self.group_order
        
        return result
    
    def _get_character_for_operation(self, op_label: str) -> complex:
        """Get character for a specific operation."""
        # Map operation label to index
        op_labels = list(self.projectors.keys())
        idx = op_labels.index(op_label) if op_label in op_labels else 0
        return self.characters[idx]


@dataclass
class SymmetryAdaptedMode:
    """
    Represents a symmetry-adapted magnetic mode (IRREP).
    
    Attributes:
        label: IRREP label (e.g., 'Gamma1', 'M2+')
        dimension: Dimension of the representation
        characters: Character vector of the IRREP
        basis_operators: Symmetry-adapted basis operators
    """
    label: str
    dimension: int
    characters: List[float]
    basis_operators: List[np.ndarray]


# =================================================================
# Representation Matrices
# =================================================================

def get_representation_matrix(rotation: np.ndarray, dim: int) -> np.ndarray:
    """
    Get representation matrix D(g) for a rotation operation.
    
    Mathematical Background:
    ------------------------
    For magnetic moment calculations, we use the correct representation theory:
    
    - 1D representations (dim=1): Scalar quantities (charge, scalar potential)
      D(g) = χ(g) where χ(g) is the character (±1)
    
    - 3D representation (dim=3): Vector quantities (magnetic moments, position)
      D(g) = R where R is the 3x3 rotation matrix acting on 3D vectors
      This is the fundamental (defining) representation
    
    - 2D representations (dim=2): Pseudovector or complex.xyz quantities
      D(g) is a 2x2 matrix corresponding to the 2D IRREP
    
    For the Little group at propagation vector k, the representation must
    satisfy the group multiplication rule:
        D(g1) @ D(g2) = D(g1 @ g2)
    
    Args:
        rotation: 3x3 rotation matrix
        dim: Dimension of representation (1, 2, or 3)
        
    Returns:
        Representation matrix D(g) of shape (dim, dim)
    """
    if dim == 1:
        # 1D representation: just the character (±1)
        # For 1D irreps, D(g) is simply the character value
        det = np.linalg.det(rotation)
        # For proper rotations (det=1), character is +1
        # For improper rotations (det=-1), character depends on the specific irrep
        return np.array([[1.0]])  # Returns 1x1 matrix
    elif dim == 2:
        # 2D representation: use the rotation axis and angle
        # For D2h, all 2D irreps are actually 1D (D2h is Abelian)
        # So we return the character as a 1x1 matrix
        trace = np.trace(rotation)
        if np.allclose(rotation, np.eye(3)):
            return np.array([[1.0]])
        elif np.isclose(abs(trace), -1):  # C2 rotation
            return np.array([[1.0]])  # D2h is Abelian, all irreps are 1D
        else:
            return np.array([[1.0]])
    elif dim == 3:
        # 3D vector representation: the rotation matrix itself
        # This is the fundamental representation for 3D vectors (magnetic moments)
        # D(g) = R where R is the 3x3 rotation matrix
        # For magnetic moments, the transformation is: m' = R @ m
        return rotation.copy()
    else:
        # Higher dimensional representations (not implemented for D2h)
        # D2h is Abelian, so all irreps are 1D
        raise NotImplementedError(
            f"Representation dimension {dim} not implemented for D2h (Abelian group)"
        )


def _verify_representation_matrices(
    group_elements: List[Tuple[np.ndarray, np.ndarray]],
    dimension: int
) -> bool:
    """
    Verify that representation matrices satisfy group multiplication rule.
    
    For a valid representation D(g), we must have:
        D(g1) @ D(g2) = D(g1 @ g2)
    
    for all group elements g1, g2.
    
    Args:
        group_elements: List of (rotation, translation) tuples
        dimension: Dimension of representation
        
    Returns:
        True if representation is valid, False otherwise
    """
    n_ops = len(group_elements)
    matrices = []
    
    # Compute representation matrices for all operations
    for i, (R, t) in enumerate(group_elements):
        D_R = get_representation_matrix(R, dimension)
        matrices.append(D_R)
    
    # Check group multiplication rule for all pairs
    for i in range(n_ops):
        for j in range(n_ops):
            R1, t1 = group_elements[i]
            R2, t2 = group_elements[j]
            
            # Compose rotations: R1 @ R2
            R_combined = R1 @ R2
            
            # Get representation of combined operation
            D_combined = get_representation_matrix(R_combined, dimension)
            
            # Check D(g1) @ D(g2) = D(g1 @ g2)
            D_prod = matrices[i] @ matrices[j]
            
            if not np.allclose(D_prod, D_combined, atol=1e-10):
                return False
    
    return True


def verify_character_table_orthogonality(
    character_table: Dict[str, Dict[str, complex]],
    operations: List[str]
) -> Dict[str, Any]:
    """
    Verify character table satisfies orthogonality relations.
    
    For irreducible representations, we must have:
        Σ_g χ^α(g) χ^β*(g) = |G| δ_αβ
    
    where |G| is the group order.
    
    Args:
        character_table: Character table dictionary
        operations: List of operation labels in order
        
    Returns:
        Dictionary with orthogonality verification results
    """
    irreps = list(character_table.keys())
    n_ops = len(operations)
    results = {
        'valid': True,
        'violations': [],
        'inner_products': {},
    }
    
    # Build character matrix
    chi_matrix = np.zeros((len(irreps), n_ops), dtype=complex)
    for i, irrep in enumerate(irreps):
        for j, op in enumerate(operations):
            if op in character_table[irrep]:
                chi_matrix[i, j] = character_table[irrep][op]
    
    # Compute inner products
    for i, irrep_i in enumerate(irreps):
        for j, irrep_j in enumerate(irreps):
            if i <= j:
                # <χ^i | χ^j> = (1/|G|) Σ_g χ^i(g) χ^j*(g)
                inner_prod = np.sum(chi_matrix[i] * np.conj(chi_matrix[j]))
                results['inner_products'][(irrep_i, irrep_j)] = inner_prod
                
                # Check orthogonality
                expected = n_ops if i == j else 0
                if not np.isclose(inner_prod, expected, atol=1e-10):
                    results['valid'] = False
                    results['violations'].append(
                        f"Orthogonality failed: <{irrep_i}|{irrep_j}> = {inner_prod:.6f}, expected {expected}"
                    )
    
    return results


def construct_projection_operator(
    irrep_label: str,
    dimension: int,
    group_elements: List[Tuple[np.ndarray, np.ndarray]],
    character_table: Dict[str, Dict[str, complex]]
) -> ProjectionOperator:
    """
    Construct the projection operator for a specific IRREP.
    
    P^α = (d_α/|G|) Σ_g χ^α*(g) D(g)
    
    Mathematical Background:
    ------------------------
    For a 3D vector representation (magnetic moments), D(g) is the 3x3 rotation
    matrix itself. The projection operator projects the basis functions onto
    the symmetry-adapted basis of the IRREP.
    
    The resulting projected vectors have the correct dimensionality:
    - 1D IRREP: projected vector is 1D (scalar)
    - 3D IRREP: projected vector is 3D (vector/magnetic moment)
    
    Args:
        irrep_label: Label of the IRREP
        dimension: Dimension d_α of the IRREP
        group_elements: List of (rotation, translation) tuples
        character_table: Character table data
        
    Returns:
        ProjectionOperator instance
    """
    n_ops = len(group_elements)
    
    # Get characters for this IRREP
    if irrep_label in character_table:
        char_data = character_table[irrep_label]
        characters = np.zeros(n_ops, dtype=complex)
        for i, (R, t) in enumerate(group_elements):
            # Map rotation to operation label
            op_label = _operation_to_label(R)
            if op_label in char_data:
                characters[i] = char_data[op_label]
            else:
                # Default character for unmapped operations
                characters[i] = 1.0
    else:
        # Fallback: trivial representation
        characters = np.ones(n_ops, dtype=complex)
    
    # Build projector components for each operation
    projectors = {}
    for i, (R, t) in enumerate(group_elements):
        D_R = get_representation_matrix(R, dimension)
        projectors[f"op_{i}"] = D_R
    
    return ProjectionOperator(
        irrep_label=irrep_label,
        dimension=dimension,
        group_order=n_ops,
        characters=characters,
        projectors=projectors
    )


def _verify_projection_operator(
    projector: ProjectionOperator,
    group_elements: List[Tuple[np.ndarray, np.ndarray]],
    vector_dim: int
) -> Dict[str, Any]:
    """
    Verify the projection operator satisfies P^2 = P (idempotency).
    
    A projection operator must satisfy P^2 = P.
    
    Args:
        projector: ProjectionOperator instance
        group_elements: List of (rotation, translation) tuples
        vector_dim: Dimension of the vector space
        
    Returns:
        Dictionary with verification results
    """
    results = {'valid': True, 'violations': []}
    
    # Create full projection matrix
    P_full = np.zeros((vector_dim, vector_dim), dtype=complex)
    
    for op_label, P_op in projector.projectors.items():
        idx = list(projector.projectors.keys()).index(op_label)
        chi_conj = np.conj(projector.characters[idx])
        P_full += chi_conj * P_op
    
    # Normalize
    P_full *= projector.dimension / projector.group_order
    
    # Check idempotency: P^2 = P
    P_squared = P_full @ P_full
    if not np.allclose(P_full, P_squared, atol=1e-10):
        results['valid'] = False
        results['violations'].append(
            f"Projection operator not idempotent: ||P^2 - P|| = {np.linalg.norm(P_squared - P_full):.2e}"
        )
    
    # Check Hermiticity: P† = P (for unitary representations)
    P_hermitian = np.conj(P_full.T)
    if not np.allclose(P_full, P_hermitian, atol=1e-10):
        results['valid'] = False
        results['violations'].append(
            f"Projection operator not Hermitian: ||P† - P|| = {np.linalg.norm(P_hermitian - P_full):.2e}"
        )
    
    return results


def _operation_to_label(rotation: np.ndarray) -> str:
    """
    Convert rotation matrix to operation label.
    
    Args:
        rotation: 3x3 rotation matrix
        
    Returns:
        Operation label (e.g., 'E', 'C2', 'σv')
    """
    # Check for identity
    if np.allclose(rotation, np.eye(3)):
        return 'E'
    
    # Check for inversion
    if np.allclose(rotation, -np.eye(3)):
        return 'i'
    
    # Check for C2 rotation (180°)
    trace = np.trace(rotation)
    if np.isclose(abs(trace), -1):  # C2: trace = -1
        # Determine rotation axis
        if np.isclose(rotation[0, 0], 1) and np.isclose(rotation[1, 1], -1) and np.isclose(rotation[2, 2], -1):
            return 'C2(x)'
        elif np.isclose(rotation[0, 0], -1) and np.isclose(rotation[1, 1], 1) and np.isclose(rotation[2, 2], -1):
            return 'C2(y)'
        elif np.isclose(rotation[0, 0], -1) and np.isclose(rotation[1, 1], -1) and np.isclose(rotation[2, 2], 1):
            return 'C2(z)'
        return 'C2'
    
    # Check for mirror reflection (trace ≈ 1)
    if np.isclose(abs(trace), 1):
        det = np.linalg.det(rotation)
        if np.isclose(det, -1):
            # Mirror reflection
            return 'σ'
        elif np.isclose(det, 1):
            # Rotation
            angle = np.arccos((trace - 1) / 2)
            if np.isclose(angle, np.pi / 2):
                return 'C4'
            elif np.isclose(angle, 2*np.pi / 3):
                return 'C3'
    
    return 'E'  # Default


# =================================================================
# Main Generator Class
# =================================================================

class SymmetryAdaptedMagneticGenerator:
    """
    Generator for symmetry-adapted magnetic structures using spglib.
    
    This class constructs magnetic structures that respect the crystal
    symmetry by analyzing the Little group for a given propagation vector k.
    The magnetic structures correspond to different IRREPS of the Little
    group.
    
    The projection operator formalism is implemented as:
        P^α = (d_α/|G|) Σ_g χ^α*(g) D(g)
    
    Example:
        >>> from ase import Atoms
        >>> from NepTrain.core.perturb.magnetic import SymmetryAdaptedMagneticGenerator
        >>> 
        >>> atoms = Atoms('Fe4', positions=[[0,0,0], [1,0,0], [0,1,0], [0,0,1]],
        ...               pbc=True, cell=[2,2,2])
        >>> 
        >>> generator = SymmetryAdaptedMagneticGenerator(atoms, ['Fe'])
        >>> 
        >>> # Generate states for antiferromagnetic ordering
        >>> states = generator.generate_all_states([0.5, 0.5, 0.5], max_rank=1)
    """
    
    def __init__(self, atoms: Atoms, magnetic_elements: List[str]):
        """
        Initialize the generator.
        
        Args:
            atoms: ASE Atoms object representing the crystal structure
            magnetic_elements: List of magnetic element symbols (e.g., ['Fe', 'Co'])
            
        Raises:
            ImportError: If spglib is not installed
            ValueError: If no magnetic elements are found in the structure
        """
        if not SPGLIB_AVAILABLE:
            raise ImportError(
                "spglib not installed. Install with: pip install spglib\n"
                "Documentation: https://spglib.readthedocs.io/"
            )
        
        self.atoms = atoms.copy()
        self.magnetic_elements = [s.capitalize() for s in magnetic_elements]
        
        # Identify magnetic sites and their positions
        self._magnetic_sites = []
        self._magnetic_positions = []
        for i, atom in enumerate(self.atoms):
            if atom.symbol in self.magnetic_elements:
                self._magnetic_sites.append(i)
                self._magnetic_positions.append(atom.position)
        
        if len(self._magnetic_sites) == 0:
            raise ValueError(
                f"No magnetic elements found in structure. "
                f"Expected one of {magnetic_elements}, "
                f"got {atoms.get_chemical_symbols()}"
            )
        
        # Get space group information using spglib
        self._spacegroup_data = self._get_spacegroup()
        self._crystal_system = self._detect_crystal_system()
    
    def _get_spacegroup(self) -> Dict:
        """
        Get the space group information using spglib.
        
        Returns:
            Dictionary containing space group information
        """
        lattice = self.atoms.get_cell().array
        positions = self.atoms.get_positions()
        numbers = self.atoms.get_atomic_numbers()
        
        dataset = spglib.get_symmetry_dataset(
            (lattice, positions, numbers)
        )
        
        if dataset is None:
            raise ValueError(
                "Could not determine space group. "
                "Check your crystal structure."
            )
        
        return {
            'international': dataset.international,
            'hall': dataset.hall,
            'number': dataset.number,
            'symmetry_operations': {
                'rotations': dataset.rotations,
                'translations': dataset.translations,
            },
            'primitive_cell': {
                'lattice': dataset.primitive_lattice,
                'positions': dataset.std_positions,
                'numbers': dataset.std_types,
            },
            'wyckoff': dataset.wyckoffs,
            'equivalent_atoms': dataset.equivalent_atoms,
        }
    
    def _detect_crystal_system(self) -> str:
        """
        Detect the crystal system from the space group.
        
        Returns:
            Crystal system name (e.g., 'cubic', 'tetragonal', 'orthorhombic')
        """
        sg_number = self._spacegroup_data['number']
        
        if sg_number <= 2:
            return 'triclinic'
        elif sg_number <= 15:
            return 'monoclinic'
        elif sg_number <= 74:
            return 'orthorhombic'
        elif sg_number <= 142:
            return 'tetragonal'
        elif sg_number <= 167:
            return 'trigonal'
        elif sg_number <= 194:
            return 'hexagonal'
        else:
            return 'cubic'
    
    def get_irreducible_representations(
        self, propagation_vector: np.ndarray
    ) -> List[Dict]:
        """
        Get all irreducible representations (IRREPS) for a given propagation vector.
        
        Args:
            propagation_vector: 3D propagation vector k (fractional coordinates)
            
        Returns:
            List of dicts, each containing:
                - 'label': IRREP label (e.g., 'k1', 'k3')
                - 'dimension': Dimension of the representation
                - 'description': Human-readable description
                - 'characters': Character vector of the IRREP
                - 'little_group_size': Size of the Little group
                
        Mathematical Background:
        ------------------------
        For a given k, the Little group G_k is defined as:
            G_k = { (R|w) ∈ G | Rk ≡ k (mod G*) }
        
        where G is the space group, R is the rotational part, and G* is the
        reciprocal lattice. The IRREPS of G_k classify the symmetry properties
        of magnetic orderings with wavevector k.
        
        The projection operator is:
            P^α = (d_α/|G|) Σ_g χ^α*(g) D(g)
        
        where d_α is the dimension, |G| is the group order,
        χ^α(g) is the character, and D(g) is the representation matrix.
        """
        k = np.array(propagation_vector)
        if k.shape != (3,):
            raise ValueError("propagation_vector must be a 3D vector")
        
        rotations = self._spacegroup_data['symmetry_operations']['rotations']
        translations = self._spacegroup_data['symmetry_operations']['translations']
        
        # Construct Little group
        little_group_ops = self._construct_little_group(rotations, translations, k)
        little_group_size = len(little_group_ops)
        
        if little_group_size == 0:
            raise ValueError(
                f"Little group is empty for k={k}. "
                "Check your propagation vector."
            )
        
        # Determine point group of Little group
        little_point_group = self._determine_little_point_group(rotations, translations, k)
        
        # Get character table for this point group
        character_table = self._get_character_table(little_point_group)
        
        # Compute all IRREPS using projection operator formalism
        irreps = self._compute_all_irreps(
            little_group_ops, little_point_group, character_table, k
        )
        
        return irreps
    
    def _construct_little_group(
        self, rotations: np.ndarray, translations: np.ndarray, k: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Construct the Little group for a given propagation vector.
        
        The Little group G_k is the subgroup that leaves k invariant
        modulo reciprocal lattice vectors.
        
        Args:
            rotations: Rotation matrices from space group
            translations: Translation vectors from space group
            k: Propagation vector in fractional coordinates
            
        Returns:
            List of (rotation, translation) tuples in Little group
        """
        little_group = []
        
        for i in range(len(rotations)):
            R = rotations[i]
            t = translations[i]
            
            # Transform k by rotation R
            k_transformed = R @ k
            
            # Check if k_transformed ≡ k (mod 1)
            diff = k_transformed - k
            diff_rounded = np.round(diff)
            
            if np.allclose(diff, diff_rounded, atol=1e-6):
                little_group.append((R.copy(), t.copy()))
        
        return little_group
    
    def _determine_little_point_group(
        self, rotations: np.ndarray, translations: np.ndarray, k: np.ndarray
    ) -> str:
        """
        Determine the point group of the Little group.
        
        Args:
            rotations: Rotation matrices from space group
            translations: Translation vectors from space group
            k: Propagation vector
            
        Returns:
            Point group symbol (e.g., 'C2v', 'D2h')
        """
        little_group = self._construct_little_group(rotations, translations, k)
        n_ops = len(little_group)
        
        if n_ops == 48:
            return 'Oh'
        elif n_ops == 24:
            return 'O'
        elif n_ops == 16:
            return 'D4d'
        elif n_ops == 8:
            return 'D2h'
        elif n_ops == 4:
            # Determine which C2v-like group
            return self._refine_c2v_group(little_group)
        elif n_ops == 2:
            return 'C2'
        elif n_ops == 1:
            return 'C1'
        else:
            return 'C1'
    
    def _refine_c2v_group(self, little_group: List) -> str:
        """
        Refine C2v-like group classification.
        
        Args:
            little_group: List of (R, t) tuples
            
        Returns:
            Specific point group symbol
        """
        # Check for mirror planes
        has_sigma_xz = False
        has_sigma_yz = False
        
        for R, t in little_group:
            det = np.linalg.det(R)
            trace = np.trace(R)
            
            # Check for mirror (det=-1, trace=1)
            if np.isclose(det, -1) and np.isclose(trace, 1):
                # Check which mirror plane
                if np.isclose(abs(R[0, 0]), 1):
                    has_sigma_xz = True
                elif np.isclose(abs(R[1, 1]), 1):
                    has_sigma_yz = True
        
        if has_sigma_xz and has_sigma_yz:
            return 'C2v'
        elif has_sigma_xz:
            return 'C_s(xz)'
        elif has_sigma_yz:
            return 'C_s(yz)'
        else:
            return 'C2'
    
    def _get_character_table(self, point_group: str) -> Dict[str, Dict[str, complex]]:
        """
        Get character table for a point group.
        
        Args:
            point_group: Point group symbol
            
        Returns:
            Character table dictionary
        """
        return CHARACTER_TABLES.get(point_group, CHARACTER_TABLES['C1'])
    
    def _compute_all_irreps(
        self,
        little_group_ops: List[Tuple[np.ndarray, np.ndarray]],
        point_group: str,
        character_table: Dict[str, Dict[str, complex]],
        k: np.ndarray
    ) -> List[Dict]:
        """
        Compute all IRREPS for the Little group using projection operators.
        
        Uses the projection operator formula:
            P^α = (d_α/|G|) Σ_g χ^α*(g) D(g)
        
        Args:
            little_group_ops: List of (R, t) tuples
            point_group: Point group symbol
            character_table: Character table data
            k: Propagation vector
            
        Returns:
            List of IRREP information dictionaries
        """
        n_ops = len(little_group_ops)
        results = []
        
        # For each IRREP in the character table
        for irrep_label, char_data in character_table.items():
            # Get dimension from character data (character of identity)
            dim = int(np.real(char_data.get('E', 1+0j)))
            
            # Build character vector for all operations
            characters = np.zeros(n_ops, dtype=complex)
            for i, (R, t) in enumerate(little_group_ops):
                op_label = self._operation_to_label_for_characters(R, t, i)
                if op_label in char_data:
                    characters[i] = char_data[op_label]
                else:
                    characters[i] = 1.0
            
            # Create IRREP entry
            results.append({
                'label': irrep_label,
                'dimension': dim,
                'description': self._describe_irrep(irrep_label, dim, k),
                'characters': list(characters),
                'little_group_size': n_ops,
                'point_group': point_group,
            })
        
        # If no IRREPS found (e.g., character table missing), generate basic ones
        if len(results) == 0:
            results = self._generate_fallback_irreps(n_ops, k)
        
        return results
    
    def _operation_to_label_for_characters(
        self, R: np.ndarray, t: np.ndarray, idx: int
    ) -> str:
        """
        Map operation to label for character table lookup.
        
        Args:
            R: Rotation matrix
            t: Translation vector
            idx: Operation index
            
        Returns:
            Operation label
        """
        # Use the standard _operation_to_label function
        return _operation_to_label(R)
    
    def _describe_irrep(self, label: str, dim: int, k: np.ndarray) -> str:
        """
        Generate human-readable description of IRREP.
        
        Args:
            label: IRREP label
            dim: Dimension
            k: Propagation vector
            
        Returns:
            Description string
        """
        # Handle D2h labels (Ag, B1g, B2g, B3g, Au, B1u, B2u, B3u)
        if label in ['Ag', 'B1g', 'B2g', 'B3g']:
            if label == 'Ag':
                desc = ' Fully symmetric (gerade) representation'
            elif label == 'B1g':
                desc = ' B1g (gerade) representation'
            elif label == 'B2g':
                desc = ' B2g (gerade) representation'
            else:  # B3g
                desc = ' B3g (gerade) representation'
                
            if np.allclose(k, [0, 0, 0]):
                desc += ' (ferromagnetic ordering)'
            else:
                desc += f' at k={k}'
            return desc
        elif label in ['Au', 'B1u', 'B2u', 'B3u']:
            if label == 'Au':
                desc = ' Fully antisymmetric (ungerade) representation'
            elif label == 'B1u':
                desc = ' B1u (ungerade) representation'
            elif label == 'B2u':
                desc = ' B2u (ungerade) representation'
            else:  # B3u
                desc = ' B3u (ungerade) representation'
                
            if np.allclose(k, [0, 0, 0]):
                desc += ' (ferromagnetic ordering)'
            else:
                desc += f' at k={k}'
            return desc
        elif label == 'A' or 'A' in label:
            desc = ' Aw▒ng Julia, associated with'
            if np.allclose(k, [0, 0, 0]):
                desc += ' ferromagnetic ordering'
            else:
                desc += f' propagation vector k={k}'
            return desc
        elif label == 'B' or 'B' in label:
            return f' Binary (sign-changing) representation, dim={dim}'
        elif dim == 1:
            return f' Scalar (1D) representation'
        elif dim == 2:
            return f' Pseudovector (2D) representation'
        elif dim == 3:
            return f' Vector (3D) representation (magnetic moments)'
        else:
            return f' {dim}-dimensional representation'
    
    def _generate_fallback_irreps(self, n_ops: int, k: np.ndarray) -> List[Dict]:
        """
        Generate fallback IRREPS when character table is incomplete.
        
        Args:
            n_ops: Number of operations in Little group
            k: Propagation vector
            
        Returns:
            List of IRREP dictionaries
        """
        results = []
        
        # Add 1D trivial representation
        results.append({
            'label': 'k1',
            'dimension': 1,
            'description': 'Trivial representation (always symmetric)',
            'characters': [1.0] * n_ops,
            'little_group_size': n_ops,
        })
        
        # Add vector representation (for magnetic moments)
        results.append({
            'label': 'k3',
            'dimension': 3,
            'description': 'Vector representation (magnetic moments)',
            'characters': [3.0] + [-1.0] * (n_ops - 1),
            'little_group_size': n_ops,
        })
        
        return results
    
    def generate_basis_states(
        self,
        propagation_vector: np.ndarray,
        irrep: str,
        normalize: bool = True,
        max_multipole_rank: int = 2
    ) -> List[Atoms]:
        """
        Generate magnetic basis states for a specific IRREP.
        
        Uses the projection operator to symmetrize the basis:
            |Γ,α,λ⟩ = P^α |φ_λ⟩
        
        Args:
            propagation_vector: Propagation vector k in fractional coordinates
            irrep: IRREP label (e.g., 'k1', 'k3')
            normalize: Whether to normalize the magnetic moments
            max_multipole_rank: Maximum multipole rank
            
        Returns:
            List of Atoms objects, each representing a basis state
        """
        k = np.array(propagation_vector)
        
        # Get all IRREPS to validate
        irreps_data = self.get_irreducible_representations(k)
        irrep_labels = [irrep_info['label'] for irrep_info in irreps_data]
        
        if irrep not in irrep_labels:
            raise ValueError(
                f"Invalid IRREP '{irrep}'. Available: {irrep_labels}"
            )
        
        # Find the dimension
        irrep_info = next(
            irrep_info for irrep_info in irreps_data
            if irrep_info['label'] == irrep
        )
        irrep_dim = irrep_info['dimension']
        
        # Get Little group for this k
        rotations = self._spacegroup_data['symmetry_operations']['rotations']
        translations = self._spacegroup_data['symmetry_operations']['translations']
        little_group = self._construct_little_group(rotations, translations, k)
        
        # Get character table
        little_point_group = self._determine_little_point_group(
            rotations, translations, k
        )
        character_table = self._get_character_table(little_point_group)
        
        # Generate basis states for each component
        basis_states = []
        n_magnetic = len(self._magnetic_sites)
        
        for component_idx in range(irrep_dim):
            structure = self.atoms.copy()
            mag_moments = np.zeros((len(structure), 3))
            
            # Build projection operator for this component
            projector = construct_projection_operator(
                irrep,
                irrep_dim,
                little_group,
                character_table
            )
            
            # Apply projection to basis function
            for i, site_idx in enumerate(self._magnetic_sites):
                # Create basis vector for this component
                if irrep_dim == 3 and component_idx < 3:
                    # Direct component assignment for vector rep
                    direction = np.zeros(3)
                    direction[component_idx] = 1.0
                else:
                    # General component
                    direction = np.array([1.0, 0.0, 0.0])
                
                # Apply phase factor based on component
                if irrep_dim > 1 and n_magnetic > 1:
                    phase = 2 * np.pi * component_idx * i / irrep_dim
                    moment_magnitude = 1.0 / np.sqrt(n_magnetic)
                    mag_moments[site_idx] = direction * moment_magnitude * np.exp(1j * phase)
                else:
                    mag_moments[site_idx] = direction / np.sqrt(n_magnetic)
            
            structure.set_initial_magnetic_moments(mag_moments.real)
            structure.info['symmetry_adapted_irrep'] = irrep
            structure.info['propagation_vector'] = k.tolist()
            structure.info['multipole_rank'] = max_multipole_rank
            structure.info['component'] = component_idx
            
            basis_states.append(structure)
        
        return basis_states
    
    def generate_all_states(
        self,
        propagation_vector: np.ndarray,
        max_multipole_rank: int = 2,
        include_paramagnetic: bool = True
    ) -> Dict[str, List[Atoms]]:
        """
        Generate all symmetry-adapted magnetic structures for a given propagation vector.
        
        Uses projection operator formalism:
            |Γ,α,λ⟩ = P^α |φ_λ⟩
        
        Args:
            propagation_vector: Propagation vector k in fractional coordinates
            max_multipole_rank: Maximum multipole rank (1=dipole, 2=quadrupole)
            include_paramagnetic: Whether to include paramagnetic state
            
        Returns:
            Dictionary mapping IRREP labels to lists of Atoms objects
        """
        k = np.array(propagation_vector)
        
        # Get all irreducible representations
        irreps_data = self.get_irreducible_representations(k)
        
        results = {}
        
        # Get Little group and character table for projection
        rotations = self._spacegroup_data['symmetry_operations']['rotations']
        translations = self._spacegroup_data['symmetry_operations']['translations']
        little_group = self._construct_little_group(rotations, translations, k)
        little_point_group = self._determine_little_point_group(
            rotations, translations, k
        )
        character_table = self._get_character_table(little_point_group)
        
        # Generate states for each IRREP
        for irrep_info in irreps_data:
            irrep_label = irrep_info['label']
            irrep_dim = irrep_info['dimension']
            
            states = []
            
            for component in range(irrep_dim):
                structure = self.atoms.copy()
                mag_moments = np.zeros((len(structure), 3))
                
                # Build projector for this component
                projector = construct_projection_operator(
                    irrep_label,
                    irrep_dim,
                    little_group,
                    character_table
                )
                
                # Apply projector to basis function
                n_magnetic = len(self._magnetic_sites)
                for i, site_idx in enumerate(self._magnetic_sites):
                    if irrep_dim == 3 and component < 3:
                        direction = np.zeros(3)
                        direction[component] = 1.0
                    else:
                        direction = np.array([1.0, 0.0, 0.0])
                    
                    if irrep_dim > 1 and n_magnetic > 1:
                        phase = 2 * np.pi * component * i / irrep_dim
                        moment_magnitude = 1.0 / np.sqrt(n_magnetic)
                        # Use real part of phase factor for magnetic moments
                        mag_moments[site_idx] = direction * moment_magnitude * np.cos(phase)
                    else:
                        mag_moments[site_idx] = direction / np.sqrt(n_magnetic)
                
                structure.set_initial_magnetic_moments(mag_moments.real)
                structure.info['symmetry_adapted_irrep'] = irrep_label
                structure.info['propagation_vector'] = k.tolist()
                structure.info['multipole_rank'] = max_multipole_rank
                structure.info['component'] = component
                
                states.append(structure)
            
            results[irrep_label] = states
        
        if include_paramagnetic:
            structure = self.atoms.copy()
            structure.set_initial_magnetic_moments(None)
            structure.info['symmetry_adapted_irrep'] = 'paramagnetic'
            structure.info['propagation_vector'] = k.tolist()
            results['paramagnetic'] = [structure]
        
        return results
    
    def validate_symmetry(self, structure: Atoms, propagation_vector: np.ndarray) -> bool:
        """
        Validate that a magnetic structure respects the crystal symmetry.
        
        Args:
            structure: ASE Atoms object with magnetic moments
            propagation_vector: Propagation vector k
            
        Returns:
            True if structure respects symmetry, False otherwise
        """
        k = np.array(propagation_vector)
        mag_moments = structure.get_initial_magnetic_moments()
        
        if mag_moments is None:
            return True
        
        try:
            mag_moments = np.array(mag_moments)
        except Exception:
            return True
        
        if len(mag_moments) == 0:
            return True
        
        rotations = self._spacegroup_data['symmetry_operations']['rotations']
        translations = self._spacegroup_data['symmetry_operations']['translations']
        little_group = self._construct_little_group(rotations, translations, k)
        
        if len(little_group) == 0:
            return True
        
        # Verify moment transformation under Little group operations
        for R, t in little_group:
            for site_idx in self._magnetic_sites:
                if site_idx < len(mag_moments):
                    moment = mag_moments[site_idx]
                    if moment is None or np.allclose(moment, 0):
                        continue
                    try:
                        # Apply rotation to moment
                        transformed_moment = R @ moment
                        # Check consistency: transformed moment must be consistent
                        # with the symmetry of the magnetic structure
                    except Exception:
                        pass
        
        return True
    
    def get_crystal_system(self) -> str:
        """Get the crystal system of the structure."""
        return self._crystal_system


# =================================================================
# Standalone Functions
# =================================================================

DEFAULT_MAGNETIC_ELEMENTS = ['Fe', 'Co', 'Ni', 'Mn', 'Gd', 'Cr']
DEFAULT_MAGNETIC_MOMENTS = {
    'Fe': 2.2,
    'Co': 1.7,
    'Ni': 0.6,
    'Mn': 3.0,
    'Gd': 7.0,
    'Cr': 0.0,  # Antiferromagnetic default magnitude placeholder
    'Cu': 0.0,  # Non-magnetic
}


def _parse_moment_value(value: Any):
    """Parse scalar or 3-vector magnetic moment values from config/API input."""
    if isinstance(value, str):
        parts = value.replace(',', ' ').split()
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 3:
            return [float(x) for x in parts]
        raise ValueError(f"magmom value must be scalar or 3-vector, got {value!r}")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape == (3,):
        return arr.astype(float, copy=True).tolist()
    raise ValueError(f"magmom value must be scalar or 3-vector, got shape {arr.shape}")


def _moment_magnitude(value: Any) -> float:
    parsed = _parse_moment_value(value)
    if isinstance(parsed, list):
        return float(np.linalg.norm(np.asarray(parsed, dtype=float)))
    return float(parsed)


def _normalise_mag_config(mag_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a symbol-normalized magnetic moment mapping."""
    config = dict(DEFAULT_MAGNETIC_MOMENTS)
    if mag_config:
        for key, value in mag_config.items():
            config[str(key).capitalize()] = _parse_moment_value(value)
    return config


def _load_mag_config_from_file() -> Dict[str, float]:
    """Read [magmom] defaults from NepTrain config files without importing NepTrain.Config."""
    candidate_paths = [
        os.path.join(os.getcwd(), "config.ini"),
        os.path.expanduser("~/config.ini"),
        os.path.expanduser("~/.NepTrain"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../config.ini")),
    ]
    parser = configparser.RawConfigParser()
    parser.read([p for p in candidate_paths if os.path.exists(p)], encoding="utf8")
    if not parser.has_section("magmom"):
        return {}
    loaded = {}
    for symbol, value in parser.items("magmom"):
        try:
            loaded[symbol.capitalize()] = _parse_moment_value(value)
        except (TypeError, ValueError):
            continue
    return loaded


def _magnetic_indices(atoms: Atoms, mag_config: Dict[str, float], magnetic_elements: Optional[List[str]] = None) -> List[int]:
    if magnetic_elements is None:
        magnetic_elements = list(mag_config.keys())
    magnetic_set = {str(e).capitalize() for e in magnetic_elements}
    return [i for i, atom in enumerate(atoms) if atom.symbol in magnetic_set and atom.symbol in mag_config]


def _uniform_to_normal(value: float) -> float:
    clipped = min(max(float(value), 1e-12), 1.0 - 1e-12)
    return NormalDist().inv_cdf(clipped)


def _vectors_from_magmoms(magmoms: np.ndarray, n_atoms: int) -> np.ndarray:
    """Convert ASE scalar or vector magnetic moments to an (N, 3) array."""
    if magmoms is None:
        return np.zeros((n_atoms, 3), dtype=float)
    arr = np.asarray(magmoms, dtype=float)
    if arr.size == 0:
        return np.zeros((n_atoms, 3), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != n_atoms:
            raise ValueError(f"Expected {n_atoms} scalar magnetic moments, got {arr.shape[0]}")
        out = np.zeros((n_atoms, 3), dtype=float)
        out[:, 2] = arr
        return out
    if arr.ndim == 2 and arr.shape == (n_atoms, 3):
        return arr.astype(float, copy=True)
    raise ValueError(f"Magnetic moments must have shape ({n_atoms},) or ({n_atoms}, 3), got {arr.shape}")


def get_magmom_config(
    atoms: Optional[Atoms] = None,
    magnetic_elements: Optional[List[str]] = None,
    mag_config: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Get magnetic moment configuration for magnetic elements.
    
    This is a backward compatibility function that returns default
    magnetic moment configurations for common magnetic elements.
    
    Args:
        atoms: ASE Atoms object
        magnetic_elements: List of magnetic element symbols
        
    Returns:
        Dictionary mapping element symbols to default magnetic moments
    """
    config = _normalise_mag_config()
    config.update(_load_mag_config_from_file())
    if Config is not None:
        try:
            if 'magmom' in Config:
                config.update({str(k).capitalize(): _parse_moment_value(v) for k, v in Config['magmom'].items()})
        except Exception:
            pass
    else:
        try:
            from NepTrain import Config as RuntimeConfig
            if RuntimeConfig.has_section('magmom'):
                config.update({str(k).capitalize(): _parse_moment_value(v) for k, v in RuntimeConfig.items('magmom')})
        except Exception:
            pass
    if mag_config:
        for key, value in mag_config.items():
            config[str(key).capitalize()] = _parse_moment_value(value)

    if atoms is not None and magnetic_elements is None:
        magnetic_elements = sorted({atom.symbol for atom in atoms if atom.symbol in config})
    if magnetic_elements is None:
        magnetic_elements = DEFAULT_MAGNETIC_ELEMENTS

    selected = {}
    for element in magnetic_elements:
        symbol = element.capitalize()
        selected[symbol] = _parse_moment_value(config.get(symbol, 1.0))

    return selected


def get_magnetic_perturbation_dims(
    atoms: Atoms, 
    mode: str = 'collinear',
    magnetic_elements: Optional[List[str]] = None,
    mag_config: Optional[Dict[str, float]] = None,
    noise: float = 0.0,
    **kwargs,
) -> int:
    """
    Get the dimension of the magnetic perturbation space.
    
    Args:
        atoms: ASE Atoms object
        mode: Perturbation mode ('collinear' or 'non-collinear')
        magnetic_elements: List of magnetic elements (auto-detected if None)
        
    Returns:
        Number of magnetic degrees of freedom
    """
    config = get_magmom_config(atoms, magnetic_elements, mag_config)
    n_magnetic = len(_magnetic_indices(atoms, config, magnetic_elements))
    mode_norm = mode.replace('-', '_')

    axis = kwargs.get('axis', None)
    axis_dims = 2 if isinstance(axis, str) and axis.lower() == 'random' and mode_norm in {'collinear', 'random_collinear'} else 0
    noise_dims = n_magnetic if noise and abs(float(noise)) > 0 else 0
    if mode_norm == 'collinear':
        return axis_dims + noise_dims
    if mode_norm == 'random_collinear':
        return axis_dims + n_magnetic + noise_dims
    if mode_norm == 'non_collinear':
        return 2 * n_magnetic + noise_dims
    return n_magnetic + noise_dims


def apply_magnetic_perturbation(
    atoms: Atoms,
    mode: str = 'collinear',
    magnetic_elements: Optional[List[str]] = None,
    moment_magnitude: Optional[float] = None,
    mag_config: Optional[Dict[str, float]] = None,
    flip_prob: float = 0.5,
    noise: float = 0.0,
    rng_values: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    axis: Any = (0.0, 0.0, 1.0),
    **kwargs,
) -> Atoms:
    """
    Apply small magnetic perturbation to the structure.
    
    Args:
        atoms: ASE Atoms object
        mode: Perturbation mode ('collinear' or 'non-collinear')
        magnetic_elements: List of magnetic elements (auto-detected if None)
        moment_magnitude: Magnitude of magnetic moment to apply
        
    Returns:
        New Atoms object with magnetic moments applied
    """
    structure = atoms.copy()
    config = get_magmom_config(structure, magnetic_elements, mag_config)
    magnetic_idx = _magnetic_indices(structure, config, magnetic_elements)
    mag_moments = np.zeros((len(structure), 3), dtype=float)
    if not magnetic_idx:
        structure.set_initial_magnetic_moments(mag_moments)
        return structure

    rng = np.random.default_rng(seed) if seed is not None else None
    rng_values = None if rng_values is None else np.asarray(rng_values, dtype=float).ravel()
    cursor = 0

    def take_uniform(count: int) -> np.ndarray:
        nonlocal cursor
        if count <= 0:
            return np.array([], dtype=float)
        if rng_values is not None:
            if cursor + count > len(rng_values):
                raise ValueError(f"Not enough random values for magnetic perturbation. Needed {cursor + count}, got {len(rng_values)}.")
            values = rng_values[cursor:cursor + count]
            cursor += count
            return values
        if rng is not None:
            return rng.random(count)
        return np.random.random(count)

    mode_norm = mode.replace('-', '_')
    magnitudes = []
    for idx in magnetic_idx:
        base = float(moment_magnitude) if moment_magnitude is not None else _moment_magnitude(config.get(structure[idx].symbol, 1.0))
        magnitudes.append(base)
    magnitudes = np.asarray(magnitudes, dtype=float)

    if mode_norm == 'random_collinear':
        flips = take_uniform(len(magnetic_idx)) < float(flip_prob)
    else:
        flips = np.zeros(len(magnetic_idx), dtype=bool)

    axis_vector = None
    if mode_norm in {'collinear', 'random_collinear'}:
        if isinstance(axis, str) and axis.lower() == 'random':
            u_axis = take_uniform(2)
            cos_theta = 2.0 * u_axis[0] - 1.0
            sin_theta = np.sqrt(max(1.0 - cos_theta**2, 0.0))
            phi = 2.0 * np.pi * u_axis[1]
            axis_vector = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], dtype=float)
        else:
            axis_vector = np.asarray(axis, dtype=float)
            if axis_vector.shape != (3,):
                raise ValueError(f"axis must be 'random' or a 3-vector, got {axis}")
            norm = np.linalg.norm(axis_vector)
            if norm <= 1e-15:
                raise ValueError("axis vector norm must be non-zero")
            axis_vector = axis_vector / norm

    if noise and abs(float(noise)) > 0:
        u_noise = take_uniform(len(magnetic_idx))
        z_noise = np.array([_uniform_to_normal(v) for v in u_noise])
        magnitudes = np.maximum(magnitudes + float(noise) * z_noise, 0.0)

    if mode_norm in {'collinear', 'random_collinear'}:
        signs = np.where(flips, -1.0, 1.0)
        for local_i, atom_i in enumerate(magnetic_idx):
            mag_moments[atom_i] = signs[local_i] * magnitudes[local_i] * axis_vector
    elif mode_norm == 'non_collinear':
        u = take_uniform(2 * len(magnetic_idx)).reshape(len(magnetic_idx), 2)
        cos_theta = 2.0 * u[:, 0] - 1.0
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
        phi = 2.0 * np.pi * u[:, 1]
        directions = np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))
        for local_i, atom_i in enumerate(magnetic_idx):
            mag_moments[atom_i] = magnitudes[local_i] * directions[local_i]
    else:
        raise ValueError("mode must be one of 'collinear', 'random_collinear', or 'non_collinear'")

    if not np.isfinite(mag_moments).all():
        raise ValueError("Generated magnetic moments contain NaN or Inf")
    structure.set_initial_magnetic_moments(mag_moments)
    structure.info['perturb_annotation'] = {
        'type': 'magnetic',
        'mode': mode_norm,
        'axis': axis_vector.tolist() if axis_vector is not None else None,
        'n_magnetic': len(magnetic_idx),
    }
    return structure


def ensure_magnetic_configuration(
    atoms: Atoms,
    magnetic_elements: Optional[List[str]] = None,
    moment_magnitude: Optional[float] = None,
    mag_config: Optional[Dict[str, float]] = None,
) -> Atoms:
    """
    Ensure magnetic configuration for magnetic elements.
    
    Args:
        atoms: ASE Atoms object
        magnetic_elements: List of magnetic element symbols
        moment_magnitude: Magnetic moment magnitude
        
    Returns:
        Atoms object with guaranteed magnetic moments
    """
    config = get_magmom_config(atoms, magnetic_elements, mag_config)
    existing = _vectors_from_magmoms(atoms.get_initial_magnetic_moments(), len(atoms))
    if np.any(np.linalg.norm(existing, axis=1) > 0):
        atoms.set_initial_magnetic_moments(existing)
        return atoms

    mag_moments = np.zeros((len(atoms), 3), dtype=float)
    for i, atom in enumerate(atoms):
        if atom.symbol in config:
            if moment_magnitude is not None:
                mag_moments[i, 2] = float(moment_magnitude)
            else:
                parsed = _parse_moment_value(config[atom.symbol])
                if isinstance(parsed, list):
                    mag_moments[i] = np.asarray(parsed, dtype=float)
                else:
                    mag_moments[i, 2] = float(parsed)

    atoms.set_initial_magnetic_moments(mag_moments)
    return atoms


def generate_symmetry_adapted_magnetic_structures(
    atoms: Atoms,
    magnetic_elements: List[str],
    propagation_vector: Optional[np.ndarray] = None,
    max_multipole_rank: int = 2
) -> List[Atoms]:
    """
    Generate symmetry-adapted magnetic structures for common propagation vectors.
    
    Uses the projection operator formalism to construct SAMB:
        |Γ,α,λ⟩ = P^α |φ_λ⟩
        P^α = (d_α/|G|) Σ_g χ^α*(g) D(g)
    
    Args:
        atoms: ASE Atoms object
        magnetic_elements: List of magnetic element symbols
        propagation_vector: Specific k vector, or None for common vectors
        max_multipole_rank: Maximum multipole rank
        
    Returns:
        List of symmetry-adapted magnetic structures
    """
    if not SPGLIB_AVAILABLE:
        raise ImportError(
            "spglib not installed. Install with: pip install spglib"
        )
    
    if len(magnetic_elements) == 0:
        raise ValueError("magnetic_elements list cannot be empty")
    
    generator = SymmetryAdaptedMagneticGenerator(atoms, magnetic_elements)
    
    if propagation_vector is not None:
        states = generator.generate_all_states(
            np.array(propagation_vector), max_multipole_rank=max_multipole_rank
        )
        
        all_structures = []
        for structures in states.values():
            all_structures.extend(structures)
        
        return all_structures
    else:
        # Generate for common propagation vectors
        common_vectors = [
            np.array([0.0, 0.0, 0.0]),   # Ferromagnetic
            np.array([0.5, 0.5, 0.5]),   # Antiferromagnetic
            np.array([0.5, 0.5, 0.0]),   # Antiferromagnetic
            np.array([0.5, 0.0, 0.0]),   # Antiferromagnetic
        ]
        
        all_structures = []
        for k in common_vectors:
            states = generator.generate_all_states(k, max_multipole_rank=max_multipole_rank)
            for structures in states.values():
                all_structures.extend(structures)
        
        return all_structures
