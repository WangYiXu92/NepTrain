"""
Chemical validation utilities for generated structures.
Checks bond lengths, coordination numbers, and atomic overlap.
"""

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList


def get_bond_lengths(atoms: Atoms, r_cutoff: float = 3.0) -> np.ndarray:
    """
    Get all bond lengths in the structure.
    
    Args:
        atoms: ASE Atoms object
        r_cutoff: Cutoff radius for bond detection (default 3.0 Angstrom)
    
    Returns:
        Array of bond lengths
    """
    nl = NeighborList([r_cutoff / 2] * len(atoms), skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    bonds = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) > 0:
            pos = atoms.positions
            cell = atoms.cell
            for j, offset in zip(indices, offsets):
                diff = pos[j] + np.dot(offset, cell) - pos[i]
                dist = np.linalg.norm(diff)
                bonds.append(dist)
    
    return np.array(bonds)


def get_min_bond_length(atoms: Atoms, r_cutoff: float = 3.0) -> float:
    """
    Get the minimum bond length in the structure.
    
    Args:
        atoms: ASE Atoms object
        r_cutoff: Cutoff radius for bond detection
    
    Returns:
        Minimum bond length
    """
    bonds = get_bond_lengths(atoms, r_cutoff)
    if len(bonds) == 0:
        return float('inf')
    return np.min(bonds)


def get_coordination_numbers(atoms: Atoms, r_cutoff: float = 3.0) -> np.ndarray:
    """
    Get coordination numbers for all atoms.
    
    Args:
        atoms: ASE Atoms object
        r_cutoff: Cutoff radius for neighbor detection
    
    Returns:
        Array of coordination numbers
    """
    nl = NeighborList([r_cutoff / 2] * len(atoms), skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    cn = np.array([len(nl.get_neighbors(i)[0]) for i in range(len(atoms))])
    return cn


def check_atomic_overlap(atoms: Atoms, min_distance: float = 0.8) -> dict:
    """
    Check for atomic overlap (atoms too close).
    
    Args:
        atoms: ASE Atoms object
        min_distance: Minimum allowed distance between atoms
    
    Returns:
        Dictionary with overlap information
    """
    # Use smaller cutoff for overlap detection
    nl = NeighborList([min_distance / 2] * len(atoms), skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    overlaps = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) > 0:
            pos = atoms.positions
            cell = atoms.cell
            for j, offset in zip(indices, offsets):
                if j > i:  # Only check each pair once
                    diff = pos[j] + np.dot(offset, cell) - pos[i]
                    dist = np.linalg.norm(diff)
                    if dist < min_distance:
                        overlaps.append({
                            'atom1': i,
                            'atom2': j,
                            'distance': dist,
                            'element1': atoms[i].symbol,
                            'element2': atoms[j].symbol
                        })
    
    return {
        'has_overlap': len(overlaps) > 0,
        'num_overlaps': len(overlaps),
        'overlaps': overlaps,
        'min_distance': min([o['distance'] for o in overlaps]) if overlaps else None
    }


def get_structure_summary(atoms: Atoms, r_cutoff: float = 3.0, min_distance: float = 0.8) -> dict:
    """
    Get a comprehensive summary of structure validity.
    
    Args:
        atoms: ASE Atoms object
        r_cutoff: Cutoff for bond detection
        min_distance: Minimum allowed distance
    
    Returns:
        Dictionary with validation summary
    """
    bond_lengths = get_bond_lengths(atoms, r_cutoff)
    cn = get_coordination_numbers(atoms, r_cutoff)
    overlap = check_atomic_overlap(atoms, min_distance)
    
    return {
        'num_atoms': len(atoms),
        'num_bonds': len(bond_lengths),
        'min_bond_length': float(np.min(bond_lengths)) if len(bond_lengths) > 0 else None,
        'max_bond_length': float(np.max(bond_lengths)) if len(bond_lengths) > 0 else None,
        'mean_bond_length': float(np.mean(bond_lengths)) if len(bond_lengths) > 0 else None,
        'coordination': {
            'mean': float(np.mean(cn)),
            'min': int(np.min(cn)),
            'max': int(np.max(cn)),
            'std': float(np.std(cn))
        },
        'overlap': overlap,
        'is_valid': not overlap['has_overlap']
    }


def filter_valid_structures(structures: list, min_bond_length: float = 1.5, max_bond_length: float = 4.0, 
                           min_distance: float = 0.8, r_cutoff: float = 3.0) -> list:
    """
    Filter a list of structures by chemical validity.
    
    Args:
        structures: List of ASE Atoms objects
        min_bond_length: Minimum expected bond length
        max_bond_length: Maximum expected bond length
        min_distance: Minimum allowed distance between atoms
        r_cutoff: Cutoff for bond detection
    
    Returns:
        List of valid structures
    """
    valid = []
    for i, atoms in enumerate(structures):
        summary = get_structure_summary(atoms, r_cutoff, min_distance)
        
        # Check criteria
        if not summary['is_valid']:
            continue
        
        if summary['min_bond_length'] is not None:
            if summary['min_bond_length'] < min_bond_length:
                continue
            if summary['min_bond_length'] > max_bond_length:
                continue
        
        valid.append(atoms)
    
    return valid
