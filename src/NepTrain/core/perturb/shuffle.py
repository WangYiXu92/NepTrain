#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ase import Atoms
from copy import deepcopy
from typing import Union, List, Dict, Any, Optional

def shuffle_element_positions(
    structure_data: Union[Atoms, Dict, Any],
    element_range: Union[List[int], List[str], str, slice],
    shuffle_method: str = 'fisher_yates',
    seed: Optional[int] = None,
    rng_values: Optional[np.ndarray] = None
) -> (Atoms, Dict[int, Dict[str, List[float]]]):
    """
    Shuffle positions of elements within the specified range in the structure.

    Parameters
    ----------
    structure_data : Union[Atoms, Dict, Any]
        The input structure data. Must be an ASE Atoms object or convertible to one.
    element_range : Union[List[int], List[str], str, slice]
        The range of elements to shuffle. Can be:
        - List of indices (e.g., [0, 1, 2])
        - List of element symbols (e.g., ['Fe', 'O'])
        - Slice object or string representation (e.g., "0:10")
    shuffle_method : str, optional
        The shuffling algorithm to use. Options:
        - 'fisher_yates' (default): Standard random permutation (O(N)).
        - 'random_swap': Perform N random swaps, where N is the number of elements.
    seed : int, optional
        Random seed for reproducibility.
    rng_values : np.ndarray, optional
        1D array of uniform random values [0, 1] for deterministic shuffling (Sobol).
        If provided, this overrides seed and shuffle_method logic to use sort-based permutation.

    Returns
    -------
    new_structure : Atoms
        The structure with shuffled positions.
    metadata : Dict
        Metadata containing mapping of moved elements' original and new positions.
        Format: {atom_index: {'original': [x, y, z], 'new': [x, y, z]}}
    """
    # 1. Validate and prepare structure data
    if isinstance(structure_data, Atoms):
        atoms = structure_data.copy()
    elif isinstance(structure_data, dict):
        # specific handling for dict if it follows some schema, otherwise try generic
        try:
            atoms = Atoms.fromdict(structure_data)
        except Exception:
             # Fallback: assume it might be a dict representation of atoms, 
             # but without explicit schema, we might fail. 
             # Let's assume the user passes Atoms or a dict that ASE can handle or has 'positions', 'numbers', 'cell'.
             # For now, simplistic conversion attempt if possible, else raise.
             try:
                 atoms = Atoms(**structure_data)
             except Exception as e:
                 raise ValueError(f"Could not convert dictionary to Atoms: {e}")
    else:
        # Try to see if it behaves like atoms
        if hasattr(structure_data, 'get_positions') and hasattr(structure_data, 'set_positions'):
            atoms = deepcopy(structure_data)
        else:
             raise TypeError("structure_data must be an ASE Atoms object or compatible dictionary.")

    # 2. Parse element_range
    indices = _parse_element_range(atoms, element_range)
    
    if not indices:
        # Handle empty range or no matching elements
        return atoms, {}

    # Check for fixed elements (if constraints exist)
    # ASE atoms can have constraints.
    indices = _filter_fixed_indices(atoms, indices)

    if len(indices) < 2:
        # Nothing to shuffle
        return atoms, {}

    # 3. Setup Randomness
    # rng = np.random.default_rng(seed) # Moved inside

    # 4. Perform Shuffling
    original_positions = atoms.get_positions()
    selected_positions = original_positions[indices].copy()
    
    new_positions_subset = selected_positions.copy()

    if rng_values is not None:
        if len(rng_values) < len(indices):
            raise ValueError(f"Not enough random values for shuffling. Needed {len(indices)}, got {len(rng_values)}.")
        
        # Sort-based permutation using Sobol values
        # We use the first N values corresponding to the N indices
        current_rng = rng_values[:len(indices)]
        
        # argsort gives the indices that would sort the array
        # This acts as a random permutation if the input values are random
        perm = np.argsort(current_rng)
        
        # Apply permutation to positions
        # Meaning: position[0] goes to where position[perm[0]] was? 
        # Or atom[0] moves to position[perm[0]]?
        # shuffle means rearranging the items.
        # If we have positions P0, P1, P2
        # And we want to shuffle them.
        # new_positions = positions[perm]
        new_positions_subset = selected_positions[perm]
        
    else:
        rng = np.random.default_rng(seed)
        if shuffle_method == 'fisher_yates':
            rng.shuffle(new_positions_subset)
        elif shuffle_method == 'random_swap':
            n = len(indices)
            for _ in range(n):
                i, j = rng.integers(0, n, 2)
                # Swap
                new_positions_subset[[i, j]] = new_positions_subset[[j, i]]
        else:
            raise ValueError(f"Unknown shuffle_method: {shuffle_method}")

    # 5. Apply changes
    # We are assigning the shuffled positions back to the atoms at 'indices'
    # So atom at indices[k] gets new_positions_subset[k]
    
    # Check consistency
    if len(indices) != len(new_positions_subset):
        raise RuntimeError("Mismatch in indices and positions count.")

    full_new_positions = original_positions.copy()
    full_new_positions[indices] = new_positions_subset
    atoms.set_positions(full_new_positions)

    # 6. Generate Metadata
    metadata = {}
    for k, atom_idx in enumerate(indices):
        # Only record if position actually changed? 
        # The requirement implies "mapping of moved elements", maybe all in range?
        # Let's record all in range for completeness.
        old_pos = selected_positions[k]
        new_pos = new_positions_subset[k]
        
        # Check if actually moved (optional, but good for "moved elements")
        if not np.allclose(old_pos, new_pos, atol=1e-8):
            metadata[atom_idx] = {
                'original': old_pos.tolist(),
                'new': new_pos.tolist()
            }
            
    return atoms, metadata

def _parse_element_range(atoms: Atoms, element_range: Union[List[int], List[str], str, slice]) -> List[int]:
    """Helper to parse element range into a list of indices."""
    n_atoms = len(atoms)
    
    if isinstance(element_range, slice):
        return list(range(*element_range.indices(n_atoms)))
    
    if isinstance(element_range, str):
        # Handle comma-separated list of symbols
        if ',' in element_range:
            parts = [x.strip() for x in element_range.split(',')]
            # Recursively call or handle list
            return _parse_element_range(atoms, parts)

        # Try parsing as slice string "start:stop:step"
        try:
            parts = [int(x) if x else None for x in element_range.split(':')]
            if len(parts) == 1:
                # Single index string "5"
                return [parts[0]]
            elif len(parts) <= 3:
                s = slice(*parts)
                return list(range(*s.indices(n_atoms)))
        except ValueError:
            # Maybe it's a single element symbol "Fe"
            if element_range in atoms.symbols:
                return [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == element_range]
            pass
            
    if isinstance(element_range, (list, tuple, set)):
        # Check content type
        element_range = list(element_range)
        if not element_range:
            return []
            
        if isinstance(element_range[0], int):
            # Validate indices
            valid_indices = [i for i in element_range if 0 <= i < n_atoms]
            return sorted(list(set(valid_indices))) # Return unique sorted
            
        if isinstance(element_range[0], str):
            # Element symbols
            target_symbols = set(element_range)
            return [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in target_symbols]

    raise ValueError(f"Invalid element_range format: {element_range}")

def _filter_fixed_indices(atoms: Atoms, indices: List[int]) -> List[int]:
    """Filter out indices that are fixed by constraints."""
    if not atoms.constraints:
        return indices
        
    # Check constraints
    # ASE constraints usually have an 'index' attribute or 'get_indices' method?
    # Or 'adjust_positions'.
    # A generic way is checking if an atom is constrained. 
    # But usually `FixAtoms` constraint has an `index` array.
    
    fixed_indices = set()
    for constraint in atoms.constraints:
        if hasattr(constraint, 'index'):
            # constraint.index can be a list or array or int
            idx = constraint.index
            if isinstance(idx, int):
                fixed_indices.add(idx)
            else:
                fixed_indices.update(idx)
        # Handle FixCartesian etc if needed, but FixAtoms is most common for "immovable"
        
    return [i for i in indices if i not in fixed_indices]
