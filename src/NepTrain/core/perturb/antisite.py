# -*- coding: utf-8 -*-
"""Antisite / substitution defect generation for perturbation sampling."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms

try:
    import spglib
    HAS_SPGLIB = True
except ImportError:
    HAS_SPGLIB = False

logger = logging.getLogger(__name__)


def get_equivalent_sites(
    atoms: Atoms,
    symprec: float = 1e-2,
) -> Dict[str, List[List[int]]]:
    """Identify symmetry-equivalent site groups using spglib.

    Parameters
    ----------
    atoms : Atoms
        Input atomic structure.
    symprec : float
        Symmetry tolerance for spglib.

    Returns
    -------
    Dict[str, List[List[int]]]
        Mapping ``{element: [[eq_group_1_indices], [eq_group_2_indices], ...]}``.
        If spglib is unavailable, each atom forms its own singleton group.
    """
    if not HAS_SPGLIB:
        # Fallback: every atom is its own group
        result: Dict[str, List[List[int]]] = {}
        for i, sym in enumerate(atoms.get_chemical_symbols()):
            result.setdefault(sym, []).append([i])
        return result

    spglib_cell = (
        atoms.get_cell().array,
        atoms.get_scaled_positions(wrap=True),
        atoms.get_atomic_numbers(),
    )
    dataset = spglib.get_symmetry_dataset(spglib_cell, symprec=symprec)
    if dataset is None:
        logger.warning("spglib symmetry analysis failed; falling back to per-atom groups")
        result = {}
        for i, sym in enumerate(atoms.get_chemical_symbols()):
            result.setdefault(sym, []).append([i])
        return result

    equiv = dataset.equivalent_atoms  # array of length N, same value = equivalent
    symbols = atoms.get_chemical_symbols()

    # Group by (element, equivalent_class)
    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, eq_id in enumerate(equiv):
        key = (symbols[i], eq_id)
        groups.setdefault(key, []).append(i)

    # Reorganise into {element: [group1, group2, ...]}
    result: Dict[str, List[List[int]]] = {}
    for (elem, _), indices in groups.items():
        result.setdefault(elem, []).append(sorted(indices))

    # Sort each element's groups by size (descending) for deterministic ordering
    for elem in result:
        result[elem].sort(key=lambda g: (-len(g), g[0]))

    return result


def generate_antisite_defects(
    structure: Atoms,
    swap_pairs: List[Tuple[str, str]],
    num_swaps: int = 1,
    mode: str = 'symmetry_aware',
    rng_values: Optional[np.ndarray] = None,
    symprec: float = 1e-2,
) -> Tuple[Atoms, Dict[str, Any]]:
    """Generate antisite defects by swapping atom types.

    Parameters
    ----------
    structure : Atoms
        Input structure (will be copied).
    swap_pairs : List[Tuple[str, str]]
        Element pairs to swap, e.g. ``[('Fe', 'Al')]``.
    num_swaps : int
        Number of swaps to perform.
    mode : str
        ``'symmetry_aware'`` – only swap between equivalent site groups.
        ``'random'`` – randomly select atoms of matching types.
    rng_values : Optional[np.ndarray]
        1-D array of *num_swaps* values in [0, 1] for deterministic
        (Sobol-friendly) selection.
    symprec : float
        Symmetry tolerance for spglib.

    Returns
    -------
    Tuple[Atoms, Dict[str, Any]]
        ``(perturbed_structure, metadata)`` where *metadata* contains
        ``{'swaps': [{'index_i', 'index_j', 'elem_i', 'elem_j'}, ...]}``.
    """
    new_struct = structure.copy()
    symbols = new_struct.get_chemical_symbols()
    swaps_meta: List[Dict[str, Any]] = []

    # Track which indices have already been swapped in this call
    used_indices: set = set()

    for swap_idx in range(num_swaps):
        # Round-robin through swap_pairs if num_swaps > len(swap_pairs)
        pair = swap_pairs[swap_idx % len(swap_pairs)]
        elem_a, elem_b = pair

        # Select rng value for this swap
        rng_val = None
        if rng_values is not None and swap_idx < len(rng_values):
            rng_val = rng_values[swap_idx]

        if mode == 'symmetry_aware':
            idx_a, idx_b = _pick_symmetry_aware(
                new_struct, symbols, elem_a, elem_b,
                rng_val, symprec, used_indices,
            )
        else:
            idx_a, idx_b = _pick_random(
                symbols, elem_a, elem_b, rng_val, used_indices,
            )

        # Perform swap
        symbols[idx_a] = elem_b
        symbols[idx_b] = elem_a
        used_indices.add(idx_a)
        used_indices.add(idx_b)

        swaps_meta.append({
            'index_i': idx_a,
            'index_j': idx_b,
            'elem_i': elem_a,
            'elem_j': elem_b,
        })

    new_struct.set_chemical_symbols(symbols)
    return new_struct, {'swaps': swaps_meta}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_symmetry_aware(
    struct: Atoms,
    symbols: List[str],
    elem_a: str,
    elem_b: str,
    rng_val: Optional[float],
    symprec: float,
    used_indices: set,
) -> Tuple[int, int]:
    """Pick one site from A and one from B within the *same* Wyckoff group."""
    equiv_sites = get_equivalent_sites(struct, symprec=symprec)

    # Find matching groups: a group of elem_a and a group of elem_b that
    # share the same Wyckoff multiplicity (same equiv_id in spglib).
    # Strategy: find the largest pair of (A_group, B_group) that are compatible.
    # We simply iterate and pick the first available pair.
    groups_a = equiv_sites.get(elem_a, [])
    groups_b = equiv_sites.get(elem_b, [])

    if not groups_a or not groups_b:
        # Fall back to random mode
        return _pick_random(symbols, elem_a, elem_b, rng_val, used_indices)

    # Filter out already-used indices
    candidates_per_pair: List[Tuple[List[int], List[int]]] = []
    for ga in groups_a:
        avail_a = [i for i in ga if i not in used_indices]
        if not avail_a:
            continue
        for gb in groups_b:
            avail_b = [i for i in gb if i not in used_indices]
            if not avail_b:
                continue
            candidates_per_pair.append((avail_a, avail_b))

    if not candidates_per_pair:
        # No more available pairs
        raise ValueError(
            f"No available {elem_a}-{elem_b} pairs for antisite swap "
            f"(all candidate sites already used)."
        )

    # Select a pair deterministically via rng_val
    if rng_val is not None:
        pair_idx = int(rng_val * len(candidates_per_pair))
        pair_idx = min(pair_idx, len(candidates_per_pair) - 1)
    else:
        pair_idx = np.random.randint(len(candidates_per_pair))

    avail_a, avail_b = candidates_per_pair[pair_idx]

    idx_a = avail_a[0]  # First available in group
    idx_b = avail_b[0]
    return idx_a, idx_b


def _pick_random(
    symbols: List[str],
    elem_a: str,
    elem_b: str,
    rng_val: Optional[float],
    used_indices: set,
) -> Tuple[int, int]:
    """Randomly pick one A and one B site."""
    indices_a = [i for i, s in enumerate(symbols) if s == elem_a and i not in used_indices]
    indices_b = [i for i, s in enumerate(symbols) if s == elem_b and i not in used_indices]

    if not indices_a:
        raise ValueError(f"No available '{elem_a}' atoms for antisite swap.")
    if not indices_b:
        raise ValueError(f"No available '{elem_b}' atoms for antisite swap.")

    if rng_val is not None:
        ia = int(rng_val * len(indices_a))
        ia = min(ia, len(indices_a) - 1)
        # Use a secondary hash to pick b independently but deterministically
        ib_val = (rng_val * 1.618033988749895) % 1.0  # golden ratio scramble
        ib = int(ib_val * len(indices_b))
        ib = min(ib, len(indices_b) - 1)
    else:
        ia = np.random.randint(len(indices_a))
        ib = np.random.randint(len(indices_b))

    return indices_a[ia], indices_b[ib]
