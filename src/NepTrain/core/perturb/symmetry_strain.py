#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Symmetry-preserving strain perturbation.

Applies strain tensors that respect the crystal symmetry, reducing the
Sobol dimensionality from 9 (full cell) to 1–6 independent parameters
depending on the crystal system.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Crystal-system → independent strain-component count
# ---------------------------------------------------------------------------
CRYSTAL_SYSTEM_STRAIN_DIMS: Dict[str, int] = {
    'cubic': 1,           # ε₁ = ε₂ = ε₃, off-diag = 0
    'tetragonal': 2,      # ε₁ = ε₂, ε₃, off-diag = 0
    'hexagonal': 2,       # ε₁ = ε₂, ε₃, off-diag = 0
    'trigonal': 2,        # ε₁ = ε₂, ε₃, off-diag = 0  (simplified)
    'orthorhombic': 3,    # ε₁, ε₂, ε₃, off-diag = 0
    'monoclinic': 4,      # ε₁, ε₂, ε₃, ε₆; ε₄ = ε₅ = 0
    'triclinic': 6,       # all 6 Voigt components
}

_VALID_SYSTEMS = set(CRYSTAL_SYSTEM_STRAIN_DIMS.keys())


def detect_crystal_system_spglib(atoms: Atoms, symprec: float = 1e-2) -> str:
    """Detect crystal system using *spglib*, with a geometry-based fallback.

    Parameters
    ----------
    atoms : Atoms
        Input structure.
    symprec : float
        Symmetry precision for spglib.

    Returns
    -------
    str
        One of the seven crystal system names.
    """
    try:
        import spglib

        dataset = spglib.get_spacegroup_type(
            spglib.get_spacegroup((atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
                                  symprec=symprec)
        ) if hasattr(spglib, 'get_spacegroup_type') else None

        # spglib >= 2.0 path
        if dataset is None:
            try:
                lattice = atoms.get_cell().array
                positions = atoms.get_scaled_positions()
                numbers = atoms.get_atomic_numbers()
                spg_ds = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=symprec)
                if spg_ds is not None:
                    hall_number = spg_ds.get('hall_number')
                    if hall_number is not None:
                        dataset = spglib.get_spacegroup_type(hall_number)
            except Exception:
                pass

        if dataset and 'crystal_system' in dataset:
            cs = dataset['crystal_system'].lower()
            if cs in _VALID_SYSTEMS:
                return cs
    except ImportError:
        logger.debug("spglib not available, falling back to geometry-based detection")
    except Exception as exc:
        logger.debug("spglib detection failed (%s), falling back to geometry", exc)

    # Fallback: geometry-based detection from grain_boundary module
    from .grain_boundary import detect_crystal_system
    return detect_crystal_system(atoms.cell)


def get_independent_strain_count(crystal_system: str) -> int:
    """Return the number of independent strain components (1–6)."""
    cs = crystal_system.lower()
    if cs not in CRYSTAL_SYSTEM_STRAIN_DIMS:
        raise ValueError(
            f"Unknown crystal system '{crystal_system}'. "
            f"Expected one of {sorted(_VALID_SYSTEMS)}"
        )
    return CRYSTAL_SYSTEM_STRAIN_DIMS[cs]


def build_symmetric_strain_tensor(
    crystal_system: str,
    independent_strains: np.ndarray,
) -> np.ndarray:
    """Construct a 3×3 symmetric strain tensor from independent components.

    Parameters
    ----------
    crystal_system : str
        Crystal system name.
    independent_strains : np.ndarray
        1-D array of independent strain values.

    Returns
    -------
    np.ndarray
        3×3 strain tensor.
    """
    cs = crystal_system.lower()
    n = CRYSTAL_SYSTEM_STRAIN_DIMS[cs]
    eta = np.asarray(independent_strains, dtype=float)
    if eta.shape != (n,):
        raise ValueError(
            f"Expected {n} independent strains for {cs}, got shape {eta.shape}"
        )

    eps = np.zeros((3, 3), dtype=float)

    if cs == 'cubic':
        # diag(η, η, η)
        np.fill_diagonal(eps, eta[0])

    elif cs in ('tetragonal', 'hexagonal', 'trigonal'):
        # diag(η₁, η₁, η₂)
        eps[0, 0] = eta[0]
        eps[1, 1] = eta[0]
        eps[2, 2] = eta[1]

    elif cs == 'orthorhombic':
        # diag(η₁, η₂, η₃)
        eps[0, 0] = eta[0]
        eps[1, 1] = eta[1]
        eps[2, 2] = eta[2]

    elif cs == 'monoclinic':
        # diag(η₁, η₂, η₃) + ε₆ → off-diag [0,1] and [1,0]
        eps[0, 0] = eta[0]
        eps[1, 1] = eta[1]
        eps[2, 2] = eta[2]
        eps[0, 1] = eps[1, 0] = eta[3]

    elif cs == 'triclinic':
        # Full Voigt: η₁…η₆ → (ε₁, ε₂, ε₃, ε₄, ε₅, ε₆)
        eps[0, 0] = eta[0]
        eps[1, 1] = eta[1]
        eps[2, 2] = eta[2]
        eps[1, 2] = eps[2, 1] = eta[3]
        eps[0, 2] = eps[2, 0] = eta[4]
        eps[0, 1] = eps[1, 0] = eta[5]

    return eps


def generate_symmetry_preserving_strain(
    atoms: Atoms,
    strain_fraction: float = 0.03,
    crystal_system: Optional[str] = None,
    symprec: float = 1e-2,
    rng_values: Optional[np.ndarray] = None,
    min_distance: float = 0.1,
) -> Tuple[Atoms, Dict[str, Any]]:
    """Generate a symmetry-preserving strained structure.

    Parameters
    ----------
    atoms : Atoms
        Original structure (will not be mutated).
    strain_fraction : float
        Maximum strain magnitude for each independent component.
    crystal_system : str or None
        If *None*, auto-detect via spglib / geometry fallback.
    symprec : float
        Symmetry precision for spglib.
    rng_values : np.ndarray or None
        Sobol-derived values in [0, 1] (one per independent component).
        If *None*, random values are drawn uniformly.
    min_distance : float
        Minimum atomic displacement added after straining.

    Returns
    -------
    Tuple[Atoms, dict]
        ``(strained_atoms, metadata)``
    """
    struct = atoms.copy()

    # 1. Detect / use crystal system
    if crystal_system is None:
        crystal_system = detect_crystal_system_spglib(struct, symprec=symprec)

    # 2. Independent component count
    n = get_independent_strain_count(crystal_system)

    # 3. Generate independent strain values ∈ [-strain_fraction, strain_fraction]
    if rng_values is not None:
        rng_values = np.asarray(rng_values, dtype=float)
        if rng_values.shape != (n,):
            raise ValueError(
                f"Expected {n} rng_values for {crystal_system}, got {rng_values.shape}"
            )
        independent_strains = 2.0 * strain_fraction * (rng_values - 0.5)
    else:
        independent_strains = np.random.uniform(
            -strain_fraction, strain_fraction, size=n
        )

    # 4. Build full strain tensor
    eps = build_symmetric_strain_tensor(crystal_system, independent_strains)

    # 5. Apply F = I + ε  →  new_cell = cell @ F^T
    cell = np.asarray(struct.get_cell(), dtype=float)
    F = np.eye(3) + eps
    new_cell = cell @ F.T
    struct.set_cell(new_cell, scale_atoms=True)

    # 6. Volume conservation: rescale to match original volume
    orig_vol = abs(np.linalg.det(cell))
    new_vol = abs(np.linalg.det(new_cell))
    if new_vol > 1e-15:
        scale = (orig_vol / new_vol) ** (1.0 / 3.0)
        struct.set_cell(struct.get_cell() * scale, scale_atoms=True)

    # 7. Small positional perturbation (min_distance jitter)
    positions = struct.get_positions()
    displacements = np.random.normal(0, min_distance * 0.1, size=positions.shape)
    struct.set_positions(positions + displacements)

    metadata = {
        'crystal_system': crystal_system,
        'independent_strains': independent_strains,
        'full_strain_tensor': eps,
    }
    return struct, metadata
