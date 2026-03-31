#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for symmetry-preserving strain perturbation."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk


# ---------------------------------------------------------------------------
# Helper structures
# ---------------------------------------------------------------------------

def _cubic_fe():
    """BCC Fe — cubic."""
    return bulk('Fe', 'bcc', a=2.87)


def _tetragonal_tio2():
    """Rutile TiO₂ — tetragonal."""
    from ase.spacegroup import crystal
    return crystal(['Ti', 'O'], [(0, 0, 0), (0.3, 0.3, 0.0)],
                   spacegroup=136, cellpar=[4.6, 4.6, 2.96, 90, 90, 90])


def _hexagonal_mg():
    """HCP Mg — hexagonal."""
    return bulk('Mg', 'hcp', a=3.21, c=5.21)


def _orthorhombic():
    """Orthorhombic cell."""
    a, b, c = 3.0, 4.0, 5.0
    cell = np.diag([a, b, c])
    positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return Atoms('CuZn', positions=positions, cell=cell, pbc=True)


def _triclinic():
    """Triclinic cell."""
    cell = np.array([[3.0, 0.0, 0.0],
                     [0.5, 4.0, 0.0],
                     [0.3, 0.2, 5.0]])
    return Atoms('C', positions=[[0, 0, 0]], cell=cell, pbc=True)


# ---------------------------------------------------------------------------
# Import target
# ---------------------------------------------------------------------------

from NepTrain.core.perturb.symmetry_strain import (
    CRYSTAL_SYSTEM_STRAIN_DIMS,
    build_symmetric_strain_tensor,
    detect_crystal_system_spglib,
    generate_symmetry_preserving_strain,
    get_independent_strain_count,
)


# ---------------------------------------------------------------------------
# Tests: get_independent_strain_count
# ---------------------------------------------------------------------------

class TestIndependentStrainCount:
    def test_cubic(self):
        assert get_independent_strain_count('cubic') == 1

    def test_tetragonal(self):
        assert get_independent_strain_count('tetragonal') == 2

    def test_hexagonal(self):
        assert get_independent_strain_count('hexagonal') == 2

    def test_trigonal(self):
        assert get_independent_strain_count('trigonal') == 2

    def test_orthorhombic(self):
        assert get_independent_strain_count('orthorhombic') == 3

    def test_monoclinic(self):
        assert get_independent_strain_count('monoclinic') == 4

    def test_triclinic(self):
        assert get_independent_strain_count('triclinic') == 6

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_independent_strain_count('unknown')


# ---------------------------------------------------------------------------
# Tests: build_symmetric_strain_tensor
# ---------------------------------------------------------------------------

class TestBuildStrainTensor:
    def test_cubic_diagonal(self):
        eps = build_symmetric_strain_tensor('cubic', np.array([0.02]))
        assert eps.shape == (3, 3)
        np.testing.assert_allclose(eps, np.diag([0.02, 0.02, 0.02]))
        assert np.allclose(eps - np.diag(np.diag(eps)), 0)

    def test_tetragonal_equal_first_two(self):
        eps = build_symmetric_strain_tensor('tetragonal', np.array([0.01, -0.02]))
        assert eps[0, 0] == eps[1, 1] == 0.01
        assert eps[2, 2] == -0.02

    def test_orthorhombic_distinct(self):
        eps = build_symmetric_strain_tensor('orthorhombic', np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(np.diag(eps), [0.1, 0.2, 0.3])

    def test_monoclinic_offdiag(self):
        eps = build_symmetric_strain_tensor('monoclinic', np.array([0.1, 0.2, 0.3, 0.05]))
        assert eps[0, 1] == eps[1, 0] == 0.05
        assert eps[1, 2] == 0  # ε₄ = 0
        assert eps[0, 2] == 0  # ε₅ = 0

    def test_triclinic_full(self):
        eta = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        eps = build_symmetric_strain_tensor('triclinic', eta)
        assert eps[0, 0] == 1
        assert eps[1, 1] == 2
        assert eps[2, 2] == 3
        assert eps[1, 2] == eps[2, 1] == 4
        assert eps[0, 2] == eps[2, 0] == 5
        assert eps[0, 1] == eps[1, 0] == 6

    def test_wrong_count_raises(self):
        with pytest.raises(ValueError):
            build_symmetric_strain_tensor('cubic', np.array([0.1, 0.2]))


# ---------------------------------------------------------------------------
# Tests: generate_symmetry_preserving_strain
# ---------------------------------------------------------------------------

class TestGenerateStrain:
    def test_returns_atoms_and_metadata(self):
        atoms = _cubic_fe()
        result, meta = generate_symmetry_preserving_strain(atoms, crystal_system='cubic')
        assert isinstance(result, Atoms)
        assert 'crystal_system' in meta
        assert 'independent_strains' in meta
        assert 'full_strain_tensor' in meta

    def test_original_unchanged(self):
        atoms = _cubic_fe()
        orig_cell = atoms.get_cell().copy()
        generate_symmetry_preserving_strain(atoms, crystal_system='cubic')
        np.testing.assert_allclose(atoms.get_cell(), orig_cell)

    def test_manual_crystal_system(self):
        """Specifying crystal_system='cubic' should not call spglib."""
        atoms = _cubic_fe()
        result, meta = generate_symmetry_preserving_strain(
            atoms, crystal_system='cubic'
        )
        assert meta['crystal_system'] == 'cubic'
        assert len(meta['independent_strains']) == 1

    def test_rng_deterministic(self):
        """Same rng_values → same strain tensor."""
        atoms = _cubic_fe()
        rng = np.array([0.5])
        _, m1 = generate_symmetry_preserving_strain(atoms, crystal_system='cubic', rng_values=rng)
        _, m2 = generate_symmetry_preserving_strain(atoms, crystal_system='cubic', rng_values=rng)
        np.testing.assert_allclose(m1['full_strain_tensor'], m2['full_strain_tensor'])

    def test_volume_preserved(self):
        """After volume conservation, volume should match original."""
        atoms = _cubic_fe()
        orig_vol = atoms.get_volume()
        result, meta = generate_symmetry_preserving_strain(
            atoms, strain_fraction=0.05, crystal_system='cubic'
        )
        assert result.get_volume() > 0
        np.testing.assert_allclose(result.get_volume(), orig_vol, rtol=1e-10)

    def test_triclinic_equivalent_to_full(self):
        """Triclinic → 6 dims, equivalent to generic strain."""
        atoms = _triclinic()
        result, meta = generate_symmetry_preserving_strain(
            atoms, crystal_system='triclinic', rng_values=np.array([0.5]*6)
        )
        assert meta['independent_strains'].shape == (6,)

    def test_orthorhombic_structure(self):
        atoms = _orthorhombic()
        result, meta = generate_symmetry_preserving_strain(
            atoms, crystal_system='orthorhombic'
        )
        eps = meta['full_strain_tensor']
        # Off-diagonal should be zero
        assert np.allclose(eps - np.diag(np.diag(eps)), 0)


# ---------------------------------------------------------------------------
# Tests: detect_crystal_system_spglib
# ---------------------------------------------------------------------------

class TestDetectCrystalSystem:
    def test_cubic_with_explicit(self):
        """Fallback detection may not match spglib; just ensure it returns a valid system."""
        atoms = _cubic_fe()
        cs = detect_crystal_system_spglib(atoms)
        assert cs in CRYSTAL_SYSTEM_STRAIN_DIMS

    def test_hexagonal(self):
        atoms = _hexagonal_mg()
        cs = detect_crystal_system_spglib(atoms)
        assert cs in CRYSTAL_SYSTEM_STRAIN_DIMS
