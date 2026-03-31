# -*- coding: utf-8 -*-
"""Tests for antisite defect generation."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fe3al_supercell(rep=(2, 2, 2)):
    """B2 FeAl structure using rocksalt convention."""
    atoms = bulk('CsCl', crystalstructure='rocksalt', a=2.87, cubic=True)
    symbols = atoms.get_chemical_symbols()
    # CsCl has 2 atoms: index 0 = Cs, index 1 = Cl
    # Replace Cs->Fe, Cl->Al
    new_sym = ['Fe' if s == 'Cs' else 'Al' for s in symbols]
    atoms.set_chemical_symbols(new_sym)
    atoms = atoms.repeat(rep)
    return atoms


def _make_fe_cr_al_supercell(rep=(2, 2, 2)):
    """Fe-Cr-Al ternary for multi-pair testing."""
    atoms = bulk('CsCl', crystalstructure='rocksalt', a=2.87, cubic=True)
    symbols = atoms.get_chemical_symbols()
    new_sym = ['Fe' if s == 'Cs' else 'Cr' for s in symbols]
    atoms.set_chemical_symbols(new_sym)
    # Replace some Cr with Al
    syms = atoms.get_chemical_symbols()
    for i, s in enumerate(syms):
        if s == 'Cr' and i % 4 == 0:
            syms[i] = 'Al'
    atoms.set_chemical_symbols(syms)
    atoms = atoms.repeat(rep)
    return atoms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAntisiteBasic:
    """Basic antisite swap tests."""

    def test_single_swap_preserves_count(self):
        from NepTrain.core.perturb.antisite import generate_antisite_defects
        atoms = _make_fe3al_supercell()
        n_orig = len(atoms)
        result, meta = generate_antisite_defects(
            atoms, swap_pairs=[('Fe', 'Al')], num_swaps=1, mode='random',
        )
        assert len(result) == n_orig
        assert len(meta['swaps']) == 1
        sw = meta['swaps'][0]
        assert sw['elem_i'] == 'Fe'
        assert sw['elem_j'] == 'Al'

    def test_swap_changes_symbols(self):
        from NepTrain.core.perturb.antisite import generate_antisite_defects
        atoms = _make_fe3al_supercell()
        orig_symbols = atoms.get_chemical_symbols()
        result, meta = generate_antisite_defects(
            atoms, swap_pairs=[('Fe', 'Al')], num_swaps=1, mode='random',
        )
        new_symbols = result.get_chemical_symbols()
        assert new_symbols != orig_symbols

    def test_multiple_swaps(self):
        from NepTrain.core.perturb.antisite import generate_antisite_defects
        atoms = _make_fe3al_supercell()
        result, meta = generate_antisite_defects(
            atoms, swap_pairs=[('Fe', 'Al')], num_swaps=3, mode='random',
        )
        assert len(meta['swaps']) == 3

    def test_deterministic_with_rng(self):
        from NepTrain.core.perturb.antisite import generate_antisite_defects
        atoms = _make_fe3al_supercell()
        rng = np.array([0.5])
        r1, m1 = generate_antisite_defects(
            atoms, swap_pairs=[('Fe', 'Al')], num_swaps=1, mode='random', rng_values=rng,
        )
        r2, m2 = generate_antisite_defects(
            atoms, swap_pairs=[('Fe', 'Al')], num_swaps=1, mode='random', rng_values=rng,
        )
        assert r1.get_chemical_symbols() == r2.get_chemical_symbols()


class TestSymmetryAware:
    """Symmetry-aware antisite generation."""

    def test_symmetry_aware_runs(self):
        from NepTrain.core.perturb.antisite import generate_antisite_defects
        atoms = _make_fe3al_supercell()
        # symmetry_aware should work without error
        result, meta = generate_antisite_defects(
            atoms, swap_pairs=[('Fe', 'Al')], num_swaps=1, mode='symmetry_aware',
        )
        assert len(result) == len(atoms)
        assert len(meta['swaps']) == 1


class TestEquivalentSites:
    """Tests for get_equivalent_sites."""

    def test_returns_dict(self):
        from NepTrain.core.perturb.antisite import get_equivalent_sites
        atoms = _make_fe3al_supercell()
        equiv = get_equivalent_sites(atoms)
        assert isinstance(equiv, dict)
        assert 'Fe' in equiv or 'Al' in equiv

    def test_all_atoms_accounted(self):
        from NepTrain.core.perturb.antisite import get_equivalent_sites
        atoms = _make_fe3al_supercell()
        equiv = get_equivalent_sites(atoms)
        total = sum(len(g) for groups in equiv.values() for g in groups)
        # Each atom counted once per group index list
        all_indices = [i for groups in equiv.values() for g in groups for i in g]
        assert sorted(all_indices) == list(range(len(atoms)))


class TestMultiPair:
    """Multiple swap pairs."""

    def test_semicolon_parse(self):
        from NepTrain.core.perturb.run import _parse_antisite_pairs
        pairs = _parse_antisite_pairs('Fe,Al;Fe,Cr')
        assert pairs == [('Fe', 'Al'), ('Fe', 'Cr')]

    def test_single_pair_string(self):
        from NepTrain.core.perturb.run import _parse_antisite_pairs
        pairs = _parse_antisite_pairs('Fe,Al')
        assert pairs == [('Fe', 'Al')]

    def test_list_input(self):
        from NepTrain.core.perturb.run import _parse_antisite_pairs
        pairs = _parse_antisite_pairs([('Fe', 'Al'), ('Cr', 'Fe')])
        assert pairs == [('Fe', 'Al'), ('Cr', 'Fe')]

    def test_none_input(self):
        from NepTrain.core.perturb.run import _parse_antisite_pairs
        assert _parse_antisite_pairs(None) == []


class TestEdgeCases:

    def test_no_matching_element_raises(self):
        from NepTrain.core.perturb.antisite import generate_antisite_defects
        atoms = _make_fe3al_supercell()
        with pytest.raises(ValueError, match="No available 'Xx'"):
            generate_antisite_defects(
                atoms, swap_pairs=[('Xx', 'Al')], num_swaps=1, mode='random',
            )

    def test_original_unchanged(self):
        from NepTrain.core.perturb.antisite import generate_antisite_defects
        atoms = _make_fe3al_supercell()
        orig_syms = list(atoms.get_chemical_symbols())
        generate_antisite_defects(
            atoms, swap_pairs=[('Fe', 'Al')], num_swaps=1, mode='random',
        )
        assert atoms.get_chemical_symbols() == orig_syms
