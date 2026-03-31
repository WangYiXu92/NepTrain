#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for perturbation compatibility checking."""
import pytest

from NepTrain.core.perturb.compatibility import validate_compatibility


class TestNoConflicts:
    """Cases that should pass without errors or warnings."""

    def test_empty_kwargs(self):
        assert validate_compatibility() == []

    def test_surface_plus_vacancy(self):
        result = validate_compatibility(
            surface=True, vac_elements=['Fe'], vac_num=1
        )
        assert result == []

    def test_dislocation_alone(self):
        assert validate_compatibility(dislocation=True) == []

    def test_sym_strain_alone(self):
        assert validate_compatibility(sym_strain=True, cell_pert_fraction=0.0) == []


class TestMutexGroups:
    """Cases that must raise ValueError for mutually exclusive combos."""

    def test_amorphous_dislocation(self):
        with pytest.raises(ValueError, match="Mutually exclusive"):
            validate_compatibility(amorphous=True, dislocation=True)

    def test_sym_strain_cell_pert(self):
        with pytest.raises(ValueError, match="Mutually exclusive"):
            validate_compatibility(sym_strain=True, cell_pert_fraction=0.03)

    def test_amorphous_twinning(self):
        with pytest.raises(ValueError, match="Mutually exclusive"):
            validate_compatibility(amorphous=True, twinning=True)

    def test_amorphous_grain_boundary(self):
        with pytest.raises(ValueError, match="Mutually exclusive"):
            validate_compatibility(amorphous=True, gb=True)

    def test_amorphous_stacking_fault(self):
        with pytest.raises(ValueError, match="Mutually exclusive"):
            validate_compatibility(amorphous=True, stacking_fault=True)

    def test_amorphous_surface(self):
        with pytest.raises(ValueError, match="Mutually exclusive"):
            validate_compatibility(amorphous=True, surface=True)

    def test_multiple_mutex_pairs(self):
        """amorphous + dislocation + surface should report multiple pairs."""
        with pytest.raises(ValueError, match="amorphous") as exc_info:
            validate_compatibility(amorphous=True, dislocation=True, surface=True)
        msg = str(exc_info.value)
        assert msg.count("Mutually exclusive") >= 2


class TestDependencyGroups:
    """Cases that must raise ValueError for missing dependencies."""

    def test_vacancy_without_elements(self):
        with pytest.raises(ValueError, match="vac_elements"):
            validate_compatibility(vac_elements=None, vac_num=1)

    def test_vacancy_with_zero_num(self):
        with pytest.raises(ValueError, match="vac_num"):
            validate_compatibility(vac_elements=['Fe'], vac_num=0)

    def test_vacancy_valid(self):
        result = validate_compatibility(vac_elements=['Fe'], vac_num=1)
        assert result == []

    def test_antisite_without_pairs(self):
        with pytest.raises(ValueError, match="antisite_pairs"):
            validate_compatibility(antisite=True, antisite_pairs=None)

    def test_antisite_valid(self):
        result = validate_compatibility(antisite=True, antisite_pairs=[('Fe', 'Al')])
        assert result == []


class TestWarnings:
    """Cases that pass but produce warnings."""

    def test_amorphous_shuffle(self):
        result = validate_compatibility(amorphous=True, shuffle_elements=['Fe'])
        assert len(result) == 1
        assert 'shuffle' in result[0]

    def test_amorphous_rigid(self):
        result = validate_compatibility(amorphous=True, rigid=True)
        assert len(result) == 1
        assert 'rigid' in result[0]

    def test_amorphous_shuffle_and_rigid(self):
        result = validate_compatibility(
            amorphous=True, shuffle_elements=['Fe'], rigid=True
        )
        assert len(result) == 2
