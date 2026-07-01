import os
import sys

import numpy as np
from ase import Atoms

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.dft.vasp.io import attach_magnetic_exyz_arrays


def test_attach_magnetic_exyz_arrays_uses_vector_magmoms_and_zero_torque():
    atoms = Atoms('Fe2', positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)
    initial_spin = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    dft_moment = np.array([[2.2, 0.1, 0.0], [0.0, 0.0, -2.1]])
    atoms.set_initial_magnetic_moments(initial_spin)
    atoms.calc = type('Calc', (), {'results': {'magmoms': dft_moment}})()

    out = attach_magnetic_exyz_arrays(atoms)

    np.testing.assert_allclose(out.arrays['spin'], initial_spin)
    np.testing.assert_allclose(out.arrays['moment'], dft_moment)
    np.testing.assert_allclose(out.arrays['torque'], np.zeros((2, 3)))


def test_attach_magnetic_exyz_arrays_converts_collinear_to_vectors():
    atoms = Atoms('Fe2', positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.set_initial_magnetic_moments([2.0, -2.0])
    atoms.calc = type('Calc', (), {'results': {'magmoms': np.array([2.1, -1.9])}})()

    out = attach_magnetic_exyz_arrays(atoms)

    np.testing.assert_allclose(out.arrays['spin'], [[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    np.testing.assert_allclose(out.arrays['moment'], [[0.0, 0.0, 2.1], [0.0, 0.0, -1.9]])
    assert out.arrays['spin'].shape == (2, 3)
    assert out.arrays['moment'].shape == (2, 3)
    assert out.arrays['torque'].shape == (2, 3)


def test_attach_magnetic_exyz_arrays_ignores_scalar_total_magmom():
    atoms = Atoms('Fe2', positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.set_initial_magnetic_moments([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    atoms.calc = type('Calc', (), {'results': {'magmom': 0.0}})()

    out = attach_magnetic_exyz_arrays(atoms)

    np.testing.assert_allclose(out.arrays['moment'], out.arrays['spin'])


def test_attach_magnetic_exyz_arrays_accepts_spin_override_for_vasprun_frames():
    atoms = Atoms('Fe2', positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.calc = type('Calc', (), {'results': {}})()
    spin = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    out = attach_magnetic_exyz_arrays(atoms, spin_values=spin)

    np.testing.assert_allclose(out.arrays['spin'], spin)
    np.testing.assert_allclose(out.arrays['moment'], spin)
