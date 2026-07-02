"""Tests for magnetic training data validator (B1).

Each test uses a temporary extxyz file with known arrays.
The validator should report per-frame and per-array issues
without crashing on real-world edge cases.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.io import write as ase_write


def _make_xyz(atoms, out, arrays=None, info=None):
    """Write a single-frame magnetic extxyz for validation."""
    atoms = atoms.copy()
    arrays = arrays or {}
    for name, arr in arrays.items():
        atoms.arrays[name] = np.asarray(arr, dtype=float)
    if info:
        for k, v in info.items():
            atoms.info[k] = v
    ase_write(out, atoms, format="extxyz")


def test_valid_magnetic_frame_passes_all_checks(tmp_path):
    from NepTrain.core.validation.magnetic import validate_magnetic_xyz

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    xyz = tmp_path / "valid.xyz"
    _make_xyz(
        atoms,
        xyz,
        arrays={
            "spin": [[0.0, 0.0, 2.2], [0.0, 0.0, -2.2]],
            "moment": [[0.0, 0.0, 2.1], [0.0, 0.0, -2.1]],
            "torque": [[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]],
            "force": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        info={"energy": -12.0},
    )

    report = validate_magnetic_xyz(xyz)

    assert report["valid"] is True
    assert report["n_frames"] == 1
    assert report["spin_shape_ok"] is True
    assert report["torque_nonzero"] is True


def test_non_magnetic_as_missing_spin_flagged(tmp_path):
    from NepTrain.core.validation.magnetic import validate_magnetic_xyz

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    xyz = tmp_path / "missing.xyz"
    _make_xyz(
        atoms,
        xyz,
        arrays={
            "force": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        info={"energy": -12.0},
    )

    from NepTrain.core.validation.magnetic import validate_magnetic_xyz

    report = validate_magnetic_xyz(xyz)
    assert report["valid"] is False
    assert report["missing_arrays"]


def test_single_frame_without_spin(tmp_path):
    from NepTrain.core.validation.magnetic import validate_magnetic_xyz

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    xyz = tmp_path / "no_spin.xyz"
    _make_xyz(
        atoms,
        xyz,
        arrays={
            "force": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        info={"energy": -12.0},
    )

    report = validate_magnetic_xyz(xyz)

    assert report["missing_arrays"]
    for key in ("spin", "moment", "torque"):
        assert report["missing_arrays"][key] == 1


def test_scalar_magmom_arrays_flagged_as_not_vector(tmp_path):
    from NepTrain.core.validation.magnetic import validate_magnetic_xyz

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    xyz = tmp_path / "scalar.xyz"
    _make_xyz(
        atoms,
        xyz,
        arrays={
            "spin": [2.2, 2.2],
            "moment": [2.1, 2.1],
            "torque": [0.0, 0.0],
            "force": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        info={"energy": -12.0},
    )

    report = validate_magnetic_xyz(xyz)

    assert report["spin_shape_ok"] is False
    assert report["valid"] is False


def test_all_zero_torque_explicitly_noted(tmp_path):
    from NepTrain.core.validation.magnetic import validate_magnetic_xyz

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    xyz = tmp_path / "zero_torque.xyz"
    _make_xyz(
        atoms,
        xyz,
        arrays={
            "spin": [[0.0, 0.0, 2.2], [0.0, 0.0, -2.2]],
            "moment": [[0.0, 0.0, 2.1], [0.0, 0.0, -2.1]],
            "torque": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "force": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        info={"energy": -12.0},
    )

    report = validate_magnetic_xyz(xyz)

    assert report["torque_nonzero"] is False
    assert report["warnings"]
    assert any("torque" in w.lower() for w in report["warnings"])


def test_nan_in_spin_flagged_invalid(tmp_path):
    from NepTrain.core.validation.magnetic import validate_magnetic_xyz

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    xyz = tmp_path / "nan.xyz"
    spin = np.array([[0.0, 0.0, 2.2], [0.0, 0.0, np.nan]])
    _make_xyz(
        atoms,
        xyz,
        arrays={
            "spin": spin,
            "moment": np.zeros((2, 3)),
            "torque": np.zeros((2, 3)),
            "force": np.zeros((2, 3)),
        },
        info={"energy": -12.0},
    )

    report = validate_magnetic_xyz(xyz)

    assert report["valid"] is False
    assert report["nan_inf_arrays"]


def test_xyz_with_energy_ok(tmp_path):
    from NepTrain.core.validation.magnetic import validate_magnetic_xyz

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    xyz = tmp_path / "with_energy.xyz"
    _make_xyz(
        atoms,
        xyz,
        arrays={
            "spin": [[0.0, 0.0, 2.2], [0.0, 0.0, -2.2]],
            "moment": [[0.0, 0.0, 2.1], [0.0, 0.0, -2.1]],
            "torque": [[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "force": np.zeros((2, 3)),
        },
        info={"energy": -12.0},
    )

    report = validate_magnetic_xyz(xyz)

    assert report["valid"] is True
    assert report.get("energies_per_frame") == [-12.0]
