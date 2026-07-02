"""Tests for magnetic spin texture coverage report (A).

Validates that the coverage report correctly computes and compares
spin texture statistics between new structures and a base training set.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write


def _make_xyz(atoms_list, out):
    """Write multi-frame magnetic extxyz."""
    frames = []
    for atoms, arrays in atoms_list:
        a = atoms.copy()
        for name, arr in arrays.items():
            a.arrays[name] = np.asarray(arr, dtype=float)
        frames.append(a)
    ase_write(out, frames, format="extxyz")


def _fe2_fm():
    """FM BCC Fe: spins aligned ↑↑."""
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    spin = np.array([[0.0, 0.0, 2.2], [0.0, 0.0, 2.2]])
    return atoms, {"spin": spin, "moment": spin * 0.95, "torque": np.zeros((2, 3))}


def _fe2_afm():
    """AFM BCC Fe: spins ↑↓."""
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    spin = np.array([[0.0, 0.0, 2.2], [0.0, 0.0, -2.2]])
    return atoms, {"spin": spin, "moment": spin * 0.95, "torque": np.zeros((2, 3))}


def _fe2_ncl():
    """Non-collinear: spins at 90°."""
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    spin = np.array([[2.2, 0.0, 0.0], [0.0, 2.2, 0.0]])
    return atoms, {"spin": spin, "moment": spin * 0.95, "torque": np.zeros((2, 3))}


def test_coverage_fm_only_reports_fm_ratio_1(tmp_path):
    from NepTrain.core.validation.coverage import magnetic_coverage_report

    base = tmp_path / "base.xyz"
    new = tmp_path / "new.xyz"
    _make_xyz([_fe2_fm(), _fe2_fm()], base)
    _make_xyz([_fe2_fm()], new)

    report = magnetic_coverage_report(new, base=base)

    assert report["n_new"] == 1
    assert report["n_base"] == 2
    assert "fm_ratio" in report["new"]
    assert report["new"]["fm_ratio"] == pytest.approx(1.0, abs=0.05)


def test_coverage_afm_has_low_pairwise_alignment(tmp_path):
    from NepTrain.core.validation.coverage import magnetic_coverage_report

    new = tmp_path / "new.xyz"
    _make_xyz([_fe2_afm()], new)

    report = magnetic_coverage_report(new)

    assert report["new"]["mean_pairwise_alignment"] == pytest.approx(-1.0, abs=0.05)


def test_coverage_distinguishes_fm_from_afm(tmp_path):
    from NepTrain.core.validation.coverage import magnetic_coverage_report

    base = tmp_path / "base.xyz"
    new = tmp_path / "new.xyz"
    _make_xyz([_fe2_fm()], base)
    _make_xyz([_fe2_afm()], new)

    report = magnetic_coverage_report(new, base=base)

    # base is all FM → high alignment; new is AFM → low alignment
    assert report["base"]["mean_pairwise_alignment"] > 0.5
    assert report["new"]["mean_pairwise_alignment"] < 0.0


def test_coverage_report_writes_json(tmp_path):
    from NepTrain.core.validation.coverage import magnetic_coverage_report

    new = tmp_path / "new.xyz"
    _make_xyz([_fe2_fm()], new)
    out_json = tmp_path / "coverage.json"

    report = magnetic_coverage_report(new, base=None, output_json=out_json)

    assert out_json.exists()
    on_disk = json.loads(out_json.read_text())
    assert on_disk["n_new"] == 1


def test_coverage_no_arrays_graceful(tmp_path):
    from NepTrain.core.validation.coverage import magnetic_coverage_report

    empty = tmp_path / "empty.xyz"
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.arrays["force"] = np.zeros((2, 3))
    ase_write(empty, atoms, format="extxyz")

    report = magnetic_coverage_report(empty)

    # Should not crash — gracefully report zero magnetic frames
    assert report["n_new"] == 1
    assert report["new"]["n_magnetic"] == 0


def test_spin_magnitude_statistics_present(tmp_path):
    from NepTrain.core.validation.coverage import magnetic_coverage_report

    new = tmp_path / "new.xyz"
    _make_xyz([_fe2_fm(), _fe2_afm()], new)

    report = magnetic_coverage_report(new)

    mags = report["new"]["spin_magnitudes"]
    assert len(mags) == 4  # 2 frames × 2 atoms
    assert abs(np.mean(mags) - 2.2) < 0.1


def test_pairwise_alignment_range(tmp_path):
    import pytest
    from NepTrain.core.validation.coverage import magnetic_coverage_report

    new = tmp_path / "new.xyz"
    _make_xyz([_fe2_fm(), _fe2_afm(), _fe2_ncl()], new)

    report = magnetic_coverage_report(new)

    # FM=-1.0..+1.0 range is covered by mixed FM/AFM/NCL
    assert report["new"]["mean_pairwise_alignment"] is not None
    # Net magnetization varies across frames
    net_mags = report["new"]["net_magnetization"]
    assert len(net_mags) == 3
