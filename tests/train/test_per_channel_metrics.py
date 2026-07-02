"""Tests for per-channel metrics enhancement (B).

Covers:
- loss.out parsing with magnetic columns
- prediction output parsing for component-wise RMSE
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


def _write_loss_out(path: Path, lines: list):
    """Write a numpy-compatible loss.out."""
    data = np.array(lines, dtype=float)
    np.savetxt(path, data, fmt="%.6e")


def test_standard_10_col_loss_still_works(tmp_path):
    from NepTrain.core.train.status import _read_nep_loss

    loss_out = tmp_path / "loss.out"
    _write_loss_out(loss_out, [
        [1, 0.5, 0.1, 0.2, 3.0, 45.0, 0.0, 5.0, 70.0, 0.0],
    ])

    result = _read_nep_loss(loss_out)
    assert result["epoch"] == 1
    assert result["rmse_energy"] == 3.0
    assert result["rmse_force"] == 45.0
    assert result["rmse_energy_test"] == 5.0
    assert result["rmse_force_test"] == 70.0


def test_magnetic_loss_columns_detected(tmp_path):
    from NepTrain.core.train.status import _read_nep_loss

    loss_out = tmp_path / "loss.out"
    # 16 columns: standard 10 + 6 magnetic (Mx,My,Mz trn; Tx,Ty,Tz trn)
    _write_loss_out(loss_out, [
        [1, 0.5, 0.1, 0.2, 3.0, 45.0, 0.0, 5.0, 70.0, 0.0,
         0.03, 0.04, 0.02, 0.001, 0.001, 0.001],
    ])

    result = _read_nep_loss(loss_out)
    assert result["epoch"] == 1
    assert result["n_columns"] == 16
    assert result["has_magnetic_loss"] is True
    assert "rmse_moment_x" in result
    assert "rmse_moment_y" in result
    assert "rmse_moment_z" in result
    assert result["rmse_moment_x"] == 0.03
    assert result["rmse_moment_z"] == 0.02


def test_prediction_parity_computes_component_rmse(tmp_path):
    from NepTrain.core.train.status import parse_prediction_metrics

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    # force.out: fmt = "fx_pred fy_pred fz_pred fx_dft fy_dft fz_dft"
    np.savetxt(pred_dir / "force.out", [
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        [-0.1, 0.0, 0.1, 0.0, 0.0, 0.0],
    ])
    # energy.out: fmt = "e_pred e_dft"
    np.savetxt(pred_dir / "energy.out", [
        [-12.0, -12.1],
        [-12.0, -11.9],
    ])

    metrics = parse_prediction_metrics(str(pred_dir))

    assert "force_x_rmse" in metrics
    assert metrics["force_x_rmse"] == pytest.approx(np.sqrt((0.1**2 + 0.1**2) / 2), abs=1e-6)
    assert "force_y_rmse" in metrics
    assert "energy_rmse" in metrics
    assert metrics["energy_rmse"] == pytest.approx(np.sqrt((0.1**2 + 0.1**2) / 2), abs=1e-6)


def test_prediction_metric_nonexistent_dir(tmp_path):
    from NepTrain.core.train.status import parse_prediction_metrics
    import pytest

    metrics = parse_prediction_metrics(str(tmp_path / "nope"))
    assert metrics == {}


def test_prediction_with_spin_output(tmp_path):
    from NepTrain.core.train.status import parse_prediction_metrics

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    # spin.out: Sx_pred Sy_pred Sz_pred Sx_dft Sy_dft Sz_dft
    np.savetxt(pred_dir / "spin.out", [
        [0.0, 0.0, 2.0, 0.0, 0.0, 2.2],
        [0.0, 0.0, -2.0, 0.0, 0.0, -2.1],
    ])
    np.savetxt(pred_dir / "force.out", [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    metrics = parse_prediction_metrics(str(pred_dir))
    assert "moment_x_rmse" in metrics
    assert metrics["moment_x_rmse"] == 0.0
    assert metrics["moment_z_rmse"] == pytest.approx(np.sqrt((0.2**2 + 0.1**2) / 2), abs=1e-6)
