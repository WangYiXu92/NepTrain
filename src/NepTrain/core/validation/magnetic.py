#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate magnetic extxyz training data for GPUMD ``model_type=4``.

Run with: ``NepTrain validate-magnetic train.xyz`` (or via CLI later).

Checks are non-fatal — the validator returns a report dict; the caller
decides what to stop on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from ase.io import read as ase_read

MAGNETIC_ARRAYS = ("spin", "moment", "torque")
MAX_WARNINGS = 20


def _array_shape_ok(arr: np.ndarray, n_atoms: int) -> bool:
    return arr.shape == (n_atoms, 3)


def _array_isfinite(arr: np.ndarray) -> bool:
    return np.isfinite(arr).all()


def validate_magnetic_xyz(path: str | Path) -> Dict[str, Any]:
    """Validate a magnetic extxyz file for GPUMD magnetic NEP training.

    Returns a dict with per-frame diagnostics. ``valid`` is True only when
    all frames pass the mandatory checks (all magnetic arrays present with
    correct (N,3) shapes and no NaN/Inf values). Warnings are informational
    and do not invalidate the report.
    """
    artifact = Path(path)
    n_frames = 0
    missing: Dict[str, int] = {"spin": 0, "moment": 0, "torque": 0}
    nan_inf: List[str] = []
    scalar_frames: List[int] = []
    spin_norms: List[float] = []
    energies: List[float] = []
    torque_ref = None
    torque_nonzero = None
    warnings: List[str] = []

    report: Dict[str, Any] = {
        "path": str(artifact),
        "exists": artifact.exists(),
        "valid": False,
        "n_frames": 0,
        "missing_arrays": {},
        "spin_shape_ok": None,
        "torque_nonzero": None,
        "nan_inf_arrays": [],
        "warnings": [],
        "spin_norms": [],
        "energies_per_frame": [],
    }

    if not artifact.exists():
        report["warnings"].append("file does not exist")
        return report

    try:
        frames = ase_read(str(artifact), index=":", format="extxyz")
    except Exception as exc:
        report["warnings"].append(f"cannot read extxyz: {exc}")
        return report

    if not isinstance(frames, list):
        frames = [frames]
    n_frames = len(frames)
    report["n_frames"] = n_frames

    for i, atoms in enumerate(frames):
        n_at = len(atoms)

        # ---- energy tracking ----
        energy = atoms.get_potential_energy()
        if energy is not None and energy != 0.0:
            energies.append(float(energy))

        # ---- check required arrays ----
        for name in MAGNETIC_ARRAYS:
            if name not in atoms.arrays:
                missing[name] += 1
                continue
            arr = np.asarray(atoms.arrays[name], dtype=float)
            if not _array_shape_ok(arr, n_at):
                if arr.ndim == 1 and arr.shape[0] == n_at:
                    scalar_frames.append(i)
                missing[name] += 1
                continue
            if not _array_isfinite(arr):
                nan_inf.append(name)

        # ---- spin norm ----
        if "spin" in atoms.arrays:
            spin = np.asarray(atoms.arrays["spin"], dtype=float)
            if _array_shape_ok(spin, n_at) and _array_isfinite(spin):
                norms = np.linalg.norm(spin, axis=1)
                spin_norms.extend(norms.tolist())

        # ---- torque statistics ----
        if "torque" in atoms.arrays:
            torq = np.asarray(atoms.arrays["torque"], dtype=float)
            if _array_shape_ok(torq, n_at) and _array_isfinite(torq):
                if torque_ref is None:
                    torque_ref = torq
                if torque_nonzero is None and not np.allclose(torq, 0.0, atol=1e-12):
                    torque_nonzero = True

    # ---- finalise ----
    report["missing_arrays"] = {k: v for k, v in missing.items() if v}
    report["nan_inf_arrays"] = list(set(nan_inf))

    # shape check: spin is our reference array
    spin_shape_ok = "spin" not in report["missing_arrays"] and not scalar_frames
    report["spin_shape_ok"] = spin_shape_ok

    if scalar_frames:
        msg = f"frames {scalar_frames[:5]} have scalar (N,) magnetic arrays instead of (N,3); convert collinear data to vectors (z-axis) before GPUMD training"
        if len(scalar_frames) > 5:
            msg += f" ({len(scalar_frames)} total)"
        report["warnings"].append(msg)

    if torque_nonzero is None:
        torque_nonzero = False
    report["torque_nonzero"] = bool(torque_nonzero)
    if not torque_nonzero:
        report["warnings"].append("all torque arrays are zero — this is OK for Phase 1 but must be explicitly marked as zero-torque dataset")

    report["spin_norms"] = spin_norms
    report["energies_per_frame"] = energies

    # valid = all frames have all three arrays with correct shape + no NaN
    has_all_arrays = not report["missing_arrays"]
    report["valid"] = bool(has_all_arrays and spin_shape_ok and not nan_inf)

    # trim warnings
    if len(report["warnings"]) > MAX_WARNINGS:
        report["warnings"] = report["warnings"][:MAX_WARNINGS]
        report["warnings"].append(f"... truncated, {len(report['warnings'])} total warnings")

    return report
