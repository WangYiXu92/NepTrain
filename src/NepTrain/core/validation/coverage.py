#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spin texture coverage report for magnetic active learning.

Answers the key question: "Did this generation add new magnetic
configurations, or just repeat geometric perturbations?"

Usage: ``NepTrain magnetic-coverage selected.xyz --base train.xyz``
       Generates a JSON report comparing new structures against the
       training set baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase.io import read as ase_read


def _get_vector_array(atoms, *names):
    """Return (N,3) array from atoms.arrays or initial_magmoms."""
    for name in names:
        arr = atoms.arrays.get(name)
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
            if arr.ndim == 1 and arr.shape[0] == len(atoms):
                out = np.zeros((len(atoms), 3), dtype=float)
                out[:, 2] = arr
                return out
        if name == "initial_magmoms":
            magmoms = atoms.get_initial_magnetic_moments()
            arr = np.asarray(magmoms, dtype=float)
            if arr.size > 0 and np.any(np.abs(arr) > 1e-8):
                if arr.ndim == 1 and arr.shape[0] == len(atoms):
                    out = np.zeros((len(atoms), 3), dtype=float)
                    out[:, 2] = arr
                    return out
                if arr.ndim == 2 and arr.shape == (len(atoms), 3):
                    return arr
    return None


def _frame_stats(atoms) -> Optional[Dict[str, Any]]:
    """Compute per-frame spin texture statistics."""
    spin = _get_vector_array(atoms, "spin", "initial_magmoms")
    if spin is None:
        return None
    n_at = len(atoms)
    norms = np.linalg.norm(spin, axis=1)
    safe = np.where(norms > 1e-12, norms, 1.0)
    dirs = spin / safe[:, None]

    # FM/AFM classification: collinear if all directions are parallel or antiparallel
    if n_at > 1:
        dot = dirs @ dirs.T
        iu = np.triu_indices(n_at, k=1)
        align_vals = dot[iu]
        mean_align = float(np.mean(align_vals))
        # FM: all alignments close to +1
        is_fm = bool(np.all(align_vals > 0.8))
        # AFM: at least one strong negative alignment
        is_afm = bool(np.any(align_vals < -0.8)) and not is_fm
    else:
        mean_align = 1.0
        is_fm = True
        is_afm = False

    net_mag = float(np.linalg.norm(spin.sum(axis=0)))
    return {
        "spin_norms": norms.tolist(),
        "mean_norm": float(np.mean(norms)),
        "mean_pairwise_alignment": mean_align,
        "net_magnetization": net_mag,
        "fm": is_fm,
        "afm": is_afm,
        "noncollinear": not is_fm and not is_afm,
    }


def _dataset_stats(frames) -> Dict[str, Any]:
    stats_list = []
    for atoms in frames:
        s = _frame_stats(atoms)
        if s:
            stats_list.append(s)

    if not stats_list:
        return {"n_frames": len(frames), "n_magnetic": 0, "error": "no magnetic arrays found in any frame"}

    all_spin_norms = []
    all_pairwise = []
    fm_count = 0
    afm_count = 0
    ncl_count = 0
    net_mags = []

    for s in stats_list:
        all_spin_norms.extend(s["spin_norms"])
        all_pairwise.append(s["mean_pairwise_alignment"])
        net_mags.append(s["net_magnetization"])
        if s["fm"]:
            fm_count += 1
        elif s["afm"]:
            afm_count += 1
        else:
            ncl_count += 1

    n_total = len(stats_list)
    return {
        "n_frames": len(frames),
        "n_magnetic": n_total,
        "fm_ratio": fm_count / n_total if n_total else None,
        "afm_ratio": afm_count / n_total if n_total else None,
        "noncollinear_ratio": ncl_count / n_total if n_total else None,
        "spin_magnitudes": all_spin_norms,
        "spin_magnitude_mean": float(np.mean(all_spin_norms)),
        "spin_magnitude_std": float(np.std(all_spin_norms)),
        "mean_pairwise_alignment": float(np.mean(all_pairwise)) if all_pairwise else None,
        "net_magnetization": net_mags,
        "net_magnetization_mean": float(np.mean(net_mags)),
    }


def magnetic_coverage_report(
    new_xyz: str | Path,
    base: str | Path | None = None,
    output_json: str | Path | None = None,
) -> Dict[str, Any]:
    """Generate a spin texture coverage report.

    Parameters
    ----------
    new_xyz
        Path to new/selected extxyz structures.
    base
        Optional path to the base training set for comparison.
    output_json
        Optional path to write the JSON report.

    Returns
    -------
    dict with ``new`` and optionally ``base`` statistics plus a
    ``comparison`` section when both are available.
    """
    new_path = Path(new_xyz)
    report: Dict[str, Any] = {
        "n_new": 0,
        "n_base": 0,
        "new": {},
        "comparison": {},
    }

    if not new_path.exists():
        report["error"] = f"{new_path} does not exist"
        if output_json:
            Path(output_json).write_text(json.dumps(report, indent=2))
        return report

    try:
        new_frames = ase_read(str(new_path), index=":", format="extxyz")
    except Exception as exc:
        report["error"] = f"cannot read {new_path}: {exc}"
        if output_json:
            Path(output_json).write_text(json.dumps(report, indent=2))
        return report

    if not isinstance(new_frames, list):
        new_frames = [new_frames]
    report["n_new"] = len(new_frames)
    report["new"] = _dataset_stats(new_frames)

    if base and Path(base).exists():
        try:
            base_frames = ase_read(str(base), index=":", format="extxyz")
        except Exception as exc:
            report["base"] = {"error": f"cannot read {base}: {exc}"}
        else:
            if not isinstance(base_frames, list):
                base_frames = [base_frames]
            report["n_base"] = len(base_frames)
            report["base"] = _dataset_stats(base_frames)

            # Comparison
            new_stats = report["new"]
            base_stats = report["base"]
            if "fm_ratio" in new_stats and "fm_ratio" in base_stats:
                report["comparison"] = {
                    "fm_ratio_diff": round(new_stats.get("fm_ratio", 0) - base_stats.get("fm_ratio", 0), 4),
                    "afm_ratio_diff": round(new_stats.get("afm_ratio", 0) - base_stats.get("afm_ratio", 0), 4),
                    "ncl_ratio_diff": round(new_stats.get("noncollinear_ratio", 0) - base_stats.get("noncollinear_ratio", 0), 4),
                    "spin_magnitude_mean_diff": round(new_stats.get("spin_magnitude_mean", 0) - base_stats.get("spin_magnitude_mean", 0), 4),
                    "coverage_assessment": _coverage_assessment(new_stats, base_stats),
                }

    if output_json:
        Path(output_json).write_text(json.dumps(report, indent=2, default=str))

    return report


def _coverage_assessment(new_stats: dict, base_stats: dict) -> str:
    """Heuristic assessment of whether new data expands magnetic coverage."""
    ncl_new = new_stats.get("noncollinear_ratio", 0)
    ncl_base = base_stats.get("noncollinear_ratio", 0)
    afm_new = new_stats.get("afm_ratio", 0)
    afm_base = base_stats.get("afm_ratio", 0)
    mag_std_new = new_stats.get("spin_magnitude_std", 0)
    mag_std_base = base_stats.get("spin_magnitude_std", 0)

    gains = []
    if ncl_new > ncl_base + 0.05:
        gains.append(f"+{ncl_new - ncl_base:.0%} non-collinear")
    if afm_new > afm_base + 0.05:
        gains.append(f"+{afm_new - afm_base:.0%} AFM")
    if mag_std_new > mag_std_base * 1.2:
        gains.append("wider spin-magnitude spread")

    if gains:
        return f"EXPANDED coverage: {', '.join(gains)}"
    if ncl_new == 0 and afm_new == 0 and ncl_base == 0 and afm_base == 0:
        return "FM-only dataset — add AFM and non-collinear structures"
    return "SIMILAR coverage — new data does not significantly expand magnetic phase space"
