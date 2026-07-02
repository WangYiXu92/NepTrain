#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check VASP INCAR readiness for non-collinear constrained-moment calculations.

These calculations are required for generating GPUMD ``model_type=4``
training data with real torque (magnetic force) labels.

Run with: ``NepTrain check-magnetic-input <calc_dir>`` (or via CLI later).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


def _parse_bool(value: str) -> bool:
    """Parse a VASP boolean/toggle value."""
    v = value.strip().upper()
    return v in (".TRUE.", "T", "TRUE", ".T.", "YES", "1", "ON")


def _read_incar(directory: Path) -> Dict[str, str]:
    """Read INCAR key-value pairs (case-insensitive keys)."""
    incar = directory / "INCAR"
    if not incar.exists():
        raise FileNotFoundError(f"INCAR not found in {directory}")
    params: Dict[str, str] = {}
    for line in incar.read_text(errors="ignore").splitlines():
        line = line.split("!", 1)[0].split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().upper()
        val = val.strip()
        if val:
            params[key] = val
    return params


def _count_magmom_values(magmom: str) -> int:
    """Count MAGMOM entries, handling multipliers like 16*2.2."""
    count = 0
    for token in magmom.split():
        token = token.strip()
        if not token:
            continue
        if "*" in token:
            left, _ = token.split("*", 1)
            try:
                count += int(left)
            except ValueError:
                count += 1
        else:
            count += 1
    return count


def check_magnetic_readiness(directory: str | Path) -> Dict[str, Any]:
    """Check a VASP calculation directory for non-collinear constrained-moment readiness.

    Returns a dict with ``ready`` (bool) and diagnostic fields.
    """
    calc = Path(directory)
    report: Dict[str, Any] = {
        "directory": str(calc),
        "ready": False,
        "noncollinear": False,
        "constrained_m": False,
        "m_con_present": False,
        "magmom_vector": False,
        "magmom_count": None,
        "missing_params": [],
        "missing_inputs": [],
        "warnings": [],
    }

    # Check inputs exist
    if not (calc / "INCAR").exists():
        report["missing_inputs"].append("INCAR")
    for fname in ("POSCAR", "OUTCAR"):
        if not (calc / fname).exists():
            report["warnings"].append(f"{fname} missing — cannot cross-validate atom count or torque output")
    if report["missing_inputs"]:
        return report

    try:
        params = _read_incar(calc)
    except FileNotFoundError:
        report["missing_inputs"].append("INCAR")
        return report

    # LNONCOLLINEAR
    lnoncollinear = params.get("LNONCOLLINEAR")
    if lnoncollinear and _parse_bool(lnoncollinear):
        report["noncollinear"] = True
    else:
        report["noncollinear"] = False
        report["missing_params"].append("LNONCOLLINEAR")

    # I_CONSTRAINED_M
    iconst_m = params.get("I_CONSTRAINED_M")
    if iconst_m and iconst_m.strip() in ("1", "2"):
        report["constrained_m"] = True
    else:
        report["missing_params"].append("I_CONSTRAINED_M")

    # M_CON
    m_con = params.get("M_CON")
    if m_con:
        report["m_con_present"] = True
    else:
        report["missing_params"].append("M_CON")

    # MAGMOM
    magmom = params.get("MAGMOM", "")
    if magmom:
        n_vals = _count_magmom_values(magmom)
        report["magmom_count"] = n_vals
        if n_vals % 3 == 0 and n_vals > 0:
            report["magmom_vector"] = True
        else:
            report["warnings"].append(f"MAGMOM has {n_vals} values (scalar per atom); non-collinear VASP prefers 3N vector format. This IS still accepted.")
    else:
        report["missing_params"].append("MAGMOM")

    report["ready"] = bool(
        report["noncollinear"]
        and report["constrained_m"]
        and report["m_con_present"]
        and report["magmom_count"] is not None
    )

    return report
