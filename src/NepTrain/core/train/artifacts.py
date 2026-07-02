#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workflow artifact validation and stage reports for NepTrain.

The automatic training loop launches expensive external jobs (NEP, GPUMD, DFT).
A file merely existing is not enough evidence that a stage completed correctly,
so every stage can write a small JSON report with artifact health summaries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ase.io import read as ase_read


XYZ_SUFFIXES = {".xyz", ".extxyz"}


def _json_safe(value: Any) -> Any:
    """Convert Path / NumPy scalar values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def validate_file_artifact(path: str | Path) -> dict[str, Any]:
    """Validate a generic file artifact by existence and non-empty size."""
    artifact = Path(path)
    exists = artifact.exists()
    size_bytes = artifact.stat().st_size if exists and artifact.is_file() else 0
    return {
        "path": str(artifact),
        "exists": exists,
        "size_bytes": size_bytes,
        "valid": bool(exists and size_bytes > 0),
    }


def _count_xyz_frames(path: Path) -> int:
    """Count XYZ/extxyz frames by scanning headers without constructing Atoms.

    This is intentionally lightweight for GPUMD trajectories that can be large;
    array-level validation still uses ASE when required arrays are requested.
    """
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            stripped = header.strip()
            if not stripped:
                continue
            try:
                n_atoms = int(stripped)
            except ValueError:
                raise ValueError(f"Malformed XYZ atom-count line: {stripped!r}")
            # comment line + n_atoms coordinate lines
            for _ in range(n_atoms + 1):
                if not handle.readline():
                    raise ValueError("Truncated XYZ frame")
            count += 1
    return count


def _frame_has_array(atoms, name: str) -> bool:
    if name in atoms.arrays:
        arr = np.asarray(atoms.arrays[name])
        return arr.shape[0] == len(atoms) and np.isfinite(arr).all()
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", {}) if calc is not None else {}
    if name in results:
        arr = np.asarray(results[name])
        return arr.size > 0 and np.isfinite(arr).all()
    return False


def validate_xyz_artifact(
    path: str | Path,
    required_arrays: Sequence[str] = (),
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Validate an extended XYZ artifact.

    Returns a compact report with frame count and per-required-field missing
    counts. Missing arrays make the artifact invalid unless ``allow_empty`` is
    true and the file has zero frames.
    """
    artifact = Path(path)
    base = validate_file_artifact(artifact)
    report: dict[str, Any] = {
        **base,
        "kind": "xyz",
        "required_arrays": list(required_arrays),
        "n_frames": 0,
        "missing_arrays": {},
        "read_error": None,
    }
    if not base["exists"]:
        report["valid"] = False
        return report
    if base["size_bytes"] == 0:
        report["valid"] = bool(allow_empty)
        return report

    if not required_arrays:
        try:
            n_frames = _count_xyz_frames(artifact)
        except Exception as exc:
            report["read_error"] = f"{type(exc).__name__}: {exc}"
            report["valid"] = False
            return report
        report["n_frames"] = n_frames
        report["valid"] = bool((n_frames > 0) or allow_empty)
        return report

    try:
        frames = ase_read(str(artifact), index=":", format="extxyz")
        if not isinstance(frames, list):
            frames = [frames]
    except Exception as exc:  # pragma: no cover - exact ASE exceptions vary
        report["read_error"] = f"{type(exc).__name__}: {exc}"
        report["valid"] = False
        return report

    report["n_frames"] = len(frames)
    missing: dict[str, int] = {}
    for name in required_arrays:
        count = sum(1 for atoms in frames if not _frame_has_array(atoms, name))
        if count:
            missing[name] = count
    report["missing_arrays"] = missing
    report["valid"] = bool((len(frames) > 0 or allow_empty) and not missing)
    return report


def validate_artifact(
    path: str | Path,
    required_arrays: Sequence[str] = (),
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Validate an artifact, using extxyz-aware checks for XYZ files."""
    artifact = Path(path)
    if artifact.suffix.lower() in XYZ_SUFFIXES:
        return validate_xyz_artifact(artifact, required_arrays, allow_empty=allow_empty)
    return validate_file_artifact(artifact)


def _report_path(work_path: str | Path, generation: int, stage: str) -> Path:
    return Path(work_path) / "workflow_reports" / f"Generation-{generation}" / f"{stage}.json"


def write_stage_report(
    work_path: str | Path,
    *,
    generation: int,
    stage: str,
    status: str,
    artifacts: Mapping[str, str | Path] | None = None,
    required_arrays: Mapping[str, Sequence[str]] | None = None,
    summary: Mapping[str, Any] | None = None,
    errors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write a JSON report for one workflow stage.

    Parameters
    ----------
    work_path
        Root training work directory, typically ``config['work_path']``.
    generation
        Training generation number.
    stage
        Stage name such as ``nep``, ``gpumd``, ``select``, ``dft``, ``pred``.
    status
        ``completed``, ``skipped``, ``failed`` or another caller-defined state.
    artifacts
        Named output files to validate.
    required_arrays
        Optional per-artifact required ASE arrays/results. Keys match
        ``artifacts`` names.
    """
    required_arrays = required_arrays or {}
    artifact_reports = {
        name: validate_artifact(path, required_arrays.get(name, ()))
        for name, path in (artifacts or {}).items()
    }
    if status == "skipped":
        valid_artifacts = True
    else:
        valid_artifacts = all(item.get("valid", False) for item in artifact_reports.values())
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(generation),
        "stage": stage,
        "status": status,
        "valid_artifacts": valid_artifacts,
        "summary": _json_safe(dict(summary or {})),
        "errors": list(errors or []),
        "artifacts": artifact_reports,
    }
    path = _report_path(work_path, generation, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _generation_number(path: Path) -> int:
    try:
        return int(path.parent.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def load_stage_reports(work_path: str | Path, generation: int | None = None) -> dict[str, dict[str, Any]]:
    """Load the latest stage reports for a work directory.

    If generation is omitted, reports from all generations are scanned and the
    newest report per stage is returned.
    """
    root = Path(work_path) / "workflow_reports"
    if not root.exists():
        return {}
    if generation is not None:
        candidates = sorted((root / f"Generation-{generation}").glob("*.json"))
    else:
        candidates = sorted(root.glob("Generation-*/*.json"), key=lambda p: (_generation_number(p), p.name))
    reports: dict[str, dict[str, Any]] = {}
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stage = data.get("stage") or path.stem
        reports[stage] = data
    return reports
