#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert completed VASP magnetic calculations to GPUMD magnetic extxyz.

The output schema is compatible with GPUMD magnetic NEP ``model_type=4``::

    Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:torque:R:3:moment:R:3

``spin`` is parsed from INCAR ``MAGMOM`` (the constrained/input magnetic
configuration). ``moment`` is parsed from OUTCAR magnetization blocks when
available. ``torque`` is currently written as zero unless an explicit torque
array is supplied by a future parser; VASP OUTCAR does not expose the GPUMD
training torque label directly in the standard paths used here.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase.io import read as ase_read


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def _floats(text: str) -> List[float]:
    return [float(x) for x in _FLOAT_RE.findall(text)]


def parse_incar_magmom(path: Path, n_atoms: int) -> np.ndarray:
    """Parse INCAR MAGMOM into an (N, 3) spin array.

    Supports both collinear scalar form (N values) and non-collinear vector form
    (3N values). Basic VASP multiplier syntax such as ``16*2.2`` is supported.
    """
    text = path.read_text(errors="ignore")
    match = re.search(r"^\s*MAGMOM\s*=\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        return np.zeros((n_atoms, 3), dtype=float)

    values: List[float] = []
    # Drop inline comments and split on whitespace.
    raw = match.group(1).split("#", 1)[0].split("!", 1)[0].split()
    for token in raw:
        if "*" in token:
            left, right = token.split("*", 1)
            try:
                count = int(left)
                value = float(right)
            except ValueError:
                continue
            values.extend([value] * count)
        else:
            try:
                values.append(float(token))
            except ValueError:
                continue

    arr = np.asarray(values, dtype=float)
    if arr.size == n_atoms:
        out = np.zeros((n_atoms, 3), dtype=float)
        out[:, 2] = arr
        return out
    if arr.size == 3 * n_atoms:
        return arr.reshape(n_atoms, 3)
    raise ValueError(f"MAGMOM in {path} has {arr.size} values; expected {n_atoms} or {3*n_atoms}")


def parse_outcar_energy(path: Path) -> float:
    """Return the final TOTEN energy from OUTCAR."""
    energy: Optional[float] = None
    for line in path.read_text(errors="ignore").splitlines():
        if "free  energy   TOTEN" in line:
            vals = _floats(line)
            if vals:
                energy = vals[0]
    if energy is None:
        raise ValueError(f"Could not find TOTEN in {path}")
    return energy


def parse_outcar_forces(path: Path, n_atoms: int) -> np.ndarray:
    """Parse final TOTAL-FORCE block from OUTCAR."""
    lines = path.read_text(errors="ignore").splitlines()
    start = None
    for i, line in enumerate(lines):
        if "TOTAL-FORCE" in line and "eV/Angst" in line:
            start = i
    if start is None:
        raise ValueError(f"Could not find TOTAL-FORCE block in {path}")

    # Next line is separator. Then N atom lines: x y z fx fy fz.
    forces = []
    for line in lines[start + 2:start + 2 + n_atoms]:
        vals = _floats(line)
        if len(vals) < 6:
            raise ValueError(f"Malformed force line in {path}: {line!r}")
        forces.append(vals[3:6])
    arr = np.asarray(forces, dtype=float)
    if arr.shape != (n_atoms, 3):
        raise ValueError(f"Parsed force shape {arr.shape}; expected ({n_atoms}, 3)")
    return arr


def _parse_magnetization_component(lines: Sequence[str], component: str, n_atoms: int) -> Optional[np.ndarray]:
    """Parse final 'magnetization (x/y/z)' component totals from OUTCAR."""
    starts = [i for i, line in enumerate(lines) if f"magnetization ({component})" in line]
    if not starts:
        return None
    start = starts[-1]

    data: List[float] = []
    # VASP block has header lines followed by rows: ion s p d tot
    for line in lines[start + 1:]:
        if "magnetization (" in line or "total charge" in line:
            break
        vals = _floats(line)
        if len(vals) >= 5 and len(data) < n_atoms:
            # First value is ion index; last value is tot.
            data.append(vals[-1])
        elif data and len(data) >= n_atoms:
            break
    if len(data) != n_atoms:
        return None
    return np.asarray(data, dtype=float)


def parse_outcar_moments(path: Path, n_atoms: int, fallback_spin: np.ndarray) -> np.ndarray:
    """Parse per-atom magnetic moment vectors from OUTCAR.

    Collinear VASP usually reports only ``magnetization (x)``; by convention we
    map a single available component to z so it remains compatible with GPUMD's
    vector moment schema. Non-collinear/SOC calculations may report x/y/z blocks;
    when all are present they are used as full vectors.
    """
    lines = path.read_text(errors="ignore").splitlines()
    comps = {c: _parse_magnetization_component(lines, c, n_atoms) for c in ("x", "y", "z")}
    present = {c: v for c, v in comps.items() if v is not None}
    if all(comps[c] is not None for c in ("x", "y", "z")):
        return np.column_stack([comps["x"], comps["y"], comps["z"]]).astype(float)
    if len(present) == 1:
        out = np.zeros((n_atoms, 3), dtype=float)
        out[:, 2] = next(iter(present.values()))
        return out
    return np.asarray(fallback_spin, dtype=float).copy()


def write_magnetic_exyz_frame(handle, atoms, energy: float, forces: np.ndarray, spin: np.ndarray, moment: np.ndarray, torque: Optional[np.ndarray] = None) -> None:
    n_atoms = len(atoms)
    torque_arr = np.zeros((n_atoms, 3), dtype=float) if torque is None else np.asarray(torque, dtype=float)
    arrays = {"force": forces, "spin": spin, "moment": moment, "torque": torque_arr}
    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (n_atoms, 3) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {name} array: shape={arr.shape}, finite={np.isfinite(arr).all()}")
        arrays[name] = arr

    cell = atoms.get_cell().array
    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    handle.write(f"{n_atoms}\n")
    handle.write(
        f'Lattice="{cell[0,0]:.10f} {cell[0,1]:.10f} {cell[0,2]:.10f} '
        f'{cell[1,0]:.10f} {cell[1,1]:.10f} {cell[1,2]:.10f} '
        f'{cell[2,0]:.10f} {cell[2,1]:.10f} {cell[2,2]:.10f}" '
        f'energy={energy:.10f} '
        f'Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:torque:R:3:moment:R:3\n'
    )
    for i, sym in enumerate(symbols):
        f = arrays["force"][i]
        s = arrays["spin"][i]
        t = arrays["torque"][i]
        m = arrays["moment"][i]
        handle.write(
            f"{sym} "
            f"{pos[i,0]:.10f} {pos[i,1]:.10f} {pos[i,2]:.10f} "
            f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f} "
            f"{s[0]:.10f} {s[1]:.10f} {s[2]:.10f} "
            f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} "
            f"{m[0]:.10f} {m[1]:.10f} {m[2]:.10f}\n"
        )


def convert_vasp_dir(calc_dir: Path, handle) -> None:
    """Convert one VASP calculation directory to one extxyz frame."""
    poscar = calc_dir / "CONTCAR"
    if not poscar.exists() or poscar.stat().st_size == 0:
        poscar = calc_dir / "POSCAR"
    incar = calc_dir / "INCAR"
    outcar = calc_dir / "OUTCAR"
    if not poscar.exists() or not incar.exists() or not outcar.exists():
        missing = [p.name for p in (poscar, incar, outcar) if not p.exists()]
        raise FileNotFoundError(f"{calc_dir}: missing {missing}")

    atoms = ase_read(poscar)
    n_atoms = len(atoms)
    spin = parse_incar_magmom(incar, n_atoms)
    energy = parse_outcar_energy(outcar)
    forces = parse_outcar_forces(outcar, n_atoms)
    moment = parse_outcar_moments(outcar, n_atoms, fallback_spin=spin)
    write_magnetic_exyz_frame(handle, atoms, energy, forces, spin, moment)


def discover_calculation_dirs(paths: Iterable[Path], recursive: bool = False) -> List[Path]:
    dirs: List[Path] = []
    for path in paths:
        if path.is_file():
            path = path.parent
        if (path / "OUTCAR").exists() and (path / "INCAR").exists():
            dirs.append(path)
        elif recursive:
            dirs.extend(sorted(p.parent for p in path.rglob("OUTCAR") if (p.parent / "INCAR").exists()))
    # Keep stable order and de-duplicate.
    seen = set()
    unique = []
    for d in dirs:
        key = d.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def convert_paths(paths: Iterable[Path], output: Path, recursive: bool = False) -> int:
    calc_dirs = discover_calculation_dirs(paths, recursive=recursive)
    if not calc_dirs:
        raise FileNotFoundError("No VASP calculation directories with OUTCAR+INCAR found")
    converted = 0
    with output.open("w") as handle:
        for calc_dir in calc_dirs:
            convert_vasp_dir(calc_dir, handle)
            converted += 1
    return converted


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert VASP magnetic OUTCAR directories to GPUMD magnetic extxyz")
    parser.add_argument("paths", nargs="+", type=Path, help="VASP calculation directories or roots")
    parser.add_argument("-o", "--output", type=Path, default=Path("train.xyz"), help="Output extxyz path")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover OUTCAR files under input roots")
    args = parser.parse_args(argv)
    count = convert_paths(args.paths, args.output, recursive=args.recursive)
    print(f"Converted {count} VASP calculations -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
