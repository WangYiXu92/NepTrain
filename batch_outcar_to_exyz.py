#!/usr/bin/env python3
"""Batch convert completed VASP magnetic OUTCAR directories to GPUMD train.xyz.

Usage:
  # Convert all configs under a project root (both phases)
  python batch_outcar_to_exyz.py /work/uk106586/fe_magnetic_nep -o train.xyz

  # Convert a single phase
  python batch_outcar_to_exyz.py /work/uk106586/fe_magnetic_nep/phase1_collinear -o train_col.xyz

  # Run on cluster via SSH
  ssh hpc "cd /work/uk106586/fe_magnetic_nep && python batch_outcar_to_exyz.py . -o train.xyz --check-only"

Validates each OUTCAR for completeness before conversion.
Reports a summary table at the end.
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Reuse the existing converter module
sys.path.insert(0, str(Path(__file__).parent / "src"))
from NepTrain.core.dft.vasp.magnetic_exyz import (
    parse_incar_magmom,
    parse_outcar_energy,
    parse_outcar_forces,
    parse_outcar_moments,
    parse_outcar_magforces,
    write_magnetic_exyz_frame,
)
from ase.io import read as ase_read


def check_outcar_complete(outcar: Path) -> tuple[bool, str]:
    """Check if OUTCAR represents a completed VASP run."""
    if not outcar.exists() or outcar.stat().st_size == 0:
        return False, "OUTCAR missing or empty"
    text = outcar.read_text(errors="ignore")
    if "Elapsed time" not in text:
        return False, "no Elapsed time (job not finished)"
    if "free  energy   TOTEN" not in text:
        return False, "no TOTEN found"
    if "TOTAL-FORCE" not in text:
        return False, "no TOTAL-FORCE block"
    return True, "OK"


def discover_configs(root: Path) -> list[Path]:
    """Find all config_* directories containing INCAR+OUTCAR(possibly incomplete)."""
    return sorted(
        d for d in root.rglob("config_*")
        if d.is_dir() and (d / "INCAR").exists()
    )


def convert_one(calc_dir: Path) -> dict:
    """Convert one VASP calc dir. Returns status dict."""
    result = {"dir": calc_dir.name, "status": "unknown"}

    outcar = calc_dir / "OUTCAR"
    incar = calc_dir / "INCAR"
    poscar = calc_dir / "CONTCAR"
    if not poscar.exists() or poscar.stat().st_size == 0:
        poscar = calc_dir / "POSCAR"

    ok, reason = check_outcar_complete(outcar)
    if not ok:
        result["status"] = "skip"
        result["reason"] = reason
        return result

    try:
        atoms = ase_read(poscar)
        n_atoms = len(atoms)
        spin = parse_incar_magmom(incar, n_atoms)
        energy = parse_outcar_energy(outcar)
        forces = parse_outcar_forces(outcar, n_atoms)
        moment = parse_outcar_moments(outcar, n_atoms, fallback_spin=spin)
        torque = parse_outcar_magforces(outcar, n_atoms)

        result["atoms"] = atoms
        result["energy"] = energy
        result["forces"] = forces
        result["spin"] = spin
        result["moment"] = moment
        result["torque"] = torque
        result["n_atoms"] = n_atoms
        result["has_torque"] = torque is not None
        result["status"] = "ok"
    except Exception as e:
        result["status"] = "error"
        result["reason"] = str(e)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert VASP OUTCAR dirs to GPUMD magnetic train.xyz"
    )
    parser.add_argument("root", type=Path, help="Project root containing config_* dirs")
    parser.add_argument("-o", "--output", type=Path, default=Path("train.xyz"))
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check completion status, don't write output"
    )
    args = parser.parse_args()

    configs = discover_configs(args.root)
    if not configs:
        print(f"No config_* directories found under {args.root}")
        return 1

    print(f"Found {len(configs)} config directories under {args.root}")
    print()

    results = []
    for cfg in configs:
        r = convert_one(cfg)
        results.append(r)

    # Summary
    status_counts = defaultdict(int)
    torque_count = 0
    total_atoms = 0
    energies = []
    for r in results:
        status_counts[r["status"]] += 1
        if r["status"] == "ok":
            if r.get("has_torque"):
                torque_count += 1
            total_atoms += r["n_atoms"]
            energies.append(r["energy"])

    print("=" * 60)
    print(f"{'Status':<12} {'Count':>6}")
    print("-" * 60)
    for s in ("ok", "skip", "error"):
        if s in status_counts:
            print(f"{s:<12} {status_counts[s]:>6}")
    print("-" * 60)
    print(f"{'Total':<12} {len(results):>6}")
    print()

    if status_counts["ok"] > 0:
        print(f"Converted:    {status_counts['ok']} / {len(results)}")
        print(f"With torque:  {torque_count} / {status_counts['ok']}")
        print(f"Total atoms:  {total_atoms}")
        print(f"Energy range: {min(energies):.4f} ~ {max(energies):.4f} eV")
        print(f"Energy span:  {max(energies) - min(energies):.4f} eV")
    print()

    # Show skipped/errored configs
    for r in results:
        if r["status"] in ("skip", "error"):
            print(f"  [{r['status']}] {r['dir']}: {r.get('reason', '?')}")

    if args.check_only:
        return 0 if status_counts["skip"] == 0 and status_counts["error"] == 0 else 1

    if status_counts["ok"] == 0:
        print("\nNo completed calculations to convert.")
        return 1

    # Write output
    n_written = 0
    with args.output.open("w") as f:
        for r in results:
            if r["status"] != "ok":
                continue
            write_magnetic_exyz_frame(
                f, r["atoms"], r["energy"], r["forces"],
                r["spin"], r["moment"], torque=r["torque"],
            )
            n_written += 1

    print(f"\nWrote {n_written} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
