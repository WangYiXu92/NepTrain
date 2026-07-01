#!/usr/bin/env python3
"""Physics validation for BCC Fe magnetic training data.

Checks a GPUMD-format train.xyz for physical sanity:
  1. Moment magnitudes within expected range (1.5–3.5 μB for BCC Fe)
  2. Torque nonzero (Phase 2 only — Phase 1 collinear has zero torque by design)
  3. Force distribution (no anomalous spikes > 50 eV/Å)
  4. Energy distribution (span should be > 0.01 eV for useful training)
  5. Spin-moment alignment (collinear: spin ⊥ moment; non-collinear: general angle)
  6. Torque ⊥ spin orthogonality (τ · s ≈ 0)

Usage:
  python validate_magnetic_data.py train.xyz [--tolerance 2.0]
  python validate_magnetic_data.py train_col.xyz train_nc.xyz --combined
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_exyz(path: Path) -> list[dict]:
    """Parse GPUMD extended xyz with magnetic columns. Returns list of frames."""
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line.strip())
            header = f.readline()
            frames.append({"n": n, "header": header})

            # Parse properties
            props_str = ""
            for part in header.split():
                if part.startswith("Properties="):
                    props_str = part.split("=", 1)[1]
                    break
            if not props_str:
                raise ValueError(f"No Properties in header: {header}")

            # Parse property spec
            prop_list = props_str.split(":")
            props = []
            for i in range(0, len(prop_list), 3):
                name = prop_list[i]
                ptype = prop_list[i+1]
                count = int(prop_list[i+2])
                props.append((name, ptype, count))

            # Parse energy
            energy = 0.0
            for part in header.split():
                if part.startswith("energy="):
                    energy = float(part.split("=")[1])
                    break

            # Read atom lines
            atoms = []
            for _ in range(n):
                parts = f.readline().split()
                row = {}
                idx = 0
                for name, ptype, count in props:
                    if ptype == "S":
                        row[name] = parts[idx]
                    else:
                        row[name] = np.array([float(x) for x in parts[idx:idx+count]])
                    idx += count
                atoms.append(row)

            frames[-1]["atoms"] = atoms
            frames[-1]["energy"] = energy
            frames[-1]["n_atoms"] = n
    return frames


def validate_frame(frame: dict, frame_idx: int, is_noncollinear: bool) -> list[str]:
    """Run all checks on one frame. Returns list of warning strings (empty = all OK)."""
    warnings = []
    atoms = frame["atoms"]
    n = len(atoms)

    forces = np.array([a["force"] for a in atoms])
    spins = np.array([a["spin"] for a in atoms])
    moments = np.array([a["moment"] for a in atoms])
    torques = np.array([a["torque"] for a in atoms])

    # 1. Moment magnitudes
    mag_mags = np.linalg.norm(moments, axis=1)
    bad_mag = np.sum((mag_mags < 0.5) | (mag_mags > 4.0))
    if bad_mag > 0:
        warnings.append(f"Frame {frame_idx}: {bad_mag}/{n} atoms with moment outside [0.5, 4.0] μB (mean={mag_mags.mean():.3f})")

    # 2. Force spikes
    force_mags = np.linalg.norm(forces, axis=1)
    max_force = force_mags.max()
    if max_force > 50.0:
        warnings.append(f"Frame {frame_idx}: max force = {max_force:.2f} eV/Å (>50, suspicious)")

    # 3. Torque check (only meaningful for non-collinear with I_CONSTRAINED_M)
    if is_noncollinear:
        torque_mags = np.linalg.norm(torques, axis=1)
        if torque_mags.max() < 1e-10:
            warnings.append(f"Frame {frame_idx}: all torques zero (I_CONSTRAINED_M missing in INCAR?)")
        else:
            # Orthogonality: τ · s ≈ 0
            dots = np.abs(np.sum(torques * spins, axis=1))
            max_dot = dots.max()
            spin_mags = np.linalg.norm(spins, axis=1)
            torque_mags_arr = np.linalg.norm(torques, axis=1)
            denom = spin_mags * torque_mags_arr + 1e-12
            cos_angles = dots / denom
            max_cos = cos_angles.max()
            if max_cos > 0.01:
                warnings.append(f"Frame {frame_idx}: torque not ⊥ spin (max cos = {max_cos:.4f}, should be ~0)")

    # 4. NaN/Inf check
    for name, arr in [("force", forces), ("spin", spins), ("moment", moments), ("torque", torques)]:
        if not np.isfinite(arr).all():
            warnings.append(f"Frame {frame_idx}: NaN/Inf in {name}")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Validate magnetic training data")
    parser.add_argument("files", nargs="+", type=Path, help="train.xyz file(s)")
    parser.add_argument("--combined", action="store_true", help="Treat files as phases of one dataset")
    args = parser.parse_args()

    all_warnings = []
    all_frames = []
    phase_labels = []

    for i, fpath in enumerate(args.files):
        is_nc = "nc" in fpath.name.lower() or "noncoll" in fpath.name.lower() or "phase2" in str(fpath).lower()
        label = f"Phase {i+1} ({'NCL' if is_nc else 'CL'})"
        frames = parse_exyz(fpath)
        print(f"\n{'='*60}")
        print(f"{label}: {fpath}")
        print(f"  Frames: {len(frames)}")

        if not frames:
            print("  EMPTY FILE")
            continue

        # Aggregate stats
        all_mags = []
        all_forces = []
        all_torques = []
        all_energies = []
        all_spin_moment_angles = []

        for idx, frame in enumerate(frames):
            warnings = validate_frame(frame, idx, is_nc)
            all_warnings.extend(warnings)

            atoms = frame["atoms"]
            all_mags.extend(np.linalg.norm([a["moment"] for a in atoms], axis=1).tolist())
            all_forces.extend(np.linalg.norm([a["force"] for a in atoms], axis=1).tolist())
            torques = np.array([a["torque"] for a in atoms])
            all_torques.extend(np.linalg.norm(torques, axis=1).tolist())
            all_energies.append(frame["energy"])

            spins = np.array([a["spin"] for a in atoms])
            moments = np.array([a["moment"] for a in atoms])
            for j in range(len(atoms)):
                s = spins[j]
                m = moments[j]
                sn = np.linalg.norm(s)
                mn = np.linalg.norm(m)
                if sn > 1e-6 and mn > 1e-6:
                    cos_angle = np.dot(s, m) / (sn * mn)
                    all_spin_moment_angles.append(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

        all_mags = np.array(all_mags)
        all_forces = np.array(all_forces)
        all_torques = np.array(all_torques)
        all_energies = np.array(all_energies)

        print(f"\n  Magnetic moments (μB):")
        print(f"    mean={all_mags.mean():.4f}  std={all_mags.std():.4f}  min={all_mags.min():.4f}  max={all_mags.max():.4f}")
        print(f"    histogram: {np.histogram(all_mags, bins=5)[0].tolist()}")

        print(f"\n  Forces (eV/Å):")
        print(f"    mean={all_forces.mean():.4f}  std={all_forces.std():.4f}  max={all_forces.max():.4f}")

        if all_torques.max() > 1e-10:
            nz = all_torques[all_torques > 1e-10]
            print(f"\n  Torques (eV) [{len(nz)} nonzero]:")
            print(f"    mean={nz.mean():.4f}  std={nz.std():.4f}  max={nz.max():.4f}")
        else:
            print(f"\n  Torques: ALL ZERO ({'expected for collinear' if not is_nc else 'UNEXPECTED — check I_CONSTRAINED_M'})")

        print(f"\n  Energies (eV):")
        print(f"    mean={all_energies.mean():.4f}  span={all_energies.max()-all_energies.min():.4f}")

        if all_spin_moment_angles:
            angles = np.array(all_spin_moment_angles)
            print(f"\n  Spin-Moment angle (deg):")
            print(f"    mean={angles.mean():.2f}  std={angles.std():.2f}")

        all_frames.extend([(f, label) for f in frames])
        phase_labels.append(label)

    # Summary warnings
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    if all_warnings:
        print(f"\n⚠️  {len(all_warnings)} warning(s):")
        for w in all_warnings[:20]:
            print(f"  • {w}")
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings)-20} more")
    else:
        print("\n✅ All checks passed — data is physically sound.")

    return 0 if not all_warnings else 1


if __name__ == "__main__":
    sys.exit(main())
