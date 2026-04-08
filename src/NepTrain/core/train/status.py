#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training progress status checker.
Reports current generation, job status, NEP loss metrics.
"""
import os
import glob
import numpy as np
from ruamel.yaml import YAML
from NepTrain import utils


def _read_restart_yaml(work_path):
    """Read restart.yaml from the work directory.

    Args:
        work_path: Path to the training work directory.

    Returns:
        dict with training configuration, or None if not found.
    """
    restart_file = os.path.join(work_path, "restart.yaml")
    if not os.path.exists(restart_file):
        return None
    with open(restart_file, "r", encoding="utf-8") as f:
        config = YAML().load(f)
    return config if isinstance(config, dict) else None


def _read_nep_loss(loss_out_path):
    """Read the last line of a NEP loss.out file.

    Args:
        loss_out_path: Path to loss.out

    Returns:
        dict with epoch, rmse_energy (meV/atom), rmse_force (meV/A), or None
    """
    if not os.path.exists(loss_out_path):
        return None
    try:
        data = np.loadtxt(loss_out_path)
        last_line = data[-1]
    except Exception:
        return None

    result = {
        "epoch": int(last_line[0]),
        "total_loss": float(last_line[1]),
        "rmse_energy": float(last_line[4]),
        "rmse_force": float(last_line[5]),
    }
    if len(last_line) >= 10:
        result["rmse_energy_test"] = float(last_line[7])
        result["rmse_force_test"] = float(last_line[8])
    return result


def _find_loss_out(work_path, generation):
    """Find the loss.out file for the current or latest generation.

    Args:
        work_path: Training work directory.
        generation: Current generation number.

    Returns:
        Path to loss.out or None.
    """
    # Try current generation first
    nep_dir = os.path.join(work_path, f"Generation-{generation}", "nep")
    loss_path = os.path.join(nep_dir, "loss.out")
    if os.path.exists(loss_path):
        return loss_path

    # Fallback: scan all generation directories
    gen_dirs = sorted(glob.glob(os.path.join(work_path, "Generation-*")), key=os.path.getmtime)
    for d in reversed(gen_dirs):
        candidate = os.path.join(d, "nep", "loss.out")
        if os.path.exists(candidate):
            return candidate
    return None


def check_status(work_path):
    """Check and report training progress.

    Args:
        work_path: Path to the training work directory.

    Returns:
        dict with status information, or None if no training found.
    """
    if not os.path.isdir(work_path):
        print(f"Work directory not found: {work_path}")
        return None

    config = _read_restart_yaml(work_path)
    if config is None:
        print(f"No restart.yaml found in {work_path}")
        return None

    generation = config.get("generation", 0)
    current_job = config.get("current_job", "unknown")
    step_times = config.get("gpumd", {}).get("step_times", [])
    max_gen = len(step_times)
    dft_software = config.get("dft", {}).get("software", "vasp")
    is_restart = config.get("restart", False)

    # Find loss.out
    loss_path = _find_loss_out(work_path, generation)
    loss_info = _read_nep_loss(loss_path) if loss_path else None

    # Build status
    status = {
        "work_path": work_path,
        "generation": generation,
        "current_job": current_job,
        "max_generation": max_gen,
        "dft_software": dft_software,
        "is_restart": is_restart,
        "loss": loss_info,
        "loss_path": loss_path,
    }

    _print_status(status)
    return status


def _print_status(status):
    """Print a human-readable status report."""
    gen = status["generation"]
    max_gen = status["max_generation"]
    job = status["current_job"]
    pct = (gen / max_gen * 100) if max_gen > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  NepTrain Training Status")
    print(f"{'=' * 50}")
    print(f"  Work directory : {status['work_path']}")
    print(f"  DFT software   : {status['dft_software']}")
    print(f"  Restart mode   : {'Yes' if status['is_restart'] else 'No'}")
    print(f"  Generation     : {gen} / {max_gen} ({pct:.0f}%)")
    print(f"  Current job    : {job}")
    print(f"  Progress bar   : [{'#' * int(pct / 5)}{'.' * (20 - int(pct / 5))}]")

    loss = status.get("loss")
    if loss:
        print(f"\n  --- NEP Training Loss (latest) ---")
        print(f"  Epoch          : {loss['epoch']}")
        print(f"  Energy RMSE    : {loss['rmse_energy']:.3f} meV/atom")
        print(f"  Force RMSE     : {loss['rmse_force']:.3f} meV/A")
        if "rmse_energy_test" in loss:
            print(f"  Energy RMSE(test): {loss['rmse_energy_test']:.3f} meV/atom")
            print(f"  Force RMSE(test) : {loss['rmse_force_test']:.3f} meV/A")
    else:
        print(f"\n  (No loss.out found yet)")

    print(f"{'=' * 50}\n")


def run_status(args):
    """CLI entry point for status check.

    Args:
        args: argparse namespace with work_path (str)
    """
    check_status(args.work_path)
