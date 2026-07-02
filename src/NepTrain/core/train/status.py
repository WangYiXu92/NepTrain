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
from .artifacts import load_stage_reports


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
        data = np.loadtxt(loss_out_path, ndmin=2)
        last_line = data[-1]
    except Exception:
        return None

    result = {
        "epoch": int(last_line[0]),
        "total_loss": float(last_line[1]),
        "rmse_energy": float(last_line[4]),
        "rmse_force": float(last_line[5]),
        "n_columns": len(last_line),
        "has_magnetic_loss": len(last_line) >= 16,
    }
    if len(last_line) >= 10:
        result["rmse_energy_test"] = float(last_line[7])
        result["rmse_force_test"] = float(last_line[8])
    if len(last_line) >= 16:
        result["rmse_moment_x"] = float(last_line[10])
        result["rmse_moment_y"] = float(last_line[11])
        result["rmse_moment_z"] = float(last_line[12])
        result["rmse_torque_x"] = float(last_line[13])
        result["rmse_torque_y"] = float(last_line[14])
        result["rmse_torque_z"] = float(last_line[15])
    return result


def parse_prediction_metrics(pred_dir: str) -> dict:
    """Parse NEP prediction outputs for per-component parity metrics.

    Reads ``energy.out``, ``force.out``, and optional ``spin.out`` from the
    prediction directory. Each file has the format::

        pred_1 pred_2 ... pred_N dft_1 dft_2 ... dft_N

    Returns a dict with per-component RMSE values.
    """
    import os

    pred = os.path.expanduser(pred_dir)
    metrics: dict = {}

    # force.out: fx_p fy_p fz_p fx_dft fy_dft fz_dft
    force_file = os.path.join(pred, "force.out")
    if os.path.exists(force_file):
        try:
            data = np.loadtxt(force_file, ndmin=2)
            n_cols = data.shape[1] // 2
            labels = ["x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy"]
            for i in range(min(n_cols, len(labels))):
                diff = data[:, i] - data[:, i + n_cols]
                metrics[f"force_{labels[i]}_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        except Exception:
            pass

    # energy.out: e_p e_dft
    energy_file = os.path.join(pred, "energy.out")
    if os.path.exists(energy_file):
        try:
            data = np.loadtxt(energy_file, ndmin=2)
            if data.shape[1] >= 2:
                diff = data[:, 0] - data[:, 1]
                metrics["energy_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        except Exception:
            pass

    # spin.out: Sx_p Sy_p Sz_p Sx_dft Sy_dft Sz_dft
    spin_file = os.path.join(pred, "spin.out")
    if os.path.exists(spin_file):
        try:
            data = np.loadtxt(spin_file, ndmin=2)
            n_cols = data.shape[1] // 2
            for i, lbl in enumerate(["x", "y", "z"]):
                if i < n_cols:
                    diff = data[:, i] - data[:, i + n_cols]
                    metrics[f"moment_{lbl}_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        except Exception:
            pass

    return metrics


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

    stage_reports = load_stage_reports(work_path, generation=generation)

    # Prediction metrics (from last pred step)
    pred_dir = os.path.join(work_path, f"Generation-{generation}", "pred")
    pred_metrics = parse_prediction_metrics(pred_dir)

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
        "pred_metrics": pred_metrics,
        "stage_reports": stage_reports,
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
        print(f"  Epoch          : {loss['epoch']} (n_columns={loss.get('n_columns', '?')})")
        print(f"  Energy RMSE    : {loss['rmse_energy']:.3f} meV/atom")
        print(f"  Force RMSE     : {loss['rmse_force']:.3f} meV/A")
        if "rmse_energy_test" in loss:
            print(f"  Energy RMSE(test): {loss['rmse_energy_test']:.3f} meV/atom")
            print(f"  Force RMSE(test) : {loss['rmse_force_test']:.3f} meV/A")
        if loss.get("has_magnetic_loss"):
            print(f"  --- Magnetic ---")
            print(f"  Mx/Mz RMSE     : {loss['rmse_moment_x']:.4f} / {loss['rmse_moment_z']:.4f} μB")
            print(f"  My RMSE        : {loss['rmse_moment_y']:.4f} μB")
            print(f"  Tx/Ty/Tz RMSE  : {loss['rmse_torque_x']:.4f} / {loss['rmse_torque_y']:.4f} / {loss['rmse_torque_z']:.4f}")
    else:
        print(f"\n  (No loss.out found yet)")

    pred = status.get("pred_metrics") or {}
    if pred:
        print(f"\n  --- Prediction Parity (per-component) ---")
        energy = pred.get("energy_rmse")
        if energy is not None:
            print(f"  Energy RMSE    : {energy:.3f} eV")
        fx = pred.get("force_x_rmse")
        if fx is not None:
            print(f"  Force X/Y/Z    : {fx:.4f} / {pred.get('force_y_rmse', 0):.4f} / {pred.get('force_z_rmse', 0):.4f} eV/A")
        mx = pred.get("moment_x_rmse")
        if mx is not None:
            print(f"  Moment X/Y/Z   : {mx:.4f} / {pred.get('moment_y_rmse', 0):.4f} / {pred.get('moment_z_rmse', 0):.4f} μB")


    stage_reports = status.get("stage_reports") or {}
    if stage_reports:
        print(f"\n  --- Workflow Stage Reports ---")
        for stage, report in sorted(stage_reports.items()):
            status_text = report.get("status", "unknown")
            valid = "valid" if report.get("valid_artifacts", False) else "check"
            summary = report.get("summary") or {}
            if not isinstance(summary, dict):
                summary = {"value": summary}
            summary_text = ", ".join(f"{k}={v}" for k, v in summary.items())
            suffix = f" ({summary_text})" if summary_text else ""
            print(f"  {stage:<8}: {status_text:<10} artifacts={valid}{suffix}")

    print(f"{'=' * 50}\n")


def run_status(args):
    """CLI entry point for status check.

    Args:
        args: argparse namespace with work_path (str)
    """
    check_status(args.work_path)
