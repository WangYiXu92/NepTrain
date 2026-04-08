#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPUMD thermodynamic data post-processing.
Parse thermo.out files and compute summary statistics.
"""
import os
import numpy as np
from NepTrain import utils
from .utils import get_dump_interval, calculate_angle, calculate_volume
from .plot import plot_md_thermo


def parse_thermo(filepath, last_n=None):
    """Parse a GPUMD thermo.out file.

    Args:
        filepath: Path to thermo.out file.
        last_n: If given, only return the last N rows (useful for skipping equilibration).

    Returns:
        dict with keys:
            - time (ndarray): time in ps
            - temperature (ndarray): temperature in K
            - kinetic_energy (ndarray)
            - potential_energy (ndarray)
            - pressure_x, pressure_y, pressure_z (ndarray): pressure in GPa
            - volume (ndarray): volume in A^3
            - num_columns (int): original column count
            - lattice_params (dict or None): Lx,Ly,Lz or angles for triclinic
    """
    data = np.loadtxt(filepath)
    if last_n is not None and last_n > 0 and last_n < len(data):
        data = data[-last_n:]

    dump_interval = get_dump_interval()
    time_ps = np.arange(0, len(data) * dump_interval / 1000, dump_interval / 1000)

    temperature = data[:, 0]
    kinetic_energy = data[:, 1]
    potential_energy = data[:, 2]
    pressure_x = data[:, 3]
    pressure_y = data[:, 4]
    pressure_z = data[:, 5]
    num_columns = data.shape[1]

    lattice_params = {}

    if num_columns == 9:
        box_length_x = data[:, 6]
        box_length_y = data[:, 7]
        box_length_z = data[:, 8]
        volume = box_length_x * box_length_y * box_length_z
        lattice_params = {
            "Lx": box_length_x, "Ly": box_length_y, "Lz": box_length_z,
        }
    elif num_columns == 12:
        box_length_x = data[:, 6]
        box_length_y = data[:, 7]
        box_length_z = data[:, 8]
        volume = box_length_x * box_length_y * box_length_z
        lattice_params = {
            "Lx": box_length_x, "Ly": box_length_y, "Lz": box_length_z,
        }
    elif num_columns == 18:
        ax, ay, az = data[:, 9], data[:, 10], data[:, 11]
        bx, by, bz = data[:, 12], data[:, 13], data[:, 14]
        cx, cy, cz = data[:, 15], data[:, 16], data[:, 17]

        a_vectors = np.column_stack((ax, ay, az))
        b_vectors = np.column_stack((bx, by, bz))
        c_vectors = np.column_stack((cx, cy, cz))

        box_length_x = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        box_length_y = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        box_length_z = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)

        box_angle_alpha = calculate_angle(b_vectors, c_vectors)
        box_angle_beta = calculate_angle(c_vectors, a_vectors)
        box_angle_gamma = calculate_angle(a_vectors, b_vectors)

        volume = calculate_volume(a_vectors, b_vectors, c_vectors)
        lattice_params = {
            "Lx": box_length_x, "Ly": box_length_y, "Lz": box_length_z,
            "alpha": box_angle_alpha, "beta": box_angle_beta, "gamma": box_angle_gamma,
        }
    else:
        volume = np.full(len(data), np.nan)
        lattice_params = None

    return {
        "time": time_ps,
        "temperature": temperature,
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "pressure_x": pressure_x,
        "pressure_y": pressure_y,
        "pressure_z": pressure_z,
        "volume": volume,
        "num_columns": num_columns,
        "lattice_params": lattice_params,
    }


def summarize_thermo(data):
    """Print summary statistics of thermodynamic data.

    Args:
        data: dict from parse_thermo()
    """
    T = data["temperature"]
    PE = data["potential_energy"]
    KE = data["kinetic_energy"]
    Px = data["pressure_x"]
    Py = data["pressure_y"]
    Pz = data["pressure_z"]
    P_avg = (Px + Py + Pz) / 3
    V = data["volume"]

    n = len(T)
    print(f"\n{'=' * 60}")
    print(f"  Thermodynamic Summary  ({n} steps)")
    print(f"{'=' * 60}")
    print(f"  {'Property':<25s} {'Mean':>12s} {'Std':>12s} {'Unit':>8s}")
    print(f"  {'-' * 57}")
    print(f"  {'Temperature':<25s} {np.mean(T):>12.2f} {np.std(T):>12.2f} {'K':>8s}")
    print(f"  {'Kinetic Energy':<25s} {np.mean(KE):>12.4f} {np.std(KE):>12.4f} {'eV':>8s}")
    print(f"  {'Potential Energy':<25s} {np.mean(PE):>12.4f} {np.std(PE):>12.4f} {'eV':>8s}")
    print(f"  {'Pressure (x)':<25s} {np.mean(Px):>12.4f} {np.std(Px):>12.4f} {'GPa':>8s}")
    print(f"  {'Pressure (y)':<25s} {np.mean(Py):>12.4f} {np.std(Py):>12.4f} {'GPa':>8s}")
    print(f"  {'Pressure (z)':<25s} {np.mean(Pz):>12.4f} {np.std(Pz):>12.4f} {'GPa':>8s}")
    print(f"  {'Pressure (avg)':<25s} {np.mean(P_avg):>12.4f} {np.std(P_avg):>12.4f} {'GPa':>8s}")

    if not np.all(np.isnan(V)):
        print(f"  {'Volume':<25s} {np.mean(V):>12.2f} {np.std(V):>12.2f} {'A^3':>8s}")

    lp = data.get("lattice_params")
    if lp:
        for key in ["Lx", "Ly", "Lz"]:
            if key in lp:
                arr = lp[key]
                print(f"  {'Lattice ' + key:<25s} {np.mean(arr):>12.4f} {np.std(arr):>12.4f} {'A':>8s}")
        for key in ["alpha", "beta", "gamma"]:
            if key in lp:
                arr = lp[key]
                print(f"  {'Angle ' + key + u' (deg)':<25s} {np.mean(arr):>12.2f} {np.std(arr):>12.2f} {'deg':>8s}")

    print(f"{'=' * 60}\n")


def run_thermo(args):
    """CLI entry point for thermo analysis.

    Args:
        args: argparse namespace with:
            - thermo_file (str): path to thermo.out
            - plot (bool): whether to generate plot
            - last_n (int): only analyze last N steps
    """
    filepath = args.thermo_file
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"thermo.out not found: {filepath}")

    last_n = args.last_n if args.last_n > 0 else None
    if last_n:
        utils.print_msg(f"Analyzing last {last_n} steps of {filepath}")

    data = parse_thermo(filepath, last_n=last_n)
    summarize_thermo(data)

    if args.plot:
        utils.print_msg(f"Generating thermo plot...")
        plot_md_thermo(filepath)
        out_dir = os.path.dirname(filepath)
        utils.print_msg(f"Plot saved to {os.path.join(out_dir, 'thermo.png')}")
