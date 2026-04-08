#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch NEP prediction: compute energies, forces, and virials for structures.
"""
import os
import numpy as np
from ase.io import read as ase_read, write as ase_write
from NepTrain import utils
from NepTrain.core.nep.calculator import Nep3Calculator


def run_predict(args):
    """Run NEP prediction on input structures.

    Args:
        args: argparse namespace with fields:
            - input_path (str): path to input structure file (extxyz)
            - nep_path (str): path to nep.txt model file
            - output_path (str): path to output file
            - append (bool): append to existing output file
    """
    input_path = args.input_path
    nep_path = args.nep_path
    output_path = args.output_path
    append = args.append

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not os.path.exists(nep_path):
        raise FileNotFoundError(f"NEP model file not found: {nep_path}")

    utils.print_msg(f"Loading NEP model from {nep_path}")
    calc = Nep3Calculator(model_file=nep_path)

    utils.print_msg(f"Reading structures from {input_path}")
    structures = ase_read(input_path, ":", format="extxyz")
    if not isinstance(structures, list):
        structures = [structures]

    utils.print_msg(f"Predicting properties for {len(structures)} structures...")
    potentials, forces_blocks, virials_blocks = calc.calculate(structures, mean_virial=True)

    utils.print_msg(f"Attaching results to structures...")
    for i, atoms in enumerate(structures):
        atoms.info["nep_total_energy"] = float(potentials[i])
        atoms.arrays["nep_energy"] = np.full(len(atoms), potentials[i] / len(atoms), dtype=np.float32)
        atoms.arrays["nep_forces"] = np.array(forces_blocks[i], dtype=np.float32)
        # virials_blocks[i] is a 9-element list for mean virial
        atoms.info["nep_virial"] = np.array(virials_blocks[i], dtype=np.float32).reshape(3, 3).tolist()

    if append and os.path.exists(output_path):
        existing = ase_read(output_path, ":", format="extxyz")
        structures = list(existing) + structures
        utils.print_msg(f"Appending {len(structures) - len(existing)} new structures to existing {output_path}")

    utils.print_msg(f"Writing {len(structures)} structures to {output_path}")
    ase_write(output_path, structures, format="extxyz")
    utils.print_msg(f"Prediction complete. Results saved to {output_path}")
