#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2026/01/23
# @Author  : TraeAI
import math
import os
import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.calculators.espresso import Espresso, EspressoProfile

from NepTrain import utils, Config, module_path
from NepTrain.core.perturb.magnetic import ensure_magnetic_configuration

from .io import read_qe_input, get_pp_files

atoms_index = 1

@utils.iter_path_to_atoms(["*.vasp", "*.xyz"], show_progress=True,
                          description="Quantum Espresso calculation progress")
def calculate_qe(atoms: Atoms, argparse):
    global atoms_index
    
    if getattr(argparse, 'use_mag', False):
        # Ensure magnetic configuration is present
        atoms = ensure_magnetic_configuration(atoms)
    else:
        atoms.set_initial_magnetic_moments(None)
    
    # Read input parameters
    if argparse.incar is not None and os.path.exists(argparse.incar):
        input_data, pseudopotentials = read_qe_input(argparse.incar)
    else:
        # Fallback to a default file if it existed, or empty dicts
        # We could look for 'pw.in' in module path if we added one
        default_in = os.path.join(module_path, "core/dft/qe/pw.in")
        if os.path.exists(default_in):
             input_data, pseudopotentials = read_qe_input(default_in)
        else:
             input_data = {'control': {'calculation': 'scf', 'restart_mode': 'from_scratch'},
                           'system': {'ecutwfc': 30, 'occupations': 'smearing', 'smearing': 'gaussian', 'degauss': 0.01},
                           'electrons': {'conv_thr': 1e-6}}
             pseudopotentials = {}

    # Update pseudopotentials from directory if not fully specified
    # We look for PPs in current dir, 'pseudo' dir, or configured path
    found_pps = get_pp_files(".")
    if not found_pps and os.path.exists("pseudo"):
        found_pps = get_pp_files("pseudo")
        
    if not found_pps and Config.has_option('environ', 'qe_pp_path'):
        qe_pp_path = os.path.expanduser(Config.get('environ', 'qe_pp_path'))
        if os.path.exists(qe_pp_path):
            found_pps = get_pp_files(qe_pp_path)
        
    # Merge found PPs with existing ones (existing take precedence if specific)
    for sym in atoms.get_chemical_symbols():
        if sym not in pseudopotentials and sym in found_pps:
            pseudopotentials[sym] = found_pps[sym]
            
    # Handle magnetic moments
    magmoms = atoms.get_initial_magnetic_moments()
    if np.any(magmoms):
        # Setup basic spin polarization if not already set
        if 'system' not in input_data:
            input_data['system'] = {}
            
        if 'nspin' not in input_data['system']:
             input_data['system']['nspin'] = 2
             
        # For non-collinear, user should provide specific input or we need more complex logic
        # Here we stick to basic collinear support mirroring Abacus implementation
        if magmoms.ndim == 2 and magmoms.shape[1] == 3:
             # Basic non-collinear setup
             input_data['system']['noncolin'] = True
             input_data['system']['lspinorb'] = True

    directory = os.path.join(argparse.directory, f"{atoms_index}-{atoms.get_chemical_formula()}")
    atoms_index += 1
    
    # Construct command profile
    # Default to 'pw.x' if not specified in config
    qe_path = Config.get('environ', 'qe_path', fallback='pw.x')
    mpirun_path = Config.get('environ', 'mpirun_path', fallback='mpirun')
    
    if "NEPTRAIN_QE_COMMAND" in os.environ:
        command = os.environ["NEPTRAIN_QE_COMMAND"]
    else:
        command = f"{mpirun_path} -n {argparse.n_cpu} {qe_path}"
        
    profile = EspressoProfile(command, pseudo_dir=directory) # Use calculation directory as pseudo_dir (ASE will copy PPs there?)

    # Setup K-points
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    kpts = (math.ceil(argparse.ka[0]/a),
            math.ceil(argparse.ka[1]/b),
            math.ceil(argparse.ka[2]/c))
            
    if argparse.kspacing is not None:
        kspacing = argparse.kspacing
        kpts = None # ASE prefers one or the other usually
    else:
        kspacing = None

    # Setup Calculator
    # ASE Espresso calculator handles writing input and reading output
    # It requires pseudopotentials to be set
    
    # Check if we have PPs for all species
    missing_pps = [sym for sym in set(atoms.get_chemical_symbols()) if sym not in pseudopotentials]
    if missing_pps:
        # Just warn or fail? ASE will fail if it can't find them.
        # We assume user put them in directory or provided in input
        pass

    calc = Espresso(
        profile=profile,
        directory=directory,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts,
        kspacing=kspacing,
        # tstress=True, # Calculate stress
        # tprnfor=True  # Calculate forces
    )
    
    # Run calculation
    # We want energy, forces, stress
    calc.calculate(atoms, properties=['energy', 'forces', 'stress'], system_changes=['positions', 'cell', 'numbers', 'pbc'])
    
    atoms.calc = calc
    
    # Process results (Virial, etc.)
    # ASE Espresso reads stress in a specific format, usually standard
    if 'stress' in calc.results:
        # Convert to virial? NepTrain uses info['virial']
        # stress in ASE is typically -virial/volume? Or just stress tensor (Voigt usually)
        # calc.results['stress'] is array of 6 elements (xx, yy, zz, yz, xz, xy) or 3x3
        # Abacus run.py does: xx, yy, zz, yz, xz, xy = -calc.results['stress'] * atoms.get_volume()
        # ASE standard: stress is -1/V * dE/depsilon
        # So virial = -stress * Volume
        vol = atoms.get_volume()
        stress = calc.results['stress']
        if stress.shape == (6,):
             xx, yy, zz, yz, xz, xy = -stress * vol
             atoms.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
        elif stress.shape == (3, 3):
             atoms.info['virial'] = -stress * vol

    if "Config_type" not in atoms.info:
        atoms.info['Config_type'] = "NepTrain scf"
    atoms.info['Weight'] = 1.0
    
    # Clean up results for lighter object if needed
    # del atoms.calc.results['stress'] # optional
    
    return atoms

def run_qe(argparse):
    result = calculate_qe(argparse.model_path, argparse)
    path = os.path.dirname(argparse.out_file_path)
    if path and not os.path.exists(path):
        os.makedirs(path)
    if len(result) and isinstance(result[0], list):
        result = [atoms for _list in result for atoms in _list]
    ase_write(argparse.out_file_path, result, format="extxyz", append=argparse.append)
    
    utils.print_success("Quantum Espresso calculation task completed!")
