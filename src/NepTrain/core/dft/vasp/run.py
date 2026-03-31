#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 19:03
# @Author  : 兵
# @email    : 1747193328@qq.com

import itertools
import math
import os.path
import logging

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.io.vasp import read_vasp

from NepTrain import utils, Config, module_path
from NepTrain.core.utils import check_env
from NepTrain.core.perturb.magnetic import ensure_magnetic_configuration, get_magmom_config
from NepTrain.core.perturb.vacancy import _filter_vacancies_for_export
from NepTrain.exceptions import CalculationError

from .io import VaspInput,write_to_xyz

logger = logging.getLogger(__name__)


@utils.iter_path_to_atoms(["*.vasp","*.xyz"],show_progress=True,
                 description="VASP calculation progress" )
def calculate_vasp(atoms: Atoms, argparse, index: int = None):
    """
    Calculate VASP single-point energy for atoms.
    
    Args:
        atoms: ASE Atoms object
        argparse: Argument namespace with calculation parameters
        index: Optional index for output directory naming. 
               If None, uses internal counter for thread safety.
               
    Returns:
        ASE Atoms with VASP calculator results, or list for MD
        
    Raises:
        CalculationError: If VASP calculation doesn't converge
    """
    # Use provided index or create thread-safe counter
    atoms_index = index
    if atoms_index is None:
        # Create thread-safe sequence if not provided
        if not hasattr(calculate_vasp, '_counter'):
            calculate_vasp._counter = itertools.count(1)
        atoms_index = next(calculate_vasp._counter)

    if getattr(argparse, 'use_mag', False):
        # Ensure magnetic configuration is present (from file or config)
        atoms = ensure_magnetic_configuration(atoms)
    else:
        atoms.set_initial_magnetic_moments(None)

    vasp = VaspInput()
    if argparse.incar is not None and os.path.exists(argparse.incar):
        vasp.read_incar(argparse.incar)
    else:
        vasp.read_incar(os.path.join(module_path, "core/dft/vasp/INCAR"))

    # Check for magnetic moments and handle collinear/non-collinear
    # Ensure magnetic configuration is set (load from config if missing)
    ensure_magnetic_configuration(atoms)
    magmoms = atoms.get_initial_magnetic_moments()
    
    is_non_collinear = False
    
    # Determine if we should apply constraints (default True, unless disabled by flag or INCAR)
    should_constrain = not getattr(argparse, 'mag_relax', False)

    if np.any(magmoms):
        # Check if 2D magmoms are effectively collinear (only z-component non-zero)
        if magmoms.ndim == 2 and magmoms.shape[1] == 3:
            if np.allclose(magmoms[:, 0], 0) and np.allclose(magmoms[:, 1], 0):
                magmoms = magmoms[:, 2]  # Convert to 1D array of z-components

        if magmoms.ndim == 2 and magmoms.shape[1] == 3:
            is_non_collinear = True
            vasp.set(
                lnoncollinear=True,
                lsorbit=True,
                saxis=(0, 0, 1),
            )
            
            # Additional settings for non-collinear calculations
            vasp.set(
                nelm=300,
                amix=0.2,
                bmix=0.0001,
                amix_mag=0.8,
                bmix_mag=0.0001,
                lasph=True,
                gga_compat=False,
                voskown=1,
                ialgo=58,
                isearch=1,
                nelmdl=10,
                addgrid=True,
                algo="All",
                prec="Accurate",
                magmom=magmoms,
            )
            
            # Apply constraints if not disabled and not already set in INCAR
            if should_constrain and vasp.int_params.get('i_constrained_m') is None:
                # Construct M_CONSTR string from magmoms vectors
                constr = []
                for vec in magmoms:
                    constr.append(f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}")
                m_constr_str = " ".join(constr)
                vasp.set(i_constrained_m=1, m_constr=m_constr_str)

            # Check for rare earth elements
            rare_earth_elements = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
            symbols = set(atoms.get_chemical_symbols())
            # Check if any intersection between symbols and rare_earth_elements
            has_rare_earth = any(s in rare_earth_elements for s in symbols)
            
            if has_rare_earth:
                vasp.set(lmaxmix=6)
            else:
                vasp.set(lmaxmix=4)

        elif vasp.int_params.get('ispin', 1) == 1:
            vasp.set(ispin=2)

            # Additional settings for collinear calculations
            vasp.set(
                nelm=300,
                amix=0.2,
                bmix=0.0001,
                amix_mag=0.8,
                bmix_mag=0.0001,
                lasph=True,
                gga_compat=False,
                voskown=1,
                ialgo=58,
                isearch=1,
                nelmdl=10,
                addgrid=True,
                algo="All",
                prec="Accurate",
                magmom=magmoms,
            )

            # Apply constraints if not disabled and not already set in INCAR
            if should_constrain and vasp.int_params.get('i_constrained_m') is None:
                # Construct M_CONSTR for collinear (0 0 m)
                constr = []
                for m in magmoms:
                    constr.append(f"0 0 {m:.6f}")
                m_constr_str = " ".join(constr)
                vasp.set(i_constrained_m=1, m_constr=m_constr_str)

    directory = os.path.join(argparse.directory, f"{atoms_index}-{atoms.get_chemical_formula()}")

    vasp_path = Config.get('environ', 'vasp_path')
    if is_non_collinear:
        if Config.has_option('environ', 'vasp_ncl_path'):
            vasp_path = Config.get('environ', 'vasp_ncl_path')
        elif 'std' in vasp_path:
            vasp_path = vasp_path.replace('std', 'ncl')
            
    command = f"{Config.get('environ','mpirun_path')} -n {argparse.n_cpu} {vasp_path}"
    if "NEPTRAIN_VASP_COMMAND" in os.environ:
        command = os.environ["NEPTRAIN_VASP_COMMAND"]

    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

    if argparse.kspacing is not None:
        vasp.set(kspacing=argparse.kspacing)
    vasp.set(
            directory = directory,
            command = command,
            kpts = (math.ceil(argparse.ka[0]/a),
                  math.ceil(argparse.ka[1]/b),
                  math.ceil(argparse.ka[2]/c)),
            gamma = argparse.use_gamma,
             )

    if vasp.int_params["ibrion"] == 0:
        # 分子动力学
        vasp.calculate(atoms, ('energy'))

        atoms_list = write_to_xyz(
            os.path.join(directory, "vasprun.xml"),
            os.path.join(directory, f"aimd_{vasp.float_params['tebeg']}k_{vasp.float_params['teend']}k.xyz"),
            "aimd", False
        )
        return atoms_list
    else:
        vasp.calculate(atoms, ('energy'))
        atoms.calc = vasp._xml_calc
        xx, yy, zz, yz, xz, xy = -vasp.results['stress'] * atoms.get_volume()  # *160.21766
        atoms.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
        # 这里没想好怎么设计config的格式化  就先使用原来的
        if "Config_type" not in atoms.info:
            atoms.info['Config_type'] = "NepTrain scf "
        atoms.info['Weight'] = 1.0
        del atoms.calc.results['stress']
        del atoms.calc.results['free_energy']
        if vasp.converged:
            return atoms
        else:
            raise CalculationError(
                f"VASP calculation did not converge for directory: {directory}",
                details={"directory": directory, "formula": atoms.get_chemical_formula()}
            )


def run_vasp(argparse):
    """
    Run VASP calculations for all structures.
    
    Args:
        argparse: Argument namespace with calculation parameters
    """
    check_env()

    result = calculate_vasp(argparse.model_path, argparse)
    path = os.path.dirname(argparse.out_file_path)
    if path and not os.path.exists(path):
        os.makedirs(path)
    
    if len(result) and isinstance(result[0], list):
        result = [atoms for _list in result for atoms in _list]
    
    # Filter vacancies before writing
    if isinstance(result, list):
        result = [_filter_vacancies_for_export(atoms) for atoms in result]
    else:
        result = _filter_vacancies_for_export(result)
        
    ase_write(argparse.out_file_path, result, format="extxyz", append=argparse.append)

    utils.print_success("VASP calculation task completed!")


if __name__ == '__main__':
    calculate_vasp("./")
