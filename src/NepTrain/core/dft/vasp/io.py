#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 15:53
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import subprocess
from pathlib import Path

import numpy as np
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from NepTrain import Config
from NepTrain.core.perturb.vacancy import _filter_vacancies_for_export


def _as_vector_magnetic_array(values, n_atoms, label):
    """Normalize scalar/vector magnetic data to shape (n_atoms, 3)."""
    if values is None:
        return np.zeros((n_atoms, 3), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros((n_atoms, 3), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != n_atoms:
            raise ValueError(f"{label} must have {n_atoms} entries, got {arr.shape[0]}")
        out = np.zeros((n_atoms, 3), dtype=float)
        out[:, 2] = arr
        return out
    if arr.ndim == 2 and arr.shape == (n_atoms, 3):
        return arr.astype(float, copy=True)
    raise ValueError(f"{label} must have shape ({n_atoms},) or ({n_atoms}, 3), got {arr.shape}")


def _calculator_result(atoms, *keys):
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", {}) if calc is not None else {}
    for key in keys:
        if key in results and results[key] is not None:
            return results[key]
    return None


def _per_atom_calculator_result(atoms, *keys):
    n_atoms = len(atoms)
    for key in keys:
        value = _calculator_result(atoms, key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.shape == (n_atoms,) or arr.shape == (n_atoms, 3):
            return value
    return None


def attach_magnetic_exyz_arrays(atoms, spin_values=None):
    """Attach spin/moment/torque arrays required by magnetic NEP training.

    `spin` is the constrained/input spin direction/moment from the ASE initial
    magnetic moments. `moment` is the DFT output magnetic moment when available
    (`magmoms`, `magmom`, or `magnetic_moments` in calculator results), falling
    back to `spin` for generated/mock structures. `torque` is taken from
    calculator results if present; otherwise it is explicitly written as zeros
    so `train.xyz` has a stable `torque:R:3` schema.
    """
    n_atoms = len(atoms)
    if spin_values is None:
        spin_values = atoms.get_initial_magnetic_moments()
    spin = _as_vector_magnetic_array(spin_values, n_atoms, "spin")
    moment_values = _per_atom_calculator_result(atoms, "magmoms", "magnetic_moments", "moments", "magmom")
    moment = spin.copy() if moment_values is None else _as_vector_magnetic_array(moment_values, n_atoms, "moment")
    torque_values = _per_atom_calculator_result(atoms, "torques", "torque", "magforces", "magnetic_forces")
    torque = np.zeros((n_atoms, 3), dtype=float) if torque_values is None else _as_vector_magnetic_array(torque_values, n_atoms, "torque")

    for label, arr in {"spin": spin, "moment": moment, "torque": torque}.items():
        if arr.shape != (n_atoms, 3) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid magnetic {label} array: shape={arr.shape}, finite={np.isfinite(arr).all()}")
        atoms.arrays[label] = arr
    return atoms

def write_to_xyz(vaspxml_path, save_path, Config_type, append=True, magnetic=False, spin=None):

    atoms_list = []
    atoms = ase_read(vaspxml_path, index=":")
    index = 1
    for atom in atoms:
        if magnetic:
            attach_magnetic_exyz_arrays(atom, spin_values=spin)
        xx, yy, zz, yz, xz, xy = - atom.calc.results['stress'] * atom.get_volume()  # *160.21766
        atom.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])

        atom.calc.results['energy'] = atom.calc.results['free_energy']

        atom.info['Config_type'] = Config_type + str(index)

        atom.info['Weight'] = 1.0

        del atom.calc.results['stress']
        del atom.calc.results['free_energy']
        atoms_list.append(atom)
        index += 1

    # Filter vacancies
    filtered_list = [_filter_vacancies_for_export(atom) for atom in atoms_list]
    ase_write(save_path, filtered_list, format='extxyz', append=append)
    return atoms_list
class VaspInput(Vasp):


    def __init__(self,*args,**kwargs):

        super(VaspInput,self).__init__(*args,**kwargs)
        self.input_params["setups"] = {"base": "recommended"}
        self.input_params["pp"] = ''
        
        os.environ[self.VASP_PP_PATH] = os.path.expanduser(Config.get("environ", "potcar_path"))

    def calculate(self,
                  atoms=None,
                  properties=('energy', ),
                  system_changes=tuple(calculator.all_changes)):
        """Do a VASP calculation in the specified directory.

        This will generate the necessary VASP input files, and then
        execute VASP. After execution, the energy, forces. etc. are read
        from the VASP output files.
        """

                      
        Calculator.calculate(self, atoms, properties, system_changes)
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(self.atoms)

        self.clear_results()

        command = self.make_command(self.command)
        self.write_input(self.atoms, properties, system_changes)

        with self._txt_outstream() as out:
            errorcode, stderr = self._run(command=command,
                                          out=out,
                                          directory=self.directory)

        if errorcode:
            raise calculator.CalculationFailed(
                '{} in {} returned an error: {:d} stderr {}'.format(
                    self.name, Path(self.directory).resolve(), errorcode,
                    stderr))

        # Read results from calculation
        self.update_atoms(atoms)
        self.read_results()
    def _run(self, command=None, out=None, directory=None):
        """Method to explicitly execute VASP"""
        if command is None:
            command = self.command
        if directory is None:
            directory = self.directory

        result = subprocess.run(command,
                                shell=True,
                                cwd=directory,
                                capture_output=True,
                                text=True)
        if out is not None:
            out.write(result.stdout)
            out.write(result.stderr)

        return result.returncode, result.stderr


if __name__ == '__main__':
    vasp=VaspInput()




    atoms=ase_read("./POSCAR",format='vasp')
    vasp.read_incar("./INCAR")
    vasp.calculate(atoms,('energy'))
    print(vasp.results)
    print(vasp.atoms.info)
    print(atoms.calc.results)
