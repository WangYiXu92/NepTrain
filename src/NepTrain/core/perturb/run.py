#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 18:12
# @Author  : 兵
# @email    : 1747193328@qq.com
import os


import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from rich.progress import track

from NepTrain import utils
from NepTrain.core.select import filter_by_bonds


def perturb_position(prim,min_distance):

    atoms = prim.copy()
    # 获取原子位置
    positions = atoms.get_positions()

    # 添加随机微扰
    perturbed_positions = positions + np.random.uniform(
        low=-min_distance,
        high=min_distance,
        size=positions.shape
    )

    # 更新结构的原子位置
    atoms.set_positions(perturbed_positions)
    return atoms

def generate_strained_structure(prim, strain_lim,min_distance):
    strains = np.random.uniform(*strain_lim, (3, ))
    atoms = prim.copy()
    cell_new = prim.cell[:] * (1 + strains)
    atoms.set_cell(cell_new, scale_atoms=True)

    return perturb_position(atoms,min_distance)


def generate_deformed_structure(prim, strain_lim,min_distance):
    R = np.random.uniform(*strain_lim, (3, 3))
    M = np.eye(3) + R
    atoms = prim.copy()
    cell_new = M @ atoms.cell[:]
    atoms.set_cell(cell_new, scale_atoms=True)

    return perturb_position(atoms,min_distance)



@utils.iter_path_to_atoms(["*.vasp","*.xyz"],show_progress=False )
def perturb(atoms:Atoms,cell_pert_fraction=0.04, min_distance=0.1,num=50):
    cell_pert_fraction=[-cell_pert_fraction,cell_pert_fraction]
    structures_rattle=[]
    for i in track(range(num),description=f"Current structure:{atoms.symbols}"):
        if i%2==0:
            structure=generate_deformed_structure(atoms,cell_pert_fraction,min_distance)
            structure.info['Config_type'] = f"perturb {i+1} deformed {cell_pert_fraction}  min_distance {min_distance}"

        else:
            structure=generate_strained_structure(atoms,cell_pert_fraction,min_distance)
            structure.info['Config_type'] = f"perturb {i+1} strained {cell_pert_fraction}  min_distance {min_distance}"

        structures_rattle.append(structure)



    return structures_rattle
def run_perturb(argparse):

    result = perturb(argparse.model_path,
                     cell_pert_fraction=argparse.cell_pert_fraction,
                     min_distance=argparse.min_distance,

                     num=argparse.num,

                     )
    path=os.path.dirname(argparse.out_file_path)

    if path and  not os.path.exists(path):
        os.makedirs(path)

    ase_write(argparse.out_file_path,[atom for _list in result for atom in _list],format="extxyz",append=argparse.append)
