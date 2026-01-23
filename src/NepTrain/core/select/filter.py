#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/5/9 19:41
# @Author  : 兵
# @email    : 1747193328@qq.com
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers,covalent_radii,chemical_symbols
from joblib import Parallel, delayed
from tqdm import tqdm

def calculate_pairwise_distances(lattice_params:np.ndarray, atom_coords:np.ndarray, fractional=True):
    """
    计算晶体中所有原子对之间的距离，考虑周期性边界条件

    参数:
    lattice_params: 晶格参数，3x3 numpy array 表示晶格向量 (a, b, c)
    atom_coords: 原子坐标，Nx3 numpy array
    fractional: 是否为分数坐标 (True) 或笛卡尔坐标 (False)

    返回:
    distances: NxN numpy array，所有原子对之间的距离
    """


    if fractional:
        atom_coords = np.dot(atom_coords, lattice_params)

    diff = atom_coords[np.newaxis, :, :] - atom_coords[:, np.newaxis, :]
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype=np.int8).T.reshape(-1, 3)
    lattice_shifts = np.dot(shifts, lattice_params)
    all_diffs = diff[:, :, np.newaxis, :] + lattice_shifts[np.newaxis, np.newaxis, :, :]
    all_distances = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
    distances = np.min(all_distances, axis=-1)
    np.fill_diagonal(distances, 0)
    return distances


def get_mini_distance_info(atoms:Atoms):
    """
    返回原子对之间的最小距离
    """
    dist_matrix = calculate_pairwise_distances(atoms.cell, atoms.positions, False)
    symbols = atoms.get_chemical_symbols()
    # 提取上三角矩阵（排除对角线）
    i, j = np.triu_indices(len(atoms), k=1)
    # 用字典来存储每种元素对的最小键长
    bond_lengths = {}
    # 遍历所有原子对，计算每一对元素的最小键长
    for idx in range(len(i)):
        atom_i, atom_j = symbols[i[idx]], symbols[j[idx]]
        # if atom_i==atom_j:
        #     continue
        # 获取当前键长
        bond_length = dist_matrix[i[idx], j[idx]]
        # if bond_length>5:
        #     continue
        # 确保元素对按字母顺序排列，避免 Cs-Ag 和 Ag-Cs 视为不同
        element_pair = tuple(sorted([atom_i, atom_j]))
        # 如果该元素对尚未存在于字典中，初始化其最小键长
        if element_pair not in bond_lengths:
            bond_lengths[element_pair] = bond_length
        else:
            # 更新最小键长
            bond_lengths[element_pair] = min(bond_lengths[element_pair], bond_length)

    return bond_lengths

def adjust_reasonable(atoms, coefficient=0.7):
    """
    根据传入系数 对比共价半径和实际键长，
    如果实际键长小于coefficient*共价半径之和，判定为不合理结构 返回False
    否则返回 True
    :param coefficient: 系数
    :return:

    """
    distance_info = get_mini_distance_info(atoms)
    for elems, bond_length in distance_info.items():

        r1=covalent_radii[chemical_symbols.index(elems[0])]
        r2=covalent_radii[chemical_symbols.index(elems[1])]
        if isinstance(r1,float) and isinstance(r2,float):

            # 相邻原子距离小于共价半径之和×系数就选中
            if (r1 + r2) * coefficient > bond_length :
                return False
    return True



def process_atom(atoms, filter_func):
    if adjust_reasonable(atoms, filter_func):
        return atoms, True
    else:
        return atoms, False

def parallel_filter_trajectory(trajectory, filter_func, n_jobs=-1):
    """使用 joblib 并行处理"""
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_atom)(atoms, filter_func)
        for atoms in tqdm(trajectory, desc="Filtering structures")
    )

    trajectory_structures = [atoms for atoms, is_reasonable in results if is_reasonable]
    filter_structures = [atoms for atoms, is_reasonable in results if not is_reasonable]

    return trajectory_structures, filter_structures


