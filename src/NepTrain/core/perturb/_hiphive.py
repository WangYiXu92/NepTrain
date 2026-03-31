#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/27 13:26
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
复制的hiphive\structure_generation\rattle.py
因为这个包依赖太多了 而且目前只用到了这个微扰  所以先只复制过来
"""
import numpy as np
from ase.neighborlist import NeighborList
from scipy.special import erf


def _probability_mc_rattle(d, d_min, width):
    """ Monte Carlo probability function as an error function.

    Parameters
    ----------
    d_min : float
        center value for the error function
    width : float
        width of error function
    """

    return (erf((d-d_min)/width) + 1.0) / 2
def mc_rattle(atoms, rattle_std, d_min, width=0.1, n_iter=10, max_attempts=5000,
              max_disp=2.0, active_atoms=None, nbr_cutoff=None, seed=42):
    """Generate displacements using the Monte Carlo rattle method

    Parameters
    ----------
    atoms : ase.Atoms
        prototype structure
    rattle_std : float
        rattle amplitude (standard deviation in normal distribution)
    d_min : float
        interatomic distance used for computing the probability for each rattle
        move. Center position of the error function
    width : float
        width of the error function
    n_iter : int
        number of Monte Carlo cycle
    max_disp : float
        rattle moves that yields a displacement larger than max_disp will
        always be rejected. This rarley occurs and is more used as a safety net
        for not generating structures where two or more have swapped positions.
    max_attempts : int
        limit for how many attempted rattle moves are allowed a single atom;
        if this limit is reached an `Exception` is raised.
    active_atoms : list
        list of which atomic indices should undergo Monte Carlo rattling
    nbr_cutoff : float
        The cutoff used to construct the neighborlist used for checking
        interatomic distances, defaults to 2 * d_min
    seed : int
        seed for setting up NumPy random state from which random numbers are
        generated

    Returns
    -------
    numpy.ndarray
        atomic displacements (`Nx3`)
    """

    # setup
    rs = np.random.RandomState(seed)

    if nbr_cutoff is None:
        nbr_cutoff = 2 * d_min

    if active_atoms is None:
        active_atoms = range(len(atoms))

    atoms_rattle = atoms.copy()
    reference_positions = atoms_rattle.get_positions()
    nbr_list = NeighborList([nbr_cutoff/2]*len(atoms_rattle), skin=0.0,
                            self_interaction=False, bothways=True)
    nbr_list.update(atoms_rattle)

    # run Monte Carlo
    for _ in range(n_iter):
        for i in active_atoms:
            i_nbrs = np.setdiff1d(nbr_list.get_neighbors(i)[0], [i])
            for n in range(max_attempts):

                # generate displacement
                delta_disp = rs.normal(0.0, rattle_std, 3)
                atoms_rattle.positions[i] += delta_disp
                disp_i = atoms_rattle.positions[i] - reference_positions[i]

                # if total displacement of atom is greater than max_disp, then reject delta_disp
                if np.linalg.norm(disp_i) > max_disp:
                    # revert delta_disp
                    atoms_rattle[i].position -= delta_disp
                    continue

                # compute min distance
                if len(i_nbrs) == 0:
                    min_distance = np.inf
                else:
                    min_distance = np.min(atoms_rattle.get_distances(i, i_nbrs, mic=True))

                # accept or reject delta_disp
                if _probability_mc_rattle(min_distance, d_min, width) > rs.rand():
                    # accept delta_disp
                    break
                else:
                    # revert delta_disp
                    atoms_rattle[i].position -= delta_disp
            else:
                raise Exception(f'Exceeded the maximum number of attempts for atom {i}')
    displacements = atoms_rattle.positions - reference_positions
    return displacements
def get_cell_perturb_matrix(cell_pert_fraction: float):
    if cell_pert_fraction < 0:
        raise RuntimeError("cell_pert_fraction can not be negative")
    e0 = np.random.rand(6)
    e = e0 * 2 * cell_pert_fraction - cell_pert_fraction
    cell_pert_matrix = np.array(
        [
            [1 + e[0], 0.5 * e[5], 0.5 * e[4]],
            [0.5 * e[5], 1 + e[1], 0.5 * e[3]],
            [0.5 * e[4], 0.5 * e[3], 1 + e[2]],
        ]
    )
    return cell_pert_matrix

def generate_mc_rattled_structures(atoms, n_structures,cell_pert_fraction, rattle_std, d_min,
                                   seed=42, **kwargs):
    """Returns list of Monte Carlo rattled configurations.

    Rattling atom `i` is carried out as a Monte Carlo move that is
    accepted with a probability determined from the minimum
    interatomic distance :math:`d_{ij}`.  If :math:`\\min(d_{ij})` is
    smaller than :math:`d_{min}` the move is only accepted with a low
    probability.

    This process is repeated for each atom a number of times meaning
    the magnitude of the final displacements is not *directly*
    connected to `rattle_std`.

    Warning
    -------
    Repeatedly calling this function *without* providing different
    seeds will yield identical or correlated results. To avoid this
    behavior it is recommended to specify a different seed for each
    call to this function.

    Notes
    ------
    The procedure implemented here might not generate a symmetric
    distribution for the displacements `kwargs` will be forwarded to
    `mc_rattle` (see user guide for a detailed explanation)

    The displacements generated will roughly be n_iter**0.5 * rattle_std
    for small values of n_iter

    Parameters
    ----------
    atoms : ase.Atoms
        prototype structure
    n_structures : int
        number of structures to generate
    rattle_std : float
        rattle amplitude (standard deviation in normal distribution);
        note this value is not connected to the final
        average displacement for the structures
    d_min : float
        interatomic distance used for computing the probability for each rattle
        move
    seed : int
        seed for setting up NumPy random state from which random numbers are
        generated
    n_iter : int
        number of Monte Carlo cycles (iterations), larger number of iterations will
        generate larger displacements (defaults to 10)

    Returns
    -------
    list of ase.Atoms
        generated structures
    """
    rs = np.random.RandomState(seed)
    atoms_list = []

    for _ in range(n_structures):
        atoms_tmp = atoms.copy()
        seed = rs.randint(1, 1000000000)


        cell_perturb_matrix = get_cell_perturb_matrix(cell_pert_fraction)

        cell = np.matmul(
            atoms_tmp.cell.array, cell_perturb_matrix
        )
        atoms_tmp.set_cell(cell,True)
        displacements = mc_rattle(atoms_tmp, rattle_std, d_min, seed=seed, **kwargs)
        atoms_tmp.positions += displacements
        atoms_list.append(atoms_tmp)
    return atoms_list

