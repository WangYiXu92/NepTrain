#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provenance metadata for magnetic training data frames.

Embed generation, stage, temperature, and selection metadata into
extxyz ``info`` fields so every frame carries its origin story.
This is invaluable for debugging bad predictions, filtering datasets,
and writing papers with transparent data provenance.
"""
from __future__ import annotations

from typing import Any


def set_provenance(atoms, /, **kwargs: Any) -> None:
    """Attach provenance metadata to an ASE Atoms object as info fields.

    All keys are prefixed with ``provenance_`` to avoid collisions
    with ASE/VASP info fields (``energy``, ``virial``, ``Config_type``).

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to annotate (mutated in-place).
    **kwargs
        Key-value provenance pairs. Common keys:

        - ``generation`` (int): active-learning generation
        - ``source_stage`` (str): ``perturb``, ``gpumd``, ``select``, ``dft``
        - ``temperature`` (float): MD temperature in K
        - ``md_time_ps`` (float): MD simulation time in ps
        - ``selection_score`` (float): FPS selection score
        - ``selection_reason`` (str): e.g. ``geom_far+spin_far``
        - ``spin_mode`` (str): ``collinear``, ``non_collinear``, ``FM``, ``AFM``
        - ``torque_source`` (str): ``vasp_magnetic_forces``, ``zero``, ``none``
        - ``dft_dir`` (str): original VASP calculation directory
    """
    for key, value in kwargs.items():
        atoms.info[f"provenance_{key}"] = value
