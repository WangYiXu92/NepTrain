#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perturbation compatibility checker.

Validates that combinations of perturbation types are mutually compatible
before the perturbation pipeline runs.
"""
from typing import List

# Mutually exclusive groups: enabling more than one raises ValueError
MUTEX_GROUPS = [
    {'sym_strain', 'cell_pert_active'},       # sym_strain and cell_pert are mutually exclusive
    {'amorphous', 'dislocation'},              # amorphous and dislocation are mutually exclusive
    {'amorphous', 'twinning'},                 # amorphous and twinning are mutually exclusive
    {'amorphous', 'grain_boundary'},           # amorphous and grain boundary are mutually exclusive
    {'amorphous', 'stacking_fault'},           # amorphous and stacking fault are mutually exclusive
    {'amorphous', 'surface'},                  # amorphous and surface are mutually exclusive
]

# Dependency groups: if the primary flag is enabled, the check must pass
DEPENDENCY_GROUPS = [
    {
        'primary': 'vacancy',
        'primary_check': lambda kwargs: (
            kwargs.get('vac_elements') is not None and kwargs.get('vac_num', 0) > 0
        ),
        'dependent': 'vac_elements',
        'message': 'vacancy requires vac_elements and vac_num > 0',
    },
    {
        'primary': 'antisite',
        'primary_check': lambda kwargs: kwargs.get('antisite_pairs') is not None,
        'dependent': 'antisite_pairs',
        'message': 'antisite requires antisite_pairs',
    },
]


def validate_compatibility(**kwargs) -> List[str]:
    """Validate compatibility of perturbation type combinations.

    Args:
        **kwargs: All parameters accepted by perturb().

    Returns:
        List[str]: Warning messages (empty list means all compatible).

    Raises:
        ValueError: If mutually exclusive combinations are detected.
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Determine vacancy and cell_pert activity from actual values
    kwargs_compat = dict(kwargs)
    if 'cell_pert_fraction' in kwargs_compat:
        kwargs_compat['cell_pert_active'] = kwargs_compat['cell_pert_fraction'] > 1e-9
    # Map alternate parameter names to mutex group keys
    if kwargs_compat.get('gb', False):
        kwargs_compat['grain_boundary'] = True
    if kwargs_compat.get('vac_elements') is not None or kwargs_compat.get('vac_num', 0) > 0:
        kwargs_compat['vacancy'] = True

    # 1. Check mutually exclusive groups
    for group in MUTEX_GROUPS:
        active = [flag for flag in group if kwargs_compat.get(flag, False)]
        if len(active) > 1:
            errors.append(
                f"Mutually exclusive perturbations: {', '.join(active)}. "
                f"These cannot be used simultaneously."
            )

    # 2. Check dependency groups
    for dep in DEPENDENCY_GROUPS:
        primary_flag = dep['primary']
        if kwargs_compat.get(primary_flag, False):
            if not dep['primary_check'](kwargs_compat):
                errors.append(dep['message'])

    # 3. Soft warnings for questionable combinations
    if kwargs_compat.get('amorphous', False) and kwargs_compat.get('shuffle_elements') is not None:
        warnings.append(
            "amorphous + shuffle: amorphous already randomizes positions, "
            "shuffle may not add meaningful diversity."
        )

    if kwargs_compat.get('amorphous', False) and kwargs_compat.get('rigid', False):
        warnings.append(
            "amorphous + rigid: amorphous breaks molecular structure, "
            "rigid body constraints may not be meaningful."
        )

    # 4. Raise if any hard errors
    if errors:
        raise ValueError(";\n".join(errors))

    return warnings
