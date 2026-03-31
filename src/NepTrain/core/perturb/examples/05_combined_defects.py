#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example 05: Combined Multiple Defects

This example demonstrates how to combine multiple types of defects
in a single structure.
"""

from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb import (
    InterstitialGenerator,
    TwinningGenerator,
    generate_interstitial,
    generate_twinning
)


def example_combined_interstitial_twinning():
    """Generate structure with both interstitial and twinning defects."""
    # Start with BCC iron structure
    fe = bulk('Fe', 'bcc', a=4.0)
    
    # Step 1: Add interstitial atoms
    fe_with_c = generate_interstitial(
        fe, 'C', n_interstitials=2, site_type='tetrahedral'
    )
    
    # Step 2: Apply twinning to the interstitial-containing structure
    fe_combined = generate_twinning(fe_with_c, twin_plane='{112}')
    
    # Save the combined structure
    write('fe_combined_interstitial_twin.vasp', fe_combined)
    print(f"Combined interstitial + twinning: {len(fe_combined)} atoms")


def example_multiple_interstitials():
    """Generate structure with multiple different interstitial elements."""
    # Start with BCC iron structure
    fe = bulk('Fe', 'bcc', a=4.0)
    
    # Add carbon interstitials first
    fe_with_c = generate_interstitial(
        fe, 'C', n_interstitials=1, site_type='octahedral'
    )
    
    # Then add nitrogen interstitials
    fe_with_cn = generate_interstitial(
        fe_with_c, 'N', n_interstitials=1, site_type='tetrahedral'
    )
    
    # Save the multi-element interstitial structure
    write('fe_multi_interstitial.vasp', fe_with_cn)
    print(f"Multi-interstitial (C+N): {len(fe_with_cn)} atoms")


def example_defect_sequence():
    """Generate a sequence of increasingly defective structures."""
    # Start with perfect structure
    al = bulk('Al', 'fcc', a=4.05)
    write('al_perfect.vasp', al)
    
    # Add 1 interstitial
    al_1 = generate_interstitial(al, 'H', n_interstitials=1)
    write('al_1_interstitial.vasp', al_1)
    
    # Add 2 interstitials
    al_2 = generate_interstitial(al, 'H', n_interstitials=2)
    write('al_2_interstitials.vasp', al_2)
    
    # Add 3 interstitials
    al_3 = generate_interstitial(al, 'H', n_interstitials=3)
    write('al_3_interstitials.vasp', al_3)
    
    print("Generated sequence of Al structures with increasing interstitials")


if __name__ == '__main__':
    print("Example 05: Combined Multiple Defects")
    print("=" * 60)
    
    example_combined_interstitial_twinning()
    print()
    example_multiple_interstitials()
    print()
    example_defect_sequence()
    
    print("=" * 60)
    print("Examples completed. Check output files.")
