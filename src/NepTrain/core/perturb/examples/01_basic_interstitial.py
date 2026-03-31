#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example 01: Basic Interstitial Defect Generation

This example demonstrates how to generate structures with interstitial defects
for common crystal structures (BCC, FCC, HCP).
"""

from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb import InterstitialGenerator


def example_bcc_interstitial():
    """Generate BCC structure with interstitial atoms."""
    # Create BCC iron structure
    fe = bulk('Fe', 'bcc', a=4.0)
    
    # Create interstitial generator
    gen = InterstitialGenerator(fe, 'C')
    
    # Generate structure with 2 interstitial atoms in tetrahedral sites
    fe_with_c = gen.generate(n_interstitials=2, site_type='tetrahedral')
    
    # Save to file
    write('fe_bcc_c_interstitial.vasp', fe_with_c)
    print(f"BCC Fe with interstitial C: {len(fe_with_c)} atoms")


def example_fcc_interstitial():
    """Generate FCC structure with interstitial atoms."""
    # Create FCC aluminum structure
    al = bulk('Al', 'fcc', a=4.05)
    
    # Create interstitial generator
    gen = InterstitialGenerator(al, 'H')
    
    # Generate structure with interstitial atoms in octahedral sites
    al_with_h = gen.generate(n_interstitials=1, site_type='octahedral')
    
    # Save to file
    write('al_fcc_h_interstitial.vasp', al_with_h)
    print(f"FCC Al with interstitial H: {len(al_with_h)} atoms")


def example_hcp_interstitial():
    """Generate HCP structure with interstitial atoms."""
    # Create HCP magnesium structure
    mg = bulk('Mg', 'hcp', a=3.2, c=5.2)
    
    # Create interstitial generator
    gen = InterstitialGenerator(mg, 'C')
    
    # Generate structure with interstitial atoms
    mg_with_c = gen.generate(n_interstitials=2, site_type='tetrahedral')
    
    # Save to file
    write('mg_hcp_c_interstitial.vasp', mg_with_c)
    print(f"HCP Mg with interstitial C: {len(mg_with_c)} atoms")


if __name__ == '__main__':
    print("Example 01: Basic Interstitial Defect Generation")
    print("=" * 60)
    
    example_bcc_interstitial()
    example_fcc_interstitial()
    example_hcp_interstitial()
    
    print("=" * 60)
    print("Examples completed. Check output files.")
