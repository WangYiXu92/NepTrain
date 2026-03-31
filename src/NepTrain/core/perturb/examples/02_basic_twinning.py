#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example 02: Basic Twinning Defect Generation

This example demonstrates how to generate structures with twinning defects
for common crystal structures (BCC, FCC, HCP).
"""

from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb import TwinningGenerator


def example_bcc_twinning():
    """Generate BCC structure with twinning defects."""
    # Create BCC iron structure
    fe = bulk('Fe', 'bcc', a=4.0)
    
    # Create twinning generator
    gen = TwinningGenerator(fe, twin_plane='{112}')
    
    # Generate structure with twinning
    fe_twin = gen.generate()
    
    # Save to file
    write('fe_bcc_twin.vasp', fe_twin)
    print(f"BCC Fe with twinning: {len(fe_twin)} atoms")


def example_fcc_twinning():
    """Generate FCC structure with twinning defects."""
    # Create FCC aluminum structure
    al = bulk('Al', 'fcc', a=4.05)
    
    # Create twinning generator
    gen = TwinningGenerator(al, twin_plane='{111}')
    
    # Generate structure with twinning
    al_twin = gen.generate()
    
    # Save to file
    write('al_fcc_twin.vasp', al_twin)
    print(f"FCC Al with twinning: {len(al_twin)} atoms")


def example_hcp_twinning():
    """Generate HCP structure with twinning defects."""
    # Create HCP magnesium structure
    mg = bulk('Mg', 'hcp', a=3.2, c=5.2)
    
    # Create twinning generator
    gen = TwinningGenerator(mg, twin_plane='{10-12}')
    
    # Generate structure with twinning
    mg_twin = gen.generate()
    
    # Save to file
    write('mg_hcp_twin.vasp', mg_twin)
    print(f"HCP Mg with twinning: {len(mg_twin)} atoms")


if __name__ == '__main__':
    print("Example 02: Basic Twinning Defect Generation")
    print("=" * 60)
    
    example_bcc_twinning()
    example_fcc_twinning()
    example_hcp_twinning()
    
    print("=" * 60)
    print("Examples completed. Check output files.")
