#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example 03: Batch Structure Generation

This example demonstrates how to generate multiple structures in batch mode
for both interstitial and twinning defects.
"""

from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb import BatchGenerator, TwinningBatchGenerator


def example_batch_interstitial():
    """Generate batch of structures with interstitial defects."""
    # Create base structures
    base_structures = [
        bulk('Fe', 'bcc', a=4.0),
        bulk('Cu', 'fcc', a=4.05),
        bulk('Mg', 'hcp', a=3.2, c=5.2),
    ]
    
    # Define perturbation parameters
    params = {
        'type': 'interstitial',
        'element': 'C',
        'n': 1,
        'site_type': 'tetrahedral'
    }
    
    # Create batch generator
    batch = BatchGenerator(base_structures, params)
    
    # Generate all structures
    structures = batch.to_list()
    print(f"Generated {len(structures)} structures with interstitials:")
    for i, structure in enumerate(structures):
        print(f"  Structure {i}: {len(structure)} atoms")
    
    # Or generate to files directly
    output_files = batch.to_file(
        output_dir='batch_interstitial',
        prefix='interstitial',
        fmt='vasp'
    )
    print(f"Saved to {len(output_files)} files in batch_interstitial/")


def example_batch_twinning():
    """Generate batch of structures with twinning defects."""
    # Create base structures
    base_structures = [
        bulk('Fe', 'bcc', a=4.0),
        bulk('Cu', 'fcc', a=4.05),
        bulk('Mg', 'hcp', a=3.2, c=5.2),
    ]
    
    # Create batch twinning generator
    batch = TwinningBatchGenerator(base_structures, n_duplicates=2)
    
    # Generate all structures
    structures = batch.to_list()
    print(f"Generated {len(structures)} twinned structures:")
    for i, structure in enumerate(structures):
        print(f"  Structure {i}: {len(structure)} atoms")
    
    # Or generate to files directly
    output_files = batch.to_file(
        output_dir='batch_twinning',
        prefix='twinning',
        fmt='vasp'
    )
    print(f"Saved to {len(output_files)} files in batch_twinning/")


if __name__ == '__main__':
    print("Example 03: Batch Structure Generation")
    print("=" * 60)
    
    example_batch_interstitial()
    print()
    example_batch_twinning()
    
    print("=" * 60)
    print("Examples completed. Check output files.")
