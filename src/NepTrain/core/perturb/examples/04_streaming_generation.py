#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example 04: Streaming Large-Scale Generation

This example demonstrates how to use streaming generation for large systems
to reduce memory usage.
"""

from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb import InterstitialGenerator, TwinningGenerator


def example_streaming_interstitial():
    """Generate many interstitial structures using streaming."""
    # Create base structure
    fe = bulk('Fe', 'bcc', a=4.0)
    
    # Create interstitial generator
    gen = InterstitialGenerator(fe, 'C')
    
    # Generate 10 structures using streaming (no memory accumulation)
    output_files = gen.generate_to_file(
        n_structures=10,
        n_interstitials=2,
        site_type='tetrahedral',
        output_dir='stream_interstitial',
        prefix='fe_interstitial',
        fmt='vasp'
    )
    
    print(f"Generated {len(output_files)} interstitial structures:")
    for f in output_files:
        print(f"  {f}")


def example_streaming_twinning():
    """Generate many twinned structures using streaming."""
    # Create base structure
    al = bulk('Al', 'fcc', a=4.05)
    
    # Create twinning generator
    gen = TwinningGenerator(al)
    
    # Generate 5 structures using streaming
    output_files = gen.generate_to_file(
        n_structures=5,
        n_duplicates=2,
        output_dir='stream_twinning',
        prefix='al_twinning',
        fmt='vasp'
    )
    
    print(f"Generated {len(output_files)} twinned structures:")
    for f in output_files:
        print(f"  {f}")


if __name__ == '__main__':
    print("Example 04: Streaming Large-Scale Generation")
    print("=" * 60)
    
    example_streaming_interstitial()
    print()
    example_streaming_twinning()
    
    print("=" * 60)
    print("Examples completed. Check output files.")
