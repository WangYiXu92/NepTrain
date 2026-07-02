
import sys
import os
import numpy as np
from ase.build import bulk
from ase.io import write

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from NepTrain.core.perturb.twinning import generate_twinning

def test_mirror_twin():
    print("Testing Mirror Twinning...")
    
    # 1. FCC Al (111) Twin
    # Stacking should be ABC|BAC
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    
    twin = generate_twinning(
        atoms, 
        miller_indices=(1, 1, 1), 
        min_dist=1.8
    )
    
    print(f"Generated Al (111) Twin: {len(twin)} atoms")
    write("test_mirror_twin_Al.xyz", twin)
    
    assert twin is not None
    assert len(twin) > 0
    
    # 2. BCC Fe (112) Twin
    atoms_bcc = bulk('Fe', 'bcc', a=2.87) # Primitive cell
    
    twin_bcc = generate_twinning(
        atoms_bcc,
        miller_indices=(1, 1, 2),
        min_dist=1.8
    )
    
    print(f"Generated Fe (112) Twin: {len(twin_bcc)} atoms")
    write("test_mirror_twin_Fe.xyz", twin_bcc)

if __name__ == "__main__":
    test_mirror_twin()
