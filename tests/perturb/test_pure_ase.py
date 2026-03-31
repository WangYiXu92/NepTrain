
import os
import sys
import numpy as np
from ase.build import bulk
from ase.io import write

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from NepTrain.core.perturb.grain_boundary import generate_grain_boundary
from NepTrain.core.perturb.twinning import generate_twinning
from NepTrain.core.perturb.run import run_perturb, perturb
from NepTrain.core.perturb import csl_core

def test_csl_core():
    print("\n--- Testing CSL Core ---")
    # Test Sigma 5
    axis = [0, 0, 1]
    sigma = 5
    angle = csl_core.get_csl_angle(sigma, axis)
    print(f"Sigma {sigma} Axis {axis} -> Angle {angle}")
    assert angle is not None
    # Sigma 5 can be 36.87 or 53.13
    assert abs(angle - 36.87) < 0.1 or abs(angle - 53.13) < 0.1

def test_grain_boundary_pure():
    print("\n--- Testing Grain Boundary (Pure ASE) ---")
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    
    # Sigma 5 Twist (001)
    gb = generate_grain_boundary(atoms, sigma=5, axis=[0,0,1], min_dist=1.8)
    
    print(f"Generated GB Sigma 5: {len(gb)} atoms")
    assert len(gb) > len(atoms)
    
    # Check annotation
    ann = gb.info.get('perturb_annotation')
    assert ann is not None
    assert ann['type'] == 'grain_boundary'
    # Check in metadata for direct call
    assert ann['metadata']['sigma'] == 5
    
    write("test_pure_gb.xyz", gb)
    print("Saved test_pure_gb.xyz")

def test_twinning_pure():
    print("\n--- Testing Twinning (Pure ASE) ---")
    atoms = bulk('Fe', 'bcc', a=2.87)
    
    # Twin (112)
    twin = generate_twinning(atoms, miller_indices=(1, 1, 2), min_dist=1.8)
    
    print(f"Generated Twin: {len(twin)} atoms")
    assert len(twin) > len(atoms)
    
    # Check annotation
    ann = twin.info.get('perturb_annotation')
    assert ann is not None
    assert ann['type'] == 'twinning'
    
    write("test_pure_twin.xyz", twin)
    print("Saved test_pure_twin.xyz")

def test_run_perturb_pure():
    print("\n--- Testing run_perturb (Pure ASE) ---")
    input_file = "temp_input_pure.xyz"
    output_file = "temp_output_pure.xyz"
    
    atoms = bulk('Cu', 'fcc', a=3.61)
    write(input_file, atoms)
    
    try:
        if os.path.exists(output_file):
            os.remove(output_file)
            
        # Run perturb with GB
        # Ensure no pymatgen is triggered
        gen = run_perturb(
            input_file,
            num=1,
            output_file=output_file,
            gb=True,
            gb_axis='0,0,1',
            gb_angle=36.87, # Sigma 5
            validate_structure=False
        )
        
        generated = list(gen)
        assert len(generated) == 1
        struct = generated[0]
        
        print(f"Generated via run_perturb: {len(struct)} atoms")
        
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    test_csl_core()
    test_grain_boundary_pure()
    test_twinning_pure()
    test_run_perturb_pure()
    print("All pure tests passed!")
