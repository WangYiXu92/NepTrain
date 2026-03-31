
import sys
import os
import numpy as np
from ase.build import bulk
from ase.io import write

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from NepTrain.core.perturb.run import perturb

def test_full_space_sampling():
    print("Testing Full Space Sampling...")
    
    atoms = bulk('Cu', 'fcc', a=3.61)
    
    # 1. Random Axis + CSL Angle
    print("\n--- Generating 5 GBs with Random Axis & CSL Angle ---")
    gen = perturb(
        atoms,
        num=5,
        gb=True,
        gb_axis='random',
        gb_angle='csl',
        gb_overlap_dist=1.2,
        validate_structure=False,
        sampler='random' # Use random sampler first
    )
    
    generated = list(gen)
    assert len(generated) == 5
    
    for i, struct in enumerate(generated):
        ann = struct.info['perturb_annotation']
        meta = ann['metadata']
        print(f"GB {i}: Axis={meta.get('axis')}, Sigma={meta.get('sigma')}, Angle={meta.get('angle'):.2f}")
        
        # Verify axis is integer-like
        axis = meta.get('axis')
        assert isinstance(axis, (list, tuple, np.ndarray))
        
    # 2. Sobol Sampler
    print("\n--- Generating 5 GBs with Sobol Sampler ---")
    gen_sobol = perturb(
        atoms,
        num=5,
        gb=True,
        gb_axis='random',
        gb_angle='csl',
        gb_overlap_dist=1.2,
        validate_structure=False,
        sampler='sobol'
    )
    
    generated_sobol = list(gen_sobol)
    assert len(generated_sobol) == 5
    
    for i, struct in enumerate(generated_sobol):
        ann = struct.info['perturb_annotation']
        meta = ann['metadata']
        print(f"Sobol GB {i}: Axis={meta.get('axis')}, Sigma={meta.get('sigma')}, Angle={meta.get('angle'):.2f}")

if __name__ == "__main__":
    test_full_space_sampling()
