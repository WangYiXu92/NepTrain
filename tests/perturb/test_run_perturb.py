
import os
import shutil
import numpy as np
from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb.run import run_perturb

def test_run_perturb_grain_boundary_csl():
    """Test run_perturb with CSL grain boundary generation."""
    input_file = "temp_input_gb.xyz"
    output_file = "temp_output_gb.xyz"
    
    # Create bulk Cu
    atoms = bulk('Cu', 'fcc', a=3.61)
    write(input_file, atoms)
    
    try:
        if os.path.exists(output_file):
            os.remove(output_file)
            
        # Run perturb with GB CSL
        # We use num=1 to generate one structure
        # We expect it to find a CSL since default axis 0,0,1 has many CSLs
        gen = run_perturb(
            input_file,
            num=1,
            output_file=output_file,
            gb=True,
            gb_axis='0,0,1',
            gb_angle='csl',
            gb_overlap_dist=1.2,
            min_distance=1.5, # Validation min distance
            validate_structure=False # Skip validation for speed/robustness in this test
        )
        
        generated = list(gen)
        assert len(generated) == 1
        struct = generated[0]
        
        # Check annotation
        assert 'perturb_annotation' in struct.info
        ann = struct.info['perturb_annotation']
        print(f"GB Annotation: {ann}")
        
        assert ann['type'] == 'grain_boundary'
        assert 'sigma' in ann
        assert ann['sigma'] is not None
        # Sigma should be an integer (e.g. 5, 13, etc.)
        assert isinstance(ann['sigma'], (int, np.integer))
        
        # Check translation support (default random)
        # The code generates a random translation if not provided
        # But we can't easily verify random values, just that it ran without error
        
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

def test_run_perturb_twinning():
    """Test run_perturb with Twinning generation."""
    input_file = "temp_input_twin.xyz"
    output_file = "temp_output_twin.xyz"
    
    # Create bulk Cu
    atoms = bulk('Cu', 'fcc', a=3.61)
    write(input_file, atoms)
    
    try:
        if os.path.exists(output_file):
            os.remove(output_file)
            
        # Run perturb with Twinning
        gen = run_perturb(
            input_file,
            num=1,
            output_file=output_file,
            twinning=True,
            twinning_indices='1,1,1', # Simple twin
            min_distance=1.5,
            validate_structure=False
        )
        
        generated = list(gen)
        assert len(generated) == 1
        struct = generated[0]
        
        # Check annotation
        assert 'perturb_annotation' in struct.info
        ann = struct.info['perturb_annotation']
        print(f"Twin Annotation: {ann}")
        
        assert ann['type'] == 'twinning'
        assert 'metadata' in ann
        assert 'indices' in ann['metadata']
        
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

def test_run_perturb_translation_fixed():
    """Test run_perturb with fixed translation."""
    input_file = "temp_input_trans.xyz"
    output_file = "temp_output_trans.xyz"
    
    # Create bulk Cu
    atoms = bulk('Cu', 'fcc', a=3.61)
    write(input_file, atoms)
    
    try:
        if os.path.exists(output_file):
            os.remove(output_file)
            
        # Fixed translation fraction
        fixed_trans = (0.25, 0.25)
        
        # Run perturb with GB and fixed translation
        gen = run_perturb(
            input_file,
            num=1,
            output_file=output_file,
            gb=True,
            gb_axis='0,0,1',
            gb_angle=36.87, # Sigma 5
            translation_frac=fixed_trans,
            validate_structure=False
        )
        
        generated = list(gen)
        struct = generated[0]
        
        # Ideally we check if translation was applied, but that's hard to verify from final struct alone
        # without comparing to 0-translation.
        # But we can check if it ran without error and maybe if we can inspect internal state?
        # For now, we rely on 'perturb_annotation' if we added it there?
        # run.py doesn't explicitly add translation to annotation for GB (it does for Twinning in metadata).
        # Let's check run.py lines 876-881. It adds 'type' and 'sigma'.
        # It DOES NOT add 'translation' to metadata for GB. This is a missing feature if we want to track it.
        # Twinning DOES add it (lines 964).
        
        # Let's verify Twinning with fixed translation then.
        
        gen_twin = run_perturb(
            input_file,
            num=1,
            output_file=output_file,
            twinning=True,
            twinning_indices='1,1,1',
            translation_frac=fixed_trans,
            validate_structure=False
        )
        
        generated_twin = list(gen_twin)
        struct_twin = generated_twin[0]
        ann = struct_twin.info['perturb_annotation']
        print(f"Twin Fixed Trans Annotation: {ann}")
        
        # Check if metadata has correct translation
        # Note: run.py passes translation_frac as tuple/list.
        # Check strict equality might be tricky with float precision, but let's try close enough
        meta_trans = ann['metadata']['translation_frac']
        assert np.allclose(meta_trans, fixed_trans)
        
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    print("Testing CSL GB...")
    test_run_perturb_grain_boundary_csl()
    print("Testing Twinning...")
    test_run_perturb_twinning()
    print("Testing Fixed Translation...")
    test_run_perturb_translation_fixed()
    print("All tests passed!")
