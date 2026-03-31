import unittest
import os
import sys
import shutil
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.dft.vasp.run import calculate_vasp
from NepTrain.core.perturb.magnetic import ensure_magnetic_configuration, get_magmom_config
from NepTrain import Config

def test_full_ncl_logic():
    print("Testing full non-collinear logic...")
    
    # 1. Test Config parsing for vector moments
    # We need to populate Config with 'magmom' section
    if not Config.has_section('magmom'):
        Config.add_section('magmom')
    Config.set('magmom', 'Fe', '3.0 0.0 0.0')
    Config.set('magmom', 'Ni', '0.0 0.0 1.0')
    
    # Verify get_magmom_config
    mag_config = get_magmom_config()
    print(f"Mag Config: {mag_config}")
    assert mag_config['Fe'] == [3.0, 0.0, 0.0]
    assert mag_config['Ni'] == [0.0, 0.0, 1.0]
    print("Config parsing: PASSED")
    
    # 2. Test ensure_magnetic_configuration
    atoms = Atoms('FeNi', positions=[[0,0,0], [2,0,0]], cell=[4,4,4], pbc=True)
    # Initially no moments
    atoms = ensure_magnetic_configuration(atoms)
    moms = atoms.get_initial_magnetic_moments()
    print(f"Applied moments: {moms}")
    assert np.allclose(moms[0], [3.0, 0.0, 0.0])
    assert np.allclose(moms[1], [0.0, 0.0, 1.0])
    print("Ensure configuration: PASSED")
    
    # 3. Test calculate_vasp settings
    args = argparse.Namespace()
    args.directory = "test_ncl_full_out"
    args.incar = None
    args.n_cpu = 1
    args.kspacing = 0.5
    args.ka = [1, 1, 1]
    args.use_gamma = True
    args.model_path = "dummy.xyz"
    args.out_file_path = "out.xyz"
    args.append = False
    
    # Mock VaspInput
    with patch('NepTrain.core.dft.vasp.run.VaspInput') as MockVaspInput:
        mock_vasp = MockVaspInput.return_value
        mock_vasp.int_params = {'ispin': 1, 'ibrion': 0} # ibrion added to avoid crash
        mock_vasp.float_params = {'tebeg': 300, 'teend': 300}
        mock_vasp.results = {'stress': np.zeros((3,3)), 'free_energy': 0.0}
        mock_vasp.converged = True
        
        with patch('NepTrain.core.utils.check_env'):
            # calculate_vasp is decorated, we need to call it properly or bypass decorator?
            # If we call calculate_vasp directly with atoms, it might work if the decorator allows passing atoms directly?
            # The decorator is @utils.iter_path_to_atoms.
            # It usually expects path pattern.
            # But let's check utils.iter_path_to_atoms implementation.
            # It takes func(atoms, argparse).
            # The wrapper takes (path_pattern, *args, **kwargs).
            # If we pass a path, it yields atoms.
            
            # So we create a file
            if not os.path.exists("test_ncl_full_gen"):
                os.makedirs("test_ncl_full_gen")
            atoms.write("test_ncl_full_gen/structure.xyz")
            args.model_path = "test_ncl_full_gen/structure.xyz"
            
            try:
                calculate_vasp("test_ncl_full_gen/structure.xyz", args)
            except Exception as e:
                print(f"Run exception (expected if mocks incomplete): {e}")
        
        # Check vasp.set calls
        # We expect: lnoncollinear=True, nelm=300, amix=0.2, lmaxmix=4 (no rare earth)
        
        calls = mock_vasp.set.call_args_list
        params_set = {}
        for call in calls:
            _, kwargs = call
            params_set.update(kwargs)
            
        print("VASP Parameters Set:")
        for k, v in params_set.items():
            print(f"  {k}: {v}")
            
        assert params_set.get('lnoncollinear') == True
        assert params_set.get('nelm') == 300
        assert params_set.get('amix') == 0.2
        assert params_set.get('lmaxmix') == 4 # Fe, Ni are not rare earth
        
        print("VASP settings: PASSED")

if __name__ == "__main__":
    test_full_ncl_logic()
