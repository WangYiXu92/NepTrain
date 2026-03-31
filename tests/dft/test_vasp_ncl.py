import os
import shutil
import sys
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.dft.vasp.run import calculate_vasp
from NepTrain import Config

def test_vasp_ncl():
    print("Testing VASP non-collinear input generation...")
    # Create atoms with non-collinear moments (vectors)
    atoms = Atoms('Fe2', positions=[[0,0,0], [2,0,0]], cell=[4,4,4], pbc=True)
    # Set moments along different directions
    # Atom 1: along x (3, 0, 0)
    # Atom 2: along z (0, 0, 3)
    moms = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    atoms.set_initial_magnetic_moments(moms)
    
    args = argparse.Namespace()
    args.directory = "test_vasp_ncl_out"
    args.incar = None
    args.n_cpu = 1
    args.kspacing = 0.5
    args.ka = [1, 1, 1]
    args.use_gamma = True
    
    if not os.path.exists("test_vasp_ncl_gen"):
        os.makedirs("test_vasp_ncl_gen")
    
    atoms.write("test_vasp_ncl_gen/structure.xyz")
    args.model_path = "test_vasp_ncl_gen/structure.xyz"
    args.directory = "test_vasp_ncl_gen"
    args.out_file_path = "test_vasp_ncl_gen/out.xyz"
    args.append = False
    
    # We need to capture the command passed to VaspInput.set or check VaspInput._run
    # VaspInput.set is called with command.
    
    # Let's mock VaspInput
    with patch('NepTrain.core.dft.vasp.run.VaspInput') as MockVaspInput:
        # Mock instance
        mock_vasp = MockVaspInput.return_value
        mock_vasp.int_params = {'ispin': 1} # default
        mock_vasp.float_params = {'tebeg': 300, 'teend': 300} # for aimd check?
        mock_vasp.results = {'stress': np.zeros((3,3)), 'free_energy': 0.0} # for post-processing
        mock_vasp.converged = True
        
        # We need to ensure vasp.calculate sets calc on atoms so atoms.calc.results works?
        # In run.py: 
        # vasp.calculate(atoms, ('energy'))
        # atoms.calc = vasp._xml_calc
        
        # We just want to check input settings.
        # But we need to allow the function to run through without crashing.
        
        # Mock check_env
        with patch('NepTrain.core.utils.check_env'):
             try:
                calculate_vasp("test_vasp_ncl_gen/structure.xyz", args)
             except Exception as e:
                print(f"Caught exception: {e}")
        
        # Check calls to vasp.set
        # We expect vasp.set(lnoncollinear=True)
        # And command should contain vasp_ncl
        
        set_calls = mock_vasp.set.call_args_list
        
        lnoncollinear_set = False
        command_set_correctly = False
        
        for call in set_calls:
            args_call, kwargs_call = call
            if kwargs_call.get('lnoncollinear') is True:
                lnoncollinear_set = True
            
            if 'command' in kwargs_call:
                cmd = kwargs_call['command']
                print(f"Command set: {cmd}")
                # We assume config has vasp_std, so replace gives vasp_ncl
                if 'vasp_ncl' in cmd:
                    command_set_correctly = True
        
        if lnoncollinear_set:
            print("LNONCOLLINEAR set to True: PASSED")
        else:
            print("LNONCOLLINEAR set to True: FAILED")
            
        if command_set_correctly:
            print("Command switched to vasp_ncl: PASSED")
        else:
            print("Command switched to vasp_ncl: FAILED (Note: requires config vasp_path='vasp_std' or similar)")

if __name__ == "__main__":
    test_vasp_ncl()
