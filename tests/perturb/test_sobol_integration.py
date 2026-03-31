import unittest
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.run import perturb
from NepTrain import Config
import os
import shutil
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

class TestSobolIntegration(unittest.TestCase):
    def setUp(self):
        # Create a dummy atoms object
        # H2O molecule
        self.atoms = Atoms('H2O', positions=[[0,0,0], [0,0,1], [0,1,0]])
        # Make a box
        self.atoms.set_cell([10, 10, 10])
        self.atoms.set_pbc(True)
        
        # Write to file
        if not os.path.exists("test_sobol_tmp"):
            os.makedirs("test_sobol_tmp")
        self.atoms.write("test_sobol_tmp/poscar.vasp")
        
        # Mock Config for magnetic
        Config['magmom'] = {'O': 2.0} # make O magnetic

    def tearDown(self):
        if os.path.exists("test_sobol_tmp"):
            shutil.rmtree("test_sobol_tmp")
        # Clean up Config
        if 'magmom' in Config:
            del Config['magmom']

    def test_full_workflow(self):
        # Arguments:
        # num=4
        # sampler='sobol'
        # mag_mode='random_collinear'
        # rotate_formula='H2O'
        
        # The decorated function takes path as first arg (or matches pattern).
        # We pass the specific file path.
        
        # Note: The decorator might behave differently depending on implementation.
        # If I look at utils.iter_path_to_atoms, it likely wraps the function.
        # Let's try calling it.
        
        gen = perturb(self.atoms, 
                      num=4, 
                      sampler='sobol',
                      mag_mode='random_collinear',
                      rotate_formula='H2O',
                      mag_noise=0.1)
                      
        results = list(gen)
        # When passing Atoms object directly, results is a list of structures
        self.assertEqual(len(results), 4)
        
        structures = results

        
        # Check properties
        for atoms in structures:
            # Check magnetic moments exist
            moms = atoms.get_initial_magnetic_moments()
            # O should be magnetic (index 2)
            # H might be 0
            # handle vector moments
            mag_val = moms[2]
            if isinstance(mag_val, (list, np.ndarray)) and len(np.shape(mag_val)) > 0:
                mag_norm = np.linalg.norm(mag_val)
            else:
                mag_norm = abs(mag_val)
                
            self.assertTrue(mag_norm > 0.5) 
            
            # Check positions changed
            # Verify it's not identical to input
            self.assertFalse(np.allclose(atoms.positions, self.atoms.positions))

if __name__ == '__main__':
    unittest.main()
