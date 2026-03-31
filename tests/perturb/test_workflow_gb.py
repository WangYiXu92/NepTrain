import unittest
import os
import shutil
from ase.build import bulk
from ase.io import read, write
from NepTrain.core.perturb.run import perturb
import numpy as np

class TestWorkflowGB(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for outputs
        self.test_dir = 'test_gb_workflow_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Create a bulk Cu structure
        self.atoms = bulk('Cu', 'fcc', a=3.6) * (2, 2, 4)
        self.test_file = os.path.join(self.test_dir, 'input.xyz')
        write(self.test_file, self.atoms)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_gb_generation(self):
        # Test GB only
        results = perturb(
            self.atoms,
            num=1,
            gb=True,
            gb_axis='0,0,1',
            gb_angle=30.0,
            gb_dist=1.5,
            skip_normal=True,
            min_distance=0.0,
            debug_plot=True
        )
        
        # results is a generator
        results = list(results)
        self.assertEqual(len(results), 1)
        atom = results[0]
        self.assertIn('gb0,0,1_30.0', atom.info['Config_type'])
        
        # Check if atoms moved (top half rotated)
        # We need original positions. 
        # Since num=1 and we skipped normal, result is just the GB structure.
        original_pos = self.atoms.get_positions()
        new_pos = atom.get_positions()
        
        # Check displacement
        # Note: input atoms (self.atoms) might differ from what perturb reads if ordering changes or if read/write changes precision
        # But for bulk Cu, it should be fine.
        # However, generate_grain_boundary wraps atoms!
        # Original self.atoms might not be wrapped.
        # Let's verify.
        
        if len(atom) == len(self.atoms):
            disp = np.linalg.norm(new_pos - original_pos, axis=1)
            # We expect some atoms to move.
            self.assertTrue(np.any(disp > 0.1), "No atoms moved after GB perturbation")
        else:
            # Atoms deleted due to overlap, perturbation successful
            pass

    def test_surface_and_gb(self):
        # Test Surface then GB
        results = perturb(
            self.atoms,
            num=1,
            surface=True,
            surface_indices='1,1,1',
            gb=True,
            gb_axis='0,0,1',
            gb_angle=45,
            skip_normal=True,
            min_distance=0.0,
            validate_structure=False
        )
        
        results = list(results)
        self.assertEqual(len(results), 1)
        atom = results[0]
        config_type = atom.info['Config_type']
        self.assertIn('surf(1,1,1)', config_type)
        self.assertIn('gb0,0,1_45.0', config_type)

if __name__ == '__main__':
    unittest.main()
