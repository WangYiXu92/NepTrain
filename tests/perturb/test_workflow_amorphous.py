import unittest
import os
import shutil
from ase.build import bulk
from ase.io import read, write
from NepTrain.core.perturb.run import perturb
import numpy as np

class TestWorkflowAmorphous(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for outputs
        self.test_dir = 'test_amorphous_workflow_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Create a bulk Cu structure
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (2, 2, 2)
        self.test_file = os.path.join(self.test_dir, 'input.xyz')
        write(self.test_file, self.atoms)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_amorphous_generation(self):
        # Test Amorphous only
        results = perturb(
            self.atoms,
            num=1,
            amorphous=True,
            amorphous_min_dist=1.8,
            skip_normal=True,
            min_distance=0.0
        )
        
        results = list(results)
        self.assertEqual(len(results), 1)
        atom = results[0]
        self.assertIn('amorphous', atom.info['Config_type'])
        
        # Check if atoms moved significantly (amorphous should look very different)
        # But we can just check if positions are different
        original_pos = self.atoms.get_positions()
        new_pos = atom.get_positions()
        
        disp = np.linalg.norm(new_pos - original_pos, axis=1)
        self.assertTrue(np.any(disp > 0.01), "No atoms moved after amorphous perturbation")
        
        # Check that volume changed (amorphous generation scales volume)
        self.assertNotEqual(atom.get_volume(), self.atoms.get_volume())

if __name__ == '__main__':
    unittest.main()
