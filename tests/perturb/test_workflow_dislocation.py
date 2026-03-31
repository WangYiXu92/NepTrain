import unittest
import os
import shutil
from ase.build import bulk
from ase.io import read, write
from NepTrain.core.perturb.run import perturb
import numpy as np

class TestWorkflowDislocation(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for outputs
        self.test_dir = 'test_dislocation_workflow_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Create a bulk Cu structure, cubic to avoid warnings
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (4, 4, 4)
        self.test_file = os.path.join(self.test_dir, 'input.xyz')
        write(self.test_file, self.atoms)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dislocation_edge(self):
        # Test Edge Dislocation
        results = perturb(
            self.atoms,
            num=1,
            dislocation=True,
            dislocation_type='edge',
            skip_normal=True,
            min_distance=0.0 # Avoid adjust_reasonable rejection
        )
        
        results = list(results)
        self.assertEqual(len(results), 1)
        atom = results[0]
        self.assertIn('disloc_edge', atom.info['Config_type'])
        
        # Check if atoms moved
        original_pos = self.atoms.get_positions()
        new_pos = atom.get_positions()
        
        # Simple check: positions should be different
        disp = np.linalg.norm(new_pos - original_pos, axis=1)
        self.assertTrue(np.any(disp > 0.01), "No atoms moved after dislocation perturbation")

    def test_dislocation_screw(self):
        # Test Screw Dislocation
        results = perturb(
            self.atoms,
            num=1,
            dislocation=True,
            dislocation_type='screw',
            dislocation_axis='0,0,1',
            dislocation_burgers='0,0,1',
            skip_normal=True,
            min_distance=0.0
        )
        
        results = list(results)
        self.assertEqual(len(results), 1)
        atom = results[0]
        self.assertIn('disloc_screw', atom.info['Config_type'])
        
        original_pos = self.atoms.get_positions()
        new_pos = atom.get_positions()
        
        # For screw dislocation along Z (axis=2), displacement is mainly in Z
        # But we just check any movement
        disp = np.linalg.norm(new_pos - original_pos, axis=1)
        self.assertTrue(np.any(disp > 0.01), "No atoms moved after screw dislocation perturbation")

    def test_surface_and_dislocation(self):
        # Test Surface then Dislocation
        results = perturb(
            self.atoms,
            num=1,
            surface=True,
            surface_indices='1,1,1',
            dislocation=True,
            dislocation_type='edge',
            skip_normal=True,
            min_distance=0.0,
            validate_structure=False
        )
        
        results = list(results)
        self.assertEqual(len(results), 1)
        atom = results[0]
        config_type = atom.info['Config_type']
        self.assertIn('surf(1,1,1)', config_type)
        self.assertIn('disloc_edge', config_type)

if __name__ == '__main__':
    unittest.main()
