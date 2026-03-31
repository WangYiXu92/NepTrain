import unittest
import os
import shutil
from ase.build import bulk
from ase.io import read, write
from NepTrain.core.perturb.run import perturb
import numpy as np

class TestWorkflowTwinning(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for outputs
        self.test_dir = 'test_twinning_workflow_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Create a bulk Cu structure
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (2, 2, 4)
        self.test_file = os.path.join(self.test_dir, 'input.xyz')
        write(self.test_file, self.atoms)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_twinning_generation(self):
        # Test Twinning only
        results = perturb(
            self.atoms,
            num=1,
            twinning=True,
            twinning_indices='1,1,1',
            twinning_z=0.5,
            skip_normal=True,
            validate_structure=False
        )
        
        results = list(results)
        # Verify results structure
        # perturb returns generator of Atoms objects directly (yield struct)
        # So results is list of Atoms
        self.assertEqual(len(results), 1)
        atom = results[0]
        self.assertIn('twin_1,1,1', atom.info['Config_type'])
        
        # Check if atoms moved
        original_pos = self.atoms.get_positions()
        new_pos = atom.get_positions()
        
        # Twinning rotates top half 180 deg.
        # Note: Twinning might remove atoms due to overlap removal.
        # So we cannot compare positions directly if atom count changes.
        if len(atom) == len(self.atoms):
            disp = np.linalg.norm(new_pos - original_pos, axis=1)
            self.assertTrue(np.any(disp > 0.01), "No atoms moved after twinning perturbation")
        else:
            # If atoms were removed, it definitely changed.
            self.assertNotEqual(len(atom), len(self.atoms))

    def test_surface_and_twinning(self):
        # Test Surface then Twinning
        results = perturb(
            self.atoms,
            num=1,
            surface=True,
            surface_indices='1,1,1',
            twinning=True,
            twinning_indices='1,1,1',
            skip_normal=True,
            min_distance=0.0
        )
        
        results = list(results)
        self.assertEqual(len(results), 1)
        atom = results[0]
        config_type = atom.info['Config_type']
        self.assertIn('surf(1,1,1)', config_type)
        self.assertIn('twin_1,1,1', config_type)
        
        # Check PBC
        self.assertTrue(atom.pbc[0])
        self.assertTrue(atom.pbc[1])
        self.assertFalse(atom.pbc[2])

if __name__ == '__main__':
    unittest.main()
