import unittest
import os
from ase.build import bulk
from ase.io import write
from NepTrain.core.perturb.run import perturb

class TestWorkflowSurface(unittest.TestCase):
    def setUp(self):
        self.atoms = bulk('Cu', 'fcc', a=3.6)
        self.test_file = 'test_surface_input.xyz'
        write(self.test_file, self.atoms)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_surface_workflow(self):
        """Test surface generation."""
        # 1. Generate surface
        results = perturb(
            self.atoms,
            num=1,
            surface=True,
            surface_indices='1,1,1',
            surface_vacuum=10.0,
            surface_layers=3,
            skip_normal=True,
            min_distance=0.0
        )
        
        results = list(results)
        self.assertEqual(len(results), 1)
        atom = results[0]
        
        # Check Config_type
        self.assertIn('surf(1,1,1)', atom.info['Config_type'])
        
        # Check PBC (should be False in Z, True in X, Y)
        # Surface generation usually sets pbc=[True, True, False]
        self.assertTrue(atom.pbc[0])
        self.assertTrue(atom.pbc[1])
        self.assertFalse(atom.pbc[2])
        
        # Check Z dimension is large (vacuum)
        # Original cell is small. New cell should have vacuum.
        self.assertTrue(atom.cell[2, 2] > 10.0)

if __name__ == '__main__':
    unittest.main()
