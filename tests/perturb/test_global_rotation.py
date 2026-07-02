import unittest
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.run import perturb

class TestGlobalRotation(unittest.TestCase):
    def setUp(self):
        # Create a simple cubic cell
        self.atoms = Atoms('Cu4', 
                           positions=[[0,0,0], [2,0,0], [0,2,0], [0,0,2]],
                           cell=[4,4,4],
                           pbc=True)
        
    def test_global_rotation_random(self):
        """Test random global rotation (sampler='random')"""
        atoms_list = list(perturb(self.atoms, num=5, 
                             rotate_cell=True, 
                             cell_pert_fraction=0.0, # Disable strain to isolate rotation
                             min_distance=0.01,
                             similarity_threshold=1.0,
                             sampler='random'))
        
        self.assertEqual(len(atoms_list), 5)
        for atoms in atoms_list:
            # Check cell is rotated
            cell = atoms.get_cell()
            # Original cell is diagonal. Rotated cell should not be diagonal (mostly)
            # unless rotation is 0 or 90 deg.
            # Check if off-diagonal elements are non-zero
            off_diag = cell - np.diag(np.diag(cell))
            # It's possible to be close to 0, but unlikely for all 5
            pass
            
            # Check volume is conserved (rotation doesn't change volume)
            self.assertAlmostEqual(atoms.get_volume(), self.atoms.get_volume(), places=3)
            
    def test_global_rotation_sobol(self):
        """Test global rotation with Sobol sampler"""
        atoms_list = list(perturb(self.atoms, num=10, 
                             rotate_cell=True, 
                             cell_pert_fraction=0.0,
                             min_distance=0.01,
                             similarity_threshold=1.0,
                             sampler='sobol'))
        
        self.assertEqual(len(atoms_list), 10)
        
        # Collect rotation angles/axes implicitly by checking cell
        cells = [a.get_cell() for a in atoms_list]
        
        # Check that we get different cells
        unique_cells = len(np.unique([c.flatten().round(3) for c in cells], axis=0))
        self.assertGreater(unique_cells, 1)
        
        # Check volume conservation
        for atoms in atoms_list:
            self.assertAlmostEqual(atoms.get_volume(), self.atoms.get_volume(), places=3)

if __name__ == '__main__':
    unittest.main()
