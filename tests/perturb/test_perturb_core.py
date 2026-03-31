import unittest
import numpy as np
from ase import Atoms
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.perturb.run import perturb_position, generate_strained_structure, generate_deformed_structure

class TestPerturbCore(unittest.TestCase):

    def setUp(self):
        self.atoms = Atoms('Fe', positions=[[0.5, 0.5, 0.5]], cell=[1,1,1], pbc=True)

    def test_perturb_position(self):
        """Test random displacement of positions."""
        np.random.seed(42)
        min_dist = 0.1
        perturbed = perturb_position(self.atoms, min_distance=min_dist)
        
        disp = perturbed.positions - self.atoms.positions
        
        # Check that displacement is within bounds [-min_dist, min_dist]
        # Implementation uses uniform low=-min, high=min
        self.assertTrue(np.all(np.abs(disp) <= min_dist))
        
        # Check it's not zero (random)
        self.assertFalse(np.allclose(disp, 0))

    def test_generate_strained_structure(self):
        """Test cell strain (diagonal)."""
        np.random.seed(42)
        strain_lim = [-0.05, 0.05]
        min_dist = 0.0 # No rattle to isolate strain
        
        strained = generate_strained_structure(self.atoms, strain_lim=strain_lim, min_distance=min_dist)
        
        # Check cell change
        orig_cell = self.atoms.get_cell()
        new_cell = strained.get_cell()
        
        # Strained cell should be diagonal (just scaling lengths)
        # Assuming original is diagonal
        # cell_new = prim.cell[:] * (1 + strains)
        # If cell is identity, new cell is diagonal.
        self.assertTrue(np.allclose(new_cell - np.diag(np.diag(new_cell)), 0))
        
        # Check limits
        strains = np.diag(new_cell) - 1.0
        self.assertTrue(np.all(strains >= strain_lim[0]))
        self.assertTrue(np.all(strains <= strain_lim[1]))

    def test_generate_deformed_structure(self):
        """Test full cell deformation."""
        np.random.seed(42)
        strain_lim = [-0.05, 0.05]
        min_dist = 0.0
        
        deformed = generate_deformed_structure(self.atoms, strain_lim=strain_lim, min_distance=min_dist)
        
        # Deformed cell might not be diagonal
        # M = I + R
        # cell_new = M @ cell
        new_cell = deformed.get_cell()
        
        # Check that it changed
        self.assertFalse(np.allclose(new_cell, self.atoms.get_cell()))
        
        # Check that volume is reasonable (small strain shouldn't collapse cell)
        self.assertTrue(deformed.get_volume() > 0.5)

if __name__ == '__main__':
    unittest.main()
