import unittest
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.rotate import rotate_fragments_by_formula
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

class TestRotateSobol(unittest.TestCase):
    def setUp(self):
        # A simple molecule, e.g. H2
        self.atoms = Atoms('H2', positions=[[0,0,0], [0,0,1]])
        
    def test_determinism(self):
        # Need 3 values per fragment.
        # H2 is one fragment if formula matches.
        rng_values = np.array([0.1, 0.2, 0.3])
        
        atoms1 = rotate_fragments_by_formula(self.atoms, 'H2', rng_values=rng_values)
        atoms2 = rotate_fragments_by_formula(self.atoms, 'H2', rng_values=rng_values)
        
        np.testing.assert_allclose(atoms1.positions, atoms2.positions)
        
        # Check it actually rotated (unless random values happen to be identity, unlikely)
        self.assertFalse(np.allclose(atoms1.positions, self.atoms.positions))
        
        # Check bond length preserved
        d1 = atoms1.get_distance(0, 1)
        d0 = self.atoms.get_distance(0, 1)
        self.assertAlmostEqual(d1, d0)

if __name__ == '__main__':
    unittest.main()
