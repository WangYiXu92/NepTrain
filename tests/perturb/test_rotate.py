import unittest
import numpy as np
from ase import Atoms
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.perturb.rotate import rotate_fragments_by_formula, get_molecules

class TestRotate(unittest.TestCase):

    def setUp(self):
        # Create two H2 molecules
        # H at (0,0,0) and (0,0,0.74)
        # H at (5,0,0) and (5,0,0.74)
        self.atoms = Atoms('H4', positions=[
            [0,0,0], [0,0,0.74],
            [5,0,0], [5,0,0.74]
        ])
        
    def test_get_molecules(self):
        """Test molecule identification."""
        mols = get_molecules(self.atoms, mult=1.2)
        self.assertEqual(len(mols), 2)
        # Indices should be [0, 1] and [2, 3] (order might vary)
        # We sort to check
        sorted_mols = sorted([sorted(list(m)) for m in mols])
        self.assertEqual(sorted_mols, [[0, 1], [2, 3]])

    def test_rotation_geometry(self):
        """Test that rotation preserves internal bond lengths."""
        # Rotate H2
        rotated = rotate_fragments_by_formula(self.atoms, formula="H2", seed=42)
        
        # Check bond lengths
        # Mol 1
        d1 = rotated.get_distance(0, 1)
        # Mol 2
        d2 = rotated.get_distance(2, 3)
        
        # Original length
        d0 = 0.74
        
        self.assertAlmostEqual(d1, d0, places=5)
        self.assertAlmostEqual(d2, d0, places=5)
        
        # Check that positions actually changed (rotation happened)
        # It's random, but with seed 42, likely not identity.
        pos_orig = self.atoms.get_positions()
        pos_new = rotated.get_positions()
        self.assertFalse(np.allclose(pos_orig, pos_new))

    def test_filter(self):
        """Test that only matching formulas are rotated."""
        # Create H2 and O2 (O-O bond ~1.2)
        # H2 at 0,0,0
        # O2 at 10,0,0
        atoms = Atoms('H2O2', positions=[
            [0,0,0], [0,0,0.74],
            [10,0,0], [10,0,1.21]
        ])
        
        # Rotate only H2
        rotated = rotate_fragments_by_formula(atoms, formula="H2", seed=42)
        
        # H2 should move
        self.assertFalse(np.allclose(atoms.positions[:2], rotated.positions[:2]))
        
        # O2 should NOT move
        np.testing.assert_allclose(atoms.positions[2:], rotated.positions[2:])

if __name__ == '__main__':
    unittest.main()
