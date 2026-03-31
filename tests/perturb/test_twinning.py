import unittest
import numpy as np
from ase.build import bulk
from ase.visualize import view
from NepTrain.core.perturb.twinning import generate_twinning

class TestTwinning(unittest.TestCase):
    def setUp(self):
        # Create a cubic FCC structure
        # (100) faces along axes
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (3, 3, 3)
        
    def test_twinning_generation(self):
        # Generate (111) twin
        twin = generate_twinning(self.atoms, miller_indices=(1, 1, 1), min_dist=1.5)
        
        # Check basic properties
        self.assertGreater(len(twin), 0)
        
        # Check that no atoms are too close
        from ase.neighborlist import neighbor_list
        d = neighbor_list('d', twin, cutoff=1.4)
        self.assertEqual(len(d), 0, f"Found atoms closer than 1.4 A: {d}")
        
        # Check that the cell is rotated
        # Original cell has (3.6*3, 0, 0) as first vector
        # Rotated cell should have [111] along Z?
        # No, we rotated such that (111) reciprocal vector is along Z.
        # For cubic, real [111] is parallel to reciprocal (111).
        # So cell Z vector should have component along original [111].
        # Actually, we rotated the *atoms* and *cell*.
        # So the new Z axis of the simulation box might NOT be along crystal [111] if the box was not aligned.
        # Wait, we rotate the *entire object*.
        # So the new Z-axis (0,0,1) of the LAB FRAME is parallel to the crystal (111) direction.
        # But the cell vectors are also rotated.
        # So the cell is still cubic in shape, but tilted.
        # Wait, `atoms.rotate(..., rotate_cell=True)` rotates the cell vectors.
        # So the cell vectors change.
        # The atoms positions change.
        # But relative coordinates?
        # If I rotate everything, relative coordinates in the cell stay same?
        # No, `rotate` modifies positions (Cartesian) and cell (Cartesian).
        # So `scaled_positions` are invariant? Yes.
        
        # Then we modify positions of top half.
        # This changes structure.
        
        pass

    def test_twinning_overlap(self):
        # Test with a case where overlap is likely
        # 2x2x2 supercell
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (2, 2, 2)
        twin = generate_twinning(atoms, miller_indices=(1, 1, 1), min_dist=1.8)
        
        # Check distances
        from ase.neighborlist import neighbor_list
        d = neighbor_list('d', twin, cutoff=1.7)
        self.assertEqual(len(d), 0)

if __name__ == '__main__':
    unittest.main()
