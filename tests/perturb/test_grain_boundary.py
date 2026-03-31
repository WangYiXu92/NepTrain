import unittest
import numpy as np
from ase.build import bulk
from NepTrain.core.perturb.grain_boundary import generate_grain_boundary

class TestGrainBoundary(unittest.TestCase):
    def setUp(self):
        # Create a simple cubic structure (Conventional cell 4 atoms)
        self.atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
        # Supercell 2x2x2 -> 32 atoms
        self.atoms = self.atoms * (2, 2, 2)
        
    def test_simple_rotation(self):
        """Test simple rotation mode (no Sigma, just angle)"""
        # Rotate by 30 degrees around Z, add vacuum to avoid periodic boundary issues
        gb_atoms = generate_grain_boundary(self.atoms, axis=[0,0,1], angle_deg=30.0, min_dist=1.5, vacuum=2.0)
        
        # Check basic properties
        # Should be roughly double the atoms (two grains stacked) minus overlaps
        # Original: 32 atoms (4 per cell * 8 cells) -> 32
        # Stacked: 64 atoms
        # With overlaps removed, maybe less.
        self.assertGreater(len(gb_atoms), 0)
        
        # Check minimum distance
        from ase.neighborlist import neighbor_list
        # Check if any atom is closer than 1.4 (min_dist=1.5)
        indices_i, indices_j, distances = neighbor_list('ijd', gb_atoms, cutoff=1.4)
        if len(distances) > 0:
            print(f"Simple Rotation: Found {len(distances)} pairs < 1.4A")
            for k in range(min(5, len(distances))):
                print(f"  {indices_i[k]} - {indices_j[k]}: {distances[k]}")
        self.assertEqual(len(distances), 0, f"Found atoms closer than 1.4 A")
        
    def test_csl_generation(self):
        """Test CSL mode (Sigma provided)"""
        # Sigma 5 Twist Boundary, add vacuum
        gb_atoms = generate_grain_boundary(self.atoms, axis=[0,0,1], sigma=5, min_dist=1.5, vacuum=2.0)
        
        self.assertGreater(len(gb_atoms), 0)
        
        # Check annotation
        self.assertIn('perturb_annotation', gb_atoms.info)
        self.assertEqual(gb_atoms.info['perturb_annotation']['metadata']['sigma'], 5)
        
        # Check minimum distance
        from ase.neighborlist import neighbor_list
        indices_i, indices_j, distances = neighbor_list('ijd', gb_atoms, cutoff=1.4)
        if len(distances) > 0:
            print(f"CSL: Found {len(distances)} pairs < 1.4A")
            for k in range(min(5, len(distances))):
                print(f"  {indices_i[k]} - {indices_j[k]}: {distances[k]}")
        self.assertEqual(len(distances), 0, f"Found atoms closer than 1.4 A")

if __name__ == '__main__':
    unittest.main()
