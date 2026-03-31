import unittest
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.perturb.shuffle import shuffle_element_positions

class TestShuffleElementPositions(unittest.TestCase):
    
    def setUp(self):
        # Create a simple structure: 4 atoms, linear chain
        # 0: Fe, 1: O, 2: Fe, 3: O
        # Positions: (0,0,0), (1,0,0), (2,0,0), (3,0,0)
        self.atoms = Atoms('Fe2O2', positions=[[0,0,0], [1,0,0], [2,0,0], [3,0,0]])
        # Fe at 0, 2
        # O at 1, 3
        
    def test_shuffle_indices(self):
        """Test shuffling by indices."""
        # Shuffle Fe atoms (0 and 2)
        # Seed 42 for reproducibility
        shuffled, metadata = shuffle_element_positions(
            self.atoms, 
            element_range=[0, 2], 
            shuffle_method='fisher_yates', 
            seed=42
        )
        
        # Check that O atoms (1, 3) did not move
        self.assertTrue(np.allclose(shuffled.positions[1], self.atoms.positions[1]))
        self.assertTrue(np.allclose(shuffled.positions[3], self.atoms.positions[3]))
        
        # Check that positions at 0 and 2 are still from the set {(0,0,0), (2,0,0)}
        pos0 = shuffled.positions[0]
        pos2 = shuffled.positions[2]
        original_set = set([(0.0,0.0,0.0), (2.0,0.0,0.0)])
        self.assertIn(tuple(pos0), original_set)
        self.assertIn(tuple(pos2), original_set)
        
        # Check metadata
        self.assertTrue(len(metadata) > 0 or np.allclose(shuffled.positions, self.atoms.positions))

    def test_shuffle_symbols(self):
        """Test shuffling by element symbols."""
        shuffled, metadata = shuffle_element_positions(
            self.atoms,
            element_range=['O'],
            seed=123
        )
        # Fe should not move
        self.assertTrue(np.allclose(shuffled.positions[0], self.atoms.positions[0]))
        self.assertTrue(np.allclose(shuffled.positions[2], self.atoms.positions[2]))
        
        # O positions should be from original O positions
        o_pos_orig = [self.atoms.positions[1], self.atoms.positions[3]]
        o_pos_new = [shuffled.positions[1], shuffled.positions[3]]
        
        # Check if they are valid
        for p in o_pos_new:
            found = any(np.allclose(p, op) for op in o_pos_orig)
            self.assertTrue(found)

    def test_shuffle_slice_string(self):
        """Test shuffling by slice string."""
        # "0:2" -> indices 0, 1
        shuffled, _ = shuffle_element_positions(
            self.atoms,
            element_range="0:2",
            seed=1
        )
        # Check 2 and 3 didn't move
        self.assertTrue(np.allclose(shuffled.positions[2], self.atoms.positions[2]))
        self.assertTrue(np.allclose(shuffled.positions[3], self.atoms.positions[3]))

    def test_reproducibility(self):
        """Test random seed reproducibility."""
        seed = 999
        s1, m1 = shuffle_element_positions(self.atoms, [0,1,2,3], seed=seed)
        s2, m2 = shuffle_element_positions(self.atoms, [0,1,2,3], seed=seed)
        
        self.assertTrue(np.allclose(s1.positions, s2.positions))
        self.assertEqual(m1, m2)

    def test_fixed_elements(self):
        """Test that fixed elements are not moved."""
        atoms = self.atoms.copy()
        # Fix atom 0
        c = FixAtoms(indices=[0])
        atoms.set_constraint(c)
        
        # Try to shuffle 0 and 1
        shuffled, metadata = shuffle_element_positions(
            atoms,
            element_range=[0, 1],
            seed=42
        )
        
        # Atom 0 should not move because it's fixed
        # But wait, if we only have 0 and 1, and 0 is fixed, then 1 has nowhere to swap with (if we exclude 0 from pool).
        # My implementation removes fixed indices. So indices=[0,1] becomes [1].
        # Length < 2, so no shuffle.
        
        self.assertTrue(np.allclose(shuffled.positions[0], atoms.positions[0]))
        self.assertTrue(np.allclose(shuffled.positions[1], atoms.positions[1]))
        self.assertEqual(len(metadata), 0)

    def test_large_structure(self):
        """Performance test stub for large structure."""
        # Create 1000 atoms
        positions = np.random.rand(1000, 3)
        atoms = Atoms('H'*1000, positions=positions)
        
        import time
        start = time.time()
        shuffled, _ = shuffle_element_positions(atoms, element_range=list(range(1000)), seed=1)
        end = time.time()
        
        self.assertLess(end - start, 1.0) # Should be very fast
        
        # Verify positions set matches
        # Sort positions and compare
        # (Floating point comparison might be tricky with sort, but exact match expected for shuffle)
        
        # Let's check sum of positions to ensure conservation
        self.assertTrue(np.allclose(np.sum(atoms.positions, axis=0), np.sum(shuffled.positions, axis=0)))

    def test_distribution(self):
        """Test that shuffling produces a somewhat uniform distribution."""
        # Setup: 3 atoms at 0, 1, 2. Shuffle all.
        # Run many times, check that atom 0 ends up at 0, 1, 2 roughly equally.
        atoms = Atoms('H3', positions=[[0,0,0], [1,0,0], [2,0,0]])
        
        counts = {0: 0, 1: 0, 2: 0}
        n_trials = 3000
        
        for i in range(n_trials):
            s, _ = shuffle_element_positions(atoms, [0, 1, 2], seed=i)
            # Find where atom 0 went.
            # Atom 0's new position
            pos = s.positions[0]
            # Match to original positions to find index
            if np.allclose(pos, [0,0,0]):
                counts[0] += 1
            elif np.allclose(pos, [1,0,0]):
                counts[1] += 1
            elif np.allclose(pos, [2,0,0]):
                counts[2] += 1
                
        # Expect roughly n_trials / 3 for each
        expected = n_trials / 3
        tolerance = 0.1 * expected # 10% tolerance
        
        for k in counts:
            self.assertTrue(abs(counts[k] - expected) < tolerance, 
                            f"Distribution check failed for pos {k}: {counts[k]} vs {expected}")

if __name__ == '__main__':
    unittest.main()
