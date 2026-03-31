import unittest
import numpy as np
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ase import Atoms
from NepTrain.core.perturb.sampler import SobolSampler
from NepTrain.core.perturb.shuffle import shuffle_element_positions

class TestSobolUniformity(unittest.TestCase):
    def test_sobol_uniformity(self):
        """Test if Sobol sequence is uniformly distributed in [0, 1]."""
        d = 2
        n = 1024
        sampler = SobolSampler(d=d, scramble=True, seed=42)
        samples = sampler.random(n=n)
        
        # Check range
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples <= 1.0))
        
        # Check mean (should be close to 0.5)
        means = np.mean(samples, axis=0)
        np.testing.assert_allclose(means, 0.5, atol=0.05)
        
        # Check variance (should be close to 1/12 ~ 0.0833)
        vars = np.var(samples, axis=0)
        np.testing.assert_allclose(vars, 1.0/12.0, atol=0.05)

    def test_shuffle_permutation_sobol(self):
        """Test if shuffle uses Sobol values to generate valid permutations."""
        atoms = Atoms('Fe10', positions=np.zeros((10, 3)))
        indices = list(range(10))
        
        # Generate 10 Sobol values
        rng_values = np.linspace(0.05, 0.95, 10) # Deterministic inputs
        # Expected sort order of linspace is 0,1,2...9 (already sorted)
        # Let's shuffle rng_values to verify sorting
        rng_values_shuffled = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0])
        # Expected permutation: indices corresponding to sorted values
        # 0.0 -> index 9
        # 0.1 -> index 1
        # ...
        
        new_atoms, meta = shuffle_element_positions(atoms, element_range=indices, rng_values=rng_values_shuffled)
        
        # Check if positions are shuffled (though all 0,0,0, we check metadata or symbols if they differed)
        # But shuffle_element_positions shuffles POSITIONS.
        # Since all positions are 0, output positions are 0.
        # But we can check if it ran without error.
        
        # Better test: Use different positions
        atoms.set_positions(np.array([[i, 0, 0] for i in range(10)]))
        new_atoms, meta = shuffle_element_positions(atoms, element_range=indices, rng_values=rng_values_shuffled)
        
        new_pos = new_atoms.get_positions()
        
        # The function shuffles POSITIONS of atoms i...j.
        # Wait, shuffle_element_positions implementation:
        # "Shuffle positions of elements within the specified range"
        # original_positions = atoms.get_positions()
        # selected_positions = original_positions[indices].copy()
        # new_positions_subset = selected_positions.copy()
        # ...
        # shuffled_indices = np.random.permutation(len(indices)) OR argsort(rng)
        # new_positions_subset = selected_positions[shuffled_indices]
        # atoms.positions[indices] = new_positions_subset
        
        # So atom at index `indices[k]` gets position `selected_positions[shuffled_indices[k]]`.
        
        # With rng_values_shuffled: [0.9, 0.1, 0.8, ... 0.0]
        # Argsort (low to high):
        # 0.0 (idx 9) -> 0
        # 0.1 (idx 1) -> 1
        # ...
        # sorted_args = [9, 1, 3, 5, 7, 8, 6, 4, 2, 0]
        # So new_positions_subset[0] comes from selected_positions[9]
        # new_positions_subset[1] comes from selected_positions[1]
        
        expected_indices = np.argsort(rng_values_shuffled)
        expected_pos = atoms.positions[indices][expected_indices]
        
        np.testing.assert_array_almost_equal(new_pos[indices], expected_pos)

if __name__ == '__main__':
    unittest.main()
