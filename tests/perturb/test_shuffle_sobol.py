import unittest
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.shuffle import shuffle_element_positions

class TestShuffleSobol(unittest.TestCase):
    def test_deterministic_shuffle(self):
        # Use valid symbols: H, He, Li, Be, B
        atoms = Atoms('HHeLiBeB')
        # Set positions explicitly to distinguish them
        # H at 0, He at 1, Li at 2, Be at 3, B at 4
        positions = np.zeros((5, 3))
        positions[:, 0] = [0, 1, 2, 3, 4]
        atoms.set_positions(positions)
        
        # rng_values: define a specific permutation via argsort
        # Values: [0.5, 0.1, 0.9, 0.2, 0.4]
        # Sorted indices: 1 (0.1), 3 (0.2), 4 (0.4), 0 (0.5), 2 (0.9)
        # Permutation: [1, 3, 4, 0, 2]
        # Original Positions: [0, 1, 2, 3, 4]
        # New Positions (subset[perm]): 
        # idx 0 gets subset[1] -> 1
        # idx 1 gets subset[3] -> 3
        # idx 2 gets subset[4] -> 4
        # idx 3 gets subset[0] -> 0
        # idx 4 gets subset[2] -> 2
        
        rng = np.array([0.5, 0.1, 0.9, 0.2, 0.4])
        
        new_atoms, _ = shuffle_element_positions(atoms, element_range='0:5', rng_values=rng)
        
        new_pos = new_atoms.get_positions()[:, 0]
        expected_pos = np.array([1.0, 3.0, 4.0, 0.0, 2.0])
        
        np.testing.assert_allclose(new_pos, expected_pos)
