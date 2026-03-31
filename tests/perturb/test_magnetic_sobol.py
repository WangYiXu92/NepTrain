import unittest
import numpy as np
from ase import Atoms
from NepTrain.core.perturb.magnetic import apply_magnetic_perturbation
from scipy.stats import norm
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

class TestMagneticSobol(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms('Fe2', positions=[[0,0,0], [2,0,0]])
        self.mag_config = {'Fe': 2.0}

    def test_noise_determinism(self):
        # 2 atoms, noise needs 2 normal values.
        # If we pass rng_values (uniform), they should be converted to normal.
        # Let's say we pass [0.5, 0.5]. norm.ppf(0.5) = 0.
        # So noise should be 0.
        
        # We need to pass enough values.
        # Noise consumes sum(is_magnetic) values. Here 2.
        rng_values = np.array([0.5, 0.5]) 
        
        atoms = apply_magnetic_perturbation(self.atoms, mode='collinear', mag_config=self.mag_config, noise=0.1, rng_values=rng_values)
        moms = atoms.get_initial_magnetic_moments()
        # Should be exactly 2.0 because noise is 0 at 0.5 quantile
        # Expect vectors along Z
        expected = np.array([[0,0,2.0], [0,0,2.0]])
        np.testing.assert_allclose(moms, expected)
        
        # Try 0.841344746 (approx +1 std dev)
        rng_values = np.array([0.841344746, 0.841344746])
        atoms = apply_magnetic_perturbation(self.atoms, mode='collinear', mag_config=self.mag_config, noise=1.0, rng_values=rng_values)
        moms = atoms.get_initial_magnetic_moments()
        # Should be approx 3.0
        expected = np.array([[0,0,3.0], [0,0,3.0]])
        np.testing.assert_allclose(moms, expected, atol=1e-4)

    def test_random_collinear_determinism(self):
        # random_collinear needs n_atoms values for flip check.
        # flip if val < prob.
        rng_values = np.array([0.1, 0.9])
        # prob = 0.5
        # 0.1 < 0.5 -> flip. 0.9 < 0.5 -> no flip.
        
        atoms = apply_magnetic_perturbation(self.atoms, mode='random_collinear', mag_config=self.mag_config, flip_prob=0.5, rng_values=rng_values)
        moms = atoms.get_initial_magnetic_moments()
        # First atom flipped (-2.0), second not (2.0)
        expected = np.array([[0,0,-2.0], [0,0,2.0]])
        np.testing.assert_allclose(moms, expected)

if __name__ == '__main__':
    unittest.main()
