import unittest
import numpy as np
from ase import Atoms
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.perturb.magnetic import apply_magnetic_perturbation

class TestMagnetic(unittest.TestCase):

    def setUp(self):
        self.atoms = Atoms('Fe2', positions=[[0,0,0], [2,0,0]])
        self.mag_config = {'Fe': 2.0}

    def test_collinear(self):
        """Test standard collinear mode sets moments correctly."""
        atoms = apply_magnetic_perturbation(self.atoms, mode='collinear', mag_config=self.mag_config)
        moms = atoms.get_initial_magnetic_moments()
        # Should be vectors along Z (default)
        expected = np.array([[0, 0, 2.0], [0, 0, 2.0]])
        np.testing.assert_allclose(moms, expected)

    def test_random_collinear(self):
        """Test random collinear flips."""
        # Force flips with probability 1.0 (should flip all to -2.0)
        # Logic: spins[flips] *= -1.
        # If prob=1.0, all flip.
        
        atoms = apply_magnetic_perturbation(self.atoms, mode='random_collinear', mag_config=self.mag_config, flip_prob=1.0)
        moms = atoms.get_initial_magnetic_moments()
        # Should be vectors along Z, flipped
        expected = np.array([[0, 0, -2.0], [0, 0, -2.0]])
        np.testing.assert_allclose(moms, expected)
        
        # Force no flips
        atoms = apply_magnetic_perturbation(self.atoms, mode='random_collinear', mag_config=self.mag_config, flip_prob=0.0)
        moms = atoms.get_initial_magnetic_moments()
        expected = np.array([[0, 0, 2.0], [0, 0, 2.0]])
        np.testing.assert_allclose(moms, expected)

    def test_non_collinear(self):
        """Test non-collinear generation produces vectors with correct magnitude."""
        atoms = apply_magnetic_perturbation(self.atoms, mode='non_collinear', mag_config=self.mag_config)
        moms = atoms.get_initial_magnetic_moments()
        # Should be (2, 3) array
        self.assertEqual(moms.shape, (2, 3))
        # Magnitudes should be close to 2.0
        magnitudes = np.linalg.norm(moms, axis=1)
        np.testing.assert_allclose(magnitudes, [2.0, 2.0])

    def test_noise(self):
        """Test that noise changes magnitude."""
        # Set a seed for reproducibility if possible, or just check deviation
        np.random.seed(42)
        atoms = apply_magnetic_perturbation(self.atoms, mode='collinear', mag_config=self.mag_config, noise=0.1)
        moms = atoms.get_initial_magnetic_moments()
        # Moms are vectors, calc magnitude
        magnitudes = np.linalg.norm(moms, axis=1)
        # Should NOT be exactly 2.0
        self.assertFalse(np.allclose(magnitudes, [2.0, 2.0]))
        # Should be close to 2.0 within noise range (roughly)
        self.assertTrue(np.all(np.abs(magnitudes - 2.0) < 0.5))

    def test_no_config(self):
        """Test behavior when element is not in config."""
        atoms = Atoms('Au2', positions=[[0,0,0], [2,0,0]])
        # Au not in mag_config
        atoms = apply_magnetic_perturbation(atoms, mode='collinear', mag_config=self.mag_config)
        moms = atoms.get_initial_magnetic_moments()
        # Should be zeros (default), but check if it's scalar or vector
        # If all zeros, get_initial_magnetic_moments might return scalars [0.0, 0.0]
        # or vectors [[0,0,0], [0,0,0]] depending on how it was set.
        # apply_magnetic_perturbation returns original atoms if no magnetic atoms found.
        # So it might not have set moments at all -> default zeros (scalar).
        # Let's handle both.
        if moms.ndim == 2:
            moms = np.linalg.norm(moms, axis=1)
        np.testing.assert_array_equal(moms, [0.0, 0.0])

    def test_vector_mag_config_is_preserved_by_ensure(self):
        """Vector config values should initialize non-collinear moments directly."""
        from NepTrain.core.perturb.magnetic import ensure_magnetic_configuration
        atoms = Atoms('Fe2', positions=[[0,0,0], [2,0,0]])
        ensure_magnetic_configuration(atoms, mag_config={'Fe': [2.0, 0.0, 0.5]})
        expected = np.array([[2.0, 0.0, 0.5], [2.0, 0.0, 0.5]])
        np.testing.assert_allclose(atoms.get_initial_magnetic_moments(), expected)

if __name__ == '__main__':
    unittest.main()
