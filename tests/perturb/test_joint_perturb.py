import unittest
import os
import shutil
import numpy as np
from ase import Atoms
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.perturb.run import perturb
from NepTrain import Config

class TestJointPerturb(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_joint_perturb_tmp"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
            
        # Create a structure: 
        # 2 Fe atoms (Magnetic)
        # 1 H2O molecule (Rotate)
        # 
        # Positions:
        # Fe: [0,0,0], [2,0,0]
        # H2O: O at [5,0,0], H at [5, 0.76, 0.59], H at [5, -0.76, 0.59] (approx)
        
        self.atoms = Atoms('Fe2OH2', positions=[
            [0,0,0], [2,0,0],         # Fe
            [5,0,0],                  # O
            [5, 0.76, 0.59],          # H
            [5, -0.76, 0.59]          # H
        ])
        self.atoms.set_cell([10, 10, 10])
        self.atoms.set_pbc(True)
        
        # Save to file
        self.input_file = os.path.join(self.test_dir, "input.xyz")
        self.atoms.write(self.input_file)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("NepTrain.core.perturb.magnetic.Config")
    def test_magnetic_and_rotate(self, mock_config):
        """Test applying both magnetic perturbation and rotation."""
        # Setup mock config for magnetic moments
        # When Config['magmom'] is accessed, it should return the section dict {'Fe': '2.2'}
        
        def getitem(key):
            if key == 'magmom':
                return {'Fe': '2.2'}
            return MagicMock()
            
        mock_config.__getitem__.side_effect = getitem
        mock_config.__contains__.side_effect = lambda key: key == 'magmom'
        
        # Run perturb
        # We use the function interface `perturb` which takes a file path or atoms
        # But `perturb` is decorated with `iter_path_to_atoms` which expects file paths.
        # Wait, the decorator handles string paths.
        
        # Call perturb
        results = perturb(
            self.atoms,
            num=2,
            mag_mode='collinear',
            rotate_formula='H2O',
            cell_pert_fraction=0.0, # Disable cell strain to check rotation easier
            min_distance=0.0 # Disable rattle to check rotation easier
        )
        
        # Results is a list of atoms (from generator)
        results = list(results)
        self.assertEqual(len(results), 2)
        # perturbed_structures = results[0] # No, results IS the list of structures
        perturbed_structures = results
        # self.assertEqual(len(perturbed_structures), 2)
        
        for s in perturbed_structures:
            # 1. Check Magnetic Moments
            moms = s.get_initial_magnetic_moments()
            # Fe atoms (indices 0, 1) should have moments ~2.2
            # Others (O, H) should be 0
            fe_moms = moms[:2]
            other_moms = moms[2:]
            
            # Since we used collinear, moments should be exactly 2.2 (if no noise)
            # But now apply_magnetic_perturbation returns 3D vectors.
            # Default axis is Z.
            expected_fe = np.array([[0.0, 0.0, 2.2], [0.0, 0.0, 2.2]])
            expected_other = np.zeros((3, 3)) # 1 O + 2 H = 3 atoms
            
            np.testing.assert_allclose(fe_moms, expected_fe, atol=1e-5)
            np.testing.assert_allclose(other_moms, expected_other, atol=1e-5)
            
            # 2. Check Rotation
            # H2O atoms (indices 2, 3, 4)
            # Positions should be different from original if rotated
            # But internal geometry (bond lengths) should be preserved
            
            # Original O-H distances
            orig_oh1 = self.atoms.get_distance(2, 3)
            orig_oh2 = self.atoms.get_distance(2, 4)
            
            # New O-H distances
            new_oh1 = s.get_distance(2, 3)
            new_oh2 = s.get_distance(2, 4)
            
            self.assertAlmostEqual(new_oh1, orig_oh1, places=4)
            self.assertAlmostEqual(new_oh2, orig_oh2, places=4)
            
            # Check positions changed (rotation happened)
            # Note: since it's random, there's a tiny chance it's close to identity, but unlikely for 2 samples
            # Also we disabled rattle/strain, so only rotation moves atoms.
            # Fe atoms should NOT move (indices 0, 1)
            np.testing.assert_allclose(s.positions[:2], self.atoms.positions[:2])
            
            # H2O should move (indices 2, 3, 4)
            # At least one coordinate should differ
            self.assertFalse(np.allclose(s.positions[2:], self.atoms.positions[2:]))

if __name__ == '__main__':
    unittest.main()
