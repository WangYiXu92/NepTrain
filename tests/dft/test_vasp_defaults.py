import unittest
import numpy as np
from ase import Atoms
from unittest.mock import patch, MagicMock
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.perturb.magnetic import ensure_magnetic_configuration
from NepTrain import Config

class TestVaspDefaults(unittest.TestCase):
    
    def setUp(self):
        self.atoms = Atoms('Fe2', positions=[[0,0,0], [2,0,0]], cell=[4,4,4], pbc=True)
        
    @patch('NepTrain.core.perturb.magnetic.Config')
    def test_ensure_magnetic_configuration_defaults(self, mock_config):
        # Setup mock config to behave like a dict for 'magmom' key
        # Config['magmom'] returns a dict-like object with items()
        
        # Configure __contains__ to return True for 'magmom'
        mock_config.__contains__.side_effect = lambda x: x == 'magmom'
        
        # Configure __getitem__ to return a dict for 'magmom'
        mock_config.__getitem__.side_effect = lambda x: {'Fe': '3.0'} if x == 'magmom' else MagicMock()
        
        # Ensure atoms have no initial moments
        self.atoms.set_initial_magnetic_moments(None)
        
        # Call function
        ensure_magnetic_configuration(self.atoms)
        
        # Verify moments were set
        moms = self.atoms.get_initial_magnetic_moments()
        
        # The function uses apply_magnetic_perturbation which sets vector moments by default (along z-axis)
        # So we expect [[0, 0, 3.0], [0, 0, 3.0]]
        
        print(f"Moments shape: {moms.shape}")
        print(f"Moments: {moms}")
        
        expected = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
        np.testing.assert_array_almost_equal(moms, expected)

if __name__ == '__main__':
    unittest.main()
