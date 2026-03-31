import unittest
import os
import shutil
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch, mock_open
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.dft.vasp.io import write_to_xyz, VaspInput
from NepTrain.core.perturb.vacancy import insert_vacancy, _filter_vacancies_for_export
from NepTrain import Config

class TestVaspIO(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_vasp_io_tmp"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        
        # Create a dummy VASP XML content (mocked later or creating minimal file if feasible)
        # But write_to_xyz uses ase.io.read(vaspxml_path), so we can mock ase.io.read
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("NepTrain.core.dft.vasp.io.ase_read")
    @patch("NepTrain.core.dft.vasp.io.ase_write")
    def test_write_to_xyz(self, mock_write, mock_read):
        """Test parsing VASP output and writing to extxyz with vacancy filtering."""
        
        # Create a mock atoms object with VASP results
        mock_atom = Atoms("Fe2O2", positions=[[0,0,0]]*4)
        mock_atom.calc = MagicMock()
        mock_atom.calc.results = {
            'stress': np.zeros((3,3)), # Note: VASP stress is usually 6-voigt or 3x3 depending on ASE version
            # ASE Vasp calculator puts stress in results
            'free_energy': -10.0
        }
        # In the code: xx, yy, zz, yz, xz, xy = - atom.calc.results['stress'] * atom.get_volume()
        # This implies results['stress'] is expected to be an array of length 6 (Voigt) or similar?
        # Actually code does: xx, yy, zz, yz, xz, xy = ...
        # So it expects unpacking 6 values. 
        # ASE standard stress is Voigt form (xx, yy, zz, yz, xz, xy).
        mock_atom.calc.results['stress'] = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
        mock_atom.get_volume = MagicMock(return_value=10.0)
        
        # Introduce a vacancy (X) to test filtering
        mock_atom_vac = Atoms("FeXO2", positions=[[0,0,0]]*4)
        mock_atom_vac.calc = MagicMock()
        mock_atom_vac.calc.results = {
            'stress': np.array([0.1]*6),
            'free_energy': -12.0
        }
        mock_atom_vac.get_volume = MagicMock(return_value=10.0)
        
        mock_read.return_value = [mock_atom, mock_atom_vac]
        
        save_path = os.path.join(self.test_dir, "output.xyz")
        
        # Call function
        result_atoms = write_to_xyz("dummy.xml", save_path, "Config_test", append=True)
        
        # Check that we processed 2 atoms
        self.assertEqual(len(result_atoms), 2)
        
        # Check info tags were added
        self.assertEqual(result_atoms[0].info['Config_type'], "Config_test1")
        self.assertEqual(result_atoms[1].info['Config_type'], "Config_test2")
        self.assertTrue('virial' in result_atoms[0].info)
        
        # Check that ase_write was called
        mock_write.assert_called_once()
        
        # Verify the list passed to ase_write was filtered
        args, kwargs = mock_write.call_args
        written_atoms_list = args[1] # 2nd arg is images
        
        self.assertEqual(len(written_atoms_list), 2)
        
        # First one should be unchanged (Fe2O2)
        self.assertEqual(written_atoms_list[0].get_chemical_formula(), "Fe2O2")
        
        # Second one should have X removed (FeXO2 -> FeO2)
        # Note: 'X' is removed by _filter_vacancies_for_export
        # Let's verify formula of second written atom
        self.assertEqual(written_atoms_list[1].get_chemical_formula(), "FeO2")

    # @patch("NepTrain.core.dft.vasp.io.os.environ") # Removed this
    @patch("NepTrain.core.dft.vasp.io.Config")
    def test_vasp_input_init(self, mock_config): # Removed mock_environ arg
        """Test VaspInput initialization and environment setup."""
        mock_config.get.return_value = "/path/to/potcar"
        
        # We need to mock Vasp superclass init to avoid needing real VASP_PP_PATH logic which might fail if env var missing
        # But we want to test that env var IS set.
        # The class sets os.environ[self.VASP_PP_PATH] = ...
        
        # VaspInput inherits from Vasp.
        # We can just instantiate it if we mock dependencies?
        # Creating a calculator usually requires parameters.
        
        # Mocking os.environ is tricky because it's a dict-like object but also os.environ.
        # The error "expected 2 arguments, got 3" usually happens when mocking __setitem__ with autospec/side_effects incorrectly?
        # Or maybe mock_environ is a Mock, and __setitem__ is called with (key, value).
        # Let's verify what mock_environ is.
        # It's a MagicMock by default if patched.
        
        # Simpler approach: Don't mock os.environ completely. Just patch dict item setting?
        # Or use patch.dict?
        with patch.dict(os.environ, {}, clear=True):
             with patch("ase.calculators.vasp.Vasp.__init__", autospec=True) as mock_super_init:
                
                 def side_effect(self, *args, **kwargs):
                     self.input_params = {}
                     self.VASP_PP_PATH = 'VASP_PP_PATH'
                     
                 mock_super_init.side_effect = side_effect
                 
                 calc = VaspInput()
                 
                 # Check Config was queried
                 mock_config.get.assert_called_with("environ", "potcar_path")
                 
                 # Check input_params
                 self.assertEqual(calc.input_params["setups"]["base"], "recommended")
                 self.assertEqual(calc.input_params["pp"], '')
                 
                 # Check env var set
                 # VaspInput uses self.VASP_PP_PATH which we set to 'VASP_PP_PATH'
                 self.assertEqual(os.environ.get('VASP_PP_PATH'), "/path/to/potcar")

    # Remove the mock_environ argument from signature since we use patch.dict inside
    # But wait, the decorator @patch("NepTrain.core.dft.vasp.io.os.environ") is still there.
    # We should remove it.

if __name__ == '__main__':
    unittest.main()
