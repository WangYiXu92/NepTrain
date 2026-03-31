import unittest
import os
import shutil
import sys
from unittest.mock import MagicMock, patch, mock_open

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from NepTrain.core.dft.qe.io import get_pp_files, read_qe_input

class TestQEIO(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_qe_io_tmp"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
            
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_pp_files(self):
        """Test finding UPF files and identifying elements."""
        # Create dummy UPF files
        # 1. Standard naming
        with open(os.path.join(self.test_dir, "Fe.pbe-n-kjpaw_psl.1.0.0.UPF"), "w") as f:
            f.write("<UPF version=\"2.0.1\">\n  <PP_HEADER\n    element=\"Fe\"\n")
            
        # 2. Lowercase extension, parsing from header
        with open(os.path.join(self.test_dir, "my_pseudo.upf"), "w") as f:
            f.write("<UPF>\n  element=\"O\"\n")
            
        # 3. Parsing from filename fallback
        with open(os.path.join(self.test_dir, "Si.pz-vbc.UPF"), "w") as f:
            f.write("Empty or weird header")
            
        pp_files = get_pp_files(self.test_dir)
        
        self.assertIn("Fe", pp_files)
        self.assertEqual(pp_files["Fe"], "Fe.pbe-n-kjpaw_psl.1.0.0.UPF")
        
        self.assertIn("O", pp_files)
        self.assertEqual(pp_files["O"], "my_pseudo.upf")
        
        self.assertIn("Si", pp_files)
        self.assertEqual(pp_files["Si"], "Si.pz-vbc.UPF")

    def test_read_qe_input_fallback(self):
        """Test fallback parser for QE input when ASE fails."""
        # Create a partial QE input file (without atomic positions, which ASE might require)
        input_content = """
&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch',
    pseudo_dir = './',
    outdir = './tempdir/'
/
&SYSTEM
    ibrav=  2, celldm(1) =10.20, nat=  2, ntyp= 1,
    ecutwfc = 60.0, 
    ecutrho = 240.0
/
&ELECTRONS
    conv_thr =  1.0d-8
    mixing_beta = 0.7
/
ATOMIC_SPECIES
 Si  28.086  Si.pbe-n-kjpaw_psl.1.0.0.UPF
"""
        input_path = os.path.join(self.test_dir, "scf.in")
        with open(input_path, "w") as f:
            f.write(input_content)
            
        # Mock ase.io.espresso.read_espresso_in to fail
        with patch("NepTrain.core.dft.qe.io.read_espresso_in", side_effect=Exception("ASE Parse Error")):
            input_data, pseudos = read_qe_input(input_path)
            
            # Check parsing
            # According to the code read, it parses sections into input_data dict
            
            # Note: The fallback parser puts keys into sections (control, system, etc.)
            self.assertIn('control', input_data)
            self.assertEqual(input_data['control']['calculation'], 'scf')
            
            self.assertIn('system', input_data)
            self.assertEqual(input_data['system']['ibrav'], 2)
            self.assertEqual(input_data['system']['ecutwfc'], 60.0)
            
            # It also handles booleans if present, and numbers.

if __name__ == '__main__':
    unittest.main()
