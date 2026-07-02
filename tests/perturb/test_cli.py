import unittest
import os
import shutil
import subprocess
import sys
from ase.build import bulk
from ase.io import write

# Path to src
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))

class TestPerturbCLI(unittest.TestCase):
    def setUp(self):
        # Create a dummy structure
        self.atoms = bulk('Fe', 'bcc', a=2.87)
        self.input_file = "test_cli_input.xyz"
        write(self.input_file, self.atoms)
        self.output_file = "test_cli_output.xyz"
        
        # Clean up output if exists
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        # Clean up files
        if os.path.exists(self.input_file):
            try:
                os.remove(self.input_file)
            except:
                pass
        if os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
            except:
                pass

    def run_cli(self, args):
        env = os.environ.copy()
        # Add src to PYTHONPATH so NepTrain module is found
        env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
        
        # Command to run: uv run python -m NepTrain.cli.cli perturb ...
        cmd = ["uv", "run", "python", "-m", "NepTrain.cli.cli", "perturb", self.input_file, "-o", self.output_file] + args
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return result

    def test_basic_perturb(self):
        print("Testing basic perturbation...")
        result = self.run_cli(["-n", "2", "--cell", "0.05", "-d", "0.1"])
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file), "Output file not created")
        self.assertGreater(os.path.getsize(self.output_file), 0)

    def test_sobol_sampler(self):
        print("Testing Sobol sampler...")
        result = self.run_cli(["-n", "2", "--sampler", "sobol"])
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

    def test_state_file(self):
        print("Testing state file...")
        state_file = "test_state.json"
        if os.path.exists(state_file):
            os.remove(state_file)
            
        result = self.run_cli(["-n", "2", "--sampler", "sobol", "--state-file", state_file])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(state_file), "State file not created")
        
        # Resume
        result = self.run_cli(["-n", "2", "--sampler", "sobol", "--state-file", state_file, "--resume"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        
        if os.path.exists(state_file):
            os.remove(state_file)

    def test_stacking_fault_cli(self):
        print("Testing stacking fault CLI...")
        # Test with random shift
        result = self.run_cli(["-n", "1", "--stacking-fault", "--sf-indices", "1,1,1", "--sf-shift", "random"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))
        
        # Test with specific shift
        result = self.run_cli(["-n", "1", "--stacking-fault", "--sf-indices", "1,1,0", "--sf-shift", "0.5,0,0"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")

    def test_dislocation_cli(self):
        print("Testing dislocation CLI...")
        result = self.run_cli(["-n", "1", "--dislocation", "--dislocation-type", "edge", "--dislocation-burgers", "1,0,0"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

    def test_grain_boundary_cli(self):
        print("Testing grain boundary CLI...")
        # Using a simple GB configuration
        result = self.run_cli(["-n", "1", "--gb", "--gb-axis", "0,0,1", "--gb-angle", "36.87", "--no-validate"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

    def test_twinning_cli(self):
        print("Testing twinning CLI...")
        # (1,1,2) twin in BCC Fe
        result = self.run_cli(["-n", "1", "--twinning", "--twinning-indices", "1,1,2"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

    def test_surface_cli(self):
        print("Testing surface CLI...")
        result = self.run_cli(["-n", "1", "--surface", "--indices", "1,0,0", "--vacuum", "10.0"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

    def test_amorphous_cli(self):
        print("Testing amorphous CLI...")
        result = self.run_cli(["-n", "1", "--amorphous"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

    def test_magnetic_cli(self):
        print("Testing magnetic CLI...")
        # Test magnetic shuffling
        result = self.run_cli(["-n", "1", "--mag-mode", "random_collinear"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

    def test_rotate_cli(self):
        print("Testing rotate CLI...")
        result = self.run_cli(["-n", "1", "--rotate-formula", "Fe"])
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

if __name__ == '__main__':
    unittest.main()
