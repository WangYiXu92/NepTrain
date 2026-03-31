import unittest
import subprocess
import sys
import os

# Ensure we can import modules if needed, but CLI test runs via subprocess
# But subprocess calls "sys.executable -m NepTrain.cli.cli"
# This requires NepTrain to be in PYTHONPATH.
# When running from root, current dir is in path.
# If we run from test_scripts/cli, we need to set PYTHONPATH.
# However, user will run this test script.
# We should probably set PYTHONPATH in subprocess env.

class TestCLIConsistency(unittest.TestCase):
    """
    Test suite to verify the consistency and functionality of the NepTrain CLI.
    """
    
    def run_cli_help(self, subcommand=None):
        """Helper to run --help for main command or subcommands."""
        # Setup env to include src
        env = os.environ.copy()
        src_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = src_path + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = src_path
            
        cmd = [sys.executable, "-m", "NepTrain.cli.cli"]
        if subcommand:
            cmd.append(subcommand)
        cmd.append("--help")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return result
        
    def test_perturb_cli_execution(self):
        """Test actual execution of perturb command via CLI."""
        # Create a dummy file
        with open("test_cli_struct.xyz", "w") as f:
            f.write("2\n\nFe 0.0 0.0 0.0\nFe 2.0 0.0 0.0\n")
            
        # Setup env
        env = os.environ.copy()
        src_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = src_path + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = src_path

        cmd = [sys.executable, "-m", "NepTrain.cli.cli", "perturb", "test_cli_struct.xyz", "--num", "1", "-o", "test_cli_out.xyz"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        # Clean up
        if os.path.exists("test_cli_struct.xyz"):
            os.remove("test_cli_struct.xyz")
        if os.path.exists("test_cli_out.xyz"):
            os.remove("test_cli_out.xyz")
            
        self.assertEqual(result.returncode, 0, f"Perturb CLI failed: {result.stderr}")
        self.assertIn("100%", result.stdout or result.stderr) # Progress bar usually prints to stderr

if __name__ == '__main__':
    unittest.main()
