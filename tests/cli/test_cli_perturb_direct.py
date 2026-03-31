import subprocess
import os

def test_cli_perturb_direct():
    input_file = "test_cli_direct.xyz"
    output_file = "test_cli_direct_out.xyz"
    with open(input_file, "w") as f:
        f.write("1\n\nH 0 0 0\n")
        
    try:
        # Run CLI command
        # Note: we need to use 'perturb' subcommand
        cmd = ["uv", "run", "python", "src/NepTrain/cli/cli.py", "perturb", input_file, "-n", "1", "-o", output_file]
        subprocess.check_call(cmd)
        
        assert os.path.exists(output_file), "Output file should be created via CLI"
        
        # Verify content
        with open(output_file, "r") as f:
            content = f.read()
        assert "H" in content
        
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    test_cli_perturb_direct()
