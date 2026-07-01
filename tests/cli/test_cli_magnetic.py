import os
import subprocess
import sys

import numpy as np
from ase import Atoms
from ase.io import read, write


def test_cli_mag_flip_prob_reaches_magnetic_perturbation(tmp_path):
    input_file = tmp_path / "fe.xyz"
    output_file = tmp_path / "out.xyz"
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)
    write(input_file, atoms, format="extxyz")

    cmd = [
        sys.executable,
        "src/NepTrain/cli/cli.py",
        "perturb",
        str(input_file),
        "-n",
        "1",
        "-o",
        str(output_file),
        "--mag-mode",
        "random_collinear",
        "--mag-flip-prob",
        "1.0",
        "--skip-normal",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, env=env)

    out = read(output_file, index=0, format="extxyz")
    moms = out.get_initial_magnetic_moments()
    assert moms.shape == (2, 3)
    assert np.all(moms[:, 2] < 0.0)
    np.testing.assert_allclose(moms[:, :2], 0.0)
    np.testing.assert_allclose(np.abs(moms[0, 2]), np.abs(moms[1, 2]))
