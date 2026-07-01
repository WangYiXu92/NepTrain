import textwrap
from pathlib import Path

import numpy as np

from NepTrain.core.dft.vasp.magnetic_exyz import (
    convert_paths,
    parse_incar_magmom,
    parse_outcar_magforces,
    parse_outcar_moments,
)


def _write_poscar(path: Path):
    path.write_text(textwrap.dedent("""\
    Fe
    1.0
    2.0 0.0 0.0
    0.0 2.0 0.0
    0.0 0.0 2.0
    Fe
    2
    Direct
    0.0 0.0 0.0
    0.5 0.5 0.5
    """))


def _write_outcar(path: Path, ncl=False, with_magforces=False):
    mag_blocks = """
 magnetization (x)
 # of ion       s       p       d       tot
 ------------------------------------------
    1        0.0     0.0     0.0     2.10
    2        0.0     0.0     0.0    -1.90
 tot         0.0     0.0     0.0     0.20
"""
    if ncl:
        mag_blocks = """
 magnetization (x)
 # of ion       s       p       d       tot
 ------------------------------------------
    1        0.0     0.0     0.0     1.00
    2        0.0     0.0     0.0     0.00
 tot         0.0     0.0     0.0     1.00
 magnetization (y)
 # of ion       s       p       d       tot
 ------------------------------------------
    1        0.0     0.0     0.0     0.00
    2        0.0     0.0     0.0     1.00
 tot         0.0     0.0     0.0     1.00
 magnetization (z)
 # of ion       s       p       d       tot
 ------------------------------------------
    1        0.0     0.0     0.0     2.00
    2        0.0     0.0     0.0    -2.00
 tot         0.0     0.0     0.0     0.00
"""
    magforces_block = ""
    if with_magforces:
        magforces_block = """
 magnetic forces in eV
 -------------------------------------------------------------------------------------
        0.000 0.000 0.000    0.010 0.020 0.000
        1.000 1.000 1.000   -0.010 -0.020 0.000
 -------------------------------------------------------------------------------------
"""
    path.write_text(textwrap.dedent(f"""\
    free  energy   TOTEN  =       -16.250000 eV

     TOTAL-FORCE (eV/Angst)
     -------------------------------------------------------------------
        0.000 0.000 0.000    0.100 0.200 0.300
        1.000 1.000 1.000   -0.100 -0.200 -0.300
     -------------------------------------------------------------------
    {mag_blocks}
    {magforces_block}
    """))


def test_parse_incar_magmom_collinear_and_vector(tmp_path):
    incar = tmp_path / "INCAR"
    incar.write_text("MAGMOM = 2*2.2\n")
    np.testing.assert_allclose(parse_incar_magmom(incar, 2), [[0, 0, 2.2], [0, 0, 2.2]])

    incar.write_text("MAGMOM = 1 0 0  0 1 0\n")
    np.testing.assert_allclose(parse_incar_magmom(incar, 2), [[1, 0, 0], [0, 1, 0]])


def test_parse_outcar_collinear_moment_maps_single_component_to_z(tmp_path):
    outcar = tmp_path / "OUTCAR"
    _write_outcar(outcar, ncl=False)
    spin = np.array([[0, 0, 2.2], [0, 0, -2.2]])
    moment = parse_outcar_moments(outcar, 2, spin)
    np.testing.assert_allclose(moment, [[0, 0, 2.1], [0, 0, -1.9]])


def test_parse_outcar_ncl_moment_uses_xyz_blocks(tmp_path):
    outcar = tmp_path / "OUTCAR"
    _write_outcar(outcar, ncl=True)
    spin = np.zeros((2, 3))
    moment = parse_outcar_moments(outcar, 2, spin)
    np.testing.assert_allclose(moment, [[1, 0, 2], [0, 1, -2]])


def test_convert_vasp_dir_to_gpumd_magnetic_exyz(tmp_path):
    calc = tmp_path / "config_001"
    calc.mkdir()
    _write_poscar(calc / "POSCAR")
    (calc / "INCAR").write_text("MAGMOM = 2.2 -2.2\n")
    _write_outcar(calc / "OUTCAR", ncl=False)

    output = tmp_path / "train.xyz"
    count = convert_paths([calc], output)

    assert count == 1
    text = output.read_text()
    header = text.splitlines()[1]
    assert "spin:R:3" in header
    assert "torque:R:3" in header
    assert "moment:R:3" in header
    assert "energy=-16.2500000000" in header
    assert "0.1000000000 0.2000000000 0.3000000000" in text
    assert "0.0000000000 0.0000000000 2.1000000000" in text


def test_parse_outcar_magforces(tmp_path):
    outcar = tmp_path / "OUTCAR"
    _write_outcar(outcar, ncl=False, with_magforces=True)
    torque = parse_outcar_magforces(outcar, 2)
    assert torque is not None
    np.testing.assert_allclose(torque, [[0.01, 0.02, 0.0], [-0.01, -0.02, 0.0]])


def test_parse_outcar_magforces_none_when_absent(tmp_path):
    outcar = tmp_path / "OUTCAR"
    _write_outcar(outcar, ncl=False, with_magforces=False)
    assert parse_outcar_magforces(outcar, 2) is None


def test_convert_vasp_dir_writes_nonzero_torque_from_magforces(tmp_path):
    calc = tmp_path / "config_001"
    calc.mkdir()
    _write_poscar(calc / "POSCAR")
    (calc / "INCAR").write_text("MAGMOM = 2.2 -2.2\n")
    _write_outcar(calc / "OUTCAR", ncl=False, with_magforces=True)

    output = tmp_path / "train.xyz"
    count = convert_paths([calc], output)

    assert count == 1
    text = output.read_text()
    # torque values should appear in the data lines (non-zero)
    assert "0.0100000000 0.0200000000 0.0000000000" in text
