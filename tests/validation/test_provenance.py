"""Tests for provenance metadata on extxyz frames (C).

Validates that provenance info survives write-read round-trip
through extxyz and that the helper sets all expected fields.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.io import read as ase_read


def test_set_provenance_writes_info_fields():
    from NepTrain.core.validation.provenance import set_provenance

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    set_provenance(
        atoms,
        generation=3,
        source_stage="gpumd",
        temperature=800,
        md_time_ps=500,
        selection_score=2.31,
        spin_mode="non_collinear",
        torque_source="vasp_magnetic_forces",
    )

    assert atoms.info["provenance_generation"] == 3
    assert atoms.info["provenance_source_stage"] == "gpumd"
    assert atoms.info["provenance_temperature"] == 800
    assert atoms.info["provenance_md_time_ps"] == 500
    assert atoms.info["provenance_selection_score"] == 2.31
    assert atoms.info["provenance_spin_mode"] == "non_collinear"
    assert atoms.info["provenance_torque_source"] == "vasp_magnetic_forces"


def test_provenance_survives_extxyz_roundtrip(tmp_path):
    from NepTrain.core.validation.provenance import set_provenance

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.arrays["force"] = np.zeros((2, 3))
    set_provenance(atoms, generation=2, source_stage="dft")

    xyz = tmp_path / "out.xyz"
    ase_write(xyz, atoms, format="extxyz")
    back = ase_read(xyz, format="extxyz")

    assert back.info["provenance_generation"] == 2
    assert back.info["provenance_source_stage"] == "dft"


def test_write_magnetic_exyz_frame_includes_provenance(tmp_path):
    from NepTrain.core.dft.vasp.magnetic_exyz import write_magnetic_exyz_frame
    from NepTrain.core.validation.provenance import set_provenance

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    spin = np.array([[0.0, 0.0, 2.2], [0.0, 0.0, -2.2]])
    set_provenance(atoms, generation=1, source_stage="dft", torque_source="vasp_magnetic_forces")

    xyz = tmp_path / "frame.xyz"
    with xyz.open("w") as f:
        write_magnetic_exyz_frame(f, atoms, energy=-12.0, forces=np.zeros((2, 3)),
                                  spin=spin, moment=spin * 0.95, torque=np.zeros((2, 3)))

    back = ase_read(xyz, format="extxyz")
    assert back.info["provenance_generation"] == 1
    assert back.info["provenance_torque_source"] == "vasp_magnetic_forces"
