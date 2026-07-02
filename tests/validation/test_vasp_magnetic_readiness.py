"""Tests for VASP constrained-magnetic readiness check (B2).

Validates INCAR parameters required for non-collinear constrained-moment
VASP calculations that produce torque (magnetic forces) labels for
GPUMD magnetic NEP training.
"""
from __future__ import annotations

import textwrap
from pathlib import Path


def _write_incar(directory: Path, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    incar = directory / "INCAR"
    incar.write_text(textwrap.dedent(content))
    return incar


def test_good_noncollinear_constrained_incar_passes(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    _write_incar(
        tmp_path,
        """
        SYSTEM = Fe
        LNONCOLLINEAR = .TRUE.
        I_CONSTRAINED_M = 1
        M_CON = 0.0 0.0 1.0 0.0 0.0 -1.0
        MAGMOM = 0.0 0.0 2.2 0.0 0.0 -2.2
        """,
    )

    report = check_magnetic_readiness(tmp_path)
    assert report["ready"] is True
    assert report["noncollinear"] is True
    assert report["constrained_m"] is True
    assert report["m_con_present"] is True


def test_missing_lconstrain_m_flagged(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    _write_incar(
        tmp_path,
        """
        SYSTEM = Fe
        LNONCOLLINEAR = .TRUE.
        MAGMOM = 0.0 0.0 2.2 0.0 0.0 -2.2
        """,
    )

    report = check_magnetic_readiness(tmp_path)
    assert report["ready"] is False
    assert "I_CONSTRAINED_M" in report["missing_params"]


def test_missing_m_con_flagged(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    _write_incar(
        tmp_path,
        """
        SYSTEM = Fe
        LNONCOLLINEAR = .TRUE.
        I_CONSTRAINED_M = 1
        MAGMOM = 0.0 0.0 2.2 0.0 0.0 -2.2
        """,
    )

    report = check_magnetic_readiness(tmp_path)
    assert report["ready"] is False
    assert "M_CON" in report["missing_params"]


def test_collinear_magmom_without_noncollinear_flagged(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    _write_incar(
        tmp_path,
        """
        SYSTEM = Fe
        I_CONSTRAINED_M = 1
        M_CON = 0.0 0.0 1.0 0.0 0.0 -1.0
        MAGMOM = 2.2 -2.2
        """,
    )

    report = check_magnetic_readiness(tmp_path)
    assert report["ready"] is False
    assert not report["noncollinear"]
    assert "LNONCOLLINEAR" in report["missing_params"]


def test_magmom_scalar_n_values_noted_but_ok(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    _write_incar(
        tmp_path,
        """
        SYSTEM = Fe
        LNONCOLLINEAR = .TRUE.
        I_CONSTRAINED_M = 1
        M_CON = 0.0 0.0 1.0 0.0 0.0 -1.0
        MAGMOM = 2.2 -2.2
        """,
    )

    report = check_magnetic_readiness(tmp_path)
    assert report["ready"] is True
    assert report["magmom_vector"] is False
    assert "scalar" in " ".join(str(w).lower() for w in report.get("warnings", []))


def test_magmom_3N_vector_preferred(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    _write_incar(
        tmp_path,
        """
        SYSTEM = Fe
        LNONCOLLINEAR = .TRUE.
        I_CONSTRAINED_M = 1
        M_CON = 0.0 0.0 1.0 0.0 0.0 -1.0
        MAGMOM = 0.0 0.0 2.2 0.0 0.0 -2.2
        """,
    )

    report = check_magnetic_readiness(tmp_path)
    assert report["ready"] is True
    assert report["magmom_vector"] is True


def test_nonexistent_directory_reports_missing(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    report = check_magnetic_readiness(tmp_path / "does_not_exist")
    assert report["ready"] is False
    assert report["missing_inputs"]


def test_missing_incar_file_reports_error(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    report = check_magnetic_readiness(tmp_path)
    assert report["ready"] is False
    assert "INCAR" in report.get("missing_inputs", [])


def test_lnoncollinear_false_explicitly_flagged(tmp_path):
    from NepTrain.core.validation.vasp_magnetic import check_magnetic_readiness

    _write_incar(
        tmp_path,
        """
        SYSTEM = Fe
        LNONCOLLINEAR = .FALSE.
        I_CONSTRAINED_M = 1
        M_CON = 0.0 0.0 1.0
        """,
    )

    report = check_magnetic_readiness(tmp_path)
    assert report["ready"] is False
    assert report["noncollinear"] is False
