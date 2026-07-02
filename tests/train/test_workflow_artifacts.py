import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io import write


def test_validate_xyz_artifact_reports_frame_count_and_required_arrays(tmp_path):
    from NepTrain.core.train.artifacts import validate_xyz_artifact

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.arrays["force"] = np.zeros((2, 3))
    atoms.arrays["spin"] = np.array([[0.0, 0.0, 2.2], [0.0, 0.0, -2.2]])
    xyz = tmp_path / "train.xyz"
    write(xyz, atoms, format="extxyz")

    report = validate_xyz_artifact(xyz, required_arrays=("force", "spin", "torque"))

    assert report["exists"] is True
    assert report["valid"] is False
    assert report["n_frames"] == 1
    assert report["missing_arrays"] == {"torque": 1}
    assert report["required_arrays"] == ["force", "spin", "torque"]


def test_write_stage_report_records_status_artifacts_and_summary(tmp_path):
    from NepTrain.core.train.artifacts import write_stage_report

    artifact = tmp_path / "selected.xyz"
    write(artifact, Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True), format="extxyz")

    report = write_stage_report(
        tmp_path,
        generation=2,
        stage="select",
        status="completed",
        artifacts={"selected": artifact},
        summary={"n_selected": 1},
    )

    report_path = tmp_path / "workflow_reports" / "Generation-2" / "select.json"
    assert report_path.exists()
    on_disk = json.loads(report_path.read_text())
    assert on_disk["stage"] == "select"
    assert on_disk["status"] == "completed"
    assert on_disk["summary"] == {"n_selected": 1}
    assert on_disk["artifacts"]["selected"]["exists"] is True
    assert report == on_disk


def test_status_includes_latest_stage_reports(tmp_path, capsys):
    from NepTrain.core.train.artifacts import write_stage_report
    from NepTrain.core.train.status import check_status
    from ruamel.yaml import YAML

    config = {
        "generation": 1,
        "current_job": "dft",
        "restart": True,
        "gpumd": {"step_times": [10, 100]},
        "dft": {"software": "vasp"},
    }
    with (tmp_path / "restart.yaml").open("w", encoding="utf-8") as handle:
        YAML().dump(config, handle)

    write_stage_report(
        tmp_path,
        generation=1,
        stage="select",
        status="completed",
        artifacts={},
        summary={"n_selected": 7},
    )

    status = check_status(str(tmp_path))
    captured = capsys.readouterr().out

    assert "stage_reports" in status
    assert status["stage_reports"]["select"]["summary"]["n_selected"] == 7
    assert "select" in captured
    assert "completed" in captured


def test_skipped_stage_report_is_not_marked_invalid_for_missing_artifact(tmp_path):
    from NepTrain.core.train.artifacts import write_stage_report

    report = write_stage_report(
        tmp_path,
        generation=1,
        stage="select",
        status="skipped",
        artifacts={"trajectorys": tmp_path / "missing.xyz"},
        summary={"reason": "empty trajectory"},
    )

    assert report["valid_artifacts"] is True
    assert report["artifacts"]["trajectorys"]["exists"] is False


def test_load_stage_reports_uses_numeric_generation_order(tmp_path):
    from NepTrain.core.train.artifacts import load_stage_reports, write_stage_report

    write_stage_report(tmp_path, generation=2, stage="select", status="completed", summary={"n": 2})
    write_stage_report(tmp_path, generation=10, stage="select", status="completed", summary={"n": 10})

    reports = load_stage_reports(tmp_path)

    assert reports["select"]["generation"] == 10
    assert reports["select"]["summary"]["n"] == 10


def test_validate_xyz_without_required_arrays_counts_frames_without_ase_read(monkeypatch, tmp_path):
    import NepTrain.core.train.artifacts as artifacts

    xyz = tmp_path / "trajectory.xyz"
    xyz.write_text(
        "1\nframe 1\nFe 0 0 0\n"
        "2\nframe 2\nFe 0 0 0\nFe 1 0 0\n",
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ASE read should not be used for frame-only validation")

    monkeypatch.setattr(artifacts, "ase_read", fail_if_called)

    report = artifacts.validate_xyz_artifact(xyz)

    assert report["valid"] is True
    assert report["n_frames"] == 2
