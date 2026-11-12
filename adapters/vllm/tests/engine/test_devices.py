"""`verity_vllm.engine.profiles.devices.device_name`: the name a GPU of a workload's row class reports, read from the
workload's sweep plan; None where the plan, its row id or the class's name is absent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.engine.profiles import devices

H200_ROW = "glm-5.3__fp8__h200__tp8__b1__i256__o32"
L40S_ROW = "smollm2-135m__bf16__l40s__tp1__b1__i256__o32"


def _workload(tmp_path: Path, doc: dict[str, Any]) -> str:
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_a_listed_class_reports_its_devices_name(tmp_path: Path) -> None:
    assert devices.device_name(_workload(tmp_path, {"sweep": {"row_id": H200_ROW}})) == "NVIDIA H200"
    assert devices.device_name_of_row(H200_ROW) == "NVIDIA H200"


@pytest.mark.parametrize("doc", [
    {},
    {"sweep": None},
    {"sweep": {}},
    {"sweep": 0},
    {"sweep": ""},
    {"sweep": False},
    {"sweep": []},
    {"sweep": {"row_id": None}},
    {"sweep": {"row_id": L40S_ROW}},
], ids=["no sweep", "null sweep", "empty sweep", "zero", "empty text", "false", "empty list", "null row id", "unlisted class"])
def test_no_name_without_a_listed_class(tmp_path: Path, doc: dict[str, Any]) -> None:
    assert devices.device_name(_workload(tmp_path, doc)) is None


@pytest.mark.parametrize("workload", [None, ""])
def test_no_workload_reads_no_file(monkeypatch: pytest.MonkeyPatch, workload: str | None) -> None:
    def refuse(path: str) -> Path:
        raise AssertionError(f"read {path}")

    monkeypatch.setattr(devices, "workload_file", refuse)
    assert devices.device_name(workload) is None
