"""`verity_vllm.engine.profiles.devices.device_name`: the name a GPU of a workload's row class reports, read from the
workload's sweep plan and the card record whose `short` is the class; None where the plan, its row id or the class's record is
absent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.engine.profiles import devices

H200_ROW = "glm-5.3__fp8__h200__tp8__b1__i256__o32"
H100_ROW = "llama32-1b__bf16__h100__tp1__b4__i256__o32__mixed__greedy__bi-eager"
L40S_ROW = "smollm2-135m__bf16__l40s__tp1__b1__i256__o32"


def _workload(tmp_path: Path, doc: dict[str, Any]) -> str:
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_a_listed_class_reports_its_devices_name(tmp_path: Path) -> None:
    assert devices.device_name(_workload(tmp_path, {"sweep": {"row_id": H200_ROW}})) == "NVIDIA H200"
    assert devices.device_name_of_row(H200_ROW) == "NVIDIA H200"
    assert devices.device_name_of_row(H100_ROW) == "NVIDIA H100 80GB HBM3"


def test_a_class_and_a_reported_name_are_read_from_the_card_records() -> None:
    """A class is the record whose `short` is its token; a token no record's `short` is (rtxpro6000, whose record's is
    rtx-pro-6000) names no device.  A reported name is registered on the record whose `driver` holds it, and on none otherwise."""
    assert [devices.class_device_name(t) for t in ("h200", "h100", "b200", "rtxpro6000", "l40s")] == \
        ["NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA B200", None, None]
    assert [devices.card_sku(n) for n in ("NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA B200", "NVIDIA H200 NVL")] == \
        ["h200-sxm-141gb", "h100-sxm-80gb", "b200-sxm-180gb", None]


def test_a_name_two_records_hold_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    two = ({"kind": "gpu", "sku": "a", "driver": [{"value": "G"}]}, {"kind": "gpu", "sku": "b", "driver": [{"value": "G"}]})
    monkeypatch.setattr(devices, "_gpu_records", lambda: two)
    with pytest.raises(ValueError, match=r"'G' is in the `driver` of 2 card records"):
        devices.card_sku("G")


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
