from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

TC_PROBE = Path(__file__).resolve().parents[3] / "calibration" / "tc_probe"
if str(TC_PROBE) not in sys.path:
    sys.path.insert(0, str(TC_PROBE))
spec = importlib.util.spec_from_file_location("tc_probe", TC_PROBE / "tc_probe.py")
tc_probe = importlib.util.module_from_spec(spec)
sys.modules["tc_probe"] = tc_probe
spec.loader.exec_module(tc_probe)


H200, H100, UNKNOWN = "NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"


class Writer:
    def __init__(self) -> None:
        self.members = []

    def measurement(self, name, value, unit, *a, member=None, extra=None, **k) -> dict:
        self.members.append(member)
        return {"name": name, "value": value, "unit": unit, "member": member, **(extra or {})}


def writer(monkeypatch: pytest.MonkeyPatch) -> Writer:
    from verity_bench import kit
    w = Writer()
    monkeypatch.setattr(kit, "writer", lambda out, *a, **k: w)
    return w

CANARY = {"canary": {"asserted_elements": 64, "total_mismatches_by_model": {"m": 0}}}


@pytest.mark.parametrize("card, member", [(H200, "mma_f16-h200"), (H100, "mma_f16-h100")])
def test_a_member_names_the_card_records_short(monkeypatch, tmp_path, card, member) -> None:
    w = writer(monkeypatch)
    tc_probe.record(tmp_path, "mma_f16", card, CANARY, ["m"])
    assert w.members == [member]


def test_a_card_no_record_names_writes_no_point_and_is_named(monkeypatch, tmp_path) -> None:
    w = writer(monkeypatch)
    with pytest.raises(SystemExit, match=re.escape(repr(UNKNOWN))):
        tc_probe.record(tmp_path, "mma_f16", UNKNOWN, CANARY, ["m"])
    assert w.members == []
