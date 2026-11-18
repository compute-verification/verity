from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "served_commit"
H200, H100, UNKNOWN = "NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"


class Writer:
    def __init__(self, card: str) -> None:
        self.stamp, self.members = {"card": card}, []

    def measurement(self, name, value, unit, *a, member=None, extra=None, **k) -> dict:
        self.members.append(member)
        return {"name": name, "value": value, "unit": unit, "member": member, **(extra or {})}


def load(monkeypatch: pytest.MonkeyPatch, file: str) -> types.ModuleType:
    stubs = {"torch": {}, "verity.commitments": {}, "verity.commitments.cuda": {"ROW_COMMIT": Path("/k/row_commit.cu")},
             "verity_vllm": {}, "verity_vllm.commit": {"step_rows": types.SimpleNamespace(ROW_BYTES=128)},
             "verity_vllm.commit.committer": {}, "verity_vllm.commit.committer.native_jit": {"CUDA_CFLAGS": []}}
    for name, attrs in stubs.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(__name__=name, **attrs))
    spec = importlib.util.spec_from_file_location(f"served_commit_{Path(file).stem}", DIR / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def writer(monkeypatch: pytest.MonkeyPatch, card: str) -> Writer:
    from verity_bench import kit
    w = Writer(card)
    monkeypatch.setattr(kit, "writer", lambda out, *a, **k: w)
    return w


def served(tmp_path: Path, record) -> Path:
    def reps(scale: float) -> dict:
        return {"reps": [{"metrics": {"tokens_per_s": 100 / scale * (1 + i / 100), "time_to_first_token_s": scale,
                                      "time_per_output_token_s": scale * (1 + i / 100)}} for i in range(5)]}
    doc = {"identity": {"arms": [{"id": record.ARM, "control": "stock"}], "preset": {"model": "unsloth/Llama-3.2-1B"}},
           "points": [{"point": {"id": "b32", "headline": True}, "arms": {record.ARM: reps(1.5), "stock": reps(1.0)}}]}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(doc))
    return path


@pytest.mark.parametrize("card, short", [(H200, "llama-3.2-1b-h200"), (H100, "llama-3.2-1b-h100"), ("none", "llama-3.2-1b-cpu")])
def test_the_served_arms_members_name_the_card_records_short(monkeypatch, tmp_path, card, short) -> None:
    record = load(monkeypatch, "record.py")
    w = writer(monkeypatch, card)
    record.main(served(tmp_path, record))
    assert w.members == [f"{short}-b32", f"{short}-b32", short]


def test_the_served_arm_writes_no_point_for_a_card_no_record_names(monkeypatch, tmp_path) -> None:
    record = load(monkeypatch, "record.py")
    w = writer(monkeypatch, UNKNOWN)
    with pytest.raises(SystemExit, match=re.escape(repr(UNKNOWN))):
        record.main(served(tmp_path, record))
    assert w.members == []


def floor(device: str, rows_floor) -> dict:
    sizes = {"128": {"matches_step_rows": True, "gb_per_s": 40.0}, "256": {"matches_step_rows": True, "gb_per_s": 50.0}}
    return {"device": device, "window_bytes": 1 << 20, "forms": {rows_floor.SERVED_FORM: {"sizes": sizes}}}


@pytest.mark.parametrize("card, short", [(H200, "h200"), (H100, "h100")])
def test_the_hash_rates_members_name_the_card_records_short(monkeypatch, tmp_path, card, short) -> None:
    rows_floor = load(monkeypatch, "rows_floor.py")
    w = writer(monkeypatch, card)
    rows_floor.record(floor(card, rows_floor), tmp_path)
    assert w.members == [f"{short}-256", f"{short}-128"]


def test_the_hash_rate_writes_no_point_for_a_card_no_record_names(monkeypatch, tmp_path) -> None:
    rows_floor = load(monkeypatch, "rows_floor.py")
    w = writer(monkeypatch, UNKNOWN)
    with pytest.raises(SystemExit, match=re.escape(repr(UNKNOWN))):
        rows_floor.record(floor(UNKNOWN, rows_floor), tmp_path)
    assert w.members == []
