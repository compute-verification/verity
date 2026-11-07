"""`capture.py --keep all` writes a recorded tensor as it is recorded only up to ``--keep-max-mib`` and while the disk keeps
``--min-free-gib`` free; any other tensor waits for the coins like ``--keep opened``'s, so a whole step can't fill the disk."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
STEP = 3


def _capture(tmp: Path, monkeypatch, **cfg):
    spec = importlib.util.spec_from_file_location("spot_capture_keep", HERE / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, capture)
    spec.loader.exec_module(capture)
    S = capture._State({"capture": str(tmp), "steps": [STEP], "mla_tokens": 0, "open": {}, "keep": "all", **cfg})
    S.step = STEP
    monkeypatch.setattr(capture, "S", S)
    return capture, S


def _blob(S, ref: dict) -> Path:
    return S.root() / "blobs" / ref["sha512"][:2] / ref["sha512"]


def test_a_tensor_over_the_keep_cap_waits_for_the_coins(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep_max_mib=1, min_free_gib=0)
    small = capture._ref(torch.arange(1 << 17, dtype=torch.int32))
    big = capture._ref(torch.arange(1 << 19, dtype=torch.int32))
    assert _blob(S, small).is_file()
    assert not _blob(S, big).exists() and big["sha512"] in S.pending
    assert not S.floor_hit


def test_past_the_disk_floor_nothing_more_is_written(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep_max_mib=64, min_free_gib=1 << 30)
    ref = capture._ref(torch.arange(1 << 10, dtype=torch.int32))
    assert S.floor_hit and not _blob(S, ref).exists() and ref["sha512"] in S.pending
