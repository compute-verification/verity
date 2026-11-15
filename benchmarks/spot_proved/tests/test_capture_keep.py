"""`capture.py` never holds a step's tensors in memory for the coins: ``--keep all`` writes one as it is recorded only up to
``--keep-max-mib`` and spools the rest, ``--keep opened`` spools each and moves only the opened layers' to the blobs once the
coins are flipped, ``--keep none`` keeps none, and past ``--min-free-gib`` nothing more is written or spooled (an opened
tensor lost to the floor is counted as missing)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
STEP = 3


def _capture(tmp: Path, monkeypatch, **cfg):
    spec = importlib.util.spec_from_file_location("spot_capture_keep", HERE / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, capture)
    spec.loader.exec_module(capture)
    S = capture._State({"capture": str(tmp), "steps": [STEP], "mla_tokens": 0, "open": {}, "keep": "all", "hash_threads": 4, **cfg})
    S.step = STEP
    monkeypatch.setattr(capture, "S", S)
    return capture, S


def _ref(capture, t, write: bool = True) -> dict:
    ref = capture._ref(t, write=write)
    capture._drain()
    return ref


def _blob(S, ref: dict) -> Path:
    return S.root() / "blobs" / ref["sha512"][:2] / ref["sha512"]


def _spooled(S, ref: dict) -> Path:
    return S.root() / "spool" / ref["sha512"]


RUNNER = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=2, first_k_dense_replace=1)))


def _open_both(capture, S, events: list[tuple[int | None, dict]]) -> dict:
    """The coins opening layer 0 (dense) and layer 1 (MoE), with ``events`` recorded at their layers."""
    S.open = {"dense": 1, "moe": 1}
    S.events = [{"seq": i, "layer": layer, "op": "x", "in": {}, "out": {"o": ref}} for i, (layer, ref) in enumerate(events)]
    d = S.root() / f"step{STEP}"
    d.mkdir(parents=True, exist_ok=True)
    return capture._commit_and_open(RUNNER, d, "0" * 128)


def test_a_tensor_over_the_keep_cap_is_spooled_until_the_coins(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep_max_mib=1, min_free_gib=0)
    small = _ref(capture, torch.arange(1 << 17, dtype=torch.int32))
    big = _ref(capture, torch.arange(1 << 19, dtype=torch.int32))
    assert _blob(S, small).is_file() and not _spooled(S, small).exists()
    assert not _blob(S, big).exists() and _spooled(S, big).is_file() and S.spooled == {big["sha512"]: 1 << 21}
    assert not S.floor_hit


def test_past_the_disk_floor_nothing_more_is_written_or_spooled(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep_max_mib=64, min_free_gib=1 << 30)
    ref = _ref(capture, torch.arange(1 << 10, dtype=torch.int32))
    assert S.floor_hit and not _blob(S, ref).exists() and not _spooled(S, ref).exists() and ref["sha512"] in S.dropped


def test_a_kv_cache_is_hashed_but_not_written_and_a_later_read_still_writes_it(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep_max_mib=64, min_free_gib=0)
    t = torch.arange(1 << 10, dtype=torch.int32)
    cache = _ref(capture, t, write=False)
    assert cache["sha512"] and not _blob(S, cache).exists() and cache["sha512"] not in S.spooled
    again = _ref(capture, t)
    assert again["sha512"] == cache["sha512"] and _blob(S, again).is_file()


def test_keep_opened_moves_only_the_opened_layers_tensors_and_deletes_the_spool(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep="opened", min_free_gib=0)
    a, b, other = (_ref(capture, torch.full((256,), x, dtype=torch.int32)) for x in (1, 2, 3))
    assert all(_spooled(S, r).is_file() for r in (a, b, other)) and not any(_blob(S, r).exists() for r in (a, b, other))
    commit = _open_both(capture, S, [(0, a), (1, b), (None, other)])
    assert _blob(S, a).is_file() and _blob(S, b).is_file() and not _blob(S, other).exists()
    assert not (S.root() / "spool" / other["sha512"]).exists()
    assert commit["kept"] == "the opened layers' 2 tensors, 2048 bytes" and commit["missing"] == {"tensors": 0, "bytes": 0}
    capture._reset()
    assert not (S.root() / "spool").exists()


def test_an_opened_tensor_lost_to_the_disk_floor_is_counted_missing(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep="opened", min_free_gib=1 << 30)
    a = _ref(capture, torch.arange(8, dtype=torch.int32))
    commit = _open_both(capture, S, [(0, a)])
    assert not _blob(S, a).exists() and commit["missing"] == {"tensors": 1, "bytes": 32}


def test_keep_none_writes_and_spools_nothing_and_opens_no_parameter(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep="none", min_free_gib=0)
    a = _ref(capture, torch.arange(8, dtype=torch.int32))
    assert a["sha512"] and not _blob(S, a).exists() and not _spooled(S, a).exists() and not S.dropped
    S.param_of = {"layers.0.w": torch.ones(4)}
    commit = _open_both(capture, S, [(0, a)])
    assert commit["opened"] == {} and commit["kept"].startswith("nothing") and not (S.root() / f"step{STEP}" / "open").exists()
    assert json.loads((S.root() / f"step{STEP}" / "commit.json").read_text())["layers"] == {"dense": [0], "moe": [1]}


def test_keep_steps_keep_one_level_less_at_the_other_steps(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, keep="opened", keep_steps=[STEP + 1])
    assert S.step_keep == "none"
    S.step = STEP + 1
    assert S.step_keep == "opened"
    S.keep = "all"
    S.step = STEP
    assert S.step_keep == "opened" and not S.keep_all
