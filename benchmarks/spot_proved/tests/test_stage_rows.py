"""`stage.py --rows-past-keys`: the long rows are read off the DSA indexer's top-k inputs (decode rows leading the batch, prefill
rows ending it, padding left out), and every population is cut to them: `cells` keeps only their (token, slot) rows, and
a population that walks a call's rows walks only theirs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage", HERE / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cap(stage, events: list, tensors: dict, tokens: int):
    C = stage.Capture

    class _Cap:
        cells, rows, slots, real_rows, tokens = C.cells, C.rows, C.slots, C.real_rows, C.tokens

        def opened(self, e, weights=False):
            return True

        def tensor(self, r, ref):
            return tensors[ref["sha512"]]

    cap = _Cap()
    cap.ranks, cap.step, cap.events, cap.meta = [0], 5611, {0: events}, {0: {"scheduled": {"a": tokens - 2, "b": 2}}}
    cap.coins, cap.per_call, cap.only = bytes(64), 0, None
    return cap


def _indexer_cap(stage, T: int = 8):
    t = {"len": np.array([[100], [3000], [50], [2049], [2048], [7], [99999], [0]], dtype=np.int32),
         "s": np.array([0, 0], dtype=np.int32), "e": np.array([1500, 2100], dtype=np.int32)}
    events = [{"seq": 0, "op": "_C::cooperative_topk", "layer": 0, "in": {"lengths": {"sha512": "len"}}, "out": {}},
              {"seq": 1, "op": "_C::top_k_per_row_prefill", "layer": 0,
               "in": {"numRows": 2, "rowStarts": {"sha512": "s"}, "rowEnds": {"sha512": "e"}}, "out": {}},
              {"seq": 2, "op": "_C::fused_add_rms_norm", "layer": 0,
               "in": {"input": {"shape": [T, 6144]}, "residual": {}, "weight": {}, "epsilon": 1e-5}, "out": {}}]
    return _cap(stage, events, t, tokens=6)


def test_the_long_rows_are_read_off_the_indexers_top_k_and_every_population_is_cut_to_them(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    cap = _indexer_cap(stage)
    # 6 tokens: 4 decode rows (the top-k's rows past them are padding), then 2 prefill rows
    long = stage.long_rows(cap, 2048)
    assert long == {1: 3000, 3: 2049, 5: 2100}
    cap.only = long
    assert cap.rows(0, 8) == [1, 3, 5]
    pairs = list(cap.cells("x", (0, 0), 12, 2, slots=2))
    assert sorted({p[0] for p in pairs}) == [2, 3, 6, 7, 10, 11] and len(pairs) == 12
    cap.per_call = 3
    drawn = list(cap.cells("x", (0, 0), 12, 2, slots=2))
    assert len(drawn) == 3 and all(p[0] in (2, 3, 6, 7, 10, 11) for p in drawn)
    assert [i[2][2] for i in stage.fused_add_rms_norm_instances(cap)] == [1, 3, 5]
    cap.only = None
    assert [i[2][2] for i in stage.fused_add_rms_norm_instances(cap)] == list(range(6))
