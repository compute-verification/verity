"""[R15 tp, rev-cb M-0359 T6-3] per-(request, engine step) attribution of a rank's committed sampled ids against the Match record
(`tp2_worker._tp2_attribution`) -- CPU-only, torch-free: the committer is a stub with `step_rows` (the rank's request->rows map) and
`_layouts` (committed TensorMeta with retained host bytes).

  * positive: every committed sampled row is the Match record's k-th generated id of the request the map names -> ok
  * NEGATIVE (equal-count request-label swap): two requests of equal (prompt, max_tokens) relabelled a<->b on one rank keep every per-request
    identity COUNT (what coverage / composition see) -- the attribution FAILS naming the rank, the step, the row and both ids
  * a step without committed sampled ids -> not ok (fail closed; attribution cannot be established)
"""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

# tp2_worker imports torch inside the methods that need it, so the module imports torch-free.  [rel int-4] a module-level
# `sys.modules.setdefault("torch", SimpleNamespace())` here poisoned the whole pytest session on a torch-free host: every later
# collected module that imports torch (cb_a torch_frontend -> `torch.utils`) then failed at COLLECTION ("'torch' is not a package")
# instead of skipping -- 2 collection errors, the session interrupted.  If tp2_worker ever needs torch at import, stub it in a
# fixture with monkeypatch.setitem(sys.modules, ...), never for the process.
try:
    w = importlib.import_module("verity_capture.bench.tp2_worker")
except ModuleNotFoundError as e:  # pragma: no cover -- only if tp2_worker grows a module-level torch import
    if e.name != "torch":
        raise
    sys.modules["torch"] = types.SimpleNamespace()
    try:
        w = importlib.import_module("verity_capture.bench.tp2_worker")
    finally:
        del sys.modules["torch"]
_cls = next(v for v in vars(w).values() if isinstance(v, type) and hasattr(v, "_tp2_attribution"))


class _Meta:
    def __init__(self, ids: list[int]) -> None:
        self.name = "runner.sampler/sampled_token_ids"
        self.dtype = "int64"
        self.shape = (len(ids), 1)
        self.host = memoryview(np.asarray(ids, dtype=np.int64).reshape(-1, 1).tobytes())
        self.nbytes = 8 * len(ids)
        self.stream_off = 0


class _Com:
    pass


class _Stub:
    _tp2_logical_request = _cls._tp2_logical_request
    _tp2_attribution = _cls._tp2_attribution


TOK = {"A": [11, 12, 13], "B": [21, 22, 23]}


def _make(match: dict, *, swap: list[str] | None = None, rank: int = 1) -> _Stub:
    s = _Stub()
    com = _Com()
    # two requests of EQUAL prompt length (5) and equal max_tokens (3): prefill step 0, decode steps 1, 2; engine ids carry vLLM's -<hex> suffix
    com.step_rows = {0: [("A-1a2b3c4d", 0, 5), ("B-9f8e7d6c", 5, 10)], 1: [("A-1a2b3c4d", 0, 1), ("B-9f8e7d6c", 1, 2)], 2: [("A-1a2b3c4d", 0, 1), ("B-9f8e7d6c", 1, 2)]}
    com._layouts = {st: [_Meta([TOK["A"][st], TOK["B"][st]])] for st in range(3)}
    rid = {"A": "r0", "B": "r1"} if not swap else {"A": "r1", "B": "r0"}
    s._tp2_commit = {"com": com, "rank": rank, "request_ids": rid, "match_tokens": match, "swap_requests": swap}
    return s


MATCH = {"r0": [11, 12, 13], "r1": [21, 22, 23]}


def test_attribution_passes_when_every_committed_sampled_row_is_the_match_records_token_of_its_request():
    r = _make(MATCH)._tp2_attribution(None)
    assert r["ok"] is True and r["steps_checked"] == 3 and r["rows_checked"] == 6 and r["mismatches_n"] == 0
    assert r["requests"] == {"r0": {"sampled_rows_committed": 3, "match_record_generated": 3}, "r1": {"sampled_rows_committed": 3, "match_record_generated": 3}}


def test_equal_count_request_label_swap_fails_naming_rank_step_row_and_ids():
    r = _make(MATCH, swap=["r0", "r1"])._tp2_attribution(None)
    assert r["ok"] is False and r["mismatches_n"] == 6
    # the swap is invisible to counts: both requests still have 3 sampled rows each
    assert {k: v["sampled_rows_committed"] for k, v in r["requests"].items()} == {"r0": 3, "r1": 3}
    assert r["mismatches"][0] == {"step": 0, "row": 0, "request": "r1", "k": 0, "committed": 11, "match_record": 21}
    assert r["why"].startswith("rank 1: 6 of 6 committed sampled rows disagree with the Match record")
    assert r["swap_requests_negative"] == ["r0", "r1"]


def test_step_without_committed_sampled_ids_is_not_ok():
    s = _make(MATCH)
    s._tp2_commit["com"]._layouts[2] = []
    r = s._tp2_attribution(None)
    assert r["ok"] is False and r["missing_steps"] == [{"step": 2, "why": "no committed sampled ids"}]


def test_match_record_shorter_than_the_run_is_a_mismatch_and_no_record_means_not_run():
    r = _make({"r0": [11, 12], "r1": [21, 22, 23]})._tp2_attribution(None)
    assert r["ok"] is False and r["mismatches"][0]["match_record"] is None and r["mismatches"][0]["request"] == "r0" and r["mismatches"][0]["k"] == 2
    s = _make(MATCH)
    s._tp2_commit["match_tokens"] = None
    assert s._tp2_attribution(None) is None


@pytest.mark.parametrize("rid,expect", [("A", "r0"), ("A-1a2b3c4d", "r0"), ("B-ffff", "r1"), ("zzz", "zzz")])
def test_logical_request_rule_exact_then_suffix_stripped_then_unique_prefix_else_raw(rid, expect):
    assert _make(MATCH)._tp2_logical_request(rid) == expect
