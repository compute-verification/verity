"""Per-(request, engine step) attribution of a rank's committed sampled ids against the Match record
(`tp2_worker._tp2_attribution`) -- CPU-only: the committer is the production per-tensor host committer (`tests.check.opened`) with
`step_rows` (the rank's request->rows map) set; the sampled ids are read through `oracle_compare.OpenedReader` (opened, verified against
the run root).

  * positive: every committed sampled row is the Match record's k-th generated id of the request the map names -> ok
  * NEGATIVE (equal-count request-label swap): two requests of equal (prompt, max_tokens) relabelled a<->b on one rank keep every per-request
    identity COUNT (what coverage / composition see) -- the attribution FAILS naming the rank, the step, the row and both ids
  * a step without committed sampled ids -> not ok (fail closed; attribution cannot be established)
  * NEGATIVE: a retained id changed after the commit to agree with the Match record -> not verified, not ok, named
"""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

# tp2_worker imports torch inside the methods that need it, so the module imports torch-free.  A module-level
# `sys.modules.setdefault("torch", SimpleNamespace())` here poisoned the whole pytest session on a torch-free host: every later
# collected module that imports torch (cb_a torch_frontend -> `torch.utils`) then failed at COLLECTION ("'torch' is not a package")
# instead of skipping -- 2 collection errors, the session interrupted.  If tp2_worker ever needs torch at import, stub it in a
# fixture with monkeypatch.setitem(sys.modules, ...), never for the process.
try:
    w = importlib.import_module("verity_vllm.pipeline.tp.rank_worker")
except ModuleNotFoundError as e:  # pragma: no cover -- only if tp2_worker grows a module-level torch import
    if e.name != "torch":
        raise
    sys.modules["torch"] = types.SimpleNamespace()
    try:
        w = importlib.import_module("verity_vllm.pipeline.tp.rank_worker")
    finally:
        del sys.modules["torch"]
_cls = next(v for v in vars(w).values() if isinstance(v, type) and hasattr(v, "_tp2_attribution"))


class _Stub:
    _tp2_logical_request = _cls._tp2_logical_request
    _tp2_attribution = _cls._tp2_attribution


TOK = {"A": [11, 12, 13], "B": [21, 22, 23]}
SAMPLED = "runner.sampler/sampled_token_ids"


def _make(match: dict, *, swap: list[str] | None = None, rank: int = 1) -> _Stub:
    """The rank's committed sampled ids (int64 [2, 1] per step) committed; `s.rc` = the run commitment, `s.ids[step]` = the committed bytes."""
    from tests.check.opened import commit_host_steps

    s = _Stub()
    s.ids = {st: np.asarray([TOK["A"][st], TOK["B"][st]], dtype=np.int64).reshape(-1, 1).tobytes() for st in range(3)}
    com, s.rc = commit_host_steps({st: [(SAMPLED, b, 0, "int64", (2, 1))] for st, b in s.ids.items()})
    # two requests of EQUAL prompt length (5) and equal max_tokens (3): prefill step 0, decode steps 1, 2; engine ids carry vLLM's -<hex> suffix
    com.step_rows = {0: [("A-1a2b3c4d", 0, 5), ("B-9f8e7d6c", 5, 10)], 1: [("A-1a2b3c4d", 0, 1), ("B-9f8e7d6c", 1, 2)], 2: [("A-1a2b3c4d", 0, 1), ("B-9f8e7d6c", 1, 2)]}
    rid = {"A": "r0", "B": "r1"} if not swap else {"A": "r1", "B": "r0"}
    s._tp2_commit = {"com": com, "rank": rank, "request_ids": rid, "match_tokens": match, "swap_requests": swap}
    return s


MATCH = {"r0": [11, 12, 13], "r1": [21, 22, 23]}


def test_attribution_passes_when_every_committed_sampled_row_is_the_match_records_token_of_its_request():
    s = _make(MATCH)
    r = s._tp2_attribution(s.rc)
    assert r["ok"] is True and r["steps_checked"] == 3 and r["rows_checked"] == 6 and r["mismatches_n"] == 0
    assert r["requests"] == {"r0": {"sampled_rows_committed": 3, "match_record_generated": 3}, "r1": {"sampled_rows_committed": 3, "match_record_generated": 3}}
    assert r["value_source"]["source"] == "opened" and r["value_source"]["reads"] == 3 and r["value_source"]["failed"] == 0


def test_equal_count_request_label_swap_fails_naming_rank_step_row_and_ids():
    s = _make(MATCH, swap=["r0", "r1"])
    r = s._tp2_attribution(s.rc)
    assert r["ok"] is False and r["mismatches_n"] == 6
    # the swap is invisible to counts: both requests still have 3 sampled rows each
    assert {k: v["sampled_rows_committed"] for k, v in r["requests"].items()} == {"r0": 3, "r1": 3}
    assert r["mismatches"][0] == {"step": 0, "row": 0, "request": "r1", "k": 0, "committed": 11, "match_record": 21}
    assert r["why"].startswith("rank 1: 6 of 6 committed sampled rows disagree with the Match record")
    assert r["swap_requests_negative"] == ["r0", "r1"]


def test_step_without_committed_sampled_ids_is_not_ok():
    s = _make(MATCH)
    s._tp2_commit["com"]._layouts[2] = []
    r = s._tp2_attribution(s.rc)
    assert r["ok"] is False and r["missing_steps"] == [{"step": 2, "why": "no committed sampled ids"}]


def test_match_record_shorter_than_the_run_is_a_mismatch_and_no_record_means_not_run():
    s = _make({"r0": [11, 12], "r1": [21, 22, 23]})
    r = s._tp2_attribution(s.rc)
    assert r["ok"] is False and r["mismatches"][0]["match_record"] is None and r["mismatches"][0]["request"] == "r0" and r["mismatches"][0]["k"] == 2
    s = _make(MATCH)
    s._tp2_commit["match_tokens"] = None
    assert s._tp2_attribution(s.rc) is None


def test_a_sampled_id_changed_after_the_commit_to_agree_with_the_match_record_is_not_verified_and_not_ok():
    """The Match record says r0's 3rd token is 99; the rank committed 13.  After the commit the retained id is changed to 99 -- a compare
    of the committer's memory would attribute every row.  The opening of step 2 does not verify against the run root: not ok, named."""
    torch = pytest.importorskip("torch")
    from tests.check.opened import retained_block

    s = _make({"r0": [11, 12, 99], "r1": [21, 22, 23]})
    retained_block(s._tp2_commit["com"], 2)[:8] = torch.frombuffer(bytearray(np.int64(99).tobytes()), dtype=torch.uint8)
    r = s._tp2_attribution(s.rc)
    assert r["ok"] is False and r["steps_checked"] == 2 and r["mismatches_n"] == 1
    assert r["mismatches"][0]["step"] == 2 and r["mismatches"][0]["why"].startswith("opened value not verified")
    assert r["why"].startswith("rank 1: step 2: opened value not verified") and r["value_source"]["failed"] == 1


@pytest.mark.parametrize("rid,expect", [("A", "r0"), ("A-1a2b3c4d", "r0"), ("B-ffff", "r1"), ("zzz", "zzz")])
def test_logical_request_rule_exact_then_suffix_stripped_then_unique_prefix_else_raw(rid, expect):
    assert _make(MATCH)._tp2_logical_request(rid) == expect
