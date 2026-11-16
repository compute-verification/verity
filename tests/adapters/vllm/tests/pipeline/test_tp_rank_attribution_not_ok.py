"""`verity_vllm.pipeline.tp.rank_worker`: what a rank's attribution of its committed sampled ids to the Match record answers
when it cannot say yes (`TP2CaptureWorkerExtension._tp2_attribution`).  A step whose ids the rank did not retain, whose
element type is none it reads, or that has no rows in the request map is a step without attributable ids; a step whose rows
are not the request map's is named by both counts, and the steps after it go on from where its requests stood; every row
that disagrees is counted, past the ones it names; a Match record longer than the run, or one naming a request the rank never
sampled, fails by the counts; and a rank with no step has checked nothing.  Each answer names the rank and its first reason.
CPU-only: the commitment is the library's own CPU core's over a few ids (`tests.check.opened.commit_steps`)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from tests.check.opened import commit_steps
from verity_vllm.pipeline.tp import rank_worker as RW

SAMPLED = "runner.sampler/sampled_token_ids"
#: two requests of three sampled ids each, as the Match record has them
MATCH = {"r0": [11, 12, 13], "r1": [21, 22, 23]}
ROWS = [("A-1a2b3c4d", 0, 1), ("B-9f8e7d6c", 1, 2)]
REFERENCE = "tp2_match capture arm tokens.json (per request generated ids; independent instrumented run)"


class _Rank(RW.TP2CaptureWorkerExtension):
    _tp2_commit: dict[str, Any]


def _served(step: int) -> list[int]:
    """The two ids the rank sampled at ``step``: each request's next id of the Match record."""
    return [MATCH["r0"][step], MATCH["r1"][step]]


def _answer(match: dict[str, list[int]] | None = None, *, steps: int = 3, ids: Any = _served, stored: Any = np.int64, dtype: str = "int64",
            shape: tuple[int, ...] = (2, 1), retain: str = "host", rows: dict[int, list[tuple[str, int, int]] | None] | None = None) -> dict[str, Any]:
    """The attribution of a rank that committed ``steps`` steps of sampled ids (``ids(step)``, stored as ``stored`` and declared
    as ``dtype`` and ``shape``) under a request map of two rows a step, except the steps ``rows`` gives."""
    com, rc = commit_steps({s: [(SAMPLED, np.asarray(ids(s), dtype=stored).tobytes(), 0, dtype, shape)] for s in range(steps)}, retain=retain)
    com.step_rows = {s: r for s, r in ({s: list(ROWS) for s in range(steps)} | (rows or {})).items() if r is not None}
    w = _Rank()
    w._tp2_commit = {"com": com, "rank": 1, "request_ids": {"A": "r0", "B": "r1"}, "match_tokens": MATCH if match is None else match}
    out = w._tp2_attribution(rc)
    assert out is not None and (out["rank"], out["reference"], out["swap_requests_negative"]) == (1, REFERENCE, None)
    assert out["value_source"]["source"] == "opened" and out["value_source"]["run_root"] == bytes(rc.run_root).hex()
    return out


def _counts(out: dict[str, Any]) -> tuple[int, int, int]:
    return out["steps_checked"], out["rows_checked"], out["mismatches_n"]


@pytest.mark.parametrize(("stored", "dtype", "shape"), [
    (np.int64, "int64", (2, 1)), (np.int64, "torch.int64", (2,)), (np.int64, "long", (2, 1)),
    (np.int32, "int32", (2, 1)), (np.int32, "torch.int32", (2,)), (np.int32, "int", (2, 1))])
def test_ids_of_either_integer_width_under_any_of_its_names_are_read_one_a_request_row(stored, dtype, shape):
    out = _answer(stored=stored, dtype=dtype, shape=shape)
    assert (out["ok"], out["why"], _counts(out)) == (True, None, (3, 6, 0)) and out["missing_steps"] == [] and out["mismatches"] == []
    assert out["requests"] == {r: {"sampled_rows_committed": 3, "match_record_generated": 3} for r in ("r0", "r1")}
    assert (out["value_source"]["reads"], out["value_source"]["not_retained"], out["value_source"]["failed"]) == (3, 0, 0)


def test_a_step_that_committed_several_ids_a_row_is_read_by_each_rows_first():
    """Two ids a request row: the row's first is its sampled id, and the second, which is no id of the Match record, is not read."""
    out = _answer(ids=lambda s: [[MATCH["r0"][s], 7001], [MATCH["r1"][s], 7002]], shape=(2, 2))
    assert (out["ok"], _counts(out)) == (True, (3, 6, 0))


def test_ids_the_rank_did_not_retain_are_steps_without_attributable_ids_and_no_read_is_counted():
    out = _answer(retain="none")
    assert (out["ok"], _counts(out)) == (False, (0, 0, 0))
    assert out["missing_steps"] == [{"step": s, "why": "committed bytes not retained (VERITY_RETAIN)"} for s in range(3)]
    assert out["why"] == "rank 1: 3 steps without attributable sampled ids (first: {'step': 0, 'why': 'committed bytes not retained (VERITY_RETAIN)'})"
    assert (out["value_source"]["reads"], out["value_source"]["not_retained"], out["value_source"]["failed"]) == (0, 3, 0)
    assert out["requests"] == {r: {"sampled_rows_committed": 0, "match_record_generated": 3} for r in ("r0", "r1")}


@pytest.mark.parametrize(("stored", "dtype"), [(np.float32, "float32"), (np.float32, "torch.float32"), (np.int16, "int16"), (np.uint8, "uint8")])
def test_ids_of_an_element_type_the_rank_does_not_read_are_steps_without_attributable_ids_by_the_types_name(stored, dtype):
    """Ids are 32 or 64 bits wide; any other element type's bytes are read as no id at all, though the step verified."""
    out = _answer(stored=stored, dtype=dtype)
    assert (out["ok"], _counts(out)) == (False, (0, 0, 0))
    assert out["missing_steps"] == [{"step": s, "why": f"sampled ids dtype {dtype} not an integer type"} for s in range(3)]
    assert out["why"].startswith(f"rank 1: 3 steps without attributable sampled ids (first: {{'step': 0, 'why': 'sampled ids dtype {dtype} not")
    assert (out["value_source"]["reads"], out["value_source"]["failed"]) == (3, 0)


@pytest.mark.parametrize("none", [None, []], ids=["absent from the map", "an empty list"])
def test_a_step_with_no_rows_in_the_request_map_is_named_and_the_others_are_still_checked(none):
    """Step 1 has no rows in the map: its ids are not read, and steps 0 and 2 are checked, step 2 against each request's second
    id of the Match record, since each request was sampled once before it."""
    out = _answer(ids=lambda s: _served({0: 0, 1: 2, 2: 1}[s]), rows={1: none})
    assert (out["ok"], _counts(out)) == (False, (2, 4, 0)) and out["missing_steps"] == [{"step": 1, "why": "no request rows recorded"}]
    assert out["why"] == "rank 1: 1 steps without attributable sampled ids (first: {'step': 1, 'why': 'no request rows recorded'})"
    assert out["value_source"]["reads"] == 2
    assert out["requests"] == {r: {"sampled_rows_committed": 2, "match_record_generated": 3} for r in ("r0", "r1")}


@pytest.mark.parametrize(("map_rows", "n"), [([*ROWS, ("C-00ff00ff", 2, 3)], 3), (ROWS[:1], 1)], ids=["a row more", "a row fewer"])
def test_a_step_whose_committed_rows_are_not_the_request_maps_is_named_by_both_counts(map_rows, n):
    """Step 1 committed two ids where the map has three rows, or one: no row of it is attributed, so no request's position moves, and
    step 2 is checked against each request's second id."""
    out = _answer(ids=lambda s: _served({0: 0, 1: 2, 2: 1}[s]), rows={1: map_rows})
    assert (out["ok"], _counts(out)) == (False, (2, 4, 1)) and out["missing_steps"] == []
    assert out["mismatches"] == [{"step": 1, "why": f"2 committed sampled rows vs {n} request rows in the map"}]
    assert out["why"] == f"rank 1: step 1: 2 committed sampled rows vs {n} request rows in the map"
    assert out["requests"] == {r: {"sampled_rows_committed": 2, "match_record_generated": 3} for r in ("r0", "r1")}


def test_ids_of_three_dimensions_are_no_rows_and_are_counted_flat_against_the_request_map():
    """Only ids of two dimensions are taken a row at a time; a step that committed (2, 2, 1) is four ids against two rows."""
    out = _answer(ids=lambda s: [[[MATCH["r0"][s]], [7001]], [[MATCH["r1"][s]], [7002]]], shape=(2, 2, 1))
    assert (out["ok"], _counts(out)) == (False, (0, 0, 3))
    assert out["mismatches"] == [{"step": s, "why": "4 committed sampled rows vs 2 request rows in the map"} for s in range(3)]


def test_every_row_that_disagrees_is_counted_and_the_first_twelve_are_named():
    """Thirteen steps whose two requests' ids are each the other's in the Match record: all 26 rows are counted, past the 24 the
    rank keeps by name, the answer lists the first twelve, and its reason is the first."""
    served = {"r0": list(range(100, 113)), "r1": list(range(200, 213))}
    out = _answer({"r0": served["r1"], "r1": served["r0"]}, steps=13, ids=lambda s: [served["r0"][s], served["r1"][s]])
    assert (out["ok"], _counts(out)) == (False, (13, 26, 26)) and out["missing_steps"] == []
    assert out["mismatches"] == [{"step": s, "row": j, "request": f"r{j}", "k": s, "committed": served[f"r{j}"][s], "match_record": served[f"r{1 - j}"][s]}
                                 for s in range(6) for j in range(2)]
    assert out["why"] == ("rank 1: 26 of 26 committed sampled rows disagree with the Match record (first: step 0 row 0 labelled r0 k=0: "
                          "committed 100 != match 200)")
    assert out["requests"] == {r: {"sampled_rows_committed": 13, "match_record_generated": 13} for r in ("r0", "r1")}


@pytest.mark.parametrize(("match", "counts"), [
    ({"r0": [11, 12, 13, 14], "r1": [21, 22, 23]}, {"r0": (3, 4), "r1": (3, 3)}),
    (MATCH | {"r2": [31]}, {"r0": (3, 3), "r1": (3, 3), "r2": (0, 1)}),
], ids=["a request generated more in the Match record", "a request the rank never sampled"])
def test_a_match_record_the_run_does_not_fill_fails_by_its_counts_though_every_row_agrees(match, counts):
    out = _answer(match)
    assert (out["ok"], _counts(out)) == (False, (3, 6, 0)) and out["missing_steps"] == [] and out["mismatches"] == []
    lens = {r: {"sampled_rows_committed": a, "match_record_generated": b} for r, (a, b) in counts.items()}
    assert out["requests"] == lens and out["why"] == f"rank 1: per-request sampled row counts != Match record lengths: {lens}"


def test_a_rank_that_committed_no_step_has_checked_nothing_and_does_not_pass():
    """A Match record of a request that generated nothing agrees with a rank of no step in every count; that is no check."""
    out = _answer({"r0": []}, steps=0)
    assert (out["ok"], _counts(out)) == (False, (0, 0, 0)) and out["missing_steps"] == [] and out["mismatches"] == []
    assert out["requests"] == {"r0": {"sampled_rows_committed": 0, "match_record_generated": 0}} and out["why"] == "rank 1: nothing checked"
