"""`verity_vllm.engine.static_schedule`: a run that declares `sampling.ignore_eos` is held to a schedule the declaration alone
decides."""
from __future__ import annotations

from typing import Any

import pytest

from verity_vllm.engine import static_schedule as ss

REQS = [{"request_id": "a", "max_tokens": 3}, {"request_id": "b", "max_tokens": 2}]


def _timing(**over):
    recs = {"a": {"arrive_step_realized": 0, "first_step": 0, "finish_step": 3, "n_generated": 3, "finish_reason": "length",
                  "prefill_steps": [0], "token_steps": [0, 1, 2]},
            "b": {"arrive_step_realized": 1, "first_step": 1, "finish_step": 3, "n_generated": 2, "finish_reason": "length",
                  "prefill_steps": [1], "token_steps": [1, 2]}}
    for rid, fields in over.items():
        recs[rid] = {**recs[rid], **fields}
    return {"engine_steps": 4, "requests": recs}


def test_declared_static_reads_ignore_eos_only():
    assert ss.declared_static({"ignore_eos": True})
    assert not ss.declared_static({"temperature": 0.0})
    assert not ss.declared_static(None)


def test_a_run_of_every_cap_passes():
    rec = ss.check(REQS, _timing(), cached_tokens={"a": 0, "b": 0}, preempted={})
    assert rec["ok"] and rec["problems"] == [] and rec["schema"] == ss.SCHEMA
    assert [r["n_generated"] for r in rec["schedule"]["requests"]] == [3, 2]


def test_an_eos_stop_fails_by_name():
    rec = ss.check(REQS, _timing(b={"n_generated": 1, "finish_reason": "stop"}))
    assert not rec["ok"]
    assert len(rec["problems"]) == 1 and rec["problems"][0].startswith("b: finished by 'stop' after 1 of its 2 tokens")


def test_n_gt_1_needs_every_child_to_finish_by_length():
    assert ss.check(REQS, _timing(a={"finish_reason": ["length", "length"]}))["ok"]
    assert not ss.check(REQS, _timing(a={"finish_reason": ["length", "stop"]}))["ok"]


def test_host_prefix_cache_hits_and_preemptions_fail():
    rec = ss.check(REQS, _timing(), cached_tokens={"a": 16, "b": 0}, preempted={"0_b": 2})
    assert not rec["ok"]
    assert any(p.startswith("a: read 16 prompt tokens' KV") for p in rec["problems"])
    assert any(p.startswith("0_b: preempted 2 time(s)") for p in rec["problems"])


def test_preemptions_count_each_drop_of_num_computed():
    snaps = [{"a": (4, 4)}, {"a": (5, 4), "b": 2}, {"a": (0, 4), "b": 3}, {"a": (4, 4), "b": 1}, {"b": 2}]
    assert ss.preemptions(snaps) == {"a": 1, "b": 1}
    assert ss.preemptions([{"a": (4, 4)}, {"a": (5, 4)}]) == {}


def test_the_schedule_digest_leaves_out_ids_and_depends_on_steps():
    t = _timing()
    renamed = {"engine_steps": 4, "requests": {"x": t["requests"]["a"], "y": t["requests"]["b"]}}
    assert ss.schedule_of(["a", "b"], t)["digest"] == ss.schedule_of(["x", "y"], renamed)["digest"]
    assert ss.schedule_of(["a", "b"], t)["digest"] != ss.schedule_of(["a", "b"], _timing(b={"finish_step": 4}))["digest"]


def test_compare_names_the_first_differing_request():
    ref = ss.schedule_of(["a", "b"], _timing())
    assert ss.compare(ref, ss.schedule_of(["a", "b"], _timing()))["ok"]
    out = ss.compare(ref, ss.schedule_of(["a", "b"], _timing(b={"first_step": 2, "token_steps": [2, 3]})))
    assert not out["ok"]
    assert "request 1: first_step 1 vs 2, token_steps [1, 2] vs [2, 3]" in out["problems"][0]


def test_against_warmup_fails_a_run_whose_schedule_differs_from_the_warmups():
    warm = ss.check(REQS, _timing())
    assert ss.against_warmup(ss.check(REQS, _timing()), warm)["content_invariance"]["ok"]
    rec = ss.against_warmup(ss.check(REQS, _timing(b={"finish_step": 4})), warm)
    assert not rec["ok"] and rec["problems"][-1].startswith("the schedule depends on the prompts' contents")
    assert ss.report_lines(rec)[0].endswith("warm-up DIFFERS)") and ss.report_lines(rec)[1].startswith("REFUSE")
    alone = ss.against_warmup(ss.check(REQS, _timing()), None)
    assert "content_invariance" not in alone and ss.report_lines(alone) == [
        f"static schedule: ok (digest {alone['schedule']['digest'][:16]}, no warm-up to compare)"]
    assert ss.against_warmup(None, warm) is None


def test_host_prefix_cache_refusal():
    assert ss.host_prefix_cache_refusal(False) is None and ss.host_prefix_cache_refusal(None) is None
    msg = ss.host_prefix_cache_refusal(True)
    assert "enable_prefix_caching=false" in msg and "circuit-native form" in msg


NO_RECORDS = [{}, {"requests": None}, {"requests": {}}, {"requests": {"a": None, "b": None}}, {"requests": {"a": {}, "b": {}}}]
NO_RECORDS_IDS = ["absent", "null", "empty", "each null", "each empty"]


@pytest.mark.parametrize("timing", NO_RECORDS, ids=NO_RECORDS_IDS)
def test_the_schedule_of_a_timing_without_request_records_is_a_row_of_nulls_per_request(timing):
    sched = ss.schedule_of(["a", "b"], timing)
    assert sched["engine_steps"] is None and sched["requests"] == [dict.fromkeys(ss.SCHEDULE_FIELDS)] * 2
    assert sched["digest"] == ss.schedule_of(["x", "y"], {})["digest"]


@pytest.mark.parametrize("timing", NO_RECORDS, ids=NO_RECORDS_IDS)
def test_a_timing_without_request_records_fails_every_request_by_name(timing):
    rec = ss.check(REQS, timing)
    assert not rec["ok"] and rec["schedule"] == ss.schedule_of(["a", "b"], {})
    assert [p.split(":")[0] for p in rec["problems"]] == ["a", "b"]
    assert rec["problems"][0].startswith("a: finished by None after None of its 3 tokens")


@pytest.mark.parametrize("without", [{}, {"requests": None}, {"requests": []}], ids=["absent", "null", "empty"])
def test_compare_counts_a_schedule_without_requests_as_none(without):
    ref = ss.schedule_of(["a", "b"], _timing())
    for a, b, counts in ((ref, {"digest": "0", **without}, "2 vs 0 requests"), ({"digest": "0", **without}, ref, "0 vs 2 requests")):
        out = ss.compare(a, b)
        why = (f"the schedule depends on the prompts' contents, not only on the declaration: engine steps "
               f"{a.get('engine_steps')} vs {b.get('engine_steps')}; {counts}")
        assert not out["ok"] and out["problems"] == [why]


@pytest.mark.parametrize("timing", [None, 0, "x", [1]], ids=["null", "zero", "text", "list"])
def test_a_timing_that_is_no_record_is_refused_before_the_ids_or_the_requests_are_read(timing):
    """The requests' records are read from the timing first, so its error is the one raised whatever else is wrong."""
    orders: tuple[Any, ...] = (None, 0, [], ["a"])
    for order in orders:
        with pytest.raises(AttributeError, match="has no attribute 'get'"):
            ss.schedule_of(order, timing)
    lists: tuple[Any, ...] = (None, 0, [], [{}], [{"request_id": "a"}], [{"request_id": "a", "max_tokens": "x"}], REQS)
    for requests in lists:
        with pytest.raises(AttributeError, match="has no attribute 'get'"):
            ss.check(requests, timing)
