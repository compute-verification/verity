"""verity_vllm.check.match.engine_steps, G8: the engine step a fold's rows state for a request's instances against the steps the
request's own sampling events cut them into, and the order of a request's instances inside an engine step.

The first tests run the whole check, in both implementations and under both criteria, on the toy of `test_global_match.py` with
the engine step of some of the fold's rows changed after the fold was loaded: nothing else of the record or the Program moves, so
the three problems G8 names are each the one the change makes.  The last one hands `within_step_order` facts written out by hand,
because no record brings them: under `sequence` a request's address is its emission position, so `check` never finds an instance
out of its request's order there."""
from __future__ import annotations

from typing import Any, cast

import pytest

from tests.check import test_global_match as tgm
from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match.attribution import Attribution
from verity_vllm.check.match.engine_steps import within_step_order
from verity_vllm.check.match.match_run import MatchRun
from verity_vllm.check.match.per_request import RequestFacts

memory = tgm.memory                                      # the toy's fixture, under its own name

IMPLS = pytest.mark.parametrize("impl", gm.IMPLS)
CRITERIA = pytest.mark.parametrize("within_step", gm.WITHIN_STEP_CRITERIA)


def _moved(rows: list[dict], request: str, step: int, to: int, last_only: bool = False) -> int:
    """The rows of `request` at engine step `step` now state step `to` (`last_only`: the last of them alone); how many changed."""
    mine = [o for o in rows if o["request"] == request and o["step"] == step]
    for o in mine[-1:] if last_only else mine:
        o["step"] = to
    return 1 if last_only else len(mine)


# what is changed in the rows (request, its step, the step stated instead, the last row alone), the request-local step G8 names, the
# engine steps it finds it in, its problem, and the checks that fail
EDITS = {
    "a second step of a request in one engine step": (
        ("r1", 1, 0, False), {"request_id": "r1", "local_step": 1, "engine_steps": [0],
                              "problem": "second TokenSelect of the request in engine step 0 (local step 0 already ended there)"}, ["G8"]),
    "a later step of a request in an earlier engine step": (
        ("r0", 2, 0, False), {"request_id": "r0", "local_step": 2, "engine_steps": [0], "problem": "engine step order decreases (after 1)"},
        ["G5", "G8"]),
    "one step of a request in two engine steps": (
        ("r1", 1, 2, True), {"request_id": "r1", "local_step": 1, "engine_steps": [1, 2], "problem": "the local step spans engine steps"},
        ["G8"]),
}


@IMPLS
@CRITERIA
@pytest.mark.parametrize("edit", EDITS)
def test_rows_that_state_another_engine_step_than_the_requests_sampling_events_cut_fail_g8_by_the_step(
        edit: str, within_step: str, impl: str, memory, tmp_path, monkeypatch):
    """r0 runs three engine steps and r1 two; each ends a step of its own with its TokenSelect.  The record's steps and the Program
    are as assembled: only what the fold's rows say of an instance's engine step differs."""
    (request, step, to, last_only), named, failed = EDITS[edit]
    monkeypatch.setenv(gm.IMPL_ENV, impl)
    gp, oracle = tgm.assemble([tgm.R0, tgm.R1], tmp_path)
    fold = gm.match_ops(impl == "fast").load_oracle(oracle, "oracle")
    as_loaded = gm.check(gp, oracle, within_step=within_step, run_x09=False, record_alternate=False, _loaded={"oracle": fold})
    assert as_loaded["verdict"] == "PASS" and tgm._check(as_loaded, "G8")["segmentation"]["disagreements"] == []
    assert _moved(cast("list[dict]", fold.oracle_rows), request, step, to, last_only) == (1 if last_only else 14)
    res = gm.check(gp, oracle, within_step=within_step, run_x09=False, record_alternate=False, _loaded={"oracle": fold})
    g8 = tgm._check(res, "G8")
    assert (res["verdict"], res["failed_checks"]) == ("FAIL", failed)
    assert g8["segmentation"]["disagreements"] == [named]
    problem = ("1 request-local step(s) whose TokenSelect segmentation disagrees with the fold rows' engine steps "
               f"(first: {named}) -- a mis-emitted sampler, not a layout")
    assert g8["problems"] == [problem]
    if "G5" in failed:
        assert tgm._check(res, "G5")["problems"] == ["r0: 3 operands produced in a LATER engine step than their consumer"]


def _one_engine_step(within_step: str) -> tuple[MatchRun, Attribution, RequestFacts]:
    """One engine step of two requests with a row each: r0's two instances, emitted in the order of its component's second and
    first, then r1's one."""
    run = MatchRun(gp={}, oracle_dir="the-fold-dir", tokens_path="", within_step=within_step, fast=False, ops=bd.record_ops(), loaded={},
                   tokens=None, tokens_optional=True, rec={"header": {}, "steps": {0: {"rows": [("r0", 0), ("r1", 0)]}}, "samples": {}},
                   res={"checks": [], "requests": []}, fold=cast("Any", None), declared={"r0": {"prompt_len": 1}, "r1": {"prompt_len": 1}})
    att = Attribution(weights_nodes=frozenset(), weights_nodes_error=None, meta_of={}, per_req={"r0": [0, 1], "r1": [2]}, unattributed=[],
                      shared_idx=[], shared_gi=frozenset(), peer_index={})
    reqs = RequestFacts(glob2loc={0: ("r0", 0, 1), 1: ("r0", 0, 0), 2: ("r1", 0, 0)}, shared_loc={}, exec_pop=3, unexercised=0, proj={},
                        comps={}, engine_step_of={0: 0, 1: 0, 2: 0}, dep_viol=[], perm_facts={})
    return run, att, reqs


@pytest.mark.parametrize("within_step, status, problems", [
    ("sequence", "FAIL", ["engine step 0: 1 instances emitted out of their request-local order"]),
    ("dag", "PASS", []),
])
def test_an_instance_emitted_before_one_its_component_has_first_fails_under_sequence_and_is_layout_under_dag(
        within_step: str, status: str, problems: list[str]):
    run, att, reqs = _one_engine_step(within_step)
    within_step_order(run, att, reqs, {0: [0, 1, 2]})
    [g8] = run.res["checks"]
    assert (g8["id"], g8["status"], g8["problems"]) == ("G8", status, problems)
    assert g8["steps_with_emission_out_of_canonical_order"] == 1
    assert g8["batch_layout"] == [{"engine_step": 0, "batch_rows_order": ["r0", "r1"], "emission_first_seen": ["r0", "r1"],
                                   "phase": "uniform", "emitted_out_of_canonical_order": 1}]
    assert (g8["segmentation"]["disagreements"], g8["dependency_order_violations"], g8["permutations"]) == ([], 0, [])
