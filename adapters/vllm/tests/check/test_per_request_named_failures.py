"""verity_vllm.check.match.per_request: two failures a request's own check names, on the batched toy of `test_global_match.py`,
in both implementations and under both criteria.

A request served past its bounded component: the global Program is assembled for a request of two tokens, and the record is of
that request run for three.  G3 names the two step counts; G4 and the request's entry differ by criterion, and are held as they
are.  Under `sequence` an instance's address is its place in the fold's order, so the step the component does not have is counted
not equal by G3 and still mapped by G4, which passes.  Under `dag` that step is matched against nothing: G4 names each of its
instances as without an address, and the request's entry says its executed steps are equal, the steps past the component not
being among the ones it matches.  G7 fails on the two populations either way.

A fold whose weights map is ambiguous: one of its rows hints another module for leaves a field already spans, so the map holds
two fields over the same leaves.  G5 says so, and no other check fails."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.check import test_global_match as tgm
from verity_vllm.check.match import global_match as gm

memory = tgm.memory                                      # the toy's fixture, under its own name

IMPLS = pytest.mark.parametrize("impl", gm.IMPLS)
CRITERIA = pytest.mark.parametrize("within_step", gm.WITHIN_STEP_CRITERIA)

(LIMIT,) = set(tgm.R1_EOS) - set(tgm.R1)                 # the toy's key of a request's declared limit
R1_THREE = {**tgm.R1, "sampled": [*tgm.R1["sampled"], 77], LIMIT: 2}   # r1 as declared, run one step past its two
NO_ADDRESS = "r1: fold instance {} (local step 2, emission {}) has no structurally identical component instance: no canonical address"


def _entry(res: dict, request: str) -> dict:
    return next(r for r in res["requests"] if r["request_id"] == request)


@IMPLS
@CRITERIA
def test_a_request_served_past_its_bounded_component_fails_g3_by_the_two_step_counts(within_step: str, impl: str, memory, tmp_path, monkeypatch):
    monkeypatch.setenv(gm.IMPL_ENV, impl)
    gp, _ = tgm.assemble([tgm.R0, tgm.R1], tmp_path / "declared")
    _, oracle = tgm.assemble([tgm.R0, R1_THREE], tmp_path / "served")     # the same workload manifest: r1's prompt and its limit of two
    res = gm.check(gp, oracle, within_step=within_step, run_x09=False, record_alternate=False)
    exceeded = "r1: executed 3 local steps but the bounded component has 2 (output interface exceeded)"
    r0, r1 = _entry(res, "r0"), _entry(res, "r1")
    assert (r1["local_steps"], r1["component"]["steps"], r1["n_instances"], r1["component"]["instances"]) == (3, 2, 54, 40)
    assert (r1["eos_truncated"], r1["unexecuted_steps"], r1["sequence_equal_steps"], r1["operands"]) == (False, 0, 2, {"equal": 84, "different": 0})
    assert (r0["local_steps"], r0["sequence_equal_steps"], r0["executed_steps_equal"], r0["n_divergent_steps"]) == (3, 3, True, 0)
    g3, g4, g7 = (tgm._check(res, cid) for cid in ("G3", "G4", "G7"))
    assert (g4["mapped"], g4["attributed"], g4["distinct"]) == (120, 120, 120)
    assert (g7["population_query"], g7["population_fold"]) == (106, 120)
    assert g7["problems"][0] == "global query exposes 106 VUs over the executed steps but the fold holds 120 attributed instances"
    if within_step == "sequence":
        assert res["failed_checks"] == ["G3", "G6", "G7"]
        assert g3["problems"] == [exceeded, "r1: 1 of 3 executed steps not canonical-equal (first: local step 2)"]
        assert g4["problems"] == []
        assert (r1["executed_steps_equal"], r1["n_divergent_steps"]) == (False, 1)
        assert r1["first_divergence"] == {"step": 2, "n_a": 0, "n_b": 14, "spec_seq_equal": False, "wiring_examples": []}
    else:
        assert res["failed_checks"] == ["G3", "G4", "G6", "G7"]
        assert g3["problems"] == [exceeded]
        # r1's fourteen instances of its third step, every other one of the fold's last engine step
        assert g4["problems"] == [NO_ADDRESS.format(93 + 2 * emission, emission) for emission in range(14)]
        assert (r1["executed_steps_equal"], r1["n_divergent_steps"], r1["within_step"]["sequence_also_equal"]) == (True, 0, False)


@IMPLS
@CRITERIA
def test_a_fold_with_two_weight_fields_over_the_same_leaves_fails_g5_and_nothing_else(within_step: str, impl: str, memory, tmp_path, monkeypatch):
    monkeypatch.setenv(gm.IMPL_ENV, impl)
    gp, oracle = tgm.assemble([tgm.R0, tgm.R1], tmp_path)
    rotary = "model.layers.0.self_attn.rotary_emb"
    fold = gm.match_ops(impl == "fast").load_oracle(oracle, "oracle")
    assert fold.oracle_wmap[-1] == (1840, 1880, rotary) and fold.oracle_wmap_overlaps == []
    rows = Path(oracle) / "instances.jsonl"
    lines = rows.read_text().splitlines()                # the header, then a row an instance
    first = next(n for n, line in enumerate(lines) if n and json.loads(line)["hint"] == rotary)
    row = json.loads(lines[first])
    assert row["operand_refs"][1] == [["p", 2, 1840, 1848]]                  # a part of the module's table
    lines[first] = json.dumps({**row, "hint": "model.layers.0.other"}, separators=(",", ":"))
    rows.write_text("\n".join(lines) + "\n")
    fold = gm.match_ops(impl == "fast").load_oracle(oracle, "oracle")
    assert fold.oracle_wmap_overlaps == [((1840, 1848, "model.layers.0.other"), (1840, 1880, rotary))]
    res = gm.check(gp, oracle, within_step=within_step, run_x09=False, record_alternate=False)
    assert (res["verdict"], res["failed_checks"]) == ("FAIL", ["G5"])
    assert tgm._check(res, "G5")["problems"] == ["weights field map has 1 overlapping ranges (ambiguous shared-value identity)"]
