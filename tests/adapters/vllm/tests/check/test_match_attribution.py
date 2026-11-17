"""`verity_vllm.check.match.attribution` after the meta channel, the paths of the global match no other CPU test runs: a fold
instance whose row states no request is shared when it reads weights alone (Gemma-2's RMSNorm scale `1 + w`), is otherwise placed
by the rows of its engine step (`occurrence.attribute`), and is otherwise attributed to no request, which G2 fails by name.

The first tests read a stub of the fold's view (no Program is decoded), the one `test_batch_decomp_named_paths.py` holds X-09's
copy of the rule to; here in both forms a loader gives an instance's operands, the record's runs and the compact form's
progressions.  The last ones run both implementations of the global match on the toy of `test_global_match.py` with GemmaRMSNorm's
weight add in every engine step (`tests/observe/test_fold_weight_only_calls.py`), through the fold and the persisted record."""
from __future__ import annotations

import dataclasses

import pytest

from tests.check import test_batch_decomp_named_paths as paths
from tests.check import test_global_match as tgm
from tests.observe import test_fold_weight_only_calls as weight_calls
from verity_vllm.check.match import batch_decomp, occurrence
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match import global_match_fast as gmf
from verity_vllm.check.match.attribution import attribute_instances, requests_observed
from verity_vllm.check.match.match_run import MatchRun
from verity_vllm.observe.events import (
    DispatchEnter,
    DispatchExit,
    RunHeader,
    StorageAlloc,
)
from verity_vllm.observe.fold.patterns import TransparentAten
from verity_vllm.observe.fold.profiles.gen_dense_gemma2_patterns import gemma2_patterns
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET

memory = tgm.memory                                      # the toy's fixture, under its own name

R0, R1, STEP0 = paths.R0, paths.R1, paths.STEP0          # one batched prefill: five token rows (r0's three, r1's two), two logits rows
W, P0, P1, EMB, SELECT = paths.W, paths.P0, paths.P1, paths.EMB, paths.SELECT
REQUESTS = paths.HEADER["workload"]["requests"]
G2 = "requests_observed: every declared request has rows + output events + instances; 0 unattributed; no undeclared request"


def _view(inst, rows, compact=False):
    """`paths._fold`'s view; `compact`: as the compact loader gives it, each run of an operand a progression of one."""
    B = paths._fold(inst, rows)
    if compact:
        B.compact = True
        for r in B.inst:
            r["args"] = [[(space, (idx, lo, hi), (0, 0, 0), 1) for space, idx, lo, hi in a] for a in r["args"]]
    return B


def _run(fold, steps, declared=(R0, R1)) -> MatchRun:
    """A run as the attribution and G2 read it: the fold's view, the record's steps, the declared requests and the document."""
    return MatchRun(gp={}, oracle_dir="the-fold-dir", tokens_path="", within_step="dag", fast=False, ops=batch_decomp.record_ops(),
                    loaded={}, tokens=None, tokens_optional=True, rec={"header": {}, "steps": steps, "samples": {}}, res={"checks": []},
                    fold=fold, declared={rid: {} for rid in declared})


@pytest.mark.parametrize("compact", [False, True], ids=["record", "compact"])
def test_an_instance_that_reads_weights_alone_is_no_requests_and_each_request_with_rows_in_its_step_holds_it(compact):
    """Nodes 3 (reads the weights alone) and 4 (reads node 3 alone) carry no request's data, nor does node 11 at engine step 1, where
    only r0 has a row.  Every other instance's row states its request."""
    B = _view([(3, "AddScalar_v1{N=2}", [W]), (4, "Scale_v1{N=2}", [[("n", 3, 0, 2)]]), (5, EMB, [P0, W]), (6, EMB, [P1, W]),
               (7, "Norm_v1{N=2}", [[("n", 5, 0, 2)], [("n", 4, 0, 2)]]), (8, "Norm_v1{N=2}", [[("n", 6, 0, 2)], [("n", 4, 0, 2)]]),
               (9, SELECT, [[("n", 7, 0, 2)]]), (10, SELECT, [[("n", 8, 0, 2)]]), (11, "AddScalar_v1{N=2}", [W])],
              [(3, 0, 0, 1, None), (4, 0, 0, 1, None), (5, 0, 0, 1, R0), (6, 0, 0, 1, R1), (7, 0, 0, 1, R0), (8, 0, 0, 1, R1),
               (9, 0, 0, 1, R0), (10, 0, 0, 1, R1), (11, 1, 0, 1, None)], compact)
    run = _run(B, {0: STEP0, 1: {"rows": [(R0, 3)], "logits_rows": [(R0, 3)]}})
    att = attribute_instances(run, REQUESTS)
    assert (att.weights_nodes, att.weights_nodes_error) == ({2}, None)
    assert att.shared_idx == [(0, 0), (1, 0), (8, 1)] and att.shared_gi == {0, 1, 8} and att.unattributed == []
    assert list(att.per_req.items()) == [(R0, [0, 1, 2, 4, 6, 8]), (R1, [0, 1, 3, 5, 7])]   # 3 own + 3 shared | 3 own + step 0's 2
    assert requests_observed(run, att, [R0, R1]) == {R0: [0, 1], R1: [0]}
    assert run.res["checks"] == [{
        "id": "G2", "what": G2, "status": "PASS", "problems": [], "n_problems": 0, "n_instances": 9, "unattributed": 0, "engine_steps": 2,
        "attributed": 11,                                           # by request: a shared instance counts for each request that holds it
        "weight_only_shared": 3, "weight_only_specs": {"AddScalar_v1": 2, "Scale_v1": 1}}]


def test_in_the_compact_form_every_node_of_a_progression_is_read():
    """Node 3 reads the one weights node in two ranges, one progression whose node does not move.  Node 6 reads nodes 3 and 4, both
    shared, as one progression; node 7 reads nodes 4 and 5 as one, and node 5 is r0's: the rows of its step place it."""
    B = _view([(3, "AddScalar_v1{N=2}", []), (4, "Scale_v1{N=2}", [[("n", 3, 0, 2)]]), (5, EMB, [P0, W]), (6, "Mix_v1{N=2}", []),
               (7, "Mix_v1{N=2}", []), (8, SELECT, [[("n", 5, 0, 2)]])],
              [(3, 0, 0, 1, None), (4, 0, 0, 1, None), (5, 0, 0, 1, R0), (6, 0, 0, 1, None), (7, 0, 0, 2, None), (8, 0, 0, 1, R0)], compact=True)
    B.inst[0]["args"] = [[("n", (2, 0, 5), (0, 5, 5), 2)]]
    B.inst[3]["args"] = [[("n", (3, 0, 2), (1, 0, 0), 2)]]
    B.inst[4]["args"] = [[("n", (4, 0, 2), (1, 0, 0), 2)]]
    att = attribute_instances(_run(B, {0: STEP0}), REQUESTS)
    assert att.shared_idx == [(0, 0), (1, 0), (3, 0)] and att.unattributed == []
    assert att.per_req == {R0: [0, 1, 2, 3, 4, 5]}                   # node 7 (instance 4) by logits row 0 of 2, which is r0's


def test_an_instance_no_rule_places_is_attributed_to_no_request_and_g2_fails_by_their_count():
    """No request in the row of nodes 5 to 9 and 12.  Node 5 reads a weight and r0's node, and node 6 no node at all: neither is
    shared, and the rows place them (token row 3 of 5 is r1's, logits row 0 of 2 is r0's).  Node 12 runs over the ten pair rows of
    five tokens: pair row 7 is token 3's, r1's.  Node 7's row count is none of the step's, node 8 has no row in the record though it
    reads the weights alone, and node 9's engine step has no rows: the three are no request's."""
    B = _view([(3, EMB, [P0, W]), (4, EMB, [P1, W]), (5, "Mix_v1", [[("n", 3, 0, 2)], W]), (6, "Table_v1", [[("p", 0, 0, 1)]]),
               (7, "Const_v1", []), (8, "AddScalar_v1{N=2}", [W]), (9, "Mix_v1", [[("n", 3, 0, 2)]]),
               (10, SELECT, [[("n", 3, 0, 2)]]), (11, SELECT, [[("n", 4, 0, 2)]]), (12, "MoeExpertGemm_v1{K=2}", [[("n", 3, 0, 2)]])],
              [(3, 0, 0, 1, R0), (4, 0, 0, 1, R1), (5, 0, 3, 5, None), (6, 0, 0, 2, None), (7, 0, 0, 7, None), (9, 9, 0, 1, None),
               (10, 0, 0, 1, R0), (11, 0, 0, 1, R1), (12, 0, 7, 10, None)])
    run = _run(B, {0: STEP0})
    att = attribute_instances(run, REQUESTS)
    assert att.shared_idx == [] and att.unattributed == [4, 5, 6]
    assert list(att.per_req.items()) == [(R0, [0, 3, 7]), (R1, [1, 2, 8, 9])]
    assert requests_observed(run, att, [R0, R1]) == {R0: [0], R1: [0]}
    assert run.res["checks"] == [{
        "id": "G2", "what": G2, "status": "FAIL", "problems": ["3 fold instances attributed to no request"], "n_problems": 1,
        "n_instances": 10, "attributed": 7, "unattributed": 3, "engine_steps": 1, "weight_only_shared": 0, "weight_only_specs": {}}]


@pytest.mark.parametrize("meta, owner", [
    ({"row": 0, "rows": 10}, (R0, "moe-pair(x2)")),                  # pair rows, pair-major: rows 0 and 1 are token 0's
    ({"row": 5, "rows": 10}, (R0, "moe-pair(x2)")),                  # token 2's, r0's last
    ({"row": 6, "rows": 10}, (R1, "moe-pair(x2)")),                  # token 3's, r1's first
    ({"row": 14, "rows": 15}, (R1, "moe-pair(x3)")),
    ({"row": 10, "rows": 10}, (None, "row-out-of-range")),
    ({"row": 0, "rows": 12}, (None, "rows=12!=tokens(5)|logits(2)")),   # no multiple of the token rows
])
def test_a_row_count_that_is_a_multiple_of_the_token_rows_is_pair_rows_and_a_pair_is_its_tokens(meta, owner):
    assert occurrence.attribute(meta, STEP0) == owner


def test_a_row_past_the_rows_of_a_step_whose_token_and_logits_rows_agree_is_out_of_range():
    decode = {"rows": [(R0, 7), (R1, 4)], "logits_rows": [(R0, 7), (R1, 4)]}
    assert occurrence.attribute({"row": 1, "rows": 2}, decode) == (R1, "token=logits")
    assert occurrence.attribute({"row": 2, "rows": 2}, decode) == (None, "row-out-of-range")


# --------------------------------------------------------------------------- on the toy, through the fold and the persisted record


def _with_weight_adds(batched_log, target):
    """`batched_log` with GemmaRMSNorm's `w.float() + 1.0` on the first norm's weight and a reader `+ 0.5` of it ahead of each engine
    step's first dispatch, as `test_fold_weight_only_calls._log` puts them; `target`: the record target the header states."""
    def log(reqs, *args, **kw):
        out, step = [], None
        for ev in batched_log(reqs, *args, **kw):
            if isinstance(ev, RunHeader) and target is not None:
                ev = dataclasses.replace(ev, profile={**ev.profile, "target": target})
            if isinstance(ev, DispatchEnter) and ev.step != step:
                step = ev.step
                w32, w1, r = (weight_calls.V(900 + 3 * step + k, (weight_calls.H,), dtype="float32") for k in range(3))
                for v in (w32, w1, r):
                    out.append(StorageAlloc(v.alloc, 0x100000 + 0x1000 * v.alloc, v.numel * 4, "cuda:0", 0, "op"))
                for i, (op, schema, args_, y) in enumerate((
                        ("aten.to.dtype", weight_calls.TO_SCHEMA, [weight_calls.W_LN1, "torch.float32"], w32),
                        ("aten.add.Tensor", weight_calls.ADD_SCHEMA, [w32, 1.0], w1), ("aten.add.Tensor", weight_calls.ADD_SCHEMA, [w1, 0.5], r))):
                    out.append(DispatchEnter(9000 + 10 * step + i, None, step, "forward", op, schema, args_, {}, 7, "model.layers.0.input_layernorm"))
                    out.append(DispatchExit(9000 + 10 * step + i, y))
            out.append(ev)
        return out
    return log


@pytest.fixture
def gemma_toy(monkeypatch):
    """`tgm.assemble` on the toy with the weight add: the record and each request's component (the request served alone) hold it.
    `per_forward`: the header declares an add in every forward; else the default, one add that every later forward reads."""
    def assemble(reqs, d, per_forward=False):
        target = DEFAULT_TARGET.with_overrides(weight_only_calls="per-forward").to_json() if per_forward else None
        toy_profile = tgm.profile

        def profile(*args, **kw):
            p = toy_profile(*args, **kw)
            return dataclasses.replace(p, patterns=[*p.patterns, *gemma2_patterns(), TransparentAten()])
        with monkeypatch.context() as m:
            m.setattr(tgm, "batched_log", _with_weight_adds(tgm.batched_log, target))
            m.setattr(tgm, "profile", profile)
            return tgm.assemble(reqs, d)
    return assemble


def _weight_only(res: dict) -> dict:
    """What the document says of the shared instances: G2's and G4's counts, and how the X-09 run inside it placed every instance."""
    g2, g4 = tgm._check(res, "G2"), tgm._check(res, "G4")
    assert g2["weight_only_shared"] == g4["weight_only_shared"] and g2["attributed"] == g4["attributed"]
    return {"instances": g2["n_instances"], "by_request": g2["attributed"], "shared": g2["weight_only_shared"], "specs": g2["weight_only_specs"],
            "mapped": g4["mapped"], "distinct": g4["distinct"], "local_identities": g4["weight_only_local_identities"],
            "x09": res["x09"]["attribution"]["how"]}


@pytest.mark.parametrize("impl", [gm, gmf], ids=["record", "fast"])
@pytest.mark.parametrize("per_forward, r1, expect", [
    # one add, at engine step 0, and its reader in each step: those of steps 0 and 1 are shared, and step 2 is r0's alone
    (False, tgm.R1, {"instances": 110, "by_request": 113, "shared": 3, "specs": {"AddScalarBf16_v1": 1, "AddScalarF32_v1": 2},
                     "mapped": 107, "distinct": 110, "local_identities": 6, "x09": {"weight-only": 3, "meta": 107}}),
    # an add and its reader in each step, r1 arriving at step 1: those of steps 1 and 2 are shared, and step 0 is r0's alone
    (True, tgm.R1_LATE, {"instances": 112, "by_request": 116, "shared": 4, "specs": {"AddScalarBf16_v1": 2, "AddScalarF32_v1": 2},
                         "mapped": 108, "distinct": 112, "local_identities": 8, "x09": {"weight-only": 4, "meta": 108}}),
], ids=["once", "per-forward-late"])
def test_on_the_toy_a_weight_add_of_a_step_two_requests_run_in_is_shared_and_has_a_local_identity_in_each(
        impl, memory, tmp_path, gemma_toy, per_forward, r1, expect):
    """The record states no request for an instance of a step two requests have rows in, and the step's one request otherwise.  A
    shared instance is among the instances of both requests, each compared with its component, and it has one local identity in each."""
    gp, oracle = gemma_toy([tgm.R0, r1], tmp_path, per_forward)
    res = impl.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "PASS" and [k["status"] for k in res["checks"]] == ["PASS"] * 8, res["reasons"]
    assert _weight_only(res) == expect
    assert sum(e["n_instances"] for e in res["requests"]) == expect["by_request"] == expect["distinct"] + expect["shared"]
    assert tgm._check(res, "G8")["permutations"] == []              # a shared instance sits where its component has it: no step is permuted


@pytest.mark.parametrize("impl", [gm, gmf], ids=["record", "fast"])
def test_on_the_toy_a_shared_instance_is_compared_with_every_requests_component(impl, memory, tmp_path, gemma_toy):
    """The record with the weight add against the Program whose components lack it, and the reverse: each request's first step
    differs from its component's by the add and its reader, and G3, G4 and G7 fail."""
    reqs = [tgm.R0, tgm.R1]
    gp_plain, oracle_plain = tgm.assemble(reqs, tmp_path / "plain")
    gp_adds, oracle_adds = gemma_toy(reqs, tmp_path / "adds")
    # shared instances of the record | r0's first step: executed instances, the fold's and the component's unmatched | r1's unmatched
    for gp, oracle, shared, (fold, comp), r0, r1 in ((gp_plain, oracle_adds, 3, (40, 38), (37, 35), (26, 24)),
                                                     (gp_adds, oracle_plain, 0, (38, 40), (35, 37), (24, 26))):
        res = impl.check(gp, oracle, record_alternate=False)
        assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G3", "G4", "G7"]
        assert tgm._check(res, "G2")["weight_only_shared"] == shared
        assert tgm._check(res, "G4")["problems"][0] == f"r0: local step 0 has {fold} executed instances but the component has {comp}"
        assert tgm._check(res, "G3")["problems"] == [
            f"r0: 3 of 3 executed steps not DAG-equal to the component (first: local step 0, {r0[0]} fold / {r0[1]} component instances unmatched)",
            f"r1: 2 of 2 executed steps not DAG-equal to the component (first: local step 0, {r1[0]} fold / {r1[1]} component instances unmatched)"]


@pytest.mark.parametrize("impl", [gm, gmf], ids=["record", "fast"])
def test_on_the_toy_one_add_issued_in_a_step_a_request_runs_alone_is_that_requests_and_a_later_reader_of_it_is_no_requests(
        impl, memory, tmp_path, gemma_toy):
    """Under the default, with r1 arriving at engine step 1: the record states r0 for the add of step 0, where r0 runs alone, so the
    readers of steps 1 and 2 read a node of r0's and no weight.  The global match has no rule that places them and fails G2 by
    their count; each request's steps then differ from its component's, which holds its own add.  X-09 places the two by data flow,
    as r0's.  What the Match says today of such a record, against this file's stand-in for a Program (each component is the request
    served alone); a test of what it says, not of what a Build declares for it."""
    gp, oracle = gemma_toy([tgm.R0, tgm.R1_LATE], tmp_path)
    res = impl.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G2", "G3", "G4", "G7"]
    g2 = tgm._check(res, "G2")
    assert g2["problems"] == ["2 fold instances attributed to no request"] and (g2["unattributed"], g2["weight_only_shared"]) == (2, 0)
    assert res["x09"]["attribution"]["how"] == {"meta": 108, "dataflow": 2} and res["x09"]["attribution"]["unattributed"] == 0
