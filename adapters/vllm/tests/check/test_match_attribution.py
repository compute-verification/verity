"""`verity_vllm.check.match.attribution` after the meta channel, the paths of the global match no other CPU test runs: a fold
instance whose row states no request is shared when it reads weights alone (Gemma-2's RMSNorm scale `1 + w`), is otherwise placed
by the rows of its engine step (`occurrence.attribute`), and is otherwise attributed to no request, which G2 fails by name.

The tests read a stub of the fold's view (no Program is decoded), the one `test_batch_decomp_named_paths.py` holds X-09's copy of
the rule to; here in both forms a loader gives an instance's operands, the record's runs and the compact form's progressions."""
from __future__ import annotations

import pytest

from tests.check import test_batch_decomp_named_paths as paths
from verity_vllm.check.match import batch_decomp, occurrence
from verity_vllm.check.match.attribution import attribute_instances, requests_observed
from verity_vllm.check.match.match_run import MatchRun

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
def test_an_instance_that_reads_weights_alone_is_no_requests_and_is_in_the_leg_of_each_request_with_rows_in_its_step(compact):
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
        "attributed": 11,                                           # by leg: a shared instance counts in each leg that holds it
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
