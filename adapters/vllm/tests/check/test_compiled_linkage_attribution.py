"""The compiled Commit's per-(step, request) attribution: the ordered boundary linkage ties every
request's rows of every committed step to ITS tokens (ids in == token(k-1) / declared prompt; argmax lm_head(final hidden row) == token(k)).
An equal-shape request swap at B>=2 relabels a step's rows together -- the sampled replay is consistent by construction, the
linkage is what catches it, so it must run over EVERY step and count the pairs it tied.  CPU-only, no engine: the check object is built
without a model and fed committed tensors through a stub."""
from __future__ import annotations

import torch

from verity_vllm.check.replay.compiled_kernel_check import CompiledKernelCheck

V, H = 64, 8


def _check(tokens: dict[str, list[int]], prompts: dict[str, list[int]], arrivals: dict[str, int], committed: dict, rows: dict) -> tuple[CompiledKernelCheck, dict]:
    ck = CompiledKernelCheck.__new__(CompiledKernelCheck)
    ck.tokens_by_req = {k: list(v) for k, v in tokens.items()}
    ck.prompts = {k: list(v) for k, v in prompts.items()}
    ck.arrivals = dict(arrivals)
    ck.request_rows = rows
    ck.pieces = 2
    ck.committed = lambda s, metas, name: committed.get((s, name))               # type: ignore[method-assign]
    ck._embed_kernel = lambda metas: "triton_poi_fused_embedding_0"              # type: ignore[method-assign]
    ck._final_kernel = lambda metas: "triton_red_fused_rms_norm_9"               # type: ignore[method-assign]
    res = {"linkage": {"ids_checked": 0, "ids_equal": 0, "argmax_checked": 0, "argmax_equal": 0, "mismatch": []}, "errors": []}
    return ck, res


class _LMHead:
    def __init__(self) -> None:
        self.weight = torch.eye(V, H).to(torch.bfloat16)      # logits = the first H coordinates: argmax(one-hot(t)) == t for t < H


def _hidden(tok: int) -> torch.Tensor:
    h = torch.zeros(H, dtype=torch.bfloat16)
    h[tok] = 1
    return h


def _world(swap_step: int | None = None):
    """Two requests, both arriving at step 0, three decode steps after a prefill: r0's tokens 1,2,3,4; r1's 5,6,7,1 (last equal to r0's first)."""
    toks = {"r0": [1, 2, 3, 4], "r1": [5, 6, 7, 1]}
    prompts = {"r0": [10, 11, 12], "r1": [20, 21]}
    committed, rows = {}, {}
    # prefill step 0: rows [0,3) r0 and [3,5) r1
    committed[(0, "c00.triton_poi_fused_embedding_0/ids")] = torch.tensor(prompts["r0"] + prompts["r1"])
    committed[(0, "c01.triton_red_fused_rms_norm_9/out")] = torch.stack([_hidden(0)] * 2 + [_hidden(toks["r0"][0])] + [_hidden(0)] + [_hidden(toks["r1"][0])])
    rows[0] = [("r0", 0, 3), ("r1", 3, 5)]
    for s in (1, 2, 3):
        order = ["r0", "r1"]
        if s == swap_step:
            order = ["r1", "r0"]                                   # the attacker's record: the two rows relabelled (equal shape)
        committed[(s, "c00.triton_poi_fused_embedding_0/ids")] = torch.tensor([toks[r][s - 1] for r in order])
        committed[(s, "c01.triton_red_fused_rms_norm_9/out")] = torch.stack([_hidden(toks[r][s]) for r in order])
        rows[s] = [("r0", 0, 1), ("r1", 1, 2)]                    # the engine's row order as recorded (unchanged: the swap is in the bytes)
    return toks, prompts, committed, rows


def test_every_step_and_every_request_row_is_tied_and_counted():
    toks, prompts, committed, rows = _world()
    ck, res = _check(toks, prompts, {"r0": 0, "r1": 0}, committed, rows)
    for s in range(4):
        ck._link_step(s, {}, _LMHead(), res)
    lk = res["linkage"]
    assert lk["ids_checked"] == lk["ids_equal"] == 2 + 3 * 2          # 2 prefill prompt gates + 3 decode steps x 2 requests
    assert lk["argmax_checked"] == lk["argmax_equal"] == 4 * 2
    assert lk["steps_checked"] == 4 and lk["pairs_checked"] == lk["pairs_total"] == 8
    assert not lk["mismatch"] and not lk.get("unmapped_rows")


def test_equal_shape_request_swap_at_an_unsampled_step_is_caught_by_the_exhaustive_linkage():
    toks, prompts, committed, rows = _world(swap_step=2)
    ck, res = _check(toks, prompts, {"r0": 0, "r1": 0}, committed, rows)
    sampled = {0, 1, 3}                                              # the swap is at step 2: no sampled step sees it
    for s in sampled:
        ck._link_step(s, {}, _LMHead(), res)
    assert res["linkage"]["ids_equal"] == res["linkage"]["ids_checked"] and res["linkage"]["argmax_equal"] == res["linkage"]["argmax_checked"]
    ck._link_step(2, {}, _LMHead(), res)                             # the exhaustive pass reaches step 2
    lk = res["linkage"]
    assert lk["ids_equal"] == lk["ids_checked"] - 2 and lk["argmax_equal"] == lk["argmax_checked"] - 2
    assert {(m["step"], m["request"]) for m in lk["mismatch"]} == {(2, "r0"), (2, "r1")}
    assert lk["pairs_checked"] == lk["pairs_total"] == 8            # every pair examined; the tally is coverage, the verdict is ids/argmax equality


def test_a_row_the_workload_does_not_name_is_counted_not_tied():
    toks, prompts, committed, rows = _world()
    rows[3] = [("r0", 0, 1), ("r9", 1, 2)]                           # the engine names a request the workload does not know
    ck, res = _check(toks, prompts, {"r0": 0, "r1": 0}, committed, rows)
    for s in range(4):
        ck._link_step(s, {}, _LMHead(), res)
    lk = res["linkage"]
    assert lk["unmapped_rows"] == 1 and lk["pairs_total"] == 8 and lk["pairs_checked"] == 7


def test_two_rows_with_the_same_prev_and_next_token_at_one_step_are_counted_as_coincident():
    toks = {"r0": [1, 2, 3, 4], "r1": [5, 2, 3, 6]}                  # at k = 2 both have (prev, next) = (2, 3)
    prompts = {"r0": [10, 11, 12], "r1": [20, 21]}
    committed, rows = {}, {}
    committed[(0, "c00.triton_poi_fused_embedding_0/ids")] = torch.tensor(prompts["r0"] + prompts["r1"])
    committed[(0, "c01.triton_red_fused_rms_norm_9/out")] = torch.stack([_hidden(0)] * 2 + [_hidden(1)] + [_hidden(0)] + [_hidden(5)])
    rows[0] = [("r0", 0, 3), ("r1", 3, 5)]
    for s in (1, 2, 3):
        committed[(s, "c00.triton_poi_fused_embedding_0/ids")] = torch.tensor([toks[r][s - 1] for r in ("r0", "r1")])
        committed[(s, "c01.triton_red_fused_rms_norm_9/out")] = torch.stack([_hidden(toks[r][s]) for r in ("r0", "r1")])
        rows[s] = [("r0", 0, 1), ("r1", 1, 2)]
    ck, res = _check(toks, prompts, {"r0": 0, "r1": 0}, committed, rows)
    for s in range(4):
        ck._link_step(s, {}, _LMHead(), res)
    lk = res["linkage"]
    assert lk["ids_equal"] == lk["ids_checked"] and lk["argmax_equal"] == lk["argmax_checked"]   # every gate passes ...
    # ... but with no positions buffer committed, step 2 cannot tell r0 from r1 (the key degrades to the tokens)
    assert lk["coincident_pairs"] == 1 and lk["coincident_sample"] == [{"step": 2, "pairs": [[2, 3, None]]}]
    assert lk["coincident_pairs_tokens"] == 1 and lk.get("pos_checked", 0) == 0


POS = "c01.triton_poi_fused_1/pos"


def _coincident_world_with_positions(bad_step: int | None = None):
    """The token-coincident world above plus the committed positions buffer of every step (prompt lengths 3 and 2, so the two decode
    rows of step k sit at positions 3+k-1 and 2+k-1: the position gate tells them apart)."""
    toks = {"r0": [1, 2, 3, 4], "r1": [5, 2, 3, 6]}
    prompts = {"r0": [10, 11, 12], "r1": [20, 21]}
    committed, rows = {}, {}
    committed[(0, "c00.triton_poi_fused_embedding_0/ids")] = torch.tensor(prompts["r0"] + prompts["r1"])
    committed[(0, "c01.triton_red_fused_rms_norm_9/out")] = torch.stack([_hidden(0)] * 2 + [_hidden(1)] + [_hidden(0)] + [_hidden(5)])
    committed[(0, POS)] = torch.tensor([0, 1, 2, 0, 1])
    rows[0] = [("r0", 0, 3), ("r1", 3, 5)]
    for s in (1, 2, 3):
        committed[(s, "c00.triton_poi_fused_embedding_0/ids")] = torch.tensor([toks[r][s - 1] for r in ("r0", "r1")])
        committed[(s, "c01.triton_red_fused_rms_norm_9/out")] = torch.stack([_hidden(toks[r][s]) for r in ("r0", "r1")])
        committed[(s, POS)] = torch.tensor([3 + s - 1, 2 + s - 1]) if s != bad_step else torch.tensor([3 + s - 1, 3 + s - 1])
        rows[s] = [("r0", 0, 1), ("r1", 1, 2)]
    return toks, prompts, committed, rows


def test_the_position_gate_tells_token_coincident_rows_apart_and_is_counted():
    toks, prompts, committed, rows = _coincident_world_with_positions()
    ck, res = _check(toks, prompts, {"r0": 0, "r1": 0}, committed, rows)
    for s in range(4):
        ck._link_step(s, {POS: object()}, _LMHead(), res)
    lk = res["linkage"]
    assert lk["pos_checked"] == lk["pos_equal"] == 2 + 3 * 2        # 2 prefill ranges + 3 decode steps x 2 rows
    assert lk["coincident_pairs_tokens"] == 1                        # the token-only residual is still named ...
    assert lk.get("coincident_pairs", 0) == 0 and lk["position_gate"] == POS   # ... and the positions resolve it
    assert lk["coincident_tokens_sample"] == [{"step": 2, "pairs": [[2, 3]], "positions_differ": True}]


def test_a_wrong_committed_position_is_a_linkage_mismatch_and_leaves_the_rows_coincident():
    toks, prompts, committed, rows = _coincident_world_with_positions(bad_step=2)
    ck, res = _check(toks, prompts, {"r0": 0, "r1": 0}, committed, rows)
    for s in range(4):
        ck._link_step(s, {POS: object()}, _LMHead(), res)
    lk = res["linkage"]
    assert lk["pos_checked"] == 8 and lk["pos_equal"] == 7
    assert [m for m in lk["mismatch"] if "positions" in m] == [{"step": 2, "request": "r1", "positions": [4], "expected": [3]}]
    assert lk["coincident_pairs"] == 1 and lk["coincident_sample"] == [{"step": 2, "pairs": [[2, 3, 4]]}]
