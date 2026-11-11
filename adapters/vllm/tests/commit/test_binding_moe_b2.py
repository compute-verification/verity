"""The fused-MoE block-stream planes at B>1 are per-token (pair-major) members split by the step's request->row
map, like every other per-token member -- never the whole batched plane bound to every request (the hidden-launch rule).

Through the committer's offline core (no engine, no CUDA): a step stream with two requests (r0 = 100 token rows, r1 = 200 rows), the six
MoE planes of one layer (rows == tokens for router_w/router_ids/up/down/sum; rows == tokens*TOPK for act) and a few dense members.

| check                                                                     | expected                         | test |
| r1's plane entry: request-local range [0, rows_1*w) at global offset rows_0*w | positive                         | test_moe_planes_split_by_request_rows |
| act plane: TOPK x the request's token rows                                | positive                         | test_moe_planes_split_by_request_rows |
| a required range beyond r1's own rows (but inside the batched plane)      | coverage FAIL (passed before)    | test_range_beyond_own_rows_is_not_covered_by_the_batched_plane |
| B=1-style map (no row map) against a two-request population              | every r1 MoE identity missing    | test_omitted_request_moe_planes_missing |
| swapped request->row map with unequal shapes                             | coverage FAIL for the larger one | test_swapped_request_rows_rejected |
"""
from __future__ import annotations

import hashlib
import random

from verity_vllm.commit import binding as B, step_rows
from verity_vllm.commit.committer.native_host import NativeHostCommitter

CHUNK = step_rows.ROW_BYTES
PD = hashlib.sha256(b"program").hexdigest()
TOPK = 8
ROWS = {"r0": (0, 100), "r1": (100, 300)}            # token rows of the step (r0 first, r1 second; 300 tokens)
TOKENS = 300
STEPS = 2


def step_stream(seed: int = 3) -> list[list[tuple[str, bytes]]]:
    """One honest step: dense per-token members (nbytes == tokens: width 1 in the offline uint8 layout) + the six MoE planes of layer 0."""
    rng = random.Random(seed)
    out = []
    for _s in range(STEPS):
        t = [("model.embed_tokens/in", rng.randbytes(TOKENS)), ("model.embed_tokens", rng.randbytes(TOKENS)),
             ("model.layers.0.self_attn.qkv_proj/0", rng.randbytes(TOKENS)),
             ("moe/L0/router_w", rng.randbytes(TOKENS)), ("moe/L0/router_ids", rng.randbytes(TOKENS)),
             ("moe/L0/up", rng.randbytes(TOKENS)), ("moe/L0/act", rng.randbytes(TOKENS * TOPK)),
             ("moe/L0/down", rng.randbytes(TOKENS)), ("moe/L0/sum", rng.randbytes(TOKENS)),
             ("runner.sampler/sampled_token_ids", rng.randbytes(2))]
        out.append(t)
    return out


def commit(stream, **map_kw):
    c = NativeHostCommitter(None, run_id="run", hash_threads=1, program_digest=hashlib.sha256(b"program").digest())
    run = c.commit_offline(stream)
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, chunk=CHUNK, **map_kw)
    return c, run, bmap


def rows_ok() -> dict:
    return {s: [("r0", *ROWS["r0"]), ("r1", *ROWS["r1"])] for s in range(STEPS)}


def moe_population(rows: dict[str, tuple[int, int]]) -> list[tuple]:
    """The manifest's reading: per request, request-local ranges over ITS token rows (x TOPK for act), whatever the routing chose."""
    pop = []
    for rid, (lo, hi) in rows.items():
        n = hi - lo
        for s in range(STEPS):
            for plane, w in (("router_w", 1), ("router_ids", 1), ("up", 1), ("act", TOPK), ("down", 1), ("sum", 1)):
                pop.append(B.identity(PD, rid, s, 0, "moe/L0", plane, 0, n * w, 0))
    return pop


def entries(bmap, rid: str, plane: str, step: int = 0):
    return [e for e in bmap["entries"] if e["identity"]["request_id"] == rid and e["identity"]["op_path"] == "moe/L0"
            and e["identity"]["output_member"] == plane and e["identity"]["step"] == step]


# ---------------------------------------------------------------------- positive

def test_moe_planes_split_by_request_rows():
    _c, _run, bmap = commit(step_stream(), request_rows=rows_ok())
    for plane, f in (("router_w", 1), ("router_ids", 1), ("up", 1), ("act", TOPK), ("down", 1), ("sum", 1)):
        e0, = entries(bmap, "r0", plane)
        e1, = entries(bmap, "r1", plane)
        assert e0["identity"]["element_range"] == [0, 100 * f] and e0["element_offset"] == 0, (plane, e0["identity"], e0["element_offset"])
        assert e1["identity"]["element_range"] == [0, 200 * f] and e1["element_offset"] == 100 * f, (plane, e1["identity"], e1["element_offset"])
        assert e0["identity"]["element_range"][1] + e1["identity"]["element_range"][1] == TOKENS * f          # the two requests tile the plane
    cov = B.coverage_check(moe_population(ROWS), bmap)
    assert cov["ok"], cov


# ---------------------------------------------------------------------- negatives

def test_range_beyond_own_rows_is_not_covered_by_the_batched_plane():
    # r1 has 200 rows; a required range of 250 rows lies inside the 300-row batched plane -> with a whole-plane binding this passed by
    # containment (r1's identity pointed at r0's rows); with per-request rows it is a missing identity
    _c, _run, bmap = commit(step_stream(), request_rows=rows_ok())
    over = [B.identity(PD, "r1", 0, 0, "moe/L0", "up", 0, 250, 0), B.identity(PD, "r1", 0, 0, "moe/L0", "act", 0, 250 * TOPK, 0)]
    cov = B.coverage_check(over, bmap)
    assert not cov["ok"] and cov["missing_n"] == 2, cov
    # and r0's own range is still covered (the split does not shrink the honest bindings)
    assert B.coverage_check([B.identity(PD, "r0", 0, 0, "moe/L0", "up", 0, 100, 0)], bmap)["ok"]


def test_omitted_request_moe_planes_missing():
    _c, _run, bmap = commit(step_stream())                                     # B=1-style map: everything binds r0
    cov = B.coverage_check(moe_population(ROWS), bmap)
    assert not cov["ok"] and {m["request_id"] for m in cov["missing"]} == {"r1"}
    assert cov["bound_n"] == len(moe_population(ROWS)) // 2


def test_swapped_request_rows_rejected():
    # the row map names r1's rows as r0's and vice versa: the request with more rows (r1: 200) is required beyond the 100 rows it now binds
    swapped = {s: [("r1", *ROWS["r0"]), ("r0", *ROWS["r1"])] for s in range(STEPS)}
    _c, _run, bmap = commit(step_stream(), request_rows=swapped)
    cov = B.coverage_check(moe_population(ROWS), bmap)
    assert not cov["ok"] and {m["request_id"] for m in cov["missing"]} == {"r1"}, cov
    assert cov["missing_n"] == 6 * STEPS                                        # every plane of r1 in every step
