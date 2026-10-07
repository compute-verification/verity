"""[R15 hopper HOP-03, M-0225] The FA3 tile rule reads the LAUNCH's max_seqlen_q, not the request's own rows.

Torch-free: the registry helpers (`parse_launch_context`, `launch_seqlen_q`, `attention_geometry` on a cc 9.0 / cc 8.9 profile) and the
workload-side derivation (`verity_vllm.pipeline.launch_context`).  The measured case: Llama-3.2-1B (NH 32, KVH 8, D 64: g = 4) -- an 8-token
prompt alone runs the 64 x 192 tile (kBlockN 192), the same prompt in a prefill launch with a 1024-token prompt runs 192 x 128 (kBlockN 128).
"""
from __future__ import annotations

import json

import pytest

from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.frontend import targets as TG
from verity_vllm.pipeline import launch_context as LC

def _prof(cc):
    return TargetProfile.default().with_overrides(compute_capability=tuple(cc))


def test_parse_launch_context_forms():
    assert TG.parse_launch_context(None) is None
    assert TG.parse_launch_context("") is None
    assert TG.parse_launch_context(1024) == {"max_seqlen_q_by_step": {0: 1024}, "conflict": False, "declared": True}
    assert TG.parse_launch_context("1024")["max_seqlen_q_by_step"] == {0: 1024}
    assert TG.parse_launch_context("[1024,1,1]")["max_seqlen_q_by_step"] == {0: 1024, 1: 1, 2: 1}
    assert TG.parse_launch_context({"0": 1024, "3": 7})["max_seqlen_q_by_step"] == {0: 1024, 3: 7}
    assert TG.parse_launch_context("conflict")["conflict"] is True
    with pytest.raises(ValueError):
        TG.parse_launch_context(0)
    with pytest.raises(ValueError):
        TG.parse_launch_context("[1024,0]")


def test_launch_seqlen_q_is_the_launch_max_never_below_own_rows():
    # no context: own rows (B1 semantics)
    assert TG.launch_seqlen_q(8, 0, None) == (8, "own rows")
    assert TG.launch_seqlen_q(1, 5, None) == (1, "own rows")
    ctx = TG.parse_launch_context([1024, 1, 1])
    # the 8-token prompt prefilled in a launch whose longest prompt has 1024 rows
    assert TG.launch_seqlen_q(8, 0, ctx) == (1024, "declared launch context")
    # its decode steps: everyone decodes -> 1
    assert TG.launch_seqlen_q(1, 1, ctx) == (1, "declared launch context")
    # [HOP-03b, rev-pm M-0326 T2] a context BELOW the request's own rows is another shape's context: refused, never corrected to max()
    with pytest.raises(TG.LaunchContextInvalid):
        TG.launch_seqlen_q(1024, 0, TG.parse_launch_context([8]))
    # [HOP-03b, rev-pm M-0326 T1] a step the context does not name is refused, never completed with own rows (the M-0225 bug via a truncated CLI)
    with pytest.raises(TG.LaunchContextInvalid):
        TG.launch_seqlen_q(1, 7, ctx)
    with pytest.raises(TG.LaunchContextConflict):
        TG.launch_seqlen_q(8, 0, TG.parse_launch_context("conflict"))


def test_fa3_tile_flips_with_the_launch_context_fa2_does_not():
    h100 = _prof((9, 0))
    assert h100.effective_flash_attn_version() == 3
    NH, KVH, D = 32, 8, 64
    alone = TG.attention_geometry(h100, D, TG.launch_seqlen_q(8, 0, None)[0], NH, KVH)
    batched = TG.attention_geometry(h100, D, TG.launch_seqlen_q(8, 0, TG.parse_launch_context([1024, 1]))[0], NH, KVH)
    assert alone["BN"] == 192 and batched["BN"] == 128           # M-0225: the measured divergence, derived vs observed
    long_alone = TG.attention_geometry(h100, D, TG.launch_seqlen_q(1024, 0, None)[0], NH, KVH)
    long_batched = TG.attention_geometry(h100, D, TG.launch_seqlen_q(1024, 0, TG.parse_launch_context([1024, 1]))[0], NH, KVH)
    assert long_alone["BN"] == long_batched["BN"] == 128        # LP * g > 64: the context changes nothing (digest unchanged)
    decode = TG.attention_geometry(h100, D, TG.launch_seqlen_q(1, 3, TG.parse_launch_context([1024, 1, 1, 1]))[0], NH, KVH)
    assert decode["BN"] == 192                                    # decode-only launch: one MMA warpgroup tile
    # a late arrival prefilling 300 rows while this request decodes: that engine step's launch runs BN 128 for the decoder too
    staggered = TG.attention_geometry(h100, D, TG.launch_seqlen_q(1, 2, TG.parse_launch_context([64, 1, 300]))[0], NH, KVH)
    assert staggered["BN"] == 128
    # threshold: LP * g <= 64 -> 16 tokens at g = 4 is the largest one-warpgroup prompt
    assert TG.fa3_kblock_n(64, 16, 4) == 192 and TG.fa3_kblock_n(64, 17, 4) == 128
    l40s = _prof((8, 9))
    assert l40s.effective_flash_attn_version() == 2
    for sq in (8, 1024):
        g2 = TG.attention_geometry(l40s, D, sq, NH, KVH)
        assert g2["BN"] == 128 and g2["fa_version"] == 2         # FA2: kBlockN is per head dim; the launch context is inert


def _wl(reqs):
    return {"requests": [{"request_id": f"r{i}", "prompt_len": lp, "max_tokens": c, "arrive_step": a} for i, (lp, c, a) in enumerate(reqs)]}


def test_shape_contexts_co_arrival_and_staggered():
    # the measured B8 mixed row: all arrive at step 0 -> every shape's step 0 sees the longest prompt, every decode step 1
    reqs = [(641, 39, 0), (203, 7, 0), (433, 2, 0), (71, 27, 0), (8, 15, 0), (1024, 69, 0), (155, 128, 0), (873, 121, 0)]
    ctx = LC.shape_contexts(_wl(reqs))
    assert ctx[(8, 14)] == [1024] + [1] * 14
    assert ctx[(1024, 68)] == [1024] + [1] * 68
    assert ctx[(155, 127)] == [1024] + [1] * 127
    assert LC.spec_for(_wl(reqs), 8, 14) == json.dumps([1024] + [1] * 14, separators=(",", ":"))
    assert LC.spec_for(_wl(reqs), 1024, 127) == ""                # not a request shape (the row's default build_request): no context
    # staggered: r1 (300 rows) arrives at engine step 2 while r0 decodes -> r0's step 2 launch has max_seqlen_q 300; r1's own step 0 = 300
    st = LC.shape_contexts(_wl([(64, 6, 0), (300, 3, 2)]))
    assert st[(64, 5)] == [64, 1, 300, 1, 1, 1]
    assert st[(300, 2)] == [300, 1, 1]
    # two requests of ONE shape co-scheduled differently -> 'conflict' (fail closed under FA3)
    cf = LC.shape_contexts(_wl([(8, 3, 0), (8, 3, 1), (1024, 2, 0)]))
    assert cf[(8, 2)] == "conflict"
    assert LC.spec_for(_wl([(8, 3, 0), (8, 3, 1), (1024, 2, 0)]), 8, 2) == "conflict"
    # the sweep-block form of the workload (input_lengths / output_caps / arrive_steps) is read too
    sw = {"sweep": {"input_lengths": [8, 1024], "output_caps": [3, 2], "arrive_steps": [0, 0]}}
    assert LC.shape_contexts(sw)[(8, 2)] == [1024, 1, 1]


def test_measured_workload_file():
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "out", "gen", "sweep", "workloads",
                     "llama32-1b__bf16__h100__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json")
    if not os.path.exists(p):
        pytest.skip("workload file not in this checkout")
    w = json.load(open(p))
    assert LC.spec_for(w, 8, 14) == json.dumps([1024] + [1] * 14, separators=(",", ":"))
    assert all(v[0] == 1024 for v in LC.shape_contexts(w).values())


def test_unreadable_manifest_is_an_error_not_no_context():
    # [HOP-03b, rev-pm M-0326 T3] a manifest the module cannot read raises (rc != 0 on the CLI); row_pod's launch_args() fails the shape's
    # Build on rc != 0 instead of treating empty stdout as "no context" (which would build the H100 B >= 2 request Program as a batch of one).
    with pytest.raises((KeyError, TypeError, ValueError)):
        LC.shape_contexts({"requests": [{"foo": 1}]})
    with pytest.raises((KeyError, TypeError, ValueError)):
        LC.shape_contexts({"sweep": {"input_lengths": [8, 1024]}})
    # a shape the workload does not have is genuinely "no context" (the row's default build_request of the longest prompt / largest cap)
    assert LC.spec_for({"sweep": {"input_lengths": [8, 1024], "output_caps": [15, 69]}}, 1024, 127) == ""
