"""Red-team (REDTEAM-ADVICE, phase 9 target 2) against ENC-REPLAY's ReplayStep_v1 (`replay_step`), the consumer side of
the prefix-cache witness. The module says it "takes each admitting row's prefix-cache hit as witness that it only
bounds"; these cases measure that freedom.

Findings, each confirmed executably below:

1. The prefix hit `hit` is a FREE WITNESS within the bound. On one committed state and one arrival the step accepts
   every hit the bound allows (`h*B < ntok`, `evict <= hit`), each producing a different, valid schedule: a larger hit
   schedules fewer tokens, so a prover minimises the compute charged by claiming the largest hit. The step derives the
   schedule FROM the hit but never the hit; pinning it is CacheStep_v1's (ENC-DERIVED), and wiring that pin into this
   input is the compile unit's.
2. The returns `ret` are a FREE WITNESS within the bound (`ret <= held`). A stopping request's return bends the free
   pool, and the pool decides later admissions, so the prover shapes the pool by its returns. The honest return is
   ENC-DERIVED's; nothing here derives it.

(The bounds themselves -- evict <= hit, h*B < ntok, ret <= held -- are enforced, and the author's suite already
regresses them; the point here is the width of the accepted band inside those bounds.)
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.core.protocols.certifier import replay_step as RS


def _engine(**kw):
    e = dict(max_num_seqs=256, max_num_batched_tokens=64, max_model_len=128, num_gpu_blocks=24, block_size=16,
             enable_chunked_prefill=True, enable_prefix_caching=True, long_prefill_token_threshold=0, watermark=0.0,
             scheduler_reserve_full_isl=True)
    e.update(kw)
    return e


def _both(st, state, inp):
    S = SimpleNamespace(**st)
    ref = RS.reference(S, state, inp)
    (got,) = RS.evaluate(st, [(state, inp)])
    assert got == ref
    return ref


def test_the_prefix_hit_is_free_witness_within_the_bound():
    """Finding 1. One empty engine, one arrival of 40 tokens (block 16). The step accepts hit 0, 1 and 2, scheduling
    40, 24 and 8 tokens; hit 3 (h*B >= ntok) is refused. A prover claims hit 2 to charge 8 tokens of compute where the
    honest hit is 0 and the real work is 40."""
    st = RS.statics(2, 2, 2, _engine(), 7)
    S = SimpleNamespace(**st)
    empty = RS.empty(S)
    arrival = (RS.Req(0, 40, 3),)

    scheduled = {}
    for h in range(4):
        out = _both(st, empty, RS.Inputs(arrival, bits=(1, 0), hit=(h, 0, 0, 0), evict=(0, 0, 0, 0)))
        if out.ok:
            assert len(out.rows) == 1
            _p, _id, _kind, start, n = out.rows[0]
            scheduled[h] = (start, n)
    assert scheduled == {0: (0, 40), 1: (16, 24), 2: (32, 8)}, scheduled
    assert 3 not in scheduled, "hit 3 reaches the last token and is refused"
    print(f"\n[replay-step] hit free within the bound: hit -> (start, tokens) = {scheduled}; "
          f"claiming hit 2 charges 8 tokens where hit 0 charges 40")


def test_the_returns_are_free_witness_within_the_bound():
    """Finding 2. A single running request that stops this step holds 3 blocks. The step accepts every return 0..3,
    each leaving a different free pool; 4 (more than held) is refused. The pool then decides later admissions, so the
    return is a lever the prover holds inside the bound."""
    st = RS.statics(2, 2, 2, _engine(num_gpu_blocks=24), 7)
    S = SimpleNamespace(**st)
    # out = O - 1 and comp = P + out - 1: one more token samples the last output and the request stops.
    done = RS.State((RS.Req(0, 40, 1, 0, 39),), (), 4)
    frees = {}
    for r in range(5):
        out = _both(st, done, RS.Inputs((), bits=(), ret=(r, 0)))
        if out.ok:
            assert not out.state.running, "the request stopped"
            frees[r] = out.state.free
    assert frees == {0: 4, 1: 5, 2: 6, 3: 7}, frees
    assert 4 not in frees, "a return past what it holds (3 blocks) is refused"
    print(f"\n[replay-step] return free within the bound: ret -> next free = {frees}; "
          f"4 (> held = 3) refused")
