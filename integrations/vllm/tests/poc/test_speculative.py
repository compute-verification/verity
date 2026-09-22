"""Lane SD: the `Accept` / `Advance` composites (`verity_vllm.registry.spec`) on the recorded speculative transitions
(`fixtures/speculative/*.boundary.json`: every step's speculative record as written by the SD capture on the pod,
6 requests, k=3), CPU only.  For each recorded decode block: Accept(candidates, argmaxes) reproduces the recorded
verdicts, accepted length and the `sampler.accept` row (committed tokens then -1); Advance reproduces the recorded
`kv.committed` / `kv.discarded` split (before termination truncation) and the block's slot liveness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "vllm-poc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verity.ir import bind  # noqa: E402
from verity.ir.evaluate import evaluate_call  # noqa: E402
from verity_vllm.registry import spec  # noqa: E402
from verity_vllm.registry.crosscheck import _standalone  # noqa: E402

FIX = ROOT / "fixtures" / "speculative"
U32 = 0xFFFFFFFF


def _blocks():
    out = []
    for f in sorted(FIX.glob("*.boundary.json")):
        b = json.loads(f.read_text())
        for s in b["steps"]:
            sp = s.get("speculative") or {}
            if sp.get("k"):
                out.append((b["request_id"], s["step"], s["positions"], sp))
    return out


BLOCKS = _blocks()


def test_fixture_covers_the_five_cases():
    acc = [sp["accepted"] for _, _, _, sp in BLOCKS]
    k = BLOCKS[0][3]["k"]
    assert k in acc, "all accepted"
    assert 0 in acc, "first rejected"
    assert any(0 < a < k for a in acc), "partial then rejected"
    assert any(len(pos) < sp["accepted"] + 1 for _, _, pos, sp in BLOCKS), "termination inside a block"
    by_engine_step: dict[int, set[int]] = {}
    for _, _, _, sp in BLOCKS:
        by_engine_step.setdefault(sp["engine_step"], set()).add(sp["accepted"])
    assert any(len(v) > 1 for v in by_engine_step.values()), "different accepted lengths in one batch step"


@pytest.mark.parametrize("rid,step,positions,sp", BLOCKS, ids=[f"{r}-s{s}" for r, s, _, _ in BLOCKS])
def test_accept_reproduces_recorded_transition(rid, step, positions, sp):
    K, a = sp["k"], sp["accepted"]
    prog, call = _standalone(bind(spec.Accept, K=K))
    cand = [int(c) for c in sp["candidates"]]
    committed = [int(x) for x in sp["committed_tokens"]]
    assert len(committed) == a + 1
    # the target argmaxes: at slots <= a they are the committed tokens (accepted candidates equal them, the bonus token
    # is the argmax at slot a); at the dead slots any value not equal to the candidate would do — the fixture does not
    # hold them (they are leaves of the replay unit), so take a value that differs from the candidate
    argmax = committed + [(cand[j] + 1) & 0x7FFFFFFF if j < K else 0 for j in range(a + 1, K + 1)]
    out = evaluate_call(prog.circuit, call, cand + argmax)
    verdict, got_a, appended = list(out[:K]), out[K], [x - (1 << 32) if x & 0x80000000 else x for x in out[K + 1:]]
    assert verdict == [1 if v else 0 for v in sp["verdicts"]]
    assert got_a == a
    assert appended == committed + [-1] * (K - a), "the sampler.accept leaf: committed tokens then -1"


@pytest.mark.parametrize("rid,step,positions,sp", BLOCKS, ids=[f"{r}-s{s}" for r, s, _, _ in BLOCKS])
def test_advance_reproduces_recorded_state_effect(rid, step, positions, sp):
    K, a, slots = sp["k"], sp["accepted"], sp["slots"]
    prog, call = _standalone(bind(spec.Advance, K=K))
    verdict = [1 if v else 0 for v in sp["verdicts"]]
    out = evaluate_call(prog.circuit, call, [slots[0]] + verdict)
    nxt, live, committed, discarded = out[0], list(out[1:K + 2]), out[K + 2], out[K + 3]
    assert committed == a + 1 and discarded == K - a
    assert nxt == slots[0] + a + 1
    # the block's liveness before termination truncation: slots 0..a live, the rest dead
    assert live == [1 if j <= a else 0 for j in range(K + 1)]
    kv_committed_block = [t for t in slots if t in set(sp["kv"]["committed"])]
    truncated = len(positions) < a + 1  # termination inside the block: the request's tape ends before slot a
    if not truncated:
        assert kv_committed_block == slots[: a + 1]
        assert sp["kv"]["discarded"] == slots[a + 1:]
        assert sp["advance"]["n_next"] == nxt
    else:
        assert kv_committed_block == positions, "termination keeps only the positions the tape reaches"
        assert set(sp["kv"]["discarded"]) >= set(slots[a + 1:])
