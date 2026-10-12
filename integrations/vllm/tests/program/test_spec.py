"""Lane SD: `Accept`/`Advance` semantics against a Python reference, and the TINY `VerifyBranch` / `ServeSpec`
programs build, evaluate and satisfy the static-schedule obligation (`accepted == A[s]`) when `A` is taken
from an evaluation of the program itself."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.ml.circuits import Program, bind  # noqa: E402
from verity.ml.circuits.evaluate import evaluate_call  # noqa: E402
from verity_vllm.program.registry import spec  # noqa: E402
from verity_vllm.program.registry.b1 import Serve  # noqa: E402
from tests.program.crosscheck import _standalone  # noqa: E402

U32 = 0xFFFFFFFF
TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
TINY_NOBIAS = {**TINY, "BIAS": False, "KVH": 2, "QKV": 96}


def ref_accept(cand: list[int], argmax: list[int]) -> tuple[list[int], int, list[int]]:
    K = len(cand)
    verdict, ok = [], True
    for p in range(K):
        ok = ok and cand[p] == argmax[p]
        verdict.append(1 if ok else 0)
    a = sum(verdict)
    appended = [argmax[j] if j <= a else -1 for j in range(K + 1)]
    return verdict, a, appended


@pytest.mark.parametrize("K", [1, 3, 5])
def test_accept_matches_reference(K):
    prog, call = _standalone(bind(spec.Accept, K=K))
    rng = np.random.default_rng(7 + K)
    cases = [([1] * K, [1] * (K + 1)), ([1] * K, [2] + [1] * K), ([5, 5, 5, 5, 5][:K], [5, 5, 5, 5, 5][:K] + [9])]
    for _ in range(20):
        cand = [int(x) for x in rng.integers(0, 3, K)]
        arg = [int(x) for x in rng.integers(0, 3, K + 1)]
        cases.append((cand, arg))
    for cand, arg in cases:
        out = evaluate_call(prog.circuit, call, cand + arg)
        verdict, a, appended = ref_accept(cand, arg)
        got_v, got_a, got_app = list(out[:K]), out[K], [x - (1 << 32) if x & 0x80000000 else x for x in out[K + 1:]]
        assert (got_v, got_a, got_app) == (verdict, a, appended), (cand, arg)


@pytest.mark.parametrize("K", [1, 3])
def test_advance_matches_reference(K):
    prog, call = _standalone(bind(spec.Advance, K=K))
    for pos in (0, 17):
        for verdict in itertools.product([0, 1], repeat=K):
            # only prefix-closed verdicts occur (Accept produces them); Advance is total anyway
            out = evaluate_call(prog.circuit, call, [pos] + list(verdict))
            live = [1] + list(verdict)
            committed = sum(live)
            assert out[0] == (pos + committed) & U32
            assert list(out[1:K + 2]) == live
            assert out[K + 2] == committed
            assert out[K + 3] == (K + 1) - committed


def _random_weights(C: dict, rng) -> list[int]:
    from verity_vllm.program.registry.b1 import weights_type
    n = weights_type(C).leaves
    w = (rng.standard_normal(n) * 0.05).astype(np.float32).view(np.uint32) >> 16
    return [int(x) for x in w]


def test_verify_branch_tiny_evaluates_and_accepts_own_argmax():
    """A branch whose candidates are the target's own greedy tokens is accepted in full; a corrupted first
    candidate is rejected at slot 0."""
    rng = np.random.default_rng(3)
    K, LP = 2, 5
    W = _random_weights(TINY, rng)
    # get the target's greedy continuation with the plain Serve program, then verify it as a branch
    serve = Program(bind(Serve, C=TINY, LP=LP - K, STEPS=K))
    prompt = [int(x) for x in rng.integers(0, TINY["V"], LP - K)]
    toks = _eval_program(serve, [prompt, W])
    assert len(toks) == K + 1
    branch = prompt + list(toks[:K])  # committed tape + K candidates (the target's own greedy tokens)
    vb = Program(bind(spec.VerifyBranch, C=TINY, LP=LP, K=K))
    out = _eval_program(vb, [branch, W])
    argmax, verdict, acc, appended = out[:K + 1], out[K + 1:2 * K + 1], out[2 * K + 1], out[2 * K + 2:]
    assert list(argmax) == list(toks), "the branch prefill reproduces the serving program's greedy tokens"
    assert list(verdict) == [1] * K and acc == K and list(appended) == list(toks)
    bad = list(branch)
    bad[LP - K] = (bad[LP - K] + 1) % TINY["V"]
    out = _eval_program(vb, [bad, W])
    verdict, acc = out[K + 1:2 * K + 1], out[2 * K + 1]
    assert list(verdict) == [0] * K and acc == 0


def _eval_program(prog: Program, inputs: list[list[int]]) -> list[int]:
    from verity.ml.circuits.evaluate import evaluate_region
    from verity.ml.circuits.layout import resolve
    C = prog.circuit
    t = {}
    for pi, vals in enumerate(inputs):
        coll = prog.input_coll(pi)
        assert coll.type.leaves == len(vals), (pi, coll.type.leaves, len(vals))
        for j, v in enumerate(vals):
            t[resolve(C.root_scope, coll.refs[j])] = v
    evaluate_region(C, C[0:prog.gates], t)
    return [t[g] for g in C.root_outputs()]


@pytest.mark.slow
def test_serve_spec_tiny_schedule_obligation():
    """Evaluate ServeSpec under a guessed schedule, read the accepted lengths it computes, rebuild under that
    schedule: the accepted outputs then equal A (the equality obligation), and the appended slots are -1
    exactly where the schedule says they are dead."""
    rng = np.random.default_rng(11)
    K, LP = 2, 3
    W, WD = _random_weights(TINY, rng), _random_weights(TINY_NOBIAS, rng)
    prompt = [int(x) for x in rng.integers(0, TINY["V"], LP)]
    A = (K, K)
    for _ in range(3):
        prog = Program(bind(spec.ServeSpec, C=TINY, CD=TINY_NOBIAS, LP=LP, K=K, A=A))
        out = _eval_program(prog, [prompt, W, WD])
        per_step = [(out[s * (K + 2)], out[s * (K + 2) + 1:(s + 1) * (K + 2)]) for s in range(len(A))]
        got = tuple(a for a, _ in per_step)
        if got == A:
            for a, appended in per_step:
                dead = [j for j, x in enumerate(appended) if x == U32]
                assert dead == list(range(a + 1, K + 1)), (a, appended)
            return
        A = got  # a step's accepted length can only depend on the steps before it, so this converges
    pytest.fail(f"schedule did not converge: {A}")
