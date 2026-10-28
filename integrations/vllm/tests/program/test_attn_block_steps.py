"""A key block cut into its softmax step and its P·V groups (`fa2_check_inf.AttnBlockSoftmax_v1`, `fa3_check_inf.AttnBlockSoftmax_v2`,
`fa2_check_inf.AttnBlockPv_v1`) is the block word for word: the softmax, then each group of PK keys in order reading the last one's
O, equals `AttnBlock_v5{FIRST=False}` and `AttnBlock_v4{FIRST=False}` at both CHECKs, and at the first visited block
`AttnBlockFirst_v1` at every key count, its last group padded to a k-step; at LAST (the head's last key block) the same less the
running max, the one output no later step reads.  Inputs are `test_dead_free`'s: bf16 edge words,
random words and normals, and all -inf keys, under running states that a first block left and f32 edge states (a running max of
-inf, where the guards differ, NaN, +inf).  And at Llama-3.2-1B's 64-dim heads over kBlockN 192, a unit each step, every unit
passes `interface.check_proof_block` with its keys and value slices a row each.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.liveness import dead_gates
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import dead_free as DF, fa2_check_inf as F2, fa3_check_inf as F3, hopper, prims as P
from verity.ml.windows.interface import MAX_ROWS, check_proof_block, unit_rows

RNG = np.random.default_rng(20261028)
BF16_EDGES = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F7F, 0xFF7F, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x3F81]
#: f32 running maxima at the edges: -inf (every key so far masked), +inf, NaN, ±0, the largest finite, a subnormal, 1
F32_MAXES = [0xFF800000, 0x7F800000, 0x7FC00000, 0x00000000, 0x80000000, 0x7F7FFFFF, 0x00000001, 0x3F800000]
HOPPER, AMPERE = hopper.HopperBF16WgmmaDot16, P.AmpereBF16TcDot16
D = 16


def _bf16(x) -> list[int]:
    return [int(w) for w in (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16)]


def _cases(n: int) -> list[tuple[list[int], list[int], list[int]]]:
    """(q, K rows, V rows) of n keys: each edge word's constant row, edge words mixed, random words, normals, all -inf keys."""
    rows = [[w] * (2 * n * D + D) for w in BF16_EDGES]
    rows += [[int(w) for w in RNG.choice(BF16_EDGES, 2 * n * D + D)], [int(w) for w in RNG.integers(0, 1 << 16, 2 * n * D + D)]]
    cases = [(r[:D], r[D:D + n * D], r[D + n * D:]) for r in rows]
    cases += [(_bf16(RNG.normal(0, 1, D)), _bf16(RNG.normal(0, 1, n * D)), _bf16(RNG.normal(0, 1, n * D))) for _ in range(3)]
    return cases + [(_bf16(q), [0xFF80] * (n * D), _bf16(RNG.normal(0, 1, n * D)))
                    for q in (RNG.normal(0, 1, D), np.abs(RNG.normal(0, 1, D)))]  # scores NaN, and every score -inf


def _states(BN: int) -> list[tuple[int, list[int], list[int]]]:
    """Running states (m, l, o) in: those `AttnBlockFirst_v1` leaves on normal and all -inf keys, and the f32 edge maxima over
    normal lane sums and O."""
    states = []
    for q, k, v in _cases(BN)[-5:]:
        out = evaluate(bind(DF.AttnBlockFirstV1, D=D, NVIS=BN, BN=BN, DOT=HOPPER), q, k, v)
        states.append((out[0], list(out[1:5]), list(out[5:])))
    f32 = lambda n: [int(w) for w in np.abs(RNG.normal(0, 4, n)).astype(np.float32).view(np.uint32)]  # noqa: E731
    return states + [(m, f32(4), f32(D)) for m in F32_MAXES]


def _cut(softmax, NVIS: int, FIRST: bool, CHECK: bool, PK: int, VS: int, DOT, q, k, v, state=(), LAST=False) -> tuple[int, ...]:
    """The block's (m, l, o) leaves, (l, o) at LAST, from its softmax step and then its P·V groups in order, each reading the last
    one's O."""
    out = evaluate(bind(softmax, D=D, NVIS=NVIS, FIRST=FIRST, DOT=DOT, CHECK=CHECK, PK=PK, LAST=LAST), q, k, *state)
    lanes = 4 if LAST else 5
    at = lanes if FIRST else lanes + D
    o = [] if FIRST else list(out[lanes:at])
    for g, (lo, hi) in enumerate(F2.pv_groups(NVIS, PK)):
        pv = bind(F2.AttnBlockPvV1, D=D, N=hi - lo, VS=VS, DOT=DOT, FIRST=FIRST and g == 0)
        o = list(evaluate(pv, out[at:at + hi - lo], v[lo * D:hi * D], *([] if FIRST and g == 0 else [o])))
        at += hi - lo
    assert at == len(out)
    return (*out[:lanes], *o)


def _dead(fn) -> int:
    return dead_gates(Program(fn), transitive=True).computed


@pytest.mark.parametrize("NVIS,BN,PK,VS,DOT", [(1, 48, 16, 4, HOPPER), (15, 48, 32, 8, HOPPER), (16, 48, 16, 16, HOPPER),
                                               (17, 48, 16, 4, HOPPER), (33, 48, 32, 4, HOPPER), (47, 48, 32, 8, AMPERE),
                                               (48, 48, 48, 4, HOPPER), (48, 48, 16, 16, AMPERE)])
def test_first_block_cut_equals_attn_block_first_v1(NVIS, BN, PK, VS, DOT):
    """The first visited block (`AttnBlockSoftmax_v1` and `_v2` at FIRST, CHECK, then its groups, the first from +0): its key
    counts 1 ... BN, its last group short of a k-step (1, 15, 17, 33, 47) or not, one group (48 at PK 48) or several; at LAST
    (a head of one key block) less its running max."""
    old = bind(DF.AttnBlockFirstV1, D=D, NVIS=NVIS, BN=BN, DOT=DOT)
    for q, k, v in _cases(NVIS):
        want = evaluate(old, q, k, v)
        for softmax in (F2.AttnBlockSoftmaxV1, F3.AttnBlockSoftmaxV2):
            for last in (False, True):
                assert _cut(softmax, NVIS, True, True, PK, VS, DOT, q, k, v, LAST=last) == want[last:], (softmax.id, q, k, v)


@pytest.mark.parametrize("softmax,block", [(F2.AttnBlockSoftmaxV1, F2.AttnBlockV5), (F3.AttnBlockSoftmaxV2, F3.AttnBlockV4)])
@pytest.mark.parametrize("CHECK", [True, False])
@pytest.mark.parametrize("BN,PK,VS", [(48, 16, 4), (48, 32, 8), (32, 32, 16)])
def test_full_block_cut_equals_attn_block(softmax, block, CHECK, BN, PK, VS):
    """A block below the first (FIRST=False, NVIS = BN): FA2's guard against `AttnBlock_v5`, FA3's against `AttnBlock_v4`, guarded
    and not, over running maxima of -inf (where the guards differ) and NaN among the states; at LAST (block 0) less its running
    max."""
    old = bind(block, D=D, NVIS=BN, FIRST=False, BN=BN, DOT=HOPPER, CHECK=CHECK)
    states = _states(BN)
    neg_inf = next(s for s in states if s[0] == F32_MAXES[0])
    for i, (q, k, v) in enumerate(_cases(BN)):
        for j, (m, l, o) in enumerate((neg_inf, states[i % len(states)], states[(i + 7) % len(states)])):
            want, last = evaluate(old, q, k, v, m, l, o), (i + j) % 2 == 1
            got = _cut(softmax, BN, False, CHECK, PK, VS, HOPPER, q, k, v, (m, l, o), LAST=last)
            assert got == want[last:], (last, q, k, v, m, l, o)


def test_the_guards_differ_where_they_should():
    """At an unguarded block whose scores are all -inf after a running max of -inf, FA2's softmax (max * scale guarded, P 0) and
    FA3's (neither guarded, P NaN) differ, so each test above holds its own guard."""
    q, k = _bf16(np.abs(RNG.normal(0, 1, D))), [0xFF80] * (16 * D)
    state = (0xFF800000, [0] * 4, [0] * D)
    fa2, fa3 = (evaluate(bind(s, D=D, NVIS=16, FIRST=False, DOT=HOPPER, CHECK=False, PK=16, LAST=True), q, k, *state)
                for s in (F2.AttnBlockSoftmaxV1, F3.AttnBlockSoftmaxV2))
    assert fa2 != fa3


@pytest.mark.parametrize("fn", [bind(F2.AttnBlockSoftmaxV1, D=D, NVIS=33, FIRST=True, DOT=HOPPER, CHECK=True, PK=16, LAST=False),
                                bind(F2.AttnBlockSoftmaxV1, D=D, NVIS=33, FIRST=True, DOT=HOPPER, CHECK=True, PK=16, LAST=True),
                                bind(F2.AttnBlockSoftmaxV1, D=D, NVIS=48, FIRST=False, DOT=HOPPER, CHECK=False, PK=32, LAST=False),
                                bind(F2.AttnBlockSoftmaxV1, D=D, NVIS=48, FIRST=False, DOT=HOPPER, CHECK=False, PK=32, LAST=True),
                                bind(F3.AttnBlockSoftmaxV2, D=D, NVIS=48, FIRST=False, DOT=HOPPER, CHECK=False, PK=32, LAST=False),
                                bind(F3.AttnBlockSoftmaxV2, D=D, NVIS=48, FIRST=False, DOT=HOPPER, CHECK=True, PK=32, LAST=True),
                                bind(F2.AttnBlockPvV1, D=D, N=17, VS=4, DOT=HOPPER, FIRST=True),
                                bind(F2.AttnBlockPvV1, D=D, N=32, VS=8, DOT=HOPPER, FIRST=False)])
def test_steps_have_no_dead_gate(fn):
    """At LAST too: the running max it no longer returns is what P and the rescale read."""
    assert _dead(fn) == 0


def test_refused_bindings():
    for kw in ({"PK": 24}, {"PK": 0}, {"NVIS": 0}, {"FIRST": True, "CHECK": False}):
        with pytest.raises(ValueError):
            bind(F2.AttnBlockSoftmaxV1,
                 **{"D": D, "NVIS": 48, "FIRST": False, "DOT": HOPPER, "CHECK": True, "PK": 16, "LAST": False, **kw})
    for kw in ({"VS": 3}, {"VS": 0}, {"N": 0}):
        with pytest.raises(ValueError):
            bind(F2.AttnBlockPvV1, **{"D": D, "N": 16, "VS": 4, "DOT": HOPPER, "FIRST": False, **kw})


def _rows(fn, port: str, n: int) -> tuple[int, int]:
    """``fn``'s proof block rows with ``port`` a row an element, and their `rows_k_log` (`check_proof_block`'s)."""
    ins, outs = unit_rows(fn, {port: n})
    return len(ins) + len(outs), check_proof_block(fn, {port: n})


def test_a_192_key_block_is_units_of_at_most_201_rows():
    """Llama-3.2-1B's 64-dim heads over kBlockN 192 (sm_90's tile, the class of record's), its value GEMM's slices of 32 columns, in
    groups of PK 96 keys: a full block is a softmax of 201 rows and two groups of 195, the first visited block at 192 keys a softmax of
    197 and groups of 194 (the first) and 195, every rows_k_log 27, a row fewer at LAST (no running max out); and at every first-block
    key count 1 ... 192 each unit passes."""
    D64, BN, VS, PK = 64, 192, 32, 96
    sm = lambda F, N, first, last=False: bind(F, D=D64, NVIS=N, FIRST=first, DOT=HOPPER, CHECK=True, PK=PK, LAST=last)  # noqa: E731
    pv = lambda N, first: bind(F2.AttnBlockPvV1, D=D64, N=N, VS=VS, DOT=HOPPER, FIRST=first)  # noqa: E731
    for F in (F2.AttnBlockSoftmaxV1, F3.AttnBlockSoftmaxV2):
        assert _rows(sm(F, BN, False), "k", BN) == (201, 27)
        assert _rows(sm(F, BN, True), "k", BN) == (197, 27)
        assert _rows(sm(F, BN, False, True), "k", BN) == (200, 27)
        assert _rows(sm(F, BN, True, True), "k", BN) == (196, 27)
    assert _rows(pv(PK, False), "v", PK * D64 // VS) == (195, 27)
    assert _rows(pv(PK, True), "v", PK * D64 // VS) == (194, 27)
    worst = [0, 0]
    for n in range(1, BN + 1):
        worst[0] = max(worst[0], _rows(sm(F2.AttnBlockSoftmaxV1, n, True), "k", n)[0])
        for g, (lo, hi) in enumerate(F2.pv_groups(n, PK)):
            worst[1] = max(worst[1], _rows(pv(hi - lo, g == 0), "v", (hi - lo) * D64 // VS)[0])
    assert worst == [197, 195] and max(worst) <= MAX_ROWS
