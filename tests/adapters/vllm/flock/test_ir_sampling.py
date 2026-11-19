"""Flock's lowering of GumbelTopPTokenSelect_v1 (``verity_flock.ir_sampling``): the pieces against the circuits' primitives, the
lane unit against the IR composites it is lowered from, a row's words against the circuits' reference evaluator on synthetic sets
(a row whose last BLAKE3 chunk is short included), the staged file's units, the native words, the pin, and the negatives."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.circuits.ops import fp32 as F
from verity.circuits.ops import mufu as P
from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import ir_sampling as S
from verity_flock import lowerings
from verity_flock import subcircuit_templates as TM
from verity_vllm.definitions.registry import ref_prims as RP

REG = lowerings.registry("C-Flock")
SPECIAL = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FC00123, 0x7F800001, 0xFFC00000, 0xFF800007, 1, 0x80000001,
           0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x3F800000, 0xBF800000, 0x3F4CCCCD, 0x7E800000, 0x7E800001, 0x00800001]


def f32_words(rng: random.Random, n: int) -> list[int]:
    out = []
    for _ in range(n):
        r, s = rng.random(), rng.getrandbits(1) << 31
        if r < 0.2:
            out.append(rng.choice(SPECIAL))
        elif r < 0.5:
            out.append(s | (rng.randint(100, 150) << 23) | rng.getrandbits(23))
        elif r < 0.6:
            out.append(s | rng.getrandbits(23))
        elif r < 0.7:
            out.append(s | (rng.randint(240, 254) << 23) | rng.getrandbits(23))
        else:
            out.append(rng.getrandbits(32))
    return out


def bf16_words(rng: random.Random, n: int) -> list[int]:
    special = [0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x3F80, 0x7F7F, 0x0080]
    return [rng.choice(special) if rng.random() < 0.15 else
            (rng.getrandbits(1) << 15) | (rng.randint(110, 140) << 7) | rng.getrandbits(7) if rng.random() < 0.6 else rng.getrandbits(16)
            for _ in range(n)]


def run_piece(piece, widths, vectors):
    C = gf2.Circuit()
    ins = [C.input(f"i{k}", w) for k, w in enumerate(widths)]
    C.output("o", piece(C, *ins))
    T = len(vectors)
    outs, _ = C.evaluate({f"i{k}": IL.slice_words([v[k] for v in vectors], w) for k, w in enumerate(widths)}, T)
    return [int(x) for x in IL.unslice_words(outs["o"], T)]


PIECES = [
    ("F32Mul_v1", (32, 32), F.F32Mul.evaluate),
    ("F32AddFtz_v1", (32, 32), F.F32AddFtz.evaluate),
    ("DivFullScaleA_v1", (32, 32), P.DivFullScaleA.evaluate),
    ("F32GtStrict_v1", (32, 32), RP.F32GtStrict.evaluate),
]


@pytest.mark.parametrize("name,widths,ref", PIECES, ids=[p[0] for p in PIECES])
def test_pieces_match_the_ir_primitives(name, widths, ref):
    rng = random.Random(name)
    a = f32_words(rng, 6000)
    b = f32_words(rng, 6000)
    vecs = list(zip(a, b)) + [(x, y) for x in SPECIAL for y in SPECIAL]
    got = run_piece(S.PIECES[name], widths, vecs)
    want = [ref(x, y) for x, y in vecs]
    assert sum(g != w for g, w in zip(got, want)) == 0


def ir_walk(target, args, resolved):
    """The IR composite on ``args`` by its own gates and the primitives' evaluators, a gate whose primitive id is in
    ``resolved`` taking that value: the reference the lane unit is held to."""
    _prog, call = IL.standalone(target)
    t = dict(zip(call.args, args))
    for g in call:
        t[g.index] = resolved[g.prim.id] if g.prim.id in resolved else g.prim.evaluate(*[t[o] for o in g.operands])
    return call, t


def _def(V: int):
    return TM.subcircuit(S.TEMPLATE, V=V).definition()


def lane_reference(V, x, temp, rcp, skip, noisy, kbit, g, best, best_i, i):
    tl, ms, gs = S._composites(_def(V))
    call, t = ir_walk(tl, [x, temp, rcp, skip], {})
    scaled = t[call.returns[0]]
    call, t = ir_walk(ms, [i, scaled, 0], {f"BitAtx{V}_v1": kbit})
    masked = t[call.returns[1]]
    call, t = ir_walk(gs, [best, best_i, i, masked, 0, noisy], {"GumbelNoiseLane_v1": g})
    nb, ni, nxt = (t[r] for r in call.returns[:3])
    if i == 0:                        # the scan's init: (y_0, 0, 1), y_0 the step's own y
        (gt,) = [x for x in call if x.prim.id == "F32GtStrict_v1"]
        nb, ni = t[gt.operands[0]], 0
    return [scaled, nb, ni, nxt]


def test_lane_unit_matches_the_ir_composites():
    V = 1024
    low = S.lane(_def(V))
    rng = random.Random(7)
    vecs = []
    for _ in range(4000):
        temp = rng.choice([0, 0x80000000, 0x3F800000, 0x3F4CCCCD, 0x3E99999A, 0x40000000, 0x7F000000, 5, 0x7F800000, 0x7FC00000,
                           *f32_words(rng, 2)])
        rcp = P.DivFullRcp.evaluate(temp) if rng.random() < 0.7 else f32_words(rng, 1)[0]
        vecs.append((bf16_words(rng, 1)[0], temp, rcp, rng.getrandbits(1), rng.getrandbits(1), rng.getrandbits(1), f32_words(rng, 1)[0],
                     f32_words(rng, 1)[0], rng.getrandbits(32), rng.choice([0, 0, 1, 5, rng.getrandbits(32)])))
    outs, ok = low.evaluate([np.array([v[j] for v in vecs], dtype=np.uint64) for j in range(len(S.IN_PORTS))])
    assert ok == (1 << len(vecs)) - 1
    bad = sum([int(o[n]) for o in outs] != lane_reference(V, *v) for n, v in enumerate(vecs))
    assert bad == 0


def test_the_lane_is_the_pinned_one():
    low = S.lane(_def(128256))
    assert S.pin_digest(low) == REG.module(S.TEMPLATE).PIN == S.PIN
    assert S.pin_digest(S.lane(_def(64))) == S.PIN
    head = low.unit_text.split("\n", 1)[0].split()
    assert head[0] == IL.FORMAT and head[1] == S.UNIT and int(head[2]) <= 1 << 13


def test_supports_and_relation():
    sub = TM.subcircuit(S.TEMPLATE, V=128256)
    assert REG.supports(sub, "frame-v3/blake3-keyed") is None
    assert REG.relation(sub, "frame-v3/blake3-keyed") == "irf:gumbel-top-p-token-select"
    assert "not wired yet" in REG.supports(sub, "vllm-v1")


# --- negatives ---------------------------------------------------------------------------------------------------------------


def test_a_mutated_circuit_description_is_caught_by_the_ir_comparison():
    V = 64
    low = S.lane(_def(V))
    rng = random.Random(3)
    vecs = []
    for _ in range(1500):
        temp = rng.choice([0x3F4CCCCD, 0x7F000000, 5, *f32_words(rng, 1)])
        vecs.append((bf16_words(rng, 1)[0], temp, P.DivFullRcp.evaluate(temp), rng.getrandbits(1), rng.getrandbits(1), rng.getrandbits(1),
                     f32_words(rng, 1)[0], f32_words(rng, 1)[0], rng.getrandbits(32), rng.choice([0, rng.getrandbits(32)])))
    ports = [np.array([v[j] for v in vecs], dtype=np.uint64) for j in range(len(S.IN_PORTS))]
    want = [lane_reference(V, *v) for v in vecs]
    first = low.layout.in_words * IL.WORD
    ands = list(range(first + 1, first + low.layout.counts["ands"]))
    caught = 0
    for k in ands[::len(ands) // 48][:48]:
        lay = IL.LoweredUnit(list(low.layout.rows_a), list(low.layout.rows_b), low.layout.const, low.layout.in_groups, low.layout.out_groups,
                        low.layout.counts)
        lay.rows_b[k] = sorted(set(lay.rows_b[k]) ^ {k - 1})
        outs, _ = IL.evaluate_layout(lay, ports, len(vecs))
        caught += sum([int(o[n]) for o in outs] != w for n, w in enumerate(want)) > 0
    assert caught >= 16                               # a third of the spread-out mutations (the rest sit in rare paths)

