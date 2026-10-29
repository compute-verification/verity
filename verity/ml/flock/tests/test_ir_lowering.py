"""C-Flock's generic IR lowering: the pieces against the IR primitives, the units read off each template's IR graph, the
lowered templates against the IR evaluator on captured #101 slices and synthetic sets, the pins, and the negatives."""
from __future__ import annotations

import itertools
import os
import random
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import REGISTRY
from verity_flock import fp, gf2
from verity_flock import ir_lower as IL

P = pytest.importorskip("verity.ml.definitions.registry.prims")
from verity_numerical.bench import lowerings  # noqa: E402
from verity_numerical.bench import templates as TM  # noqa: E402
from verity_numerical.bench.generate import generate  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402

SLICES = Path(__file__).resolve().parents[4] / "benchmarks" / "numerical" / "tests" / "bench" / "data" / "captured-101" / "slices"
REG = lowerings.registry("C-Flock")
LOWERED = {"rope-head": {"D": 64}, "silu-mul": {"I": 8192}, "silu-mul-v2": {"I": 256}}


SPECIAL = [0, 0x80000000, 0x7F800000, 0xFF800000, 1, 0x80000001, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x3F800000, 0xBF800000,
           0x7FC00000, 0xFFC00000, 0x7FC00001, 0xFFC12345, 0x7F800001, 0xFF812345, 0x7FFFFFFF, 0xFFBFFFFF]   # quiet and signalling NaNs


def f32_words(rng: random.Random, n: int) -> list[int]:
    special = SPECIAL
    out = []
    for _ in range(n):
        r, s = rng.random(), rng.getrandbits(1) << 31
        if r < 0.15:
            out.append(rng.choice(special))
        elif r < 0.3:
            out.append(s | rng.getrandbits(23))
        elif r < 0.45:
            out.append(s | (rng.randint(1, 30) << 23) | rng.getrandbits(23))
        elif r < 0.6:
            out.append(s | (rng.randint(220, 254) << 23) | rng.getrandbits(23))
        elif r < 0.8:
            out.append(s | (rng.randint(120, 134) << 23) | rng.getrandbits(23))
        else:
            out.append(rng.getrandbits(32))
    return out


def near(rng: random.Random, a: int) -> int:
    """An operand within two binades of a, of either sign, so additions cancel."""
    e = min(254, max(0, ((a >> 23) & 0xFF) + rng.randint(-2, 2)))
    return ((a ^ (rng.getrandbits(1) << 31)) & 0x807FFFFF & ~(0xFF << 23)) | (e << 23)


def bf16_words(rng: random.Random, n: int) -> list[int]:
    special = [0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0x3F80, 0x7F7F, 0x0080]
    return [rng.choice(special) if rng.random() < 0.1 else
            (rng.getrandbits(1) << 15) | (rng.randint(100, 150) << 7) | rng.getrandbits(7) if rng.random() < 0.5 else rng.getrandbits(16)
            for _ in range(n)]


def run_piece(piece, widths, out_width, vectors):
    C = gf2.Circuit()
    ins = [C.input(f"i{k}", w) for k, w in enumerate(widths)]
    C.output("o", piece(C, *ins))
    T = len(vectors)
    outs, _ = C.evaluate({f"i{k}": IL.slice_words([v[k] for v in vectors], w) for k, w in enumerate(widths)}, T)
    return [int(x) for x in IL.unslice_words(outs["o"], T)]


PIECES = [
    ("F32Add_v1", (32, 32), 32, lambda C, a, b: IL.PIECES["F32Add_v1"](C, a, b), P.F32Add.evaluate),
    ("F32Mul_v1", (32, 32), 32, lambda C, a, b: IL.PIECES["F32Mul_v1"](C, a, b), P.F32Mul.evaluate),
    ("F32Fma_v1", (32, 32, 32), 32, lambda C, a, b, c: IL.PIECES["F32Fma_v1"](C, a, b, c), P.F32Fma.evaluate),
    ("F2fpBf16_v1", (32,), 16, lambda C, a: IL.PIECES["F2fpBf16_v1"](C, a), P.F2fpBf16.evaluate),
    ("F32ToBf16Rn_v1", (32,), 16, lambda C, a: IL.PIECES["F32ToBf16Rn_v1"](C, a), P.F32ToBf16Rn.evaluate),
]
from verity_flock.boolean_export import prims as _registered  # noqa: E402

#: the tail primitives (tail_pieces): attention's softmax, the norms' row scalars, the sampler lane's F32 ops
TAIL = {"F32AddFtz_v1": 2, "F32SubFtz_v1": 2, "F32MulFtz_v1": 2, "F32FmaFtz_v1": 3, "F32FmaSubFtz_v1": 3, "F32Max_v1": 2,
        "GuardNegInfZero_v1": 1, "MufuEx2Ftz_v1": 1, "MufuRcpFtz_v1": 1, "Fa2InvSum_v1": 1, "RsqrtApprox_v1": 1,
        "MufuSqrtFtz_v1": 1, "DivFullRcp_v1": 1, "DivFullScaleA_v1": 2, "F32GtStrict_v1": 2}
PIECES += [(name, (32,) * k, 1 if name == "F32GtStrict_v1" else 32, IL.PIECES[name], _registered()[name].evaluate)
           for name, k in TAIL.items()]
#: the `_v2` FP32 primitives (every NaN result 0x7FFFFFFF) and their pieces
V2 = {"F32Add_v2": 2, "F32Mul_v2": 2, "F32Fma_v2": 3, "F32Div_v2": 2, "F32AddFtz_v2": 2, "F32SubFtz_v2": 2, "F32MulFtz_v2": 2,
      "F32FmaFtz_v2": 3, "F32FmaSubFtz_v2": 3, "F32Sub_v2": 2, "F32FmaRm_v2": 3, "DivFullScaleA_v2": 2}
PIECES += [(name, (32,) * k, 32, IL.PIECES[name], REGISTRY[name].evaluate) for name, k in V2.items()]


@pytest.mark.parametrize("name,widths,out,piece,ref", PIECES, ids=[p[0] for p in PIECES])
def test_pieces_match_the_ir_primitives(name, widths, out, piece, ref):
    rng = random.Random(20260926)
    vecs = []
    for _ in range(6000):
        v = f32_words(rng, len(widths))
        if len(v) >= 2 and rng.random() < 0.3:
            v[1] = near(rng, v[0])
        if len(v) == 3 and rng.random() < 0.3:
            v[2] = near(rng, P.f32_mul_bits(v[0], v[1]))
        vecs.append(v)
    got = run_piece(piece, widths, out, vecs)
    bad = [(v, g) for v, g in zip(vecs, got) if g != ref(*v)]
    assert bad == []                                  # exact bits: a NaN's sign and payload too


def test_a_primitive_without_a_piece_lowers_through_the_boolean_definition_of_its_word():
    """A primitive C-Flock has no piece for, computed by a registered Boolean Definition (``.word`` the primitive), lowers by
    running that Definition's ``build``: found through core's registry, not imported from its owner."""
    from types import SimpleNamespace

    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits.types import Array
    from verity.ml.circuits.boolean.forms import NOT
    from verity.ml.circuits.boolean.trace import definition

    U4, word = Array(4, BL.BIT), SimpleNamespace(id="FlockTestNot4_v1")
    assert word.id not in IL.PIECES
    definition("FlockTestNot4Bits", 1, lambda C, a: [NOT(b) for b in a], (("a", U4),), U4, word=word)
    assert run_piece(IL.PIECES[word.id], (4,), 4, [[a] for a in range(16)]) == [~a & 15 for a in range(16)]


@pytest.mark.parametrize("name", sorted(V2))
def test_the_v2_pieces_are_their_primitives_on_every_special_word(name):
    """Every pair of the special words (every triple of every other one): a NaN operand of either sign and any payload, an
    invalid operation, and for `DivFullScaleA_v2` a NaN dividend over a huge, tiny or ordinary divisor."""
    k = V2[name]
    vecs = [list(v) for v in itertools.product(SPECIAL if k == 2 else SPECIAL[::2], repeat=k)]
    ref = REGISTRY[name].evaluate
    assert [(v, g) for v, g in zip(vecs, run_piece(IL.PIECES[name], (32,) * k, 32, vecs)) if g != ref(*v)] == []


@pytest.mark.parametrize("name", ["MufuEx2Ftz_v1", "MufuRcpFtz_v1", "Fa2InvSum_v1", "RsqrtApprox_v1", "MufuSqrtFtz_v1",
                                  "DivFullRcp_v1"])
def test_mufu_pieces_match_on_every_exponent(name):
    """Each MUFU piece (a table read inside exponent logic) against its primitive on every exponent, both signs, the
    mantissa's edges and random mantissas; the reads come from the pinned tables."""
    rng = random.Random(5)
    vecs = [[(s << 31) | (e << 23) | (rng.getrandbits(23) if k > 2 else [0, 1, 0x7FFFFF][k])] for e in range(256) for s in (0, 1)
            for k in range(24)]
    from verity_flock.boolean_export import prims
    ref = prims()[name].evaluate
    assert [g for v, g in zip(vecs, run_piece(IL.PIECES[name], (32,), 32, vecs)) if g != ref(*v)] == []


@pytest.mark.parametrize("name,widths,gen", [
    ("GumbelStreamKey_v1", (64, 64), lambda r: [r.getrandbits(64), r.choice([r.getrandbits(64), r.randint(0, 5000)])]),
    ("GumbelNoiseLane_v1", (32, 32), lambda r: [r.getrandbits(32), r.choice([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
                                                                            r.randint(0, 128255), r.getrandbits(32)])]),
    ("GumbelNoiseLane_v2", (32, 32), lambda r: [r.getrandbits(32), r.choice([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
                                                                            r.randint(0, 128255), r.getrandbits(32)])]),
    ("F32Eq_v1", (32, 32), lambda r: (lambda v: [v[0], r.choice([v[1], v[0], v[0] ^ 0x80000000])])(f32_words(r, 2))),
    ("BitOr_v1", (1, 1), lambda r: [r.getrandbits(1), r.getrandbits(1)]),
    ("BitNot_v1", (1,), lambda r: [r.getrandbits(1)]),
])
@pytest.mark.slow
def test_sampler_pieces_match_the_ir_primitives(name, widths, gen):
    """The sampler's Philox stream key and Gumbel noise lane (libdevice logf / log1pf, its true toward-zero add) and the row's
    bit logic, exact bits."""
    rng = random.Random(9)
    vecs = [gen(rng) for _ in range(300 if name.startswith("Gumbel") else 3000)]
    ref = _registered()[name].evaluate
    assert [(v, g) for v, g in zip(vecs, run_piece(IL.PIECES[name], widths, 32, vecs)) if g != ref(*v)] == []


@pytest.mark.parametrize("name", ["NvLogf_v1", "NvLogf_v2"])
def test_the_logf_pieces_match_the_ir_primitives_outside_the_domain_too(name):
    """``_v1``'s NaN (main's pinned 0xFFC00000 for a negative word, a NaN input's payload) and ``_v2``'s (0x7FFFFFFF for both), on
    negative words, NaNs of both signs and the positive range."""
    rng = random.Random(name)
    vecs = [[w] for w in f32_words(rng, 1500)] + [[w | 0x80000000] for w in f32_words(rng, 300)]
    vecs += [[w] for w in (0x7FC00123, 0xFFC00123, 0x7F800001, 0xFF800001, 0x80000000, 0x80000001, 0xFF800000, 0x7F800000)]
    ref = _registered()[name].evaluate
    assert [(v, g) for v, g in zip(vecs, run_piece(IL.PIECES[name], (32,), 32, vecs)) if g != ref(*v)] == []


#: the other rows' primitives (tail_pieces): FA3's inverse sum, the greedy argmax, the FP8 linear's quantizer, the MoE router
MORE = ["Fa3InvSum_v1", "Bf16GtStrict_v1", "SelectBf16_v1", "F32Fabs_v1", "F32Neg_v1", "F32Fmaxf_v1", "F32Fminf_v1", "F32ToE4m3Sat_v1",
        "F32Div_v1", "F32Sub_v1", "F32FmaRm_v1", "F32Sat_v1", "F32BitsShl23_v1", "F32IsFinite_v1", "I32Eq_v1"]


@pytest.mark.parametrize("name", MORE)
def test_the_other_rows_pieces_match_the_ir_primitives(name):
    """Exact bits, NaN payloads included, on each operand kind's specials and random words (f32 operands near each other so
    sums cancel, the e4m3 grid's edges for the quantizer, divisors of every magnitude)."""
    p = _registered()[name]
    widths = [w for _, t in p.params for w in t.leaf_widths()]
    rng = random.Random(31)
    vecs = []
    for i in range(6000):
        v = []
        for w in widths:
            v.append(rng.getrandbits(1) if w == 1 else bf16_words(rng, 1)[0] if w == 16 else f32_words(rng, 1)[0])
        f = [j for j, w in enumerate(widths) if w == 32]
        if len(f) >= 2 and rng.random() < 0.3:
            v[f[1]] = near(rng, v[f[0]])
        if name == "F32ToE4m3Sat_v1" and i % 2:
            v[0] = (rng.getrandbits(1) << 31) | (rng.randint(110, 140) << 23) | rng.choice([0, 0x080000, 0x100000, rng.getrandbits(23)])
        if name == "F32FmaRm_v1" and rng.random() < 0.3:
            v[2] = near(rng, P.f32_mul_bits(v[0], v[1]))
        if name == "I32Eq_v1" and rng.random() < 0.3:
            v[1] = v[0]
        vecs.append(v)
    got = run_piece(IL.PIECES[name], widths, sum(p.ret.leaf_widths()), vecs)
    assert [(v, g) for v, g in zip(vecs, got) if g != p.evaluate(*v)] == []


def _e2m1_vectors(rng: random.Random) -> list[list[int]]:
    from verity.ml.circuits.silicon.cast import E2M1_UP
    words = [b + d for b, _ in E2M1_UP for d in range(-4, 5)] + [b | 1 << i for b, _ in E2M1_UP for i in range(21)]
    words += [0, 1, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x7F800000] + [0x7F800000 | 1 << i for i in range(23)]
    words += [e << 23 | rng.getrandbits(3) << 20 | rng.choice([0, 1, rng.getrandbits(20)]) for e in range(118, 134) for _ in range(16)]
    words += [w & 0x7FFFFFFF for w in f32_words(rng, 4000)]
    return [[w | s] for w in words for s in (0, 0x80000000)]


#: NVFP4 quantization's casts and the bit AND (PoUW's units): (primitive, operand widths, result width, vectors)
NVFP4 = [("E4m3ToF32_v1", (8,), 32, lambda rng: [[b] for b in range(256)]),
         ("F32ToE2m1Sat_v1", (32,), 4, _e2m1_vectors),
         ("BitAnd_v1", (1, 1), 1, lambda rng: [[a, b] for a in (0, 1) for b in (0, 1)])]


@pytest.mark.parametrize("name,widths,out,gen", NVFP4, ids=[n[0] for n in NVFP4])
def test_the_nvfp4_cast_and_bit_and_pieces_match_the_ir_primitives(name, widths, out, gen):
    """Exact bits: the E4M3 widening on every byte (NaN 0x7FFFE000), the AND on every pair of bits, and the E2M1 cast on each
    boundary of its grid (the six ties and 5, where saturation begins) within 4 ulps and with each of its low 21 bits set, on
    ±0, subnormals, ±inf and a NaN per payload bit, across the exponents around the grid and on random words, both signs."""
    import verity.ml.circuits.ops.prims  # noqa: F401  (registers the casts)
    import verity.ml.circuits.ops.scalar  # noqa: F401  (registers BitAnd_v1)

    vecs = gen(random.Random(name))
    ref = REGISTRY[name].evaluate
    assert [(v, g) for v, g in zip(vecs, run_piece(IL.PIECES[name], widths, out, vecs)) if g != ref(*v)] == []


@pytest.mark.parametrize("name", ["HopperE4m3QgmmaDot32_v1", "BlackwellE4m3QmmaDot32_v1"])
def test_the_fp8_tensor_core_step_matches_the_ir_on_every_encoding(name):
    """``fp.tc_dot_e4m3`` = ``HopperE4m3QgmmaDot32`` and sm_120's ``BlackwellE4m3QmmaDot32`` (the total models): E4M3 NaNs (0x7F,
    0xFF), zeros, subnormals, the largest magnitudes, accumulators of every class (Hopper's 14-bit adder truncates them too, sm_120's
    26-bit one does not), bit for bit."""
    import verity.ml.circuits.ops.prims  # noqa: F401  (registers the steps)

    Q = REGISTRY[name]

    rng = random.Random(12)
    specials8 = [0x00, 0x80, 0x7F, 0xFF, 0x01, 0x81, 0x07, 0x7E, 0xFE, 0x08, 0x38]
    specials32 = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F7FFFFF, 0xFF7FFFFF, 1, 0x00800000]

    def e4(small):
        r = rng.random()
        return rng.choice(specials8) if r < (0.01 if small else 0.08) else (rng.getrandbits(1) << 7) | (rng.randint(4, 10) << 3) \
            | rng.getrandbits(3) if small else rng.getrandbits(8) & (0x7E if r < 0.5 else 0xFF)

    vecs = []
    for i in range(1500):
        small = i % 3 == 0
        acc = (rng.getrandbits(1) << 31) | (rng.randint(110, 140) << 23) | rng.getrandbits(23) if small else \
            rng.choice(specials32) if rng.random() < 0.1 else rng.getrandbits(32)
        vecs.append([acc] + [e4(small) for _ in range(64)])
    got = run_piece(IL.PIECES[name], [32] + [8] * 64, 32, vecs)
    assert [(v, g) for v, g in zip(vecs, got) if g != Q.evaluate(*v)] == []


def _fp4_vectors(rng: random.Random, n: int, scale) -> list[list[int]]:
    """Accumulators of every class (NaN payloads, infinities, subnormals, huge and ordinary), E2M1 codes (random, mostly zero, and
    pairs cancelling within each group, or none at all), and ``scale(rng, family)`` bytes."""
    special32 = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC12345, 0x7F800001, 0x7F7FFFFF, 0xFF7FFFFF, 1, 0x80000001]

    def acc() -> int:
        r, s = rng.random(), rng.getrandbits(1) << 31
        return rng.choice(special32) if r < 0.08 else s | rng.getrandbits(23) if r < 0.2 else \
            s | (rng.randint(90, 160) << 23) | rng.getrandbits(23) if r < 0.45 else s | (rng.randint(0, 254) << 23) | rng.getrandbits(23)

    vecs = []
    for t in range(n):
        fam = t % 7
        a = [rng.choice([0, 8]) if fam == 1 and rng.random() < 0.8 else rng.randint(0, 15) for _ in range(64)]
        b = [x ^ 8 if i % 2 else x for i, x in enumerate(a)] if fam == 2 else [rng.randint(0, 15) for _ in range(64)]
        if fam == 6:
            a = b = [0] * 64
        vecs.append([acc()] + a + b + scale(rng, fam))
    return vecs


def test_the_nvf4_step_matches_the_ir_on_every_encoding():
    """``unit_fp4.step_total`` = ``BlackwellNvf4OmmaDot64``: NaN scale bytes (low seven bits 0x7F, either bit 7), bit 7 set on
    ordinary scales (ignored), zero and subnormal scale mantissas, the top exponent field, NaN and infinite accumulators, cancelling
    groups, bit for bit."""
    from verity.ml.circuits.ops.prims import BlackwellNvf4OmmaDot64 as Q

    def scale(rng, fam):
        def one():
            r = rng.random()
            return rng.choice([0x7F, 0xFF]) if r < 0.02 else (rng.getrandbits(1) << 7) | rng.randint(0, 7) if r < 0.2 else \
                rng.choice([0x78, 0x79, 0x7E, 0x70, 0xF8, 0xFE]) if r < 0.3 else rng.randint(0, 0xFF)
        return [one() for _ in range(8)]

    vecs = _fp4_vectors(random.Random(13), 3000, scale)
    got = run_piece(IL.PIECES["BlackwellNvf4OmmaDot64_v1"], [32] + [4] * 128 + [8] * 8, 32, vecs)
    assert [(v, g) for v, g in zip(vecs, got) if g != Q.evaluate(*v)] == []


def test_the_mxf4_step_matches_the_ir_on_every_encoding():
    """``unit_fp4.step_total(ue8m0=True)`` = ``BlackwellMxf4OmmaDot64``: UE8M0 NaN scales, scales deep enough that the -174 floor binds and sums
    fall below the subnormal grid (+0), and large enough that they leave binary32 (the infinity of the sum's sign), bit for bit."""
    def scale(rng, fam):
        lo, hi = {3: (0, 20), 4: (230, 254), 5: (100, 150)}.get(fam, (0, 254))
        return [0xFF if rng.random() < 0.03 else rng.randint(lo, hi) for _ in range(4)]

    from verity.ml.circuits.ops.prims import BlackwellMxf4OmmaDot64 as Q

    vecs = _fp4_vectors(random.Random(14), 3500, scale)
    got = run_piece(IL.PIECES["BlackwellMxf4OmmaDot64_v1"], [32] + [4] * 128 + [8] * 4, 32, vecs)
    want = [Q.evaluate(*v) for v in vecs]
    assert {0x7F800000, 0xFF800000, 0} <= set(want) and any(w & 0x7F800000 == 0 and w & 0x7FFFFF for w in want)
    assert [(v, g, w) for v, g, w in zip(vecs, got, want) if g != w] == []


@pytest.mark.parametrize("name,table", [("MufuTanh_v1", "tanh_mufu"), ("TanhF32Rn_v1", "tanh_rn")])
@pytest.mark.slow
def test_the_tanh_pieces_match_the_ir_primitives(name, table):
    """Gemma's tanh (a table read over |x|): each region's bounds and their neighbours, both signs, NaNs and infinities, and random
    words across the table's region, exact bits."""
    from verity_flock import tail_pieces as TP

    lo, hi = TP.TANH_BOUNDS[table]
    rng = random.Random(8)
    words = [lo - 1, lo, lo + 1, hi - 1, hi, hi + 1, 0, 1, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0x3F800000, 0x41200000]
    words += [rng.randrange(lo, hi) for _ in range(6000)] + [rng.randrange(0, 0x7F800000) for _ in range(1000)]
    vecs = [[w | (rng.getrandbits(1) << 31)] for w in words] + [[w | 0x80000000] for w in words[:14]]
    ref = _registered()[name].evaluate
    assert [(v, g) for v, g in zip(vecs, run_piece(IL.PIECES[name], (32,), 32, vecs)) if g != ref(*v)] == []


def test_the_gelu_piece_matches_the_ir_primitive_on_every_gate_word():
    """``GeluTanhMulBf16``: every bf16 gate word (the activation table's whole domain), each with a random up word, exact bits."""
    rng = random.Random(3)
    ups = bf16_words(rng, 1 << 16)
    vecs = [[g, u] for g, u in zip(range(1 << 16), ups)]
    ref = _registered()["GeluTanhMulBf16_v1"].evaluate
    assert [(v, g) for v, g in zip(vecs, run_piece(IL.PIECES["GeluTanhMulBf16_v1"], (16, 16), 16, vecs)) if g != ref(*v)] == []


def test_bit_at_selects_the_bit_or_zero():
    """``BitAtx{W}`` (a width family ``ir_lower.PIECES`` resolves): bit i of the word, 0 for i outside [0, W)."""
    from verity.ml.definitions.registry.sampling import bit_at
    rng = random.Random(4)
    W = 1000
    vecs = [[rng.getrandbits(W), rng.choice([rng.randint(0, W - 1), 0, W - 1, W, 0xFFFFFFFF, 0x80000000])] for _ in range(64)]
    C = gf2.Circuit()
    word, i = C.input("w", W), C.input("i", 32)
    C.output("o", IL.PIECES[f"BitAtx{W}_v1"](C, word, i))
    outs, _ = C.evaluate({"w": [sum(((v[0] >> b) & 1) << t for t, v in enumerate(vecs)) for b in range(W)],
                          "i": IL.slice_words([v[1] for v in vecs], 32)}, len(vecs))
    assert [int(x) for x in IL.unslice_words(outs["o"], len(vecs))] == [bit_at(W).evaluate(*v) for v in vecs]


#: the pinned rcp table read inline: 16,712 + 584 decoder rows (14 low and 9 high index bits), 9,753 nonempty product rows and
#: 24 copy rows, its 7 constant value bits wired to the constant; its 93,528,819 set bits
READ_ROWS, READ_NNZ = 27_073, 93_528_819


def test_a_table_read_is_laid_out_as_plain_rows():
    """A table read is laid out inline as rows of the unit (``ir_lower._Read``), no slot: the rcp table's decoders, a product row
    per nonempty (value bit, high minterm) and a copy row per live value bit, its set bits in the product rows' forms."""
    C = gf2.Circuit()
    x = C.input("in0", 32)
    C.output("out0", IL.PIECES["MufuRcpFtz_v1"](C, x))
    assert C.counts()["lookups"] == 1
    lay = IL.layout(C, ["in0"], ["out0"])
    assert (lay.counts["reads"], lay.counts["read_rows"], lay.counts["read_nnz_a"]) == (1, READ_ROWS, READ_NNZ)

    def latest(r):
        cols = np.asarray(r, dtype=np.int64)
        cols = cols[cols != lay.const]
        return int(cols.max()) if len(cols) else -1

    first = lay.in_words * IL.WORD
    assert all(latest(lay.rows_a[i]) < i and latest(lay.rows_b[i]) < i for i in range(first, lay.useful) if i != lay.const)


def _rows_eval(lay, words: list[int]) -> tuple[list[int], bool]:
    """The laid-out rows run on 64 lanes of one 32-bit input port (numpy, for rows with millions of set bits): the output
    words and whether every row holds."""
    T, mask = 64, np.uint64(0xFFFFFFFFFFFFFFFF)
    z = np.zeros(lay.useful, dtype=np.uint64)
    at = lay.in_port_cols[0]
    for b in range(32):
        z[at + b] = np.uint64(sum(((w >> b) & 1) << t for t, w in enumerate(words)))
    z[lay.const] = mask

    def ev(r):
        a = np.asarray(r, dtype=np.int64)
        return np.bitwise_xor.reduce(z[a]) if len(a) else np.uint64(0)

    for i in range(lay.in_words * IL.WORD, lay.useful):
        if i != lay.const:
            z[i] = ev(lay.rows_a[i]) & ev(lay.rows_b[i])
    holds = all(ev(lay.rows_a[i]) & ev(lay.rows_b[i]) == z[i] for i in range(lay.useful))
    o, nb = lay.out_port_cols[0], lay.out_bits[0]
    return [sum(((int(z[o + b]) >> t) & 1) << b for b in range(nb)) for t in range(T)], holds


@pytest.mark.parametrize("prim", ["MufuRcpFtz_v1", "MufuEx2Ftz_v1", "MufuSqrtFtz_v1", "RsqrtApprox_v1"])
def test_a_library_reads_rows_compute_its_piece(prim):
    """A read of a library table, its constant value bits wired to the constant and 14 low index bits: the laid-out rows give
    the gate circuit's outputs on random and edge words, and every row holds."""
    C = gf2.Circuit()
    x = C.input("in0", 32)
    C.output("out0", IL.PIECES[prim](C, x))
    lay = IL.layout(C, ["in0"], ["out0"])
    rng = random.Random(12)
    words = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 1, 0x00800000, 0x3F800000] + [rng.getrandbits(32) for _ in range(56)]
    outs, _ = C.evaluate({"in0": IL.slice_words(words, 32)}, 64)
    got, holds = _rows_eval(lay, words)
    assert holds and got == [int(v) for v in IL.unslice_words(outs["out0"], 64)]


def test_a_program_table_keeps_a_row_for_every_live_bit(monkeypatch):
    """A table off core's library list keeps its constant bits as rows, so its layout shows nothing of its content."""
    words = [(i * 2654435761) & 0x7 | 0x8 for i in range(1 << 6)]
    monkeypatch.setitem(gf2.TABLES, "toy_program", lambda: words)
    monkeypatch.setitem(gf2._TABLE_WORDS, "toy_program", words)
    C = gf2.Circuit()
    x = C.input("in0", 6)
    v = C.lookup("toy_program", x, 4)
    C.output("out0", v)
    assert IL._library_table("rcp") and not IL._library_table("toy_program")
    r = IL._Read(C, 0, [0, 1, 2, 3], 0)
    assert r.const_bits == {} and r.bits == [0, 1, 2, 3]


@pytest.mark.parametrize("prim", ["Bf16Add_v1", "Bf16AddF2fp_v1"])
def test_bf16_adds_match_the_ir_primitives(prim):
    rng = random.Random(7)
    vecs = [bf16_words(rng, 2) for _ in range(6000)]
    ref = {"Bf16Add_v1": P.Bf16Add, "Bf16AddF2fp_v1": P.Bf16AddF2fp}[prim].evaluate
    assert run_piece(IL.PIECES[prim], (16, 16), 16, vecs) == [ref(*v) for v in vecs]


def test_template_units_are_read_off_the_ir():
    rope = TM.subcircuit("rope-head", D=64)
    U = IL.units(rope.definition())
    assert U.n == 32 and U.in_widths == [16] * 4 and U.out_widths == [16, 16]
    assert U.in_src[5] == [("leaf", 5), ("leaf", 37), ("leaf", 69), ("leaf", 101)] and U.out_src[5] == [("ret", 5), ("ret", 37)]
    silu = TM.subcircuit("silu-mul", I=8192)
    U = IL.units(silu.definition())
    assert U.n == 8192 and U.in_src[3] == [("leaf", 3), ("leaf", 8195)] and U.out_src[3] == [("ret", 3)]


def test_rope_unit_matches_its_primitives():
    low = REG.module("rope-head").lowering(TM.subcircuit("rope-head", D=8))
    rng = random.Random(3)
    vecs = [bf16_words(rng, 4) for _ in range(8000)]
    (lo, hi), ok = IL.evaluate_layout(low.layout, [[v[j] for v in vecs] for j in range(4)], len(vecs))
    assert ok == (1 << len(vecs)) - 1
    assert [int(x) for x in lo] == [P.RopeOut.evaluate(*v) for v in vecs]
    assert [int(x) for x in hi] == [P.RopeOutAdd.evaluate(*v) for v in vecs]


def test_silu_unit_covers_the_whole_table():
    """u = 1.0 makes the unit's output its silu table entry, so every gate word g is checked; then random (g, u)."""
    low = REG.module("silu-mul").lowering(TM.subcircuit("silu-mul", I=4))
    g = list(range(1 << 16))
    (out,), ok = IL.evaluate_layout(low.layout, [g, [0x3F80] * len(g)], len(g))
    assert ok == (1 << len(g)) - 1
    assert [int(x) for x in out] == [P.SiluMulBf16.evaluate(w, 0x3F80) for w in g]
    rng = random.Random(5)
    vecs = [bf16_words(rng, 2) for _ in range(8000)]
    (out,), _ = IL.evaluate_layout(low.layout, [[v[0] for v in vecs], [v[1] for v in vecs]], len(vecs))
    assert [int(x) for x in out] == [P.SiluMulBf16.evaluate(*v) for v in vecs]


#: up words where `SiluMulBf16_v2` is not a plain product: -0 (``+ 0.0f`` makes it +0), +-inf, NaNs of either sign, a subnormal
SILU_V2_UPS = (0x3F80, 0x0000, 0x8000, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81, 0x0001, 0x4049)


def test_silu_v2_unit_is_the_served_function_on_every_gate_word():
    """`SiluMul_v2`'s unit against `SiluMulBf16_v2`, the primitive the template's Definition registers, on every gate word at
    each of :data:`SILU_V2_UPS`, then on random (g, u). The two versions differ on NaN, -inf, -0 and the gates whose
    ``expf(-g)`` overflows, so a unit with `SiluMul_v1`'s semantics fails here."""
    low = REG.module("silu-mul-v2").lowering(TM.subcircuit("silu-mul-v2", I=4))
    served = REGISTRY["SiluMulBf16_v2"].evaluate
    assert low.units.n == 4 and low.layout.counts["ands"] == 1380
    g = list(range(1 << 16))
    differs = {w for w in g if served(w, 0x3F80) != P.SiluMulBf16.evaluate(w, 0x3F80)}
    assert {0x7FC0, 0xFF80, 0x8000, 0xC2B2} <= differs        # NaN, -inf, -0, -89
    for u in SILU_V2_UPS:
        (out,), ok = IL.evaluate_layout(low.layout, [g, [u] * len(g)], len(g))
        assert ok == (1 << len(g)) - 1
        assert [w for w, x in zip(g, out) if int(x) != served(w, u)] == [], hex(u)
    rng = random.Random(7)
    vecs = [bf16_words(rng, 2) for _ in range(8000)]
    (out,), ok = IL.evaluate_layout(low.layout, [[v[0] for v in vecs], [v[1] for v in vecs]], len(vecs))
    assert ok == (1 << len(vecs)) - 1
    assert [int(x) for x in out] == [served(*v) for v in vecs]


def test_silu_v2_circuit_units_are_a_power_of_two():
    """`verity/flock-circuit` holds a power-of-two number of units an instance (both verifiers refuse another count), so at
    GLM-5.3's dense I = 1536 a `silu-mul-v2` unit is three elements, 512 units, each still `SiluMulBf16_v2`; at I = 256 it is
    the circuit-checked unit."""
    import json

    from verity_flock import circuit as FC

    mod = REG.module("silu-mul-v2")
    assert mod.circuit_lowering(TM.subcircuit("silu-mul-v2", I=256)).sha256 == mod.PINS[256]
    sub = TM.subcircuit("silu-mul-v2", I=1536)
    low = mod.circuit_lowering(sub)
    served = REGISTRY["SiluMulBf16_v2"].evaluate
    assert low.sha256 == mod.CIRCUIT_PINS[1536]
    assert (low.name, low.units.n, low.layout.counts["ands"]) == ("silu-mul-v2/bf16/element/x3", 512, 3 * 1380)
    rng = random.Random(11)
    fin = np.array([bf16_words(rng, 2 * 1536) for _ in range(4)], dtype=np.uint64)
    got, _, ok = low.evaluate(fin)
    assert ok == (1 << (4 * 512)) - 1
    assert got.tolist() == [[served(int(r[i]), int(r[1536 + i])) for i in range(1536)] for r in fin]
    meta = json.loads(FC.compose(low, sub).split("\n", 2)[1].removeprefix("META "))
    (unit,) = [r for r in meta["ranges"] if r["net"] == "unit"]
    assert unit["count"] == meta["vus_per_block"] * 512 and unit["count"] & (unit["count"] - 1) == 0


@pytest.mark.parametrize("which", ["gemm-tc-step", "silu-mul"])
def test_a_units_nonzeros_are_its_netlist_rows_inside_its_slot(which):
    """`Layout.nonzeros` on a GEMM coordinate's tensor-core step and on SiLU·mul (its table read's rows): the (row, column)
    pairs of every row of the unit's ``flock-ir-unit/v2`` text, which the block places in the unit's slot, each below
    ``useful``."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as G
    from verity_flock import circuit as FC

    if which == "gemm-tc-step":
        low = IL.lowering(bind(G.GemmCoordinate, K=64, DOT=P.AmpereBF16TcDot16), "gemm-coordinate/k64/tc-step", unit_fn=IL.tc_units)
    else:
        low = REG.module("silu-mul").lowering(TM.subcircuit("silu-mul", I=4))
    lay = low.layout
    a, b = lay.nonzeros()
    assert (len(a), len(b)) == (lay.counts["nnz_a"], lay.counts["nnz_b"])
    assert all(x.dtype == np.int64 and x.shape[1] == 2 and x.min() >= 0 and x.max() < lay.useful for x in (a, b))
    assert lay.useful < 1 << FC._slot_log(lay.useful)
    got = {"a": [[] for _ in range(lay.useful)], "b": [[] for _ in range(lay.useful)]}
    for side, x in (("a", a), ("b", b)):
        for r, c in x.tolist():
            got[side][r].append(c)
    rows = low.text.splitlines()[1:1 + lay.useful]
    assert rows == [" ".join(map(str, [len(ra), *ra, len(rb), *rb])) for ra, rb in zip(got["a"], got["b"])]


def _check_set(s: InputSet) -> int:
    sub = s.subcircuit
    low = REG.module(sub.template.id).lowering(sub)
    fin, want = IL.flat_ports(s, sub.inputs, 0, s.n), IL.flat_ports(s, sub.outputs, 0, s.n)
    cuts = IL.cut_words(low, fin) if low.units.cut else None
    got, cut_out, ok = low.evaluate(fin, cuts)
    assert ok == (1 << (s.n * low.units.n)) - 1
    bad = (got != want).any(axis=1)
    if cuts is not None:
        bad |= (cut_out != cuts).any(axis=1)
    return int(bad.sum())


@pytest.mark.parametrize("name", ["rope-head-d64", "silumul-v1-i8192"])
def test_captured_slices(name):
    s = InputSet.open(SLICES / name)
    assert s.source == "captured" and s.n > 0
    assert _check_set(s) == 0


@pytest.mark.parametrize("template", sorted(LOWERED))
def test_synthetic_sets(template, tmp_path):
    sub = TM.subcircuit(template, **LOWERED[template])
    n = 24 if template == "rope-head" else 3
    summary = generate(sub, n, 20260926, tmp_path)
    s = InputSet.open(summary["path"])
    assert s.source == "synthetic"
    assert _check_set(s) == 0


@pytest.mark.parametrize("template", sorted(LOWERED))
def test_lowering_is_the_pinned_one(template):
    mod = REG.module(template)
    sub = TM.subcircuit(template, **LOWERED[template])
    (param,) = LOWERED[template].values()
    low = mod.lowering(sub)
    assert low.sha256 == mod.PINS[param]
    head = low.text.split("\n", 1)[0].split()
    assert head[0] == IL.FORMAT and head[1] == mod.UNIT and int(head[2]) <= 1 << 13
    if template == "silu-mul":
        assert mod.frame_lowering(sub).sha256 == mod.FRAME_PINS[param]


@pytest.mark.parametrize("template", sorted(LOWERED))
def test_pinned_rows_read_earlier_columns(template):
    """What the soundness package's `Rows` needs beyond the parser's checks (`soundness/README.md` §1.5): every computed
    row reads only earlier columns or the constant, and never an input padding column (`A = B = 0`, forced zero)."""
    mod = REG.module(template)
    sub = TM.subcircuit(template, **LOWERED[template])
    lows = [mod.lowering(sub)] + ([mod.frame_lowering(sub)] if template == "silu-mul" else [])
    for low in lows:
        lay = low.layout
        n_in = lay.in_words * IL.WORD
        padding = {c for c in range(n_in) if not lay.rows_a[c] and not lay.rows_b[c]}
        bad = [(r, c) for r in range(n_in, lay.const) for c in lay.rows_a[r] + lay.rows_b[r]
               if not ((c < r and c not in padding) or c == lay.const)]
        assert lay.const + 1 == lay.useful and not bad, (low.name, bad[:5])


def test_the_pin_covers_the_leaf_maps():
    """IR6: the netlist's LEAVES line places every unit on its instance's words (flat input and output leaves, each port's
    [leaves, bits]); another arrangement of the same unit is another pin with the same rows."""
    import json

    mod = REG.module("rope-head")
    a, b = mod.lowering(TM.subcircuit("rope-head", D=64)), mod.lowering(TM.subcircuit("rope-head", D=32))
    line = a.text.splitlines()[-1]
    assert line.startswith("LEAVES ") and json.loads(line.removeprefix("LEAVES ")) == a.leaves
    assert a.leaves["in"][1] == [1, 33, 65, 97] and a.leaves["out"][1] == [1, 33]
    assert a.leaves["in_ports"] == [[64, 16], [64, 16]] and a.leaves["out_ports"] == [[64, 16]]
    assert a.sha256 != b.sha256 and a.text.splitlines()[:-1] == b.text.splitlines()[:-1]


INGRESS = {"MODEL": "m", "LP": 3, "V": 1000}


def test_the_ingress_parse_is_one_unit_equal_to_its_definition(tmp_path):
    """``ingress-prompt`` (the I/O shell's ``IngressPrompt_v1``) lowers as ONE unit (``ir_lower.whole``) whose input row is the
    body, whole SHA-512 blocks: its rows equal the IR on a synthetic set of schema bodies, and give ``io_shell.reference``'s
    (ok, ids, max_tokens) on the same bodies with one byte changed, which leaves the schema or moves a field."""
    from verity.ml.definitions.registry import io_shell as X

    mod, sub = REG.module("ingress-prompt"), TM.subcircuit("ingress-prompt", **INGRESS)
    s = InputSet.open(generate(sub, 4, 20261024, tmp_path)["path"])
    assert _check_set(s) == 0
    low = mod.lowering(sub)
    assert low.units.n == 1 and sum(low.units.in_widths) % 1024 == 0 and low.sha256 == mod.PINS[tuple(INGRESS.values())]
    rng, LP = random.Random(3), INGRESS["LP"]
    fin = IL.flat_ports(s, sub.inputs, 0, s.n).copy()
    for row in fin:
        row[rng.randrange(len(row))] ^= 1 + rng.randrange(255)
    got, _cuts, ok = low.evaluate(fin)
    want = [X.reference(bytes(int(b) for b in row), *INGRESS.values()) for row in fin]
    assert ok == (1 << s.n) - 1 and [(int(r[0]), [int(x) for x in r[1:1 + LP]], int(r[1 + LP])) for r in got] == want
    assert any(w[0] == 0 for w in want)


def test_the_greedy_select_is_its_chunk_and_combine_units(tmp_path):
    """``token-select-chunk`` and ``token-select-combine`` (``dead_free.TokenSelect_v4``'s two calls) lower as one unit each
    (``ir_lower.whole``), the chunk's input row 64 bf16 logits, one SHA-512 block: their rows equal the IR on synthetic sets, and
    the chunks' units feeding the combine's give ``TokenSelect_v1``'s token on rows of V = 200 whose maximum is tied across a
    chunk boundary, behind a NaN at a chunk's first logit, or alone in the last, short chunk."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.definitions.registry import b1

    V = 200
    lows = {}
    for t, p in (("token-select-chunk", {"C": 64}), ("token-select-combine", {"V": V, "C": 64})):
        mod, sub = REG.module(t), TM.subcircuit(t, **p)
        assert _check_set(InputSet.open(generate(sub, 3, 20261024, tmp_path / t)["path"])) == 0
        lows[t] = low = mod.lowering(sub)
        assert low.units.n == 1 and low.sha256 == mod.PINS[tuple(p.values()) if len(p) > 1 else p["C"]]
    assert sum(lows["token-select-chunk"].units.in_widths) == 1024
    rng = np.random.default_rng(7)
    x = rng.integers(0, 1 << 16, (6, V)).astype(np.uint64)
    x[1] = 0xBF80
    x[1, [63, 64]] = 0x3F80
    x[2] = 0x3F80
    x[2, 128], x[2, 150] = 0x7FC0, 0x4000
    x[3] = 0xFF80
    x[3, 197] = 0xBF80
    best, idx = [], []
    for lo, hi in b1.chunks(V, 64):
        low = lows["token-select-chunk"] if hi - lo == 64 else REG.module("token-select-chunk").lowering(
            TM.subcircuit("token-select-chunk", C=hi - lo))
        got, _cuts, ok = low.evaluate(x[:, lo:hi])
        assert ok == (1 << len(x)) - 1
        best.append(got[:, 0])
        idx.append(got[:, 1])
    got, _cuts, ok = lows["token-select-combine"].evaluate(np.column_stack([x[:, 0], *best, *idx]))
    want = np.asarray(evaluate_batch(bind(b1.TokenSelect, V=V), x)).reshape(len(x))
    assert ok == (1 << len(x)) - 1 and got[:, 0].tolist() == want.tolist()
    assert want[1:4].tolist() == [63, 150, 197]


def test_supports_says_what_is_missing():
    for template, params in LOWERED.items():
        sub = TM.subcircuit(template, **params)
        assert REG.supports(sub, "frame-v3/blake3-keyed") is None and REG.relation(sub, "frame-v3/blake3-keyed") == f"irf:{template}"
        assert "not wired yet" in REG.supports(sub, "vllm-v1")
    for template in ("rmsnorm-fused-cuda", "rmsnorm-triton"):
        sub = TM.subcircuit(template, N=2048, EPS=1e-5)
        assert REG.supports(sub, "frame-v3/blake3-keyed") is None and REG.relation(sub, "frame-v3/blake3-keyed") == f"irf:{template}"
        assert "not wired yet" in REG.supports(sub, "vllm-v1")


def test_stage_writes_the_instance_file(tmp_path):
    s = InputSet.open(SLICES / "rope-head-d64")
    path = IL.stage(REG.module("rope-head").lowering(s.subcircuit), s, 0, 2, tmp_path)
    head, body = path.read_bytes().split(b"\n", 1)
    import json

    h = json.loads(head)
    assert h["format"] == IL.INSTANCES_FORMAT and h["units"] == 64 and h["in_words"] == 1 and h["out_words"] == 1
    assert len(body) == 64 * (1 + 1) * 16
    assert (tmp_path / "net.txt").read_text().startswith(IL.FORMAT)
    first = int.from_bytes(body[:16], "little")
    x = IL.flat_ports(s, s.subcircuit.inputs, 0, 1)[0]
    assert first == int(x[0]) | int(x[32]) << 16 | int(x[64]) << 32 | int(x[96]) << 48
    y = IL.flat_ports(s, s.subcircuit.outputs, 0, 1)[0]
    outs = body[64 * 16:]
    assert [int.from_bytes(outs[16 * k:16 * k + 16], "little") for k in range(2)] == [int(y[0]) | int(y[32]) << 16, int(y[1]) | int(y[33]) << 16]


# --- negatives ---------------------------------------------------------------------------------------------------------------

def test_a_flipped_witness_bit_leaves_its_lane_unsatisfied():
    low = REG.module("rope-head").lowering(TM.subcircuit("rope-head", D=8))
    x = np.arange(16, dtype=np.uint64)[None, :] * 0x101 + 0x3F00
    first_and = low.layout.in_words * IL.WORD
    for row in (first_and, first_and + 1000, low.layout.useful - 2):
        _, _, ok = low.evaluate(x, flip=(row, 2))
        assert ok == ((1 << 4) - 1) & ~(1 << 2)


def test_a_mutated_netlist_is_caught_by_the_ir_comparison():
    """Negating one output row, or dropping one term of one AND row, changes the unit's function on the stratified words. A
    few AND rows sit in logic this composition cannot reach (RoPE's fma addend is RN(y*s), never more than 16 significant
    bits; the pieces' NaN is already the torch word, so the cast's NaN select is a no-op), so mutating them is not
    observable by any input; the sample bound allows for them."""
    low = REG.module("rope-head").lowering(TM.subcircuit("rope-head", D=8))
    rng = random.Random(11)
    vecs = [bf16_words(rng, 4) for _ in range(4000)]
    ports = [[v[j] for v in vecs] for j in range(4)]
    want = ([P.RopeOut.evaluate(*v) for v in vecs], [P.RopeOutAdd.evaluate(*v) for v in vecs])
    lay = low.layout
    first_and = lay.in_words * IL.WORD

    def caught_after(r, toggle_const=False):
        saved = lay.rows_a[r]
        if toggle_const:
            lay.rows_a[r] = saved[:-1] if saved[-1] == lay.const else saved + [lay.const]
        else:
            lay.rows_a[r] = saved[:-1] if len(saved) > 1 else saved + [lay.const]
        (lo, hi), _ = IL.evaluate_layout(lay, ports, len(vecs))
        lay.rows_a[r] = saved
        return [int(x) for x in lo] != want[0] or [int(x) for x in hi] != want[1]

    assert all(caught_after(lay.out_port_cols[j] + b, toggle_const=True) for j in range(2) for b in (0, 7, 15))
    assert sum(caught_after(r) for r in rng.sample(range(first_and, first_and + lay.counts["ands"]), 12)) >= 8


def test_a_wrong_table_entry_is_caught(monkeypatch):
    table = list(IL.silu_table())
    table[0x3F90] ^= 1
    monkeypatch.setattr(IL, "silu_table", lambda: tuple(table))
    low = IL.lowering(TM.subcircuit("silu-mul", I=4).definition(), "tampered")
    (out,), _ = IL.evaluate_layout(low.layout, [[0x3F90, 0x3F91], [0x3F80, 0x3F80]], 2)
    assert int(out[0]) != P.SiluMulBf16.evaluate(0x3F90, 0x3F80)
    assert int(out[1]) == P.SiluMulBf16.evaluate(0x3F91, 0x3F80)


def test_a_reduction_is_one_component():
    assert IL.units(TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5).definition()).n == 1


# --- RMSNorm: warps as units, the row tail native ---------------------------------------------------------------------------

SLOW = pytest.mark.skipif(not os.environ.get("FLOCK_IR_SLOW"), reason="builds a 2^19/2^21-row unit: set FLOCK_IR_SLOW=1")
RMS = {"rmsnorm-fused-cuda": "rmsnormfusedcuda-v2-n2048-00e4e4aa", "rmsnorm-triton": "rmsnormtriton-v1-n2048-00e4e4aa"}


def _tail(template, n):
    from collections import Counter

    sub = TM.subcircuit(template, N=n, EPS=1e-5)
    mod = REG.module(template)
    U = IL.units(sub.definition(), IL.reduction_cut(mod.width(n)))
    return U, Counter(g.prim.id for g in U.tail if not g.prim.id.startswith("Const"))


def test_rmsnorm_units_are_warps_with_a_native_tail():
    U, tail = _tail("rmsnorm-fused-cuda", 2048)
    assert U.n == 32 and len(U.cut) == 33
    assert tail == {"F32Add_v1": 32, "F32Div_v1": 1, "RsqrtApprox_v1": 1}
    assert U.in_src[0][-1] == ("cut", 32) and U.out_src[0][-1] == ("cut", 0)
    U, tail = _tail("rmsnorm-triton", 2048)
    assert U.n == 8 and len(U.cut) == 9
    assert tail == {"F32Add_v1": 8, "F32Fma_v1": 1, "MufuSqrtFtz_v1": 1, "DivFullRcp_v1": 2, "DivFullScaleA_v1": 1, "F32Mul_v1": 1}


def test_rmsnorm_widths_refuse_unequal_warps():
    fused, triton = REG.module("rmsnorm-fused-cuda"), REG.module("rmsnorm-triton")
    assert fused.width(2048) == 128 and fused.width(576) == 64 and fused.width(1536) is None
    assert triton.width(2048) == 256 and triton.width(1536) is None
    assert "not N = 1536" in REG.supports(TM.subcircuit("rmsnorm-fused-cuda", N=1536, EPS=1e-5), "frame-v3/blake3-keyed")


@pytest.mark.slow
def test_rmsnorm_fused_small_row_against_the_ir(tmp_path):
    """N = 64: two warps of 32 one-element threads; spine synthetic rows, outputs and warp aggregates both compared."""
    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 6, 20260926, tmp_path)["path"])
    assert REG.module("rmsnorm-fused-cuda").lowering(sub).units.n == 2
    assert _check_set(s) == 0


def test_a_wrong_row_scalar_changes_the_outputs(tmp_path):
    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 2, 7, tmp_path)["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    fin = IL.flat_ports(s, sub.inputs, 0, s.n)
    cuts = IL.cut_words(low, fin)
    good, _, _ = low.evaluate(fin, cuts)
    cuts[:, -1] ^= 1 << 12
    bad, _, _ = low.evaluate(fin, cuts)
    assert (good != bad).any(axis=1).all()


@pytest.mark.slow
def test_the_netlist_accounts_for_every_cut_word(tmp_path):
    """IR1 (the Python half; the Rust verifier refuses the rest, `flock-ir-block selftest`): the netlist's CUT line lists each
    unit's cut ports and the native tail; every cut word is computed exactly once, by a unit or by the tail, from words defined
    before. The instance header only counts them."""
    import json

    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 2, 3, tmp_path / "set")["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    head = json.loads(IL.stage(low, s, 0, 2, tmp_path / "st").read_bytes().split(b"\n", 1)[0])
    cut = json.loads((tmp_path / "st" / "net.txt").read_text().splitlines()[-1].removeprefix("CUT "))
    assert cut == low.cut and "cut" not in head
    assert head["cut_words"] == cut["words"] == len(low.units.cut) == 3 and len(cut["in"]) == len(cut["out"]) == low.units.n == 2
    produced = [k for unit in cut["out"] for _, k in unit]
    tail_out = [op[0] for op in cut["tail"] if op[0] < cut["words"]]
    assert sorted(produced + tail_out) == list(range(cut["words"]))
    defined = set(produced)
    for op in cut["tail"]:
        assert set(op[2:] if op[1] != "Const" else []) <= defined
        defined.add(op[0])
    assert {op[1] for op in cut["tail"]} <= {"Const", *IL.TAIL_PRIMS}
    rs = InputSet.open(SLICES / "rope-head-d64")
    rope = REG.module("rope-head").lowering(rs.subcircuit)
    head = json.loads(IL.stage(rope, rs, 0, 1, tmp_path / "r").read_bytes().split(b"\n", 1)[0])
    assert rope.cut is None and head["cut_words"] == 0 and not rope.text.splitlines()[-1].startswith("CUT")


@pytest.mark.slow
def test_the_pin_covers_the_cut_structure():
    """IR4: N and eps are tail constants in the netlist's CUT line, so another eps is another pin; the unit rows do not
    depend on eps (the row scalar is a cut word)."""
    import struct

    f32 = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]  # noqa: E731
    mod = REG.module("rmsnorm-fused-cuda")
    a = mod.lowering(TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5))
    b = mod.lowering(TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-3))
    consts = {op[2] for op in a.cut["tail"] if op[1] == "Const"}
    assert {f32(1e-5), f32(64.0)} <= consts and f32(1e-3) not in consts
    assert a.sha256 != b.sha256 and a.text.splitlines()[1:-1] == b.text.splitlines()[1:-1]
    assert a.text.splitlines()[-1] != b.text.splitlines()[-1]


def test_a_producers_file_is_not_the_verifiers_own(tmp_path):
    """IR2: the verifier stages its own file; a producer's file with one output word changed differs from it."""
    s = InputSet.open(SLICES / "rope-head-d64")
    low = REG.module("rope-head").lowering(s.subcircuit)
    path = IL.stage(low, s, 0, 2, tmp_path)
    assert IL.check_staged(low, s, path) == []
    b = bytearray(path.read_bytes())
    b[-3] ^= 1
    path.write_bytes(bytes(b))
    assert IL.check_staged(low, s, path) == [f"body differs from the verifier's own staging of {s.name} [0, 2)"]


def test_the_tables_are_the_pinned_ones(tmp_path):
    assert IL.write_tables(tmp_path) == IL.TABLES
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ex2.u32", "rcp.u32", "rsq.u32", "sqrt.u32"]


@SLOW
@pytest.mark.parametrize("template", sorted(RMS))
def test_rmsnorm_captured_slices_and_pins(template):
    s = InputSet.open(SLICES / RMS[template])
    assert _check_set(s) == 0
    p = s.subcircuit.params
    assert REG.module(template).lowering(s.subcircuit).sha256 == REG.module(template).PINS[(int(p["N"]), float(p["EPS"]))]


#: The served workloads' other parameters (#73 Qwen3-4B on the H100, #60 Mistral-7B on the L40S, GLM-5.3's dense layers at TP8
#: for `silu-mul-v2`): the same units read off the IR, so RoPE and SiLU·mul keep the granted rows and only their leaf maps (the
#: LEAVES line) differ; the RMSNorm warps widen.
WORKLOAD = [("rope-head", {"D": 128}), ("silu-mul", {"I": 9728}), ("silu-mul", {"I": 14336}), ("silu-mul-v2", {"I": 1536})]
WORKLOAD_RMS = [("rmsnorm-fused-cuda", {"N": 4096, "EPS": 1e-05}), ("rmsnorm-triton", {"N": 4096, "EPS": 1e-05}),
                ("rmsnorm-triton", {"N": 128, "EPS": 1e-06})]


@pytest.mark.parametrize("template,params", WORKLOAD)
def test_workload_pins_keep_the_granted_rows(template, params):
    mod = REG.module(template)
    (param,) = params.values()
    low, granted = mod.lowering(TM.subcircuit(template, **params)), mod.lowering(TM.subcircuit(template, **LOWERED[template]))
    assert low.sha256 == mod.PINS[param]
    if template == "silu-mul":
        assert mod.frame_lowering(TM.subcircuit(template, **params)).sha256 == mod.FRAME_PINS[param]
    rows = lambda t: [line for line in t.splitlines() if not line.startswith("LEAVES ")]  # noqa: E731
    assert rows(low.text) == rows(granted.text) and low.sha256 != granted.sha256


@SLOW
@pytest.mark.parametrize("template,params", WORKLOAD_RMS)
def test_workload_rmsnorm_pins(template, params):
    mod = REG.module(template)
    assert mod.lowering(TM.subcircuit(template, **params)).sha256 == mod.PINS[(params["N"], params["EPS"])]


def test_rmsnorm_warp_widths():
    """Identical warp units exist when every warp holds the same columns: Triton N a multiple of 1024 (8 per lane per chunk) or a
    power of two under 1024 (N/128 per lane); fused CUDA when min(N, 1024) threads divide N. N = 2560 is neither."""
    tr, fu = REG.module("rmsnorm-triton"), REG.module("rmsnorm-fused-cuda")
    assert [tr.width(n) for n in (96, 128, 256, 512, 1024, 2048, 2560, 4096)] == [None, 32, 64, 128, 256, 256, None, 256]
    assert (fu.width(4096), fu.width(2560)) == (256, None)
    assert REG.supports(TM.subcircuit("rmsnorm-triton", N=128, EPS=1e-06), "frame-v3/blake3-keyed") is None
    for template in ("rmsnorm-triton", "rmsnorm-fused-cuda"):
        assert "not N = 2560" in REG.supports(TM.subcircuit(template, N=2560, EPS=1e-06), "frame-v3/blake3-keyed")


# --- RMSNorm's width-8 CUDA kernels (GLM-5.3): whole warps a unit, a power-of-two count of them -----------------------------

VEC8 = ("rmsnorm-fused-cuda-vec8", "rmsnorm-cuda-vec8")
#: GLM-5.3's launches at hidden 6144 (``cuda_rmsnorm.launch_block``: under 256 tokens min(N, 1024) threads fused and N / 8
#: plain, else 256): 24 warps of one vector, or 8 warps of three
VEC8_GLM = [("rmsnorm-fused-cuda-vec8", 1024), ("rmsnorm-fused-cuda-vec8", 256), ("rmsnorm-cuda-vec8", 768), ("rmsnorm-cuda-vec8", 256)]
#: the scheme of ``verity/flock-circuit`` (``verity_flock.circuit.SCHEME``), the only one these templates are proved under
CIRCUIT_SCHEME = "frame-v3-sha512/hm96-sha512"


@pytest.mark.parametrize("template,block", VEC8_GLM)
def test_vec8_glm_launches_are_eight_units_of_whole_warps(template, block):
    """At N = 6144 each launch splits into 8 identical units of 768 elements, each reading the row scalar and computing its
    warps' aggregates (cut words), whatever the warp count: the odd part of 24 warps (three) is a unit. The tail is the warp
    aggregates' sum, ``/ N``, ``+ eps`` and RsqrtApprox."""
    from collections import Counter

    mod = REG.module(template)
    sub = TM.subcircuit(template, N=6144, EPS=1e-5, BLOCK=block)
    count, per_thread = mod.warps(6144, block)
    width, group = mod.cut(6144, block)
    assert (count, per_thread, group) == ((24, 1, 3) if block > 256 else (8, 3, 1))
    assert width == (2 if mod.FUSED else 1) * 256 * per_thread
    U = IL.units(sub.definition(), IL.reduction_cut(width), group)
    ports = 3 if mod.FUSED else 2
    assert U.n == 8 and len(U.cut) == count + 1
    tail = Counter(g.prim.id for g in U.tail if not g.prim.id.startswith("Const"))
    assert tail == {"F32Add_v1": count, "F32Div_v1": 1, "RsqrtApprox_v1": 1}
    assert all(Counter(k for k, _ in src) == {"leaf": ports * 768, "cut": 1} for src in U.in_src)
    assert all(Counter(k for k, _ in src) == {"ret": (ports - 1) * 768, "cut": group} for src in U.out_src)
    leaves = sorted(p for src in U.in_src for k, p in src if k == "leaf")
    assert leaves == list(range(ports * 6144))
    assert REG.supports(sub, CIRCUIT_SCHEME) is None and REG.relation(sub, CIRCUIT_SCHEME) == "verity/flock-circuit"
    assert "verity/flock-circuit" in REG.supports(sub, "frame-v3/blake3-keyed")


def test_vec8_warps_refuse_unequal_threads():
    """Identical warps need min(BLOCK, N / 8) threads in whole warps, each holding as many vectors: DeepSeek-V4-Pro's hidden
    7168 at BLOCK 256 gives some threads three vectors and others four; the fused kernel's threads beyond N / 8 hold nothing."""
    fused, plain = (REG.module(t) for t in VEC8)
    assert fused.warps(7168, 1024) == (28, 1) and fused.cut(7168, 1024) == (512, 7) and plain.cut(7168, 896) == (256, 7)
    assert fused.warps(7168, 256) is None and plain.warps(6144, 512) is None and fused.warps(6144, 1000) == (24, 1)
    assert fused.warps(6144 + 8, 1024) is None and fused.warps(128, 1024) is None
    for template in VEC8:
        why = REG.supports(TM.subcircuit(template, N=7168, EPS=1e-5, BLOCK=256), CIRCUIT_SCHEME)
        assert "not N = 7168, BLOCK = 256" in why


def _vec8_rows(fused: bool, n: int, rng: random.Random) -> np.ndarray:
    """Rows (x, res, w), or (x, w), where the norm's special cases decide the outputs: an all-zero row, x + res cancelling to
    +0, -0 everywhere, a NaN or an infinity in the second warp, a residual add overflowing to inf, a square overflowing f32,
    subnormal x (squares underflow to 0), a subnormal variance, one dominant element (the summation order rounds it), w's
    special words; then finite random rows over a wide exponent range."""
    def finite(lo=100, hi=150):
        return [(rng.getrandbits(1) << 15) | (rng.randint(lo, hi) << 7) | rng.getrandbits(7) for _ in range(n)]

    rows = []
    for case in range(13):
        x, res, w = finite(), finite(), finite(120, 134)
        if case == 0:
            x, res = [0] * n, [0] * n
        elif case == 1:
            res = [v ^ 0x8000 for v in x]
        elif case == 2:
            x, res = [0x8000] * n, [0x8000] * n
        elif case == 3:
            x[300] = 0x7FC1
        elif case == 4:
            x[5] = 0x7F80
        elif case == 5:
            res[400] = 0xFF80
        elif case == 6:
            x[7] = res[7] = 0x7F7F
        elif case == 7:
            x[9], res[9] = 0x5F80, 0
        elif case == 8:
            x, res = [0x0001 | (rng.getrandbits(1) << 15) for _ in range(n)], [0] * n
        elif case == 9:
            x, res = [0x1F80] * n, [0] * n
        elif case == 10:
            x, res = [0x4B80] + [0x3F80] * (n - 1), [0] * n
        elif case == 11:
            w[3:8] = [0x7FC0, 0x7F80, 0x0000, 0x8000, 0x0001]
        rows.append(x + res + w if fused else x + w)
    for _ in range(8):
        rows.append(finite() + finite() + finite(120, 134) if fused else finite() + finite(120, 134))
    rows.append(bf16_words(rng, (3 if fused else 2) * n))
    return np.array(rows, dtype=np.uint64)


@pytest.mark.slow
@pytest.mark.parametrize("template", VEC8)
def test_vec8_units_are_the_definition_on_edge_and_random_rows(template):
    """The lowered units with the IR's cut words against the Definition's gate reference, outputs and warp aggregates, on
    :func:`_vec8_rows`; every honest lane satisfied. Builds a 1.2M-row unit (about 30 s): a vec8 warp is 256 elements, and
    nothing smaller exercises the width-8 thread partials, the square-pair sums or the FMA chain."""
    from verity.ml.circuits.evaluation import evaluate_batch

    sub = TM.subcircuit(template, N=512, EPS=1e-5, BLOCK=64)            # two warps of one vector a thread, a unit each
    low = REG.module(template).lowering(sub)
    fused = REG.module(template).FUSED
    assert low.units.n == 2 and len(low.units.cut) == 3
    fin = _vec8_rows(fused, 512, random.Random(20261016))
    want = np.asarray(evaluate_batch(sub.definition(), *np.split(fin, 3 if fused else 2, axis=1), kernel="reference"),
                      dtype=np.uint64).reshape(len(fin), -1)
    cuts = IL.cut_words(low, fin)
    got, cut_out, ok = low.evaluate(fin, cuts)
    assert ok == (1 << (len(fin) * low.units.n)) - 1
    assert [i for i in range(len(fin)) if (got[i] != want[i]).any() or (cut_out[i] != cuts[i]).any()] == []
    normed = want[:, :512]
    assert (normed[0] & 0x7FFF == 0).all()                                 # the zero row: +-0 out
    assert ((normed[3] & 0x7F80 == 0x7F80) & (normed[3] & 0x7F != 0)).all()  # one NaN: every output NaN


@SLOW
@pytest.mark.parametrize("template,block", VEC8_GLM)
def test_vec8_glm_pins(template, block):
    """The GLM-5.3 units (about 3.5M rows each): their pins (``PINS``, keyed (N, EPS, BLOCK)) and rows of 768 elements."""
    mod = REG.module(template)
    low = mod.lowering(TM.subcircuit(template, N=6144, EPS=1e-5, BLOCK=block))
    assert low.sha256 == mod.PINS[(6144, 1e-5, block)] and low.units.n == 8
    assert low.layout.useful < 1 << 22


def test_units_refuse_components_of_different_shapes():
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.refs import array_of
    from verity.ml.circuits.types import Array, Value

    mixed = CompositeDefinition("FlockLoweringMixed", 1, (), lambda S: ((("x", Array(4, Value(32))),), Array(2, Value(32))),
                                lambda B, S, x: array_of([B.call(P.F32Add, x[0], x[1]), B.call(P.F32Mul, x[2], x[3])]),
                                register=False)
    with pytest.raises(ValueError, match="not all the same unit"):
        IL.units(bind(mixed))


def test_a_primitive_without_a_piece_is_named():
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.types import Value

    odd = PrimitiveDefinition("FlockLoweringTestOnly", 1, (("a", Value(8)),), Value(8), lambda a: a ^ 0x55, register=False)
    C = gf2.Circuit()
    with pytest.raises(IL.NoPiece, match="FlockLoweringTestOnly_v1"):
        IL.lower(C, odd, [C.input("a", 8)])


def test_piece_counts_are_what_the_design_says():
    """Constant propagation specializes the general pieces: a bf16 x bf16 product needs far fewer ANDs than f32 x f32."""
    def count(f, widths):
        C = gf2.Circuit()
        C.output("o", f(C, *[C.input(f"i{k}", w) for k, w in enumerate(widths)]))
        return C.counts()["ands"]

    general = count(fp.f32_mul, (32, 32))
    narrow = count(lambda C, a, b: fp.f32_mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b)), (16, 16))
    assert narrow * 3 < general


# --- verity/flock-ir-frame/v3: the frame-v3 binding --------------------------------------------------------------------------

@pytest.mark.slow
def test_frame_plans_take_the_largest_run_that_fits():
    """RoPE and SiLU·mul read whole chunks; an RMSNorm block holds runs of its chunks' compressions (fused: 4 of 16, its four
    warps' elements; Triton: 8 of 16, one warp's), within 2^22 bits."""
    from verity_flock import ir_frame as F

    rope = TM.subcircuit("rope-head", D=64)
    pl = F.plan(REG.module("rope-head").frame_lowering(rope), rope)
    assert (pl.G, pl.nb, pl.chunk_nb, pl.comp_slots, pl.units_per_block, pl.unit_pos0, pl.k_log) == (2, 2, 2, 8, 64, 64, 20)
    assert pl.wiring[0] == [(0, 0), (0, 64), (1, 0), (1, 64)]
    silu = TM.subcircuit("silu-mul", I=8192)
    low = REG.module("silu-mul").frame_lowering(silu)
    pl = F.plan(low, silu)
    assert low.units.n == 4096 and (pl.G, pl.nb, pl.chunk_nb, pl.comp_slots, pl.units_per_block, pl.unit_pos0, pl.k_log) == (1, 16, 16, 32, 256, 256, 22)
    assert pl.comps[3][0] == [(0, 3, 0), (0, 19, 0)]
    fused = TM.subcircuit("rmsnorm-fused-cuda", N=2048, EPS=1e-5)
    pl = F.plan(REG.module("rmsnorm-fused-cuda").lowering(fused), fused)
    assert (pl.nb, pl.chunk_nb, pl.G, pl.chunk_slots, pl.chunks_per_comp, pl.units_per_block, pl.k_log) == (4, 16, 1, 8, 6, 4, 22)
    assert pl.comps[1] == ([(p, c, 1) for p in range(3) for c in (0, 2)], [4, 5, 6, 7])
    triton = TM.subcircuit("rmsnorm-triton", N=2048, EPS=1e-5)
    pl = F.plan(REG.module("rmsnorm-triton").lowering(triton), triton)
    assert (pl.nb, pl.chunk_nb, pl.G, pl.chunk_slots, pl.units_per_block, pl.unit_pos0, pl.k_log) == (8, 16, 1, 2, 1, 1, 22)


def test_frame_staging_commits_rows_and_outputs(tmp_path):
    import json

    import blake3
    from verity.ml.commitments.rowleaf import ROLE_X, blake3_row_key

    s = InputSet.open(SLICES / "rope-head-d64")
    path = REG.module("rope-head").stage(s, 0, 1, tmp_path)
    head, body = path.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    assert h["format"] == "flock-ir-frame/v3" and h["scheme"] == "frame-v3/blake3-keyed" and h["units"] == 32
    assert sorted(h["frame_v3"]["roots"]) == ["cs", "out", "x"] and h["frame_v3"]["schemas"]["x"] == "blake3-keyed/row/v2"
    x = IL.flat_ports(s, s.subcircuit.inputs, 0, 1)
    row0 = x[0, :64].astype("<u2").tobytes()
    assert body[:128] == row0
    assert body[256:256 + 32] == blake3.blake3(row0, key=blake3_row_key(ROLE_X)).digest()
    assert h["blocks"] == [[[[0, 0, 0, 0], [0, 1, 0, 0], None, None], list(range(32)) + [None] * 32]]


@pytest.mark.slow
def test_frame_staging_carries_the_cut_words(tmp_path):
    """A cut template's frame file ends with every instance's cut words (the IR's), which the verifier checks against the
    netlist's pinned tail; the empty run slot of a three-port component block is padding."""
    import json

    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 3, 5, tmp_path / "set")["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    path = REG.module("rmsnorm-fused-cuda").stage(s, 0, 3, tmp_path / "st")
    head, body = path.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    assert h["cut_words"] == 3 and "cut" not in h and h["layout"]["chunks_per_block"] == 4
    assert h["blocks"][0][0] == [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], None]
    cuts = np.frombuffer(body[-3 * 3 * 4:], dtype="<u4").reshape(3, 3)
    assert (cuts == IL.cut_words(low, IL.flat_ports(s, sub.inputs, 0, 3))).all()


def test_non_gemm_results_carry_no_gemm_target():
    from verity_numerical.bench import contract

    base = dict(domain="total", B=4, proof_class="NON_ZK_PROOF", authentication="included-hash", zk_mode="none",
                security={"target": -128}, hardware={"gpu": None, "cpu": {"model": "x"}}, software={"backend": {"name": "b", "commit": None},
                "toolchain": {}}, preprocessing="none", retention="retained", instances=None)
    ok = contract.fingerprint(profile=None, K=None, subcircuit="rope-head/d64/neox-bf16", **base)
    assert contract.validate_fingerprint(ok) == []
    assert any("profile must be null" in p for p in contract.validate_fingerprint({**ok, "profile": contract.PROFILE}))
    assert any("profile must be a target" in p for p in contract.validate_fingerprint({**ok, "subcircuit": None}))
    rates = [contract.measurement(contract.OVERHEAD, 1.0)]
    assert contract._check_rates(ok, rates) and "GEMM rates" in contract._check_rates(ok, rates)[0]


def test_an_ir_cell_plans_through_the_spine():
    from verity_numerical.bench import cell as C

    two_machines = {"prover": {"machine_id": "m-p", "public_ip": "1.2.3.5"}, "verifier": {"machine_id": "m-v", "public_ip": "1.2.3.4"},
                    "link": {"peer": "1.2.3.4:7400", "peer_ip": "1.2.3.4"}}
    c = C.plan("C-interactive", "rope-head/d64/neox-bf16+frame-v3/blake3-keyed", str(SLICES / "rope-head-d64"), "vy-p", verifier="vy-v",
               verifier_addr="1.2.3.4:7400", rtt_target="1.2.3.4:22", lane="flock-ir-lowering", points="2 4", runs=5, placement=two_machines)
    assert c.relation == "irf:rope-head"
    v, p = c.jobs
    assert v["command"][:3] == ["bash", "verity/ml/flock/pod/33-ir-cell.sh", "ROLE=verifier"] and "TEMPLATE=rope-head" in v["command"]
    assert "--send" in v["argv"] and v["argv"][v["argv"].index("--send") + 1] == p["argv"][p["argv"].index("--send") + 1]
    assert "VERIFIER=1.2.3.4:7400" in p["command"] and "LOOPBACK=1" in p["command"]


# --- attention: tensor-core k-steps as units, the softmax native -------------------------------------------------------------

def test_the_tensor_core_step_matches_the_ir_on_every_encoding():
    """``fp.tc_dot16`` = ``AmpereBF16TcDot16`` (the total model): attention-range words, full-range words (overflowing groups),
    NaN / inf / inf x 0 operands and accumulators, bit for bit."""
    rng = random.Random(11)
    specials16 = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 1, 0x8001, 0x7F7F, 0xFF7F]
    specials32 = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F7FFFFF, 0xFF7FFFFF, 1]

    def bf(att):
        if att:
            return (rng.getrandbits(1) << 15) | (rng.randint(110, 135) << 7) | rng.getrandbits(7)
        r = rng.random()
        return rng.choice(specials16) if r < 0.05 else (rng.getrandbits(1) << 15) | (rng.randint(200, 254) << 7) | rng.getrandbits(7) \
            if r < 0.3 else rng.getrandbits(16)

    vecs = []
    for i in range(1500):
        att = i % 3 == 0
        acc = (rng.getrandbits(1) << 31) | (rng.randint(110, 140) << 23) | rng.getrandbits(23) if att else \
            rng.choice(specials32) if rng.random() < 0.1 else rng.getrandbits(32)
        vecs.append([acc] + [bf(att) for _ in range(32)])
    got = run_piece(lambda C, acc, *ab: IL.PIECES["AmpereBF16TcDot16_v1"](C, acc, *ab), [32] + [16] * 32, 32, vecs)
    assert [(v, g) for v, g in zip(vecs, got) if g != P.AmpereBF16TcDot16.evaluate(*v)] == []


def test_attention_units_are_tensor_core_steps():
    """Every AmpereBF16TcDot16 gate is a unit (b from K/V leaves or a zero word where the IR pads a masked key; a, acc and the
    result cut words; the query public), the rest is the tail, which returns the output; on the captured T = 129 head the laid-out
    units reproduce every tensor-core value of the IR."""
    s = InputSet.open(SLICES / "attention-head-fa2-d64-bn128")
    sub = s.subcircuit
    (T,) = {len(x) // 64 for x in s.port("k")}
    low = IL.lowering(sub.definition(T), f"attention-head/tc-step/t{T}", unit_fn=IL.tc_units)
    U = low.units
    assert (T, U.n, len(U.public), low.layout.useful) == (129, 4 * 129 + 64 * 9, 64, 9473)
    assert sum(k == "zero" for u in U.in_src for k, _ in u) == 64 * 15
    assert low.leaves["out"] == [[]] * U.n and low.cut["outputs"] and len(low.cut["outputs"]) == 64
    fin = np.stack([np.concatenate([s.port("q")[i], s.port("k")[i], s.port("v")[i]]).astype(np.uint64) for i in range(s.n)])
    cuts = IL.cut_words(low, fin)
    _res, cut_out, ok = low.evaluate(fin, cuts)
    assert ok == (1 << U.n) - 1 and (cut_out == cuts).all()


def test_attention_frame_places_a_block_per_16_keys(tmp_path):
    """A 16-key group per block: run slots [K keys | V keys] (one key = one run of two BLAKE3 blocks), unit slots [QK steps | PV
    columns]; keys past T are empty run slots, their QK steps empty unit slots, and a PV step's masked port reads the empty
    slot; the query is a public port and the tail computes the outputs, which the file carries."""
    import json

    from verity_flock import ir_frame as F

    s = InputSet.open(SLICES / "attention-head-fa2-d64-bn128")
    mod = REG.module("attention-head")
    assert mod.key_count(s) == 129
    low = mod.lowering_for_set(s)
    pl = F.plan_tc(low, s.subcircuit)
    assert (pl.nb, pl.chunk_slots, pl.units_per_block, pl.unit_pos0, pl.k_log, pl.public_ports, len(pl.comps)) == (2, 32, 128, 128, 22, [0], 9)
    runs, units = pl.comps[8]
    assert runs[0] == (1, 16, 0) and runs[1:16] == [None] * 15 and runs[16] == (2, 16, 0) and units[4:64] == [None] * 60
    assert pl.wiring[5] == [(1, 32 + 2 * t) for t in range(16)] and pl.wiring[64 + 7] == [(16 + t, 14) for t in range(16)]
    path = mod.stage(s, 0, 1, tmp_path)
    head, body = path.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    assert h["public_ports"] == [0] and h["outputs"] == "tail" and h["layout"]["ret_group"] is None and len(h["blocks"]) == 9
    out = np.frombuffer(body[-4 * h["cut_words"] - 128:-4 * h["cut_words"]], dtype="<u2")
    assert (out == s.port("out")[0]).all()


def test_attention_key_classes_and_the_class_manifest(tmp_path):
    """Key-count classes [1, 128], [129, 256], [257, 512], ...; a class manifest lists every T of its range with its v3 netlist's
    sha256 and the unit rows they share (the pin is the manifest's sha256); a class cell's sub-batches split at T boundaries."""
    import hashlib
    import json

    from verity_flock import ir_bench
    from verity_flock.templates import attention_head as A

    assert [A.key_class(t) for t in (1, 128, 129, 256, 257, 287, 512, 513)] == [(1, 128)] * 2 + [(129, 256)] * 2 + [(257, 512)] * 3 + [(513, 1024)]
    sub = TM.subcircuit("attention-head", D=64, BN=128)
    m = A.class_manifest(sub, 1, 3)
    d = json.loads(m)
    assert d["format"] == A.CLASS_FORMAT and d["T"] == [1, 3] and sorted(d["nets"]) == ["1", "2", "3"]
    assert m == (json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n").encode()
    for T in (1, 2, 3):
        low = A.cached_lowering(sub, T)
        assert d["nets"][str(T)] == low.sha256 == hashlib.sha256(low.text.encode()).hexdigest()
        assert A.unit_rows_sha256(low.text) == d["unit_rows"]

    class _K(ir_bench.KeyClass):
        def __init__(self, T):
            self.T = T

    assert _K([1, 1, 1, 2, 2, 5]).subs(6, 2) == [(0, 2), (2, 3), (3, 5), (5, 6)]
    assert _K([1, 1, 1, 2, 2, 5]).subs(4, 16) == [(0, 3), (3, 4)]


def test_an_ir_run_result_is_a_point_not_a_second_cell():
    """The runner publishes a run's result.json under its own kind: ir_bench marks it a bench-point/v1 (a known kind), so the
    cell's one bench-result/v1 is bench.cell register's, and verity_flock.register drops the kind it read."""
    from pathlib import Path

    from research.store.kinds import KINDS
    from verity_flock import ir_bench, register

    assert ir_bench.POINT_KIND == "bench-point/v1" and ir_bench.POINT_KIND in KINDS
    assert 'raw.pop("kind", None)' in Path(register.__file__).read_text()


def test_hopper_wgmma_piece_is_the_ir_primitive():
    """``PIECES["HopperBF16WgmmaDot16_v1"]`` (fp.tc_dot16 with HOPPER_BF16_WGMMA_K16's groups (16,), W 26, F -133) equals the IR
    primitive on special-heavy vectors (NaN payloads, both infinities, subnormals, cancellation)."""
    import random

    from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16 as P

    rng = random.Random(7)
    spec16 = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x3F80]
    spec32 = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 1, 0x7F7FFFFF, 0x3F800000]
    w16 = lambda: rng.choice(spec16) if rng.random() < 0.2 else rng.getrandbits(16)  # noqa: E731
    cases = []
    for _ in range(300):
        a, b = [w16() for _ in range(16)], [w16() for _ in range(16)]
        if rng.random() < 0.3:
            for j in range(0, 16, 2):
                a[j + 1], b[j + 1] = a[j] ^ 0x8000, b[j]
        cases.append((rng.choice(spec32) if rng.random() < 0.3 else rng.getrandbits(32), a, b))
    C = gf2.Circuit()
    ins = [C.input(f"in{j}", 16) for j in range(32)] + [C.input("acc", 32)]
    C.output("d", IL.PIECES["HopperBF16WgmmaDot16_v1"](C, ins[32], *ins[:32]))
    vals = {f"in{j}": [sum((((c[1] + c[2])[j] >> i) & 1) << t for t, c in enumerate(cases)) for i in range(16)] for j in range(32)}
    vals["acc"] = [sum(((c[0] >> i) & 1) << t for t, c in enumerate(cases)) for i in range(32)]
    outs, _ = C.evaluate(vals, len(cases))
    got = [sum(((outs["d"][i] >> t) & 1) << i for i in range(32)) for t in range(len(cases))]
    assert got == [P.evaluate(c[0], *c[1], *c[2]) & 0xFFFFFFFF for c in cases]


def test_a_cells_domain_is_the_one_its_statement_proves():
    """A red-team finding: ``ir_bench`` and ``bench`` wrote ``domain: finite`` for every cell.
    An IR lowering asserts nothing, so every encoding is proven to give the IR's output: ``total``.  A pure unit asserts its
    operands (and accumulator) finite: ``finite-only``, unless it is a total unit."""
    from types import SimpleNamespace

    from verity_flock import lowering as FL

    low = REG.module("silu-mul").frame_lowering(TM.subcircuit("silu-mul", I=8192))
    assert not low.circuit.asserts and low.domain == "total"
    assert IL.Lowering.domain.fget(SimpleNamespace(circuit=SimpleNamespace(asserts=[("x not finite", 0)]))) == "finite-only"
    for relation, pipe in FL.PIPES.items():
        assert pipe.domain == ("finite-only" if FL._layout(pipe)[0].asserts else "total"), relation
    assert {p.domain for p in FL.PIPES.values()} == {"finite-only"}
    assert FL.Pipe("t", None, (8, 8), 25, -132, True, "AMPERE_BF16_M16N8K16", total=True).domain == "total"


# --- the registered compiler's dead-free Definitions: RMSNormTriton_v5 and AttentionHead_v18 ----------------------------------

#: ``unit_rows_sha256`` of AttentionHead_v18's one tensor-core step, at every (D, BN, T)
V18_UNIT_ROWS = "68063483886a889a27180764bcf913cabb1a2599be427752f7fd2e681b75f50c"


def _gate_shape(U):
    by = {g.index: k for k, g in enumerate(U.tail)}
    cut = {g: k for k, g in enumerate(U.cut)}
    return [(g.prim.id, [("t", by[o]) if o in by else ("c", cut[o]) for o in g.operands]) for g in U.tail]


@pytest.mark.parametrize("n", [2048, 4096])
def test_rmsnorm_triton_v5_is_v1s_lowering(n):
    """v5 drops the 129 butterfly adds a warp that v1's lowering already left out (no output depends on them): the same warp units
    (signature, leaf maps, cut words) and the same tail, so v5 takes v1's unit netlist and its pins."""
    v5, v1 = (IL.units(TM.subcircuit(t, N=n, EPS=1e-5).definition(), IL.reduction_cut(256)) for t in ("rmsnorm-triton-v5", "rmsnorm-triton"))
    assert (v5.n, len(v5.cut)) == (n // 256, n // 256 + 1)
    assert v5.signature == v1.signature and (v5.in_src, v5.out_src) == (v1.in_src, v1.out_src)
    assert _gate_shape(v5) == _gate_shape(v1)
    assert sum(g.prim.id == "F32Add_v1" for g in v5.gates) == 31
    mod = REG.module("rmsnorm-triton-v5")
    assert mod.UNIT == REG.module("rmsnorm-triton").UNIT and mod.PINS[(n, 1e-05)] == REG.module("rmsnorm-triton").PINS[(n, 1e-05)]


def test_dead_free_templates_say_what_they_prove_under():
    v5 = REG.module("rmsnorm-triton-v5")
    assert [v5.width(n) for n in (128, 1024, 1536, 2048, 4096, 5120)] == [None, 256, None, 256, 256, 256]
    assert "not N = 128" in REG.supports(TM.subcircuit("rmsnorm-triton-v5", N=128, EPS=1e-6), CIRCUIT_SCHEME)
    for sub in (TM.subcircuit("rmsnorm-triton-v5", N=2048, EPS=1e-5), TM.attention_head_v18(64, 128), TM.attention_head_v18(128, 64)):
        assert REG.supports(sub, CIRCUIT_SCHEME) is None
        assert "not under frame-v3/blake3-keyed" in REG.supports(sub, "frame-v3/blake3-keyed")
        assert REG.relation(sub, CIRCUIT_SCHEME) == "verity/flock-circuit"
    assert "not D = 24" in REG.supports(TM.attention_head_v18(24, 32), CIRCUIT_SCHEME)


@pytest.mark.parametrize("D,BN,T", [(64, 128, 1), (64, 128, 129), (64, 128, 512), (128, 64, 65)])
def test_attention_v18_units_are_sm120_tensor_core_steps(D, BN, T):
    """Every HopperBF16WgmmaDot16 gate is a unit: D/16 QK steps a key, D PV steps per k-step of 16 visible keys; v18 pads only the
    first visited block's value rows of its last k-step (zero words), none of its key rows. The unit rows are one step's at
    every binding (a class manifest's ``unit_rows``)."""
    from verity_flock.templates import attention_head as A

    low = REG.module("attention-head-v18").lowering(TM.attention_head_v18(D, BN), T)
    U = low.units
    nvis = T - (T - 1) // BN * BN
    ksteps = (T - nvis) // 16 + -(-nvis // 16)
    assert U.signature == ("HopperBF16WgmmaDot16_v1",) and U.n == D // 16 * T + D * ksteps
    assert sum(k == "zero" for u in U.in_src for k, _ in u) == D * (-nvis % 16)
    assert len(U.public) == D and low.leaves["out"] == [[]] * U.n and len(low.cut["outputs"]) == D
    assert {op[1] for op in low.cut["tail"]} <= {"Const", *IL.TAIL_PRIMS}
    assert A.unit_rows_sha256(low.text) == V18_UNIT_ROWS


@pytest.mark.parametrize("D,BN,T", [(64, 128, 129), (128, 64, 65)])
def test_attention_v18_units_reproduce_the_ir_on_a_masked_from_head(D, BN, T, tmp_path):
    """Two key blocks at a decode launch: MASKED_FROM = 1, so the second block rescales by the running max as it is. On synthetic
    heads the laid-out units reproduce every tensor-core value of the IR."""
    sub = TM.attention_head_v18(D, BN, T=f"{T}:{T}")
    s = InputSet.open(generate(sub, 2, 20260926, tmp_path)["path"])
    mod = REG.module("attention-head-v18")
    assert mod.key_count(s) == T and sub.definition(T).statics["MASKED_FROM"] == 1
    low = mod.lowering_for_set(s)
    fin = IL.flat_ports(s, sub.inputs, 0, s.n)
    cuts = IL.cut_words(low, fin)
    _res, cut_out, ok = low.evaluate(fin, cuts)
    assert ok == (1 << (s.n * low.units.n)) - 1 and (cut_out == cuts).all()


@SLOW
@pytest.mark.parametrize("n", [2048, 4096])
def test_rmsnorm_triton_v5_pins_are_v1s(n):
    sub = TM.subcircuit("rmsnorm-triton-v5", N=n, EPS=1e-5)
    mod = REG.module("rmsnorm-triton-v5")
    low = mod.circuit_lowering(sub)
    assert low.sha256 == mod.PINS[(n, 1e-05)] and low.text == REG.module("rmsnorm-triton").lowering(TM.subcircuit("rmsnorm-triton", N=n, EPS=1e-5)).text
