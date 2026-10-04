"""PoUW's own primitives on bits (`verity_pouw.circuit.boolean`) are their word primitives (`words`) bit for bit: on every input
where the input space allows (the bf16 negation, Ẽ over E2M1 × U8), else on each operand's edges (±0, the subnormals' ends,
every binade's ends, ±inf, NaNs, the rounding and threshold boundaries) and dense seeded random words."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.evaluation.bits import evaluate_bits, evaluate_words
from verity.ml.boolean.forms import Circuit
from verity_pouw.circuit import boolean as B
from verity_pouw.circuit import words as W

M32, M576 = (1 << 32) - 1, (1 << 576) - 1
#: every FP32 class's ends: ±0, the subnormals', the normals', ±inf, quiet, signalling and all-ones NaNs
F32_EDGES = [0, 1, 2, 3, 0x7FFFFF, 0x800000, 0x800001, 0xFFFFFF, 0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x40000000, 0x7F7FFFFF,
             0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF]
F32_EDGES += [w | 0x80000000 for w in F32_EDGES]
#: each exponent field's smallest, next and largest mantissa, both signs
F32_BINADES = [s | e << 23 | m for s in (0, 1 << 31) for e in range(256) for m in (0, 1, 0x7FFFFF)]


def _words(d, ref, *cols):
    got = evaluate_words(d, *[np.asarray(c, dtype=np.uint64).reshape(-1, 1) for c in cols]).reshape(-1).tolist()
    want = [ref.evaluate(*a) for a in zip(*[[int(v) for v in c] for c in cols])]
    bad = [i for i in range(len(want)) if got[i] != want[i]]
    assert not bad, [(tuple(hex(int(c[i])) for c in cols), hex(got[i]), hex(want[i])) for i in bad[:5]]
    return len(want)


def _bits(vals, w):
    return np.array([[(v >> j) & 1 for j in range(w)] for v in vals], dtype=np.uint8)


def _wide(d, ref, cols, widths):
    out = evaluate_bits(d, *[_bits(c, w) for c, w in zip(cols, widths)])
    got = [sum(int(b) << j for j, b in enumerate(np.atleast_1d(row))) for row in out]
    want = [ref.evaluate(*a) for a in zip(*cols)]
    bad = [i for i in range(len(want)) if got[i] != want[i]]
    assert not bad, [tuple(hex(c[i]) for c in cols) for i in bad[:3]]


def test_each_names_its_word_primitive_and_is_boolean():
    """Each is the next version of its word primitive and names it; each but `PouwSqCeil_v2` and the windows (traced in
    their own, slow tests) is Boolean through its body."""
    from verity.ir.boolean import is_boolean
    from verity.ml.boolean.trace import word_of

    for d, w in ((B.PouwF32SqrtRn, W.PouwF32SqrtRn), (B.PouwETilde, W.PouwETilde), (B.PouwBf16Neg, W.PouwBf16Neg),
                 (B.PouwSqAcc, W.PouwSqAcc), (B.PouwSqCeil, W.PouwSqCeil), (B.PouwSqGe, W.PouwSqGe),
                 (B.PouwBlackwellE4m3Window, W.PouwBlackwellE4m3Window), (B.PouwHopperE4m3Window, W.PouwHopperE4m3Window),
                 (B.PouwWindowWord, W.PouwWindowWord), (B.PouwBlackwellE4m3SaltDead, W.PouwBlackwellE4m3SaltDead),
                 (B.PouwHopperE4m3SaltDead, W.PouwHopperE4m3SaltDead)):
        assert word_of(d) is w and d.id == w.id.replace("_v1", "_v2")
        assert d in (B.PouwSqCeil, B.PouwBlackwellE4m3Window, B.PouwHopperE4m3Window) or is_boolean(d.bind())


@pytest.mark.slow
def test_the_windows_are_boolean():
    """Each window's body, its builder traced and emitted (4 to 7 s each), is Boolean."""
    from verity.ir.boolean import is_boolean

    assert is_boolean(B.PouwBlackwellE4m3Window.bind()) and is_boolean(B.PouwHopperE4m3Window.bind())


def test_building_a_tile_check_registers_them():
    """C-Flock's lowering finds a primitive's Boolean Definition only in core's registry, so importing `pc4` (and `pc8`,
    which imports it) must register all eleven, in a fresh interpreter where nothing else imported `boolean`."""
    import subprocess
    import sys

    ids = [w.id.replace("_v1", "_v2") for w in (W.PouwF32SqrtRn, W.PouwETilde, W.PouwBf16Neg, W.PouwSqAcc, W.PouwSqCeil,
                                                W.PouwSqGe, W.PouwBlackwellE4m3Window, W.PouwHopperE4m3Window,
                                                W.PouwWindowWord, W.PouwBlackwellE4m3SaltDead, W.PouwHopperE4m3SaltDead)]
    code = ("import sys, verity_pouw.circuit.pc8\nfrom verity.ir.defs import REGISTRY\n"
            f"missing = [i for i in {ids!r} if i not in REGISTRY]\nsys.exit(','.join(missing) if missing else 0)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


#: E4M3's ends: ±0, the subnormals', the normals', ±1, the largest finite, and the two NaNs
E4M3_EDGES = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x7E, 0xFE, 0x7F, 0xFF]


@pytest.mark.parametrize("name, ands", [("PouwBlackwellE4m3Window", 102_409), ("PouwHopperE4m3Window", 67_623)])
def test_window_on_finite_cancelling_edge_and_nan_codes(name, ands):
    """A window's masked words on random finite codes, products that cancel pairwise (+0 words, and totals of either sign
    near zero), small codes (subnormal sums), E4M3's ends (largest products, both zeros, NaNs) and finite windows with one
    NaN code (NaN in exactly the words whose atoms read it).  The bit form's builder runs on a plain circuit, as C-Flock's
    lowering does (no common-subexpression pass), and has exactly the traced Definition's ANDs: each atom's products are
    built once for all its words, where the 15 atom calls are 145,651 ANDs on sm_120 and 109,735 on the H100."""
    rng = np.random.default_rng(20261008)
    finite = np.array([c for c in range(256) if c & 0x7F != 0x7F])
    a, b = rng.choice(finite, (96, 128)), rng.choice(finite, (96, 128))
    a[16:32, 1::2] = a[16:32, ::2] ^ 0x80
    b[16:32, 1::2] = b[16:32, ::2]
    a[32:48], b[32:48] = rng.choice(finite[(finite & 0x78) == 0], (2, 16, 128))
    a[48:72], b[48:72] = rng.choice(E4M3_EDGES, (2, 24, 128))
    a[np.arange(72, 96), rng.integers(0, 128, 24)] = 0x7F
    C = Circuit()
    ins = [C.input(f"in{t}", 8) for t in range(2 * W.WINDOW)]
    C.output("out", getattr(B, name).build(C, *ins))
    assert C.counts()["ands"] == ands
    cols = np.concatenate([a, b], axis=1).T
    out, _ = C.evaluate({f"in{t}": [sum(((int(v) >> j) & 1) << i for i, v in enumerate(col)) for j in range(8)]
                         for t, col in enumerate(cols)}, len(a))
    got = [sum(((x >> i) & 1) << j for j, x in enumerate(out["out"])) for i in range(len(a))]
    want = [getattr(W, name).evaluate(*map(int, a[i]), *map(int, b[i])) for i in range(len(a))]
    bad = [i for i in range(len(a)) if got[i] != want[i]]
    assert not bad, bad[:5]
    words = [[(w >> 32 * m) & M32 for m in range(W.MASKS)] for w in want]
    assert any(0x7FFFFFFF in ws for ws in words) and any(0 in ws for ws in words[16:32])


@pytest.mark.parametrize("name, ands", [("PouwBlackwellE4m3SaltDead", 2_493), ("PouwHopperE4m3SaltDead", 1_806)])
def test_salt_dead_on_every_code_and_each_accumulators_edges(name, ands):
    """Every flat code (both signs of β, ±0, subnormals, ±448 and the NaNs) against FP32's edges and random accumulators of
    the forming's range, and every binade's ends against random codes.  The builder runs on a plain circuit, as C-Flock's
    lowering does, and has exactly the traced Definition's ANDs: the two ends share their 32 equal products and everything
    that doesn't read F's sign, where two atom calls and casts are 20,062 ANDs on sm_120 and 15,230 on the H100."""
    rng = random.Random(f"salt-dead-{name}")
    d, w = getattr(B, name), getattr(W, name)
    C = Circuit()
    C.output("out", d.build(C, C.input("acc", 32), C.input("cb", 8)))
    assert C.counts()["ands"] == ands
    accs = F32_EDGES + [int(np.float32(rng.uniform(-448, 448)).view(np.uint32)) for _ in range(30)]
    acc = [a for a in accs for _ in range(256)] + F32_BINADES
    cb = [c for _ in accs for c in range(256)] + [rng.getrandbits(8) for _ in F32_BINADES]
    got = evaluate_words(d, *[np.asarray(c, dtype=np.uint64).reshape(-1, 1) for c in (acc, cb)]).reshape(-1).tolist()
    bad = [i for i in range(len(acc)) if got[i] != w.evaluate(acc[i], cb[i])]
    assert not bad, [(hex(acc[i]), hex(cb[i])) for i in bad[:5]]
    assert 0 < sum(got) < len(got)


def test_window_word_on_every_mask():
    """Every mask of random, zero and all-ones windows; mask 15 is +0."""
    R = random.Random(20261009)
    xs = [0, (1 << 480) - 1] + [R.getrandbits(480) for _ in range(30)]
    _wide(B.PouwWindowWord, W.PouwWindowWord, [[x for x in xs for _ in range(16)], [m for _ in xs for m in range(16)]],
          [480, 4])


def test_bf16_neg_on_every_word():
    assert _words(B.PouwBf16Neg, W.PouwBf16Neg, np.arange(1 << 16)) == 1 << 16


def test_e_tilde_on_every_e2m1_code_and_ue4m3_byte():
    """Every E2M1 code (both zeros, code 0x8 −0) against every byte (both signs, the subnormals, the NaNs 0x7F and 0xFF)."""
    e, ae = np.meshgrid(np.arange(16), np.arange(256), indexing="ij")
    assert _words(B.PouwETilde, W.PouwETilde, e.ravel(), ae.ravel()) == 16 * 256


def test_sqrt_on_every_binades_ends_and_dense_random_words():
    """sqrt.rn.f32 on ±0, ±inf, NaNs, negative words, both ends of the subnormals and of every binade, and dense random words:
    2^17 over all words, 2^15 subnormals, 2^14 in each of the lowest and highest normal binades and 2^14 around 1."""
    rng = np.random.default_rng(20261003)
    x = np.concatenate([F32_EDGES, F32_BINADES, rng.integers(0, 1 << 32, 1 << 17), rng.integers(1, 1 << 23, 1 << 15),
                        rng.integers(0x800000, 0x1000000, 1 << 14), rng.integers(0x7F000000, 0x7F800000, 1 << 14),
                        rng.integers(0x3F000000, 0x40800000, 1 << 14)])
    _words(B.PouwF32SqrtRn, W.PouwF32SqrtRn, x)


def test_sq_ge_on_each_side_of_its_threshold():
    """1 iff up48(2^300 · x²) >= θ: for each x, θ at the value's encoding, one above and one below, 0 and every bit set, and
    random θ; x over the edges, every binade's ends and random words (the sign and exponent 255 decode as any other)."""
    rng = np.random.default_rng(20261004)
    x = np.concatenate([F32_EDGES, F32_BINADES, rng.integers(0, 1 << 32, 1 << 14), rng.integers(0, 1 << 23, 1 << 12)])
    v = np.array([W.up48(W.sq300(int(t))) for t in x], dtype=np.uint64)
    th = np.concatenate([v, v + np.uint64(1), np.where(v > 0, v - np.uint64(1), v), np.zeros_like(v),
                         np.full_like(v, (1 << 64) - 1), rng.integers(0, 1 << 63, len(x)).astype(np.uint64) << np.uint64(1)])
    _words(B.PouwSqGe, W.PouwSqGe, np.tile(x, 6), th)


def test_sq_acc_on_edges_wrapping_and_random_sums():
    """acc + 2^300 · x² mod 2^576: x over the edges and every binade's ends, acc 0, every bit set and one below a wrap; then
    random x against random accumulators of every length and accumulators near 2^576."""
    R = random.Random(20261005)
    xs = F32_EDGES + F32_BINADES[::7]
    accs = [R.choice([0, M576, (M576 - W.sq300(x) + 1) & M576, R.getrandbits(576)]) for x in xs]
    xs += [R.getrandbits(32) for _ in range(1500)]
    accs += [R.choice([R.getrandbits(576), R.getrandbits(R.randrange(1, 577)), M576 - R.getrandbits(560)]) for _ in range(1500)]
    _wide(B.PouwSqAcc, W.PouwSqAcc, [accs, xs], [576, 32])


@pytest.mark.slow
def test_sq_ceil_at_its_rounding_boundaries():
    """up48(⌈64 t / n⌉) (tracing its 582-step division takes 15 to 35 s): n = 0 (every bit set), 1, small, a power of two and
    2^32 − 1 against t at 0, 1, 2^576 − 1 and the sum bound; t where 64 t / n is exact or one past, where the ceiling has 48 or
    49 bits or rounds up into a new binade (m · 2^e with m = 2^48 − 1 plus one), and random t of every length."""
    R = random.Random(20261006)
    ts, ns = [], []
    for t in (0, 1, 2, 63, M576, (1 << 571) - 1, 1 << 300, (1 << 48) - 1, 1 << 48):
        for n in (0, 1, 2, 3, 64, 1000, 8192, 1 << 31, M32):
            ts.append(t)
            ns.append(n)
    for _ in range(600):
        n = R.choice([R.getrandbits(32) or 1, R.randrange(1, 1 << 14), R.randrange(1, 64)])
        top = ((1 << 48) - 1) << R.randrange(1, 500)
        q = R.choice([(1 << 48) - 1, 1 << 48, (1 << 48) + 1, top, top + 1, R.getrandbits(R.randrange(1, 570))])
        t = R.choice([-(-q * n // 64), q * n // 64, R.getrandbits(R.randrange(0, 577))])
        ts.append(t & M576)
        ns.append(n)
    from verity.ir.boolean import is_boolean

    assert is_boolean(B.PouwSqCeil.bind())
    _wide(B.PouwSqCeil, W.PouwSqCeil, [ts, ns], [576, 32])
