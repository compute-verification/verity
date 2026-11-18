"""PoUW's own primitives on bits (`verity.circuits.ops.boolean.words`) are their word primitives (`words`) bit for bit: on every input
where the input space allows (the bf16 negation, Ẽ over E2M1 × U8), else on each operand's edges (±0, the subnormals' ends,
every binade's ends, ±inf, NaNs, the rounding and threshold boundaries) and dense seeded random words."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.circuits.boolean import is_boolean
from verity.circuits.boolean.forms import Circuit, const_bits
from verity.circuits.boolean.trace import word_of
from verity.circuits.evaluation.bits import evaluate_bits, evaluate_words
from verity.circuits.ops.boolean import words as B
from verity.circuits.ops import words as W

M32, M576 = (1 << 32) - 1, (1 << 576) - 1
#: every FP32 class's ends: ±0, the subnormals', the normals', ±inf, quiet, signalling and all-ones NaNs
F32_EDGES = [0, 1, 2, 3, 0x7FFFFF, 0x800000, 0x800001, 0xFFFFFF, 0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x40000000, 0x7F7FFFFF,
             0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF]
F32_EDGES += [w | 0x80000000 for w in F32_EDGES]
#: each exponent field's smallest, next and largest mantissa, both signs
F32_BINADES = [s | e << 23 | m for s in (0, 1 << 31) for e in range(256) for m in (0, 1, 0x7FFFFF)]
#: the hiding tile's Keccak words (`hashes`), each with its word primitive
KECCAK = ((B.PouwXor32, W.PouwXor32), (B.PouwChi32, W.PouwChi32), (B.PouwRotl64Lo, W.PouwRotl64Lo),
          (B.PouwRotl64Hi, W.PouwRotl64Hi))
#: half-lanes whose bits take both values at every position: 0, all ones, alternating bits, each half, the end bits
U32_EDGES = [0, M32, 0x55555555, 0xAAAAAAAA, 0x0000FFFF, 0xFFFF0000, 1, 1 << 31, M32 ^ 1, M32 ^ 1 << 31]


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
    for d, w in ((B.PouwF32SqrtRn, W.PouwF32SqrtRn), (B.PouwETilde, W.PouwETilde), (B.PouwBf16Neg, W.PouwBf16Neg),
                 (B.PouwSqAcc, W.PouwSqAcc), (B.PouwSqCeil, W.PouwSqCeil), (B.PouwSqGe, W.PouwSqGe),
                 (B.PouwBlackwellE4m3Window, W.PouwBlackwellE4m3Window), (B.PouwHopperE4m3Window, W.PouwHopperE4m3Window),
                 (B.PouwWindowWord, W.PouwWindowWord), (B.PouwBlackwellE4m3SaltDead, W.PouwBlackwellE4m3SaltDead),
                 (B.PouwHopperE4m3SaltDead, W.PouwHopperE4m3SaltDead)) + KECCAK:
        assert word_of(d) is w and d.id == w.id.replace("_v1", "_v2")
        assert d in (B.PouwSqCeil, B.PouwBlackwellE4m3Window, B.PouwHopperE4m3Window) or is_boolean(d.bind())


@pytest.mark.slow
def test_the_windows_are_boolean():
    """Each window's body, its builder traced and emitted (4 to 7 s each), is Boolean."""
    assert is_boolean(B.PouwBlackwellE4m3Window.bind()) and is_boolean(B.PouwHopperE4m3Window.bind())


#: E4M3's ends: ±0, the subnormals', the normals', ±1, the largest finite, and the two NaNs
E4M3_EDGES = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x7E, 0xFE, 0x7F, 0xFF]


@pytest.mark.parametrize("name, ands", [("PouwBlackwellE4m3Window", 102_409), ("PouwHopperE4m3Window", 67_623)])
def test_window_on_finite_cancelling_edge_and_nan_codes(name, ands):
    """A window's masked words on random finite codes, products that cancel pairwise (+0 words, and totals of either sign
    near zero), small codes (subnormal sums), E4M3's ends (largest products, both zeros, NaNs) and finite windows with one
    NaN code (NaN in exactly the words whose atoms read it).  The bit form's builder runs on a plain circuit, as Flock's
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
    the forming's range, and every binade's ends against random codes.  The builder runs on a plain circuit, as Flock's
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

    assert is_boolean(B.PouwSqCeil.bind())
    _wide(B.PouwSqCeil, W.PouwSqCeil, [ts, ns], [576, 32])


def test_xor_and_chi_on_every_edge_triple_and_random_words():
    """a xor b and Keccak's χ, a xor (not b and c), on every pair and triple of edge half-lanes (so every bit position takes
    each of χ's eight cases) and on 2^14 random triples."""
    rng = np.random.default_rng(20261007)
    e = np.array(U32_EDGES, dtype=np.uint64)
    a, b, c = (np.concatenate([g.ravel(), rng.integers(0, 1 << 32, 1 << 14, dtype=np.uint64)])
               for g in np.meshgrid(e, e, e, indexing="ij"))
    assert _words(B.PouwXor32, W.PouwXor32, a, b) == _words(B.PouwChi32, W.PouwChi32, a, b, c) == len(e) ** 3 + (1 << 14)


def test_rotations_by_every_amount():
    """Both halves of hi:lo rotated left by c mod 64: every c below 128 (the amount wraps at 64) and c with bits above its
    low six set, on every single-bit word (so each bit lands where the rotation sends it), edge pairs and random pairs."""
    R = random.Random(20261008)
    pairs = [((1 << j) & M32, (1 << j) >> 32) for j in range(64)]
    pairs += [(x, y) for x in U32_EDGES[:4] for y in U32_EDGES[:4]] + [(R.getrandbits(32), R.getrandbits(32)) for _ in range(32)]
    amounts = list(range(128)) + [R.getrandbits(32) for _ in range(16)] + [M32, 1 << 31 | 5]
    lo, hi, c = zip(*[(x, y, s) for x, y in pairs for s in amounts])
    for d, w in KECCAK[2:]:
        assert _words(d, w, lo, hi, c) == len(pairs) * len(amounts)


def test_a_constant_rotation_is_wiring():
    """Every rotation `hashes` builds has a constant amount, and on one the rotator folds to wiring: no AND is built, and each
    output bit is the input bit the rotation moves there."""
    for s in list(range(64)) + [64, 65, M32]:
        C = Circuit(cse=True)
        lo, hi = C.input("lo", 32), C.input("hi", 32)
        x = lo + hi
        out = B.rotl64(C, lo, hi, const_bits(s, 32))
        assert "and" not in C.kind
        assert [w.key() for w in out] == [x[(i - s) % 64].key() for i in range(64)]


def test_chi_commits_its_output_in_forms_and_traces_no_gate_for_it():
    """In `forms` (Flock's pieces run ``.build``) each χ output bit is a committed bit of its own, one row more than its AND
    (64 for a half-lane); the Definition's gates are its 32 Ands, and its pin counts those."""
    C = Circuit()
    a, b, c = (C.input(n, 32) for n in "abc")
    out = B.PouwChi32.build(C, a, b, c)
    C.output("o", out)
    assert C.counts()["ands"] == 64 and all(len(x.s) == 1 and x.c == 0 for x in out)
    assert B.PouwChi32.traced().counts == {"and": 32, "xor": 32, "not": 32}


def _rows(d, ref, rows):
    """d on each row of flat word arguments (an array parameter's words in a row) equals its word primitive."""
    params, _ = ref.signature()
    n = [t.n if hasattr(t, "n") else 1 for _, t in params]
    cut = np.cumsum([0] + n)
    arrs = [np.asarray([r[cut[i]:cut[i + 1]] for r in rows], dtype=np.uint64) for i in range(len(n))]
    got = evaluate_words(d, *arrs).reshape(-1).tolist()
    want = [ref.evaluate(*r) for r in rows]
    bad = [i for i in range(len(rows)) if got[i] != want[i]]
    assert not bad, [(rows[i], hex(got[i]), hex(want[i])) for i in bad[:3]]


#: the bodies of `words`' bytes, int7 entries, accumulator, SHA-512 words and dequantization, and each one's ANDs
MECHANICAL = {"PouwPackBE4": 0, "PouwNoiseByte": 23, "PouwDot16U8S8": 2_079, "PouwFormX": 7, "PouwFormX3": 20, "PouwAdd8": 7,
              "PouwSub8": 7, "PouwAddSub8": 14, "PouwNegBetaSum16": 153, "PouwAdd64Lo": 31, "PouwAdd64Hi": 63, "PouwCh32": 32,
              "PouwMaj32": 32, "PouwI32ToF32Rn": 368,
              **{f"PouwSha512{n}{h}": 0 for n in ("BigSigma0", "BigSigma1", "SmallSigma0", "SmallSigma1") for h in ("Hi", "Lo")}}
I7_ALL, U8_ALL = range(128), range(256)
#: int7's and s8's ends: 0, ±1, the largest and smallest
I7_EDGES, U8_EDGES = [0, 1, 0x3F, 0x40, 0x7F], [0, 1, 0x7F, 0x80, 0xFF]


@pytest.mark.parametrize("name", sorted(MECHANICAL))
def test_each_is_the_next_version_of_its_word_found_by_the_lookup_with_its_ands(name):
    from verity.circuits.ops.boolean import versions as V
    d, w = getattr(B, name), getattr(W, name)
    assert word_of(d) is w and d.id == w.id.replace("_v1", "_v2") and V.boolean_version(w).id == d.id
    assert is_boolean(d.bind()) and d.traced().counts["and"] == MECHANICAL[name]


@pytest.mark.parametrize("name", ["PouwFormX", "PouwAdd8", "PouwSub8"])
def test_the_two_operand_bytes_on_every_input(name):
    _rows(getattr(B, name), getattr(W, name), [(a, e) for a in I7_ALL for e in U8_ALL])


@pytest.mark.parametrize("name", ["PouwFormX3", "PouwAddSub8"])
def test_the_three_operand_bytes_on_every_int7_and_edge_bytes_and_random(name):
    R = random.Random(20261118)
    rows = [(a, e, f) for a in I7_ALL for e in U8_EDGES for f in U8_EDGES]
    rows += [(R.randrange(128), R.randrange(256), R.randrange(256)) for _ in range(4000)]
    _rows(getattr(B, name), getattr(W, name), rows)


def test_pack_on_edge_entries_and_random():
    R = random.Random(1)
    rows = [(a, b, c, d) for a in I7_EDGES for b in I7_EDGES for c in I7_EDGES for d in I7_EDGES]
    _rows(B.PouwPackBE4, W.PouwPackBE4, rows + [tuple(R.randrange(128) for _ in range(4)) for _ in range(2000)])


def test_noise_byte_on_every_low_six_bits_of_each_byte_and_random():
    """Each byte position, every value of its low six bits under random other bits, and c's high bits ignored."""
    R = random.Random(2)
    rows = [((v | R.getrandbits(2) << 6) << 8 * c | R.getrandbits(32) & ~(0xFF << 8 * c), c | R.getrandbits(30) << 2)
            for c in range(4) for v in range(64)]
    _rows(B.PouwNoiseByte, W.PouwNoiseByte, rows + [(R.getrandbits(32), R.getrandbits(32)) for _ in range(2000)])


def test_dot_on_extreme_and_random_products_and_wrapping_accumulators():
    """The largest products of either sign (u8 255 by s8 −128 and 127), zero, accumulators at the wrap, and random."""
    R = random.Random(3)
    rows = [(acc, *[x] * 16, *[y] * 16) for acc in (0, 1, M32, 1 << 31, (1 << 31) - 1) for x in U8_EDGES for y in U8_EDGES]
    rows += [(R.getrandbits(32), *[R.randrange(256) for _ in range(32)]) for _ in range(2000)]
    _rows(B.PouwDot16U8S8, W.PouwDot16U8S8, rows)


def test_neg_beta_sum_on_extreme_and_random_entries():
    R = random.Random(4)
    rows = [(acc, *[y] * 16) for acc in (0, 1, M32, 1 << 31) for y in U8_EDGES]
    rows += [(R.getrandbits(32), *[R.randrange(256) for _ in range(16)]) for _ in range(2000)]
    _rows(B.PouwNegBetaSum16, W.PouwNegBetaSum16, rows)


def test_the_64_bit_add_on_every_carry_edge_and_random():
    """Low halves that carry and don't, high halves that wrap, and random."""
    R = random.Random(5)
    rows = [(ah, al, bh, bl) for ah in U32_EDGES[:4] for al in U32_EDGES for bh in U32_EDGES[:4] for bl in U32_EDGES]
    rows += [tuple(R.getrandbits(32) for _ in range(4)) for _ in range(2000)]
    _rows(B.PouwAdd64Hi, W.PouwAdd64Hi, rows)
    _rows(B.PouwAdd64Lo, W.PouwAdd64Lo, [(al, bl) for _ah, al, _bh, bl in rows])


@pytest.mark.parametrize("name", ["PouwCh32", "PouwMaj32"])
def test_choose_and_majority_on_every_edge_triple_and_random(name):
    R = random.Random(6)
    rows = [(a, b, c) for a in U32_EDGES for b in U32_EDGES for c in U32_EDGES]
    _rows(getattr(B, name), getattr(W, name), rows + [tuple(R.getrandbits(32) for _ in range(3)) for _ in range(2000)])


@pytest.mark.parametrize("name", [f"PouwSha512{n}{h}" for n in ("BigSigma0", "BigSigma1", "SmallSigma0", "SmallSigma1")
                                  for h in ("Hi", "Lo")])
def test_each_sigma_half_on_every_single_bit_word_and_random(name):
    """Every one-bit word (so each input bit lands where its rotations and shift send it), edges and random words."""
    R = random.Random(7)
    rows = [(x >> 32, x & M32) for x in [1 << i for i in range(64)] + [0, (1 << 64) - 1]]
    _rows(getattr(B, name), getattr(W, name), rows + [(R.getrandbits(32), R.getrandbits(32)) for _ in range(2000)])


def test_the_i32_conversion_on_every_binades_ends_ties_and_random():
    """Zero, ±1, each power of two and its neighbours, the halfway points past 2^24 (ties to even both ways), the most
    negative word, and random words."""
    R = random.Random(8)
    xs = {0, 1, M32, 1 << 31, (1 << 31) - 1}
    for e in range(31):
        for v in ((1 << e) - 1, 1 << e, (1 << e) + 1):
            xs |= {v & M32, -v & M32}
    for e in range(25, 31):
        for q in (0x800000, 0x800001, 0xFFFFFF):
            for r in (-1, 0, 1):
                v = (q << (e - 23)) + (1 << (e - 24)) + r
                xs |= {v & M32, -v & M32}
    _rows(B.PouwI32ToF32Rn, W.PouwI32ToF32Rn, [(x,) for x in sorted(xs)] + [(R.getrandbits(32),) for _ in range(4000)])
