"""Pearl-C's tile check as a Program (`verity.ml.protocols.pouw.circuit.pc8`): its primitives are the scheme's operations on the domain
the row's admission bit admits, its units evaluate to `schemes.pearl_c`'s forming v1, clean-up and checked words (the pinned
``test-16-v1`` vector on the H100, the scheme itself on both sm_120 records), its admission bit is the scheme's on every row
(finiteness, the noise floor and `live_row`, each at its boundary), and its partition computes every value in exactly one
unit; its hiding Programs publish only admission bits and tile digests, their units' ports are `PORTS`, and the verifier
derives the anchored inputs (`anchors.TileAnchors`).  A short last tile row (m = 80, 112, and decode's 32 at the scheme's
tile, the pinned decode-32 case among them) is the scheme's too, the row cut refuses it, and whole-tile Programs and layouts
are pinned."""
from __future__ import annotations

import hashlib
import json
import random
import struct
from collections import Counter
from dataclasses import replace

import pytest

import circuit_fixtures as CF
import pearl_c_vectors as V
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.program import Program
from verity.ml.circuits.ops.fp32 import F32FmaV2
from verity.ml.circuits.ops.prims import (BlackwellE4m3QmmaDot32, E4m3ToF32, F2fpBf16, F32ToE4m3Sat, HopperE4m3QgmmaDot32)
from verity.ml.circuits.ops.scalar import I32Le
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32, HOPPER_E4M3_K32
from verity.ml.commitments.rowleaf import sha512_row_seg_digest
from verity.ml.randomness import derive
from verity.ml.protocols.pouw.circuit import pc8, plan
from verity.ml.protocols.pouw.circuit.anchors import TileAnchors
from verity.ml.protocols.pouw.circuit import words as W
from verity.core.protocols.pouw.protocol import DomainError, Tile
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_kw as P
from verity.ml.protocols.pouw.schemes.pearl_c_device import DEVICES, SM120_UNPROMOTED

K = 128
CASE = next(c for c in json.loads(V.OUT.read_text())["cases"] if c["case"] == "test-16-v1")
SEED = bytes(range(32))
MIXED_ZERO = {0: 0x80000000, 16: 0, 3: 0x80000000, 5: 0, 9: 0x80000000}


def _flat(lines) -> list[int]:
    return [c for ln in lines for c in ln]


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _finite(rng: random.Random, count: int) -> list[int]:
    out = []
    while len(out) < count:
        w = rng.getrandbits(32)
        if (w >> 23) & 0xFF != 0xFF:
            out.append(w)
    return out


def _codes(rng: random.Random, count: int) -> list[int]:
    return [c for c in (rng.getrandbits(8) for _ in range(4 * count)) if c & 0x7F != 0x7F][:count]


# ---- the primitives and the float forms -------------------------------------------------------------------------------

@pytest.mark.parametrize("prim, model", [(HopperE4m3QgmmaDot32, HOPPER_E4M3_K32),
                                         (BlackwellE4m3QmmaDot32, BLACKWELL_SM120_E4M3_M16N8K32)])
def test_core_atoms_are_the_devices_atoms(prim, model):
    """The forming atom's accumulator is α·x (|α·x| <= 448), the chain's a promoted window's or the running total."""
    rng = random.Random(1)
    accs = [0, 0x80000000] + [_bits(rng.uniform(-448, 448)) for _ in range(300)] + [_bits(rng.uniform(-1e9, 1e9))
                                                                                      for _ in range(100)]
    for acc in accs:
        a, b = _codes(rng, 32), _codes(rng, 32)
        assert evaluate(prim, acc, a, b) == (model.step(acc, a, b),)


def test_the_casts_are_pearls_on_finite_words():
    """`F32ToE4m3Sat` is Pearl's nearest-even saturating `f32_to_fp8` (subnormal and saturating ranges and every midpoint),
    `E4m3ToF32` its exact decode, and `F2fpBf16` its RNE to BF16, exact on an E4M3 value."""
    rng = random.Random(2)
    mids = [_bits((P.fp8_to_f32(c) + P.fp8_to_f32(c + 1)) / 2) for c in range(0x7E)]
    words = _finite(rng, 4000) + [w + d for w in mids for d in (-1, 0, 1)] + [_bits(v) for v in (448.0, 464.0, 1e30)]
    for w in words + [x | 0x80000000 for x in words]:
        if (w >> 23) & 0xFF != 0xFF:
            assert F32ToE4m3Sat.evaluate(w) == P.f32_to_fp8(P.f32_of_bits(w)), hex(w)
    for c in range(256):
        if c & 0x7F != 0x7F:
            assert E4m3ToF32.evaluate(c) == _bits(P.fp8_to_f32(c))
            assert F2fpBf16.evaluate(E4m3ToF32.evaluate(c)) << 16 == E4m3ToF32.evaluate(c) == _bits(P.fp8_to_f32(c))
            assert F2fpBf16.evaluate(E4m3ToF32.evaluate(c)) == C._bf16_of_code(c)
    for w in _finite(rng, 4000):
        if abs(P.f32_of_bits(w)) < 3.3e38:
            assert F2fpBf16.evaluate(w) == P.f32_to_bf16(P.f32_of_bits(w)), hex(w)


def test_the_exact_squares_decide_live_rows_dead_entries():
    """2^300·x² is `live_row`'s scaled square, and `PouwSqGe(x, PouwSqCeil(t, n))` is n·2^300·x² >= 64·t at every t
    around the boundary, for depths whose k/8 is and is not a power of two."""
    rng = random.Random(3)
    xs = _finite(rng, 600) + [0, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, _bits(8.0), _bits(2.0 ** -140)]
    for x in xs:
        v = P.f32_of_bits(x)
        assert W.sq300(x) == int(v * v * 2.0 ** 300)
        acc = W.PouwSqAcc.evaluate(0, x)
        assert acc == W.sq300(x) and W.PouwSqAcc.evaluate(acc, x ^ 0x80000000) == 2 * W.sq300(x)
    for x in xs:
        s = W.sq300(x)
        for n in (12, 16, 144, 1000, 8192):
            exact = s * n // 64
            for t in {0, 1, exact - 1, exact, exact + 1, exact + 2, rng.randrange(max(1, 2 * exact + 2))}:
                if t >= 0:
                    assert W.PouwSqGe.evaluate(x, W.PouwSqCeil.evaluate(t, n)) == (n * s >= 64 * t), (hex(x), n, t)


def test_the_floor_is_the_sign_of_one_fma():
    """fma(ρ, α, −1) >= +0 iff the exact ρ·α >= 1, at and beside ρ·α = 1 and far from it."""
    rng = random.Random(4)
    for _ in range(3000):
        rho = _bits(rng.uniform(1e-6, 1e6))
        inv = P.f32_of_bits(_bits(1 / P.f32_of_bits(rho)))
        for alpha in (_bits(inv) + d for d in (-2, -1, 0, 1, 2)):
            want = P.f32_of_bits(rho) * P.f32_of_bits(alpha) >= P.SIGMA_MIN
            assert I32Le.evaluate(0, F32FmaV2.evaluate(rho, alpha, pc8.MINUS_ONE)) == want
    assert I32Le.evaluate(0, F32FmaV2.evaluate(_bits(0.5), _bits(2.0), pc8.MINUS_ONE)) == 1
    assert I32Le.evaluate(0, F32FmaV2.evaluate(0, _bits(448 * 2.0 ** 32), pc8.MINUS_ONE)) == 0


# ---- the units against the scheme -------------------------------------------------------------------------------------

def _case(case: str, shape: tuple[int, int, int], dev: str, index: int = 3, weight: int = 9, tile=(16, 16),
          hashing: str = "h0"):
    s = C.PearlC(k_min=32, tile=tile, forming="v1", device=DEVICES[dev], hashing=hashing)
    ctx = V.ctx_of(case, shape, index, weight, hashing == "h3")
    m, k, n = shape
    a, b = V.rows_f32(case, "A", m, k), V.rows_f32(case, "B", n, k)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, Tile(tuple(range(m)), tuple(range(n))))
    fa, fb = C._basis(noise.seed_b, 0, k, C.LINE_NORM_V1), C._basis(noise.seed_b, 1, k, C.LINE_NORM_V1)
    g = _flat(C.gram(noise.seed_b, k, C.LINE_NORM_V1, s.device))
    words = lambda rows: [w for r in rows for w in C.parse_f32_row(r, k)]  # noqa: E731
    return s, ctx, ops, words(a), _flat(ops.e_a), words(b), _flat(ops.e_b), _flat(fa), _flat(fb), g


def _evaluate(shape: pc8.Shape, a, ea, b, eb, fa, fb, g):
    m, k, n = shape.m, shape.k, shape.n
    wout = evaluate(pc8.weight(shape), b, eb, fa, fb, g)
    bc, pb, ok_b = list(wout[:n * k]), list(wout[n * k:-n]), list(wout[-n:])
    cout = evaluate(pc8.call(shape), a, ea, fa, fb, bc, pb)
    assert evaluate(pc8.check(shape), a, ea, b, eb, fa, fb, g) == tuple(cout) + tuple(ok_b)
    mn = m * n
    c, u = [cout[i * n:(i + 1) * n] for i in range(m)], [cout[mn + i * n:mn + (i + 1) * n] for i in range(m)]
    row_a = pc8.templates(k, shape.tile, shape.dev)["row_a"]
    outs = [evaluate(row_a, a[i * k:(i + 1) * k], ea[32 * i:32 * (i + 1)], fa, fb) for i in range(m)]
    return {"b_codes": bc, "pb": [pb[64 * j:64 * (j + 1)] for j in range(n)], "c": c, "u": u, "ok_a": list(cout[2 * mn:]),
            "ok_b": ok_b, "a_codes": [x for o in outs for x in o[:k]], "pa": [list(o[k:-1]) for o in outs]}


def test_the_programs_evaluate_to_the_pinned_vector():
    """On the H100 record, `Pc8Weight` gives B̃ and P_B from the anchored G, `Pc8Call` reads them and gives the vector's C̃
    and U, `Pc8Check` is the two in one root, and the A rows' codes and P_A are the vector's; every row is admitted.  The
    16 × 16 block is cut into 4 × 8 tiles, so a word in the wrong tile or transposed within one changes the digests."""
    m, k, n = CASE["shape"]
    _s, _ctx, ops, *inputs = _case(CASE["case"], (m, k, n), "h100", CASE["index"], CASE["weight"])
    got = _evaluate(pc8.Shape(m, k, n, tile=(4, 8), dev="h100"), *inputs)
    assert V.sha(bytes(got["b_codes"])) == CASE["b_codes_sha256"] and V.sha(V.halves(got["pb"])) == CASE["pb_sha256"]
    assert V.sha(V.words(got["c"])) == CASE["c_sha256"] and V.sha(V.words(got["u"])) == CASE["u_sha256"]
    assert V.sha(bytes(got["a_codes"])) == CASE["a_codes_sha256"] and V.sha(V.halves(got["pa"])) == CASE["pa_sha256"]
    assert got["ok_a"] == [1] * m and got["ok_b"] == [1] * n and got["pa"][0][:8] == CASE["pa_0"]


@pytest.mark.parametrize("dev, k", [("sm120", 128), ("sm120-unpromoted", 160)])
def test_the_programs_are_the_scheme_on_sm120(dev, k):
    """On sm_120's records (promotion every 4 atoms, and none, where 32 | k and k/8 = 20 is not a power of two) the Programs
    give the scheme's own B̃, P_B, A′, P_A, C̃ and U."""
    s, ctx, ops, *inputs = _case(f"pc8-{dev}", (16, k, 16), dev)
    assert s.admissible(ctx, ops)
    checked = s.checked(ctx, ops)
    got = _evaluate(pc8.Shape(16, k, 16, tile=(8, 16), dev=dev), *inputs)
    assert got["b_codes"] == _flat(ops.b.codes) and got["pb"] == ops.pb
    assert got["a_codes"] == _flat(ops.a.codes) and got["pa"] == ops.pa
    assert [list(r) for r in got["c"]] == checked.c and [list(r) for r in got["u"]] == checked.u
    assert got["ok_a"] == [1] * 16 and got["ok_b"] == [1] * 16


def _scheme_row(words: list[int], side: str, dev: str):
    """The scheme on one row: its codes and peel row, or None where it refuses the row (encode's rejections, then
    `admissible`'s floor and liveness)."""
    d, fa, fb = DEVICES[dev], C._basis(SEED, 0, K, C.LINE_NORM_V1), C._basis(SEED, 1, K, C.LINE_NORM_V1)
    e = [P.sample_line(SEED, 0 if side == "row_a" else 1, 0, 5)]
    try:
        x = C.parse_f32_row(C.f32_row(words), K)
        formed = C.form_v1([x], e, fa if side == "row_a" else fb, d)
        be = C.scaled_noise(formed, e)
        peel = (C.peel_factors_a if side == "row_a" else C.peel_factors_b)(formed.codes, be, SEED, K, C.LINE_NORM_V1, d)
    except P.Reject:
        return None
    if not (P.f32_of_bits(formed.rho[0]) * P.f32_of_bits(formed.alpha[0]) >= P.SIGMA_MIN and C.live_row(x)):
        return None
    return formed.codes[0], peel[0]


def _flip(words: list[int], at: int, admitted) -> tuple[list[int], list[int]]:
    """Two rows that differ only at position ``at`` in adjacent positive FP32 words, the first ``admitted`` and the
    second not: bisection on the word between 1.0 and 2^100."""
    lo, hi = _bits(1.0), _bits(2.0 ** 100)
    row = lambda w: [w if t == at else v for t, v in enumerate(words)]  # noqa: E731
    assert admitted(row(lo)) and not admitted(row(hi))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if admitted(row(mid)) else (lo, mid)
    return row(lo), row(hi)


def _rows(dev: str) -> dict[str, list[int]]:
    """Honest rows and rows the scheme refuses: non-finite words at sampled (0, 8, …) and unsampled positions, an overflowing
    sum of squares (ρ not finite: `pc8Sem_squaresOverflow`), a zero row, an all-(−0) row, a subnormal row, the largest word, and both sides of the noise floor and
    of liveness (one word bisected to its flip; and a row whose third dead entry meets the threshold with equality); and an
    honest row with +0 and −0 words at sampled and unsampled positions."""
    base = [C.parse_f32_row(r, K) for r in V.rows_f32("pc8-circuit", "A", 3, K)]
    at = lambda w, updates: [updates.get(t, v) for t, v in enumerate(w)]  # noqa: E731
    ok = lambda w: _scheme_row(w, "row_a", dev) is not None  # noqa: E731
    floor_in, floor_out = _flip(base[0], 1, ok)
    ones = [_bits(1.0)] * K
    two_dead = at(ones, {1: _bits(9.0), 2: _bits(9.0)})
    live_in, live_out = _flip(two_dead, 3, ok)
    return {"honest-0": base[0], "honest-1": base[1], "negated": [w ^ 0x80000000 for w in base[2]],
            "inf-unsampled": at(base[0], {1: 0x7F800000}), "minus-inf-sampled": at(base[0], {0: 0xFF800000}),
            "nan-unsampled": at(base[0], {3: 0x7FC00000}), "nan-sampled": at(base[0], {8: 0xFFC00001}),
            "squares-overflow": at(base[0], {p: _bits(2.0 ** 126) for p in range(0, K, 8)}), "zero": [0] * K,
            "minus-zero": [0x80000000] * K, "mixed-zero": at(base[0], MIXED_ZERO), "subnormal": [1] * K, "max-unsampled": at(base[0], {1: 0x7F7FFFFF}), "floor-in": floor_in, "floor-out": floor_out,
            "live-in": live_in, "live-out": live_out, "live-equal": at(two_dead, {3: _bits(8.0)}),
            "live-below": at(two_dead, {3: _bits(8.0) - 1})}


@pytest.mark.parametrize("side, dev", [("row_a", "h100"), ("row_b", "h100"), ("row_a", "sm120")])
def test_the_row_units_admission_is_the_schemes_and_their_outputs_the_schemes(side, dev):
    """``ok`` is the scheme's admission on every row, and on an admitted row the unit's codes and peel row are `form_v1`'s
    and `peel_factors_a` / `_b`'s.  Nothing is claimed about a refused row's other outputs: the scheme refuses it."""
    fa, fb = C._basis(SEED, 0, K, C.LINE_NORM_V1), C._basis(SEED, 1, K, C.LINE_NORM_V1)
    own, other = (fa, fb) if side == "row_a" else (fb, fa)
    extra = [] if side == "row_a" else [_flat(C.gram(SEED, K, C.LINE_NORM_V1, DEVICES[dev]))]
    fn = pc8.templates(K, pc8.TILE, dev)[side]
    e = P.sample_line(SEED, 0 if side == "row_a" else 1, 0, 5)
    admitted = set()
    for name, words in _rows(dev).items():
        out = evaluate(fn, words, e, _flat(own), _flat(other), *extra)
        want = _scheme_row(words, side, dev)
        assert out[-1] == (want is not None), name
        if want is not None:
            admitted.add(name)
            assert list(out[:K]) == want[0] and list(out[K:-1]) == want[1], name
    assert admitted == {"honest-0", "honest-1", "negated", "mixed-zero", "floor-in", "live-in", "live-below"}


@pytest.mark.parametrize("side", ["row_a", "row_b"])
def test_a_row_unit_reads_minus_zero_as_zero(side):
    """A row unit's outputs on a row with −0 words are its outputs with +0 in their place, as the Lean's `pc8Sem` reads
    a row by value, and the scheme's (the admission test above)."""
    fa, fb = C._basis(SEED, 0, K, C.LINE_NORM_V1), C._basis(SEED, 1, K, C.LINE_NORM_V1)
    own, other = (fa, fb) if side == "row_a" else (fb, fa)
    extra = [] if side == "row_a" else [_flat(C.gram(SEED, K, C.LINE_NORM_V1, DEVICES["h100"]))]
    fn, e = pc8.templates(K, pc8.TILE, "h100")[side], P.sample_line(SEED, 0 if side == "row_a" else 1, 0, 5)
    run = lambda w: evaluate(fn, w, e, _flat(own), _flat(other), *extra)  # noqa: E731
    rows = _rows("h100")
    plus = lambda w: [0 if x == 0x80000000 else x for x in w]  # noqa: E731
    assert run(rows["mixed-zero"]) == run(plus(rows["mixed-zero"])) and run(rows["mixed-zero"])[-1] == 1
    assert run(rows["minus-zero"]) == run(rows["zero"]) and run(rows["minus-zero"])[-1] == 0


# ---- the partition ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hidden", [False, True], ids=["plain", "hidden"])
@pytest.mark.parametrize("side", ["check", "weight", "call"])
def test_the_partition_computes_every_value_in_exactly_one_unit(side, hidden):
    """One unit per row of B and of A, and per tile (G is anchored, so no unit computes it); `validate_unit_cut` certifies
    every gate once with no recompute, and the partition object verifies; so do the hiding Programs' (`pc4`'s test)."""
    shape = pc8.Shape(4, 128, 6, tile=(2, 3), dev="sm120", hidden=hidden)
    fn = {"check": pc8.check, "weight": pc8.weight, "call": pc8.call}[side](shape)
    prog, q = Program(fn), pc8.query(shape, side)
    us = pc8.units(fn)
    full = {"row_b": shape.n, "row_a": shape.m, "tile": 2 * 2}
    assert Counter(u.role for u in us) == {r: full[r] for r in pc8.ROLES[side]}
    part = PO.evaluate(prog, q)
    tp = pc8.templates(shape.k, shape.tile, shape.dev, hidden)
    assert [t for _k, t, _a, n in part.ranges for _ in range(n)] == [PO.template_id(tp[u.role]) for u in us]
    assert [u.template for u in us] == [PO.template_id(tp[u.role]) for u in us]
    assert PO.verify(PO.build(prog, q), prog).ok
    _g, _owner, verdict = pc8.unit_cut(fn)
    assert verdict.ok, verdict.codes
    assert verdict.detail["certified"] == verdict.detail["computed_gates"]


# ---- the hiding Programs ----------------------------------------------------------------------------------------------

HIDDEN = pc8.Shape(4, 128, 6, tile=(2, 3), dev="sm120", hidden=True)


def test_the_hiding_programs_publish_only_admission_bits_and_tile_digests():
    """`pc4`'s hiding test on Pearl-C's: the weight's Program is `Pc8Weight`, and `Pc8CallHidden` and `Pc8CheckHidden`
    return each tile's digest of the plain call's C̃ and U and the admission bits."""
    m, k, n = HIDDEN.m, HIDDEN.k, HIDDEN.n
    _s, _ctx, _ops, a, ea, b, eb, fa, fb, g = _case("pc8-hidden", (m, k, n), HIDDEN.dev)
    plain = replace(HIDDEN, hidden=False)
    assert pc8.weight(HIDDEN).id == pc8.weight(plain).id
    wout = evaluate(pc8.weight(plain), b, eb, fa, fb, g)
    bc, pb, ok_b = wout[:n * k], wout[n * k:-n], wout[-n:]
    cout = evaluate(pc8.call(plain), a, ea, fa, fb, bc, pb)
    c, u = ([cout[f * m * n + i * n:f * m * n + (i + 1) * n] for i in range(m)] for f in range(2))
    want = CF.tile_digests(HIDDEN.tile, c, u) + tuple(cout[2 * m * n:])
    assert evaluate(pc8.call(HIDDEN), a, ea, fa, fb, bc, pb) == want
    assert evaluate(pc8.check(HIDDEN), a, ea, b, eb, fa, fb, g) == want + tuple(ok_b)


@pytest.mark.parametrize("role", ["row_b", "row_a", "tile"])
def test_the_ports_are_each_hiding_units_committed_rows(role):
    """`PORTS` lists every parameter and output of each hiding unit once: fp32 input rows, fp8-p rows, ok rows and the tile
    digest, and the anchored inputs as public input regions."""
    tm, tn = HIDDEN.tile
    CF.check_ports(pc8.templates(K, HIDDEN.tile, HIDDEN.dev, True)[role], pc8.PORTS[role], pc8.DERIVATIONS, K,
                   {"ac": tm, "pa": tm, "bc": tn, "pb": tn})


def test_the_port_check_holds_advice_to_unread_parameters():
    """Advice is a port's third class (`PORTS`): a row's port also named advice is listed twice, and an output named
    advice in place of its row is no parameter, so the port check fails on both."""
    tm, tn = HIDDEN.tile
    tile, ports = pc8.templates(K, HIDDEN.tile, HIDDEN.dev, True)["tile"], pc8.PORTS["tile"]
    reads = {"ac": tm, "pa": tm, "bc": tn, "pb": tn}
    for bad in ({**ports, "advice": ("pa",)}, {**ports, "rows": ports["rows"][:-1], "advice": (0,)}):
        with pytest.raises(AssertionError):
            CF.check_ports(tile, bad, pc8.DERIVATIONS, K, reads)


def _row_seg_arguments(row: bytes, n_bits: int) -> list[int]:
    """`Sha512RowSeg_v1{N_BITS}`'s argument, the row's bits in leaf order: bit i is bit i % 8 of byte i // 8 (#1438's
    `verity.ml.commitments.gates.rowleaf.arguments`, which is not on main yet)."""
    return [(row[i >> 3] >> (i & 7)) & 1 for i in range(n_bits)]


def _element_bits(words, width: int) -> list[int]:
    """Words as a Boolean lowering carries them (`boolean.bit_type`): each its ``width`` bits, least significant first."""
    return [(w >> j) & 1 for w in words for j in range(width)]


@pytest.mark.parametrize("shape, tile", [((4, K, 6), (2, 3)), ((16, K, 64), (16, 64)),
                                         pytest.param((64, K, 64), (64, 64), marks=pytest.mark.slow)],
                         ids=["2x3", "16x64", "64x64"])
def test_the_h3_hiding_tile_returns_the_row_its_leaf_hashes(shape, tile):
    """`Pc8TileHiddenH3` on each of the -h3 scheme's tiles (its A′, P_A, B̃ and P_B) returns that tile's messages: its
    words' bits, as a Boolean lowering carries them, are `Sha512RowSeg_v1`'s argument over ``b"".join(messages)``, whose
    `sha512_row_seg_digest` is the scheme's -h3 leaf.  At 16 x 64 each row is two 32-column chunks, so a word in the wrong
    chunk or field moves; 64 x 64 is the scheme's tile, 262,144 bits in 32 segments.  `Sha512RowSeg_v1`'s gates over this value are #1438's; the composed test lands with it."""
    m, k, n = shape
    s, ctx, *_ = _case("pc8-h3-tile", shape, "sm120", tile=tile, hashing="h3")
    noise = s.noise(ctx)
    a, b = V.rows_f32("pc8-h3-tile", "A", m, k), V.rows_f32("pc8-h3-tile", "B", n, k)
    fn = pc8.templates(k, tile, "sm120", True, "h3")["tile"]
    n_bits = 64 * tile[0] * tile[1]
    assert fn.id == f"Pc8TileHiddenH3_v1{{K={k},TM={tile[0]},TN={tile[1]},DEV='sm120'}}"
    assert BL.bit_type(fn.signature()[1]).leaves == n_bits
    for t in s.tiles(ctx.matmul.shape):
        ops = s.encode(ctx, noise, [a[i] for i in t.rows], [b[j] for j in t.cols], t)
        assert s.admissible(ctx, ops)
        checked = s.checked(ctx, ops)
        out = evaluate(fn, _flat(ops.a.codes), _flat(ops.b.codes), _flat(ops.pa), _flat(ops.pb))
        row = b"".join(C.messages(checked))
        assert len(row) * 8 == n_bits and _element_bits(out, 32) == _row_seg_arguments(row, n_bits)
        assert sha512_row_seg_digest(row, n_bits) == s.digest(ctx, noise, checked) == C.leaf(checked, "h3")


def test_the_h3_tile_is_hiding_only_and_leaves_h0_as_it_is():
    """-h3 changes only the hiding tile: the row units and the plain tile are h0's Definitions, h0's hiding tile is
    `Pc8TileHidden` (its Programs are pinned in `WHOLE`), and a plain tile under -h3 or an unknown hashing is refused."""
    h0, h3 = pc8.templates(K, (2, 3), "sm120", True), pc8.templates(K, (2, 3), "sm120", True, "h3")
    assert h0["row_a"] == h3["row_a"] and h0["row_b"] == h3["row_b"]
    assert h0["tile"].defn is pc8.Pc8TileHiddenDef and h3["tile"].defn is pc8.Pc8TileHiddenH3Def
    for bad in ((False, "h3"), (True, "h1")):
        with pytest.raises(ValueError, match="h3 only hiding"):
            pc8.templates(K, (2, 3), "sm120", *bad)


def test_the_window_tile_reads_its_side_rows_index_fields():
    """`Pc8TileSide` is `Pc8TileHidden` with each row's index field (`leaves.INDEX`) as one more value: its digest is the
    hiding tile's whatever the indices, and the rows its ports make (`SIDE_PORTS`: codes, P, then the index field, each
    leaf's bits in order) are the window's side rows (`window.side_row`), A's row-seg and B's v2, k/2 + 72 words each."""
    from verity.core.protocols.pouw import window as Wd
    from verity.ml.protocols.pouw.circuit import leaves as L
    from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2
    tm, tn = 2, 3
    fn, hidden = pc8.side_tile(K, (tm, tn)), pc8.templates(K, (tm, tn), "sm120", True)["tile"]
    assert fn.id == f"Pc8TileSide_v1{{K={K},TM={tm},TN={tn},DEV='sm120'}}" and fn.defn is pc8.Pc8TileSideDef
    assert fn.signature()[0] == hidden.signature()[0] + (("ia", pc8.Array(tm, L.INDEX)), ("jb", pc8.Array(tn, L.INDEX)))
    assert fn.signature()[1] == hidden.signature()[1]
    rng = random.Random(11)
    ac, bc = ([[rng.getrandbits(8) & 0x77 for _ in range(K)] for _ in range(r)] for r in (tm, tn))
    pa, pb = ([[rng.getrandbits(16) & 0x3FFF for _ in range(64)] for _ in range(r)] for r in (tm, tn))
    want = list(evaluate(hidden, _flat(ac), _flat(bc), _flat(pa), _flat(pb)))
    for ii, jj in (([0, 8191], [0, 1, 131071]), ([5, 64], [70000, 3, 4])):
        ia, jb = [L.words_index(i) for i in ii], [L.words_index(j) for j in jj]
        assert list(evaluate(fn, _flat(ac), _flat(bc), _flat(pa), _flat(pb), _flat(ia), _flat(jb))) == want
        for codes, p, idx, i in [*zip(ac, pa, ia, ii), *zip(bc, pb, jb, jj)]:
            bits = sum(c << (8 * t) for t, c in enumerate(codes)) | sum(w << (8 * K + 16 * t) for t, w in enumerate(p + idx))
            words = [(bits >> (16 * w)) & 0xFFFF for w in range(L.ROW_WORDS["fp8-pi"](K))]
            assert words == Wd.side_row(L.words_fp8_p(codes, p), i) == L.words_fp8_pi(codes, p, i)
    reads = {p: tm for p in ("ac", "pa", "ia")} | {p: tn for p in ("bc", "pb", "jb")}
    bits = 16 * (K // 2 + 72)
    assert L.statement_rows(pc8.SIDE_PORTS, K, reads) == {"row_schemas": [[SCHEMA_HM96_SHA512_ROW_SEG, bits]] * tm
                                                         + [[SCHEMA_HM96_SHA512_ROW_V2, bits]] * tn
                                                         + [[SCHEMA_HM96_SHA512_ROW_V2, 256]]}
    assert (L.ROW_SCHEMA["fp8-pi", ROLE_X], L.ROW_SCHEMA["fp8-pi", ROLE_W]) == (L.ROW_SCHEMA["fp8-p", ROLE_X],
                                                                                L.ROW_SCHEMA["fp8-p", ROLE_W])


@pytest.mark.parametrize("dev", ["h100", "sm120"])
def test_the_verifier_derives_the_anchored_inputs(dev):
    """`TileAnchors.of` derives the scheme's E_A, E_B, F_A, F_B and G (on the record's chain) from its seeds, and the tile
    layout is `pc4`'s on Pearl-C's Programs."""
    shape = pc8.Shape(16, K, 16, tile=(8, 16), dev=dev)
    s, ctx, _ops, a, ea, b, eb, fa, fb, g = _case(f"pc8-anchors-{dev}", (16, K, 16), dev)
    t = TileAnchors.of(s, ctx, shape)
    assert t.inputs(pc8.weight(shape)) == {"eb": eb, "fa": fa, "fb": fb, "g": g}
    assert t.inputs(pc8.call(shape)) == {"ea": ea, "fa": fa, "fb": fb}
    assert t.regions("row_a", 3) == {"e": ea[96:128], "fa": fa, "fb": fb}
    assert t.regions("row_b", 5) == {"e": eb[160:192], "fb": fb, "fa": fa, "g": [h for w in g for h in (w & 0xFFFF, w >> 16)]}
    with pytest.raises(DomainError):
        t.check_inputs(pc8.weight(shape), [b, eb, fa, fb, g[:-1] + [g[-1] ^ 1]])
    lay = plan.tile_layout(replace(shape, hidden=True), 1)
    lay.check()
    assert lay.closure == {16: (16, *range(8)), 17: (17, *range(8, 16))} and lay.b_rows == {16: tuple(range(16)),
                                                                                             17: tuple(range(16))}


# ---- a short last tile row --------------------------------------------------------------------------------------------

def _scheme_leaves(s: C.PearlC, ctx, case: str) -> tuple[int, ...]:
    """The scheme's leaf of each of its tiles (`PearlC.tiles`, row-major; a short tile's over its real rows), flat as the
    hiding call returns its digests."""
    m, k, n = ctx.matmul.shape.m, ctx.matmul.shape.k, ctx.matmul.shape.n
    noise, a, b = s.noise(ctx), V.rows_f32(case, "A", m, k), V.rows_f32(case, "B", n, k)
    out: list[int] = []
    for t in s.tiles(ctx.matmul.shape):
        ops = s.encode(ctx, noise, [a[i] for i in t.rows], [b[j] for j in t.cols], t)
        d = s.digest(ctx, noise, s.checked(ctx, ops))
        out += [int.from_bytes(d[4 * w:4 * w + 4], "little") for w in range(8)]
    return tuple(out)


@pytest.mark.slow
@pytest.mark.parametrize("m", [80, 112, 32])
def test_a_short_last_tile_row_is_the_schemes(m):
    """At 64-row tiles where 64 ∤ m (m = 80 and 112: a 16- and a 48-row last tile row; m = 32: one 32-row tile row) the
    Programs give the scheme's A′, P_A, C̃ and U on every word and admit every row, `Pc8Check` is the two sides in one
    root, and the hiding call gives each of the scheme's tiles its leaf (`PearlC.tiles`), the short ones' over their real
    rows."""
    tile, case = (64, 16), "pc8-short"
    s, ctx, ops, *inputs = _case(case, (m, K, 16), "sm120", tile=tile)
    s.check(ctx.matmul)
    assert s.admissible(ctx, ops)
    shape = pc8.Shape(m, K, 16, tile=tile)
    assert shape.tile_rows() == tuple(len(range(i, min(i + 64, m))) for i in range(0, m, 64))
    assert [len(t.rows) for t in s.tiles(ctx.matmul.shape)] == list(shape.tile_rows())
    checked = s.checked(ctx, ops)
    got = _evaluate(shape, *inputs)
    assert got["a_codes"] == _flat(ops.a.codes) and got["pa"] == ops.pa and got["ok_a"] == [1] * m
    assert [list(r) for r in got["c"]] == checked.c and [list(r) for r in got["u"]] == checked.u
    a, ea, _b, _eb, fa, fb, _g = inputs
    hidden = evaluate(pc8.call(replace(shape, hidden=True)), a, ea, fa, fb, got["b_codes"], _flat(got["pb"]))
    assert hidden == _scheme_leaves(s, ctx, case) + (1,) * m


@pytest.mark.slow
def test_the_decode_32_case_is_one_short_tile_row():
    """`pearl_c.json`'s decode-32 case (32 × 1,024 × 64 on the H100) through
    `Pc8Weight` and `Pc8Call` at the scheme's own 64 × 64 tile, one 32-row tile row: the vector's B̃ codes, P_B, C̃ and U,
    every row admitted.  About 40 s: the only test that takes a pinned production-depth case through the circuit."""
    case = next(c for c in json.loads(V.OUT.read_text())["cases"] if c["case"] == "decode-32-v1")
    (m, k, n), base = case["shape"], case["case"]
    s, ctx, _ops, a, ea, b, eb, fa, fb, g = _case(base, (m, k, n), "h100", case["index"], case["weight"])
    noise = s.noise(ctx)
    assert (noise.seed_a.hex(), noise.seed_b.hex()) == (case["seed_a"], case["seed_b"])
    shape = pc8.Shape(m, k, n, dev="h100")
    assert shape.tile == (64, 64) and shape.tile_rows() == (32,)
    wout = evaluate(pc8.weight(shape), b, eb, fa, fb, g)
    bc, pb = list(wout[:n * k]), list(wout[n * k:-n])
    cout = evaluate(pc8.call(shape), a, ea, fa, fb, bc, pb)
    c, u = ([cout[f * m * n + i * n:f * m * n + (i + 1) * n] for i in range(m)] for f in range(2))
    assert V.sha(bytes(bc)) == case["b_codes_sha256"] and V.sha(V.halves([pb[64 * j:64 * (j + 1)] for j in range(n)])) == case["pb_sha256"]
    assert V.sha(V.words(c)) == case["c_sha256"] and V.sha(V.words(u)) == case["u_sha256"]
    assert list(wout[-n:]) == [1] * n and list(cout[2 * m * n:]) == [1] * m


SHORT = pc8.Shape(80, K, 2, tile=(64, 1), dev="sm120")


@pytest.mark.parametrize("hidden", [False, pytest.param(True, marks=pytest.mark.slow)], ids=["plain", "hidden"])
@pytest.mark.parametrize("side", ["check", "call"])
def test_a_short_tile_row_is_its_own_template_instance(side, hidden):
    """At m = 80 the call's tiles are two template instances of one Definition, `Pc8Tile` (or `Pc8TileHidden`) at
    64 × 1 and at 16 × 1, in that order in the query, the short ones at tile row 1; the partition still computes every
    value in exactly one unit."""
    shape = replace(SHORT, hidden=hidden)
    whole, short = pc8.tile_templates(shape)
    tp = pc8.templates(K, (16, 1), shape.dev, hidden)
    assert whole == pc8.templates(K, shape.tile, shape.dev, hidden)["tile"] and short == tp["tile"]
    assert whole.defn is short.defn and shape.tile_rows() == (64, 16)
    fn = {"check": pc8.check, "call": pc8.call}[side](shape)
    prog, q = Program(fn), pc8.query(shape, side)
    assert q["params"]["templates"][-2:] == [PO.template_id(whole), PO.template_id(short)]
    us = pc8.units(fn)
    tiles = [(u.template, u.where) for u in us if u.role == "tile"]
    assert tiles == [(PO.template_id(whole), (0, 0)), (PO.template_id(whole), (0, 1)), (PO.template_id(short), (1, 0)),
                     (PO.template_id(short), (1, 1))]
    assert Counter(u.role for u in us) == {"row_a": 80, "tile": 4} | ({"row_b": 2} if side == "check" else {})
    assert PO.verify(PO.build(prog, q), prog).ok
    _g, _owner, verdict = pc8.unit_cut(fn)
    assert verdict.ok, verdict.codes


def test_a_short_tile_is_proved_with_its_rows_in_its_own_stratum():
    """`plan.tile_layout` at m = 80 proves a 16-row tile with its 16 `RowA` units, in its own stratum at 16/64 of a whole
    tile's W_ref.  That is two tile strata, which the work law refuses (`check_work_strata`, the window pin's `hone`) and
    the count law draws; a W_ref with no whole quarter is refused.  At m = 32, one tile row, the work law draws, and
    `unsound_work_bound` reads the short tile's W_ref."""
    key = derive(bytes(32), plan.DRAW_DOMAIN, {"test": "pc8-short"})
    lay = plan.tile_layout(replace(SHORT, hidden=True), 64)
    lay.check()
    us = lay.units
    assert [(s.role, len(s.units), s.work) for s in lay.strata] == [("row_a", 80, 0), ("tile", 2, 2 * 64), ("tile", 2, 2 * 16)]
    for t in (u for u in us if u.role == "tile"):
        rs, cs = t.where
        rows = range(64) if rs == 0 else range(64, 80)
        assert [(us[i].role, us[i].where) for i in lay.closure[t.index][1:]] == [("row_a", (i,)) for i in rows]
        assert lay.b_rows[t.index] == (cs,)
    for refused in (lambda: plan.WorkLaw().sizes(lay.strata), lambda: plan.work_table(lay), lambda: plan.draw(key, lay, plan.WorkLaw())):
        with pytest.raises(ValueError, match="one stratum"):
            refused()
    p = plan.draw(key, lay, plan.CountLaw(4))
    assert set(p.proved) == set(p.drawn) | {i for t in p.drawn if us[t].role == "tile" for i in lay.closure[t]}
    with pytest.raises(DomainError):
        plan.tile_layout(SHORT, 2)
    decode = plan.tile_layout(pc8.Shape(32, K, 2, tile=(64, 1), hidden=True), 64)
    tile = next(s for s in decode.strata if s.role == "tile")
    assert [(s.role, s.work) for s in decode.strata] == [("row_a", 0), ("tile", 2 * 32)]
    p = plan.draw(key, decode, plan.WorkLaw(2))
    assert set(p.proved) == set(range(34)) and p.proved_b == (0, 1)
    assert 32 <= plan.unsound_work_bound(decode, {tile.name: 1}, 0.25) < 64


#: each whole-tile shape's Programs (their ids, queries and, up to 4,096 words, digests) and tile layout, pinned: a short
#: last tile row leaves every whole-tile shape as it is, with the row cut (``row_k``) or without
WHOLE = {
    pc8.Shape(1, 128, 1, tile=(1, 1)): "c906341a1377f29d1d03abda8cd10f9e45e0c5df4fc76e9c8418a7503e25f076",
    pc8.Shape(1, 128, 1, tile=(1, 1), hidden=True): "bfcdf4bda1ae9cec62b2184b5fb3f0794e7a48b3186963c7b850c31522160f22",
    pc8.Shape(4, 128, 6, tile=(2, 3)): "bb12e1f6ee0cfa62c4fc6136b7b7172987ffcb347e732edb20e42f7389337cc4",
    pc8.Shape(4, 128, 6, tile=(2, 3), hidden=True): "9ac3bada057d19b71ad95001f8c2a4b2010fbd1321a0ae17d882090d60ff3083",
    pc8.Shape(16, 128, 16, tile=(4, 8), dev="h100"): "5d7f419bdc59301605dc7869ceb664a2346172d9893b5cd13c4191df51010ebd",
    pc8.Shape(16, 160, 16, tile=(8, 16), dev="sm120-unpromoted"):
        "ebbcf53de05b062e9a0e4fbddb59813f23227510730db5d8a9b5ee361a5f665b",
    pc8.Shape(32, 128, 64, tile=(32, 64), hidden=True): "53a0e0a4ba17c301d5b2c2342696f9e85317ccd9eca43e529224985c05a78195",
    pc8.Shape(64, 128, 64, hidden=True): "46bbd94ebd4d295805b18aa3900d902a10dab58029f4a616d96a757a61469ccf",
    pc8.Shape(128, 128, 128): "42bf79753583e3318a6ee174dc9eecf279eabd011963a715d1fb79028dd445cd",
    pc8.Shape(128, 128, 128, hidden=True): "8df370c86fb0876347efc0efd99aa338656f73d91e3ea6bf7ba511e4e2ac5c67",
    pc8.Shape(2, 384, 2, tile=(1, 1), dev="h100", row_k=128):
        "6e57e2e6bfa3ac03b0ec43028e24740643e2ad9a1861cc061f8503d31115848f",
    pc8.Shape(2, 384, 2, tile=(1, 1), hidden=True, row_k=128):
        "533372e79842f43f1494b08255cc2ad65b28622bc10fb78b8df79cceee6fe66f",
    pc8.Shape(2, 96, 2, tile=(1, 1), dev="sm120-unpromoted", row_k=32):
        "52dc8f86afba68200cbc9f6e58666de2c52d14d61ccd3df4c033e3dba810b489",
    pc8.Shape(4, 384, 6, tile=(2, 3), hidden=True, row_k=128):
        "98527b5e389229acebc30c4dae44d757f666255a7b8ff28b45eb40adba2feda6",
    pc8.Shape(64, 256, 64, hidden=True, row_k=128): "f7140c9770354efbfd59c4c4db0f1977a04f4fb25942ba6c2ccb5b4a6abf7788",
}


@pytest.mark.parametrize("shape", list(WHOLE), ids=[f"{s.m}x{s.k}x{s.n}-tile{s.tile[0]}x{s.tile[1]}-{s.dev}"
                                                    f"{'-hidden' * s.hidden}{f'-rowk{s.row_k}' * (s.row_k is not None)}"
                                                    for s in WHOLE])
def test_whole_tile_programs_and_layouts_are_pinned(shape):
    """`Pc8Weight`, `Pc8Call` and `Pc8Check` (or their hiding forms), their queries and `plan.tile_layout` at W_ref 4,096
    are the pinned ones on every whole-tile shape."""
    row = {"layout": CF.layout_sha(plan.tile_layout(shape, 4096))}
    for side, build in (("weight", pc8.weight), ("call", pc8.call), ("check", pc8.check)):
        fn = build(shape)
        row[side] = {"id": fn.id, "query": json.dumps(pc8.query(shape, side), sort_keys=True)}
        if shape.m * shape.n <= 4096:
            row[side]["program"] = PO.program_sha512(Program(fn))
    assert hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() == WHOLE[shape]


def test_shapes_outside_the_circuit_are_refused():
    """The scheme's domain: whole tiles across, and down the rows whole 16-row ticket subtiles (whole tiles under 16 rows),
    so 40 and 72 rows at 64-row tiles are refused and 16, 32, 80 and 112 admitted."""
    assert SM120_UNPROMOTED.window == 32
    pc8.Shape(1, 96, 1, tile=(1, 1), dev="sm120-unpromoted").check()
    for m in (16, 32, 80, 112):
        pc8.Shape(m, 128, 64).check()
    for shape in (pc8.Shape(1, 96, 1, tile=(1, 1)), pc8.Shape(0, 128, 1, tile=(1, 1)), pc8.Shape(1, 128, 0, tile=(1, 1)),
                  pc8.Shape(1, 1 << 17, 1, tile=(1, 1)), pc8.Shape(2, 128, 2, tile=(4, 1)), pc8.Shape(4, 128, 3, tile=(2, 2)),
                  pc8.Shape(1, 128, 1, tile=(0, 1)), pc8.Shape(1, 128, 1, tile=(1, 1), dev="b200"), pc8.Shape(40, 128, 64),
                  pc8.Shape(72, 128, 64), pc8.Shape(80, 128, 96), pc8.Shape(24, 128, 64, tile=(16, 64))):
        for build in (pc8.check, pc8.weight, pc8.call):
            with pytest.raises(DomainError):
                build(shape)


def test_the_row_cut_refuses_a_short_tile_row():
    """The row cut (``row_k``) takes whole tiles only: 80 rows and decode's 32 at 64-row tiles, admitted without it, are
    refused with it by every root and by the tile layout, and 64 rows are admitted with it."""
    pc8.Shape(64, 256, 64, row_k=128).check()
    for m in (80, 32):
        pc8.Shape(m, 256, 64).check()
        shape = pc8.Shape(m, 256, 64, row_k=128)
        for refused in (pc8.check, pc8.weight, pc8.call, lambda s: plan.tile_layout(s, 4096)):
            with pytest.raises(DomainError, match="row cut takes whole tiles only"):
                refused(shape)
        with pytest.raises(DomainError, match="row cut takes whole tiles only"):
            plan.tile_layout(replace(shape, hidden=True), 4096)
