"""Pearl-C's tile check as a Program (`verity_pouw.circuit.pc8`): its primitives are the scheme's operations on the domain
the row's admission bit admits, its units evaluate to `schemes.pearl_c`'s forming v1, clean-up and checked words (the pinned
``test-16-v1`` vector on the H100, the scheme itself on both sm_120 records), its admission bit is the scheme's on every row
(finiteness, the noise floor and `live_row`, each at its boundary), and its partition computes every value in exactly one
unit; its hiding Programs publish only admission bits and tile digests, their units' ports are `PORTS`, and the verifier
derives the anchored inputs (`anchors.TileAnchors`)."""
from __future__ import annotations

import json
import random
import struct
from collections import Counter
from dataclasses import replace

import pytest

import circuit_fixtures as CF
import pearl_c_vectors as V
from verity.evaluation import evaluate
from verity.ir import partition_object as PO
from verity.ir.program import Program
from verity.ml.fp32 import F32FmaV2
from verity.ml.prims import (BlackwellE4m3QmmaDot32, E4m3ToF32, F2fpBf16, F32ToE4m3Sat, HopperE4m3QgmmaDot32)
from verity.ml.scalar import I32Le
from verity.ml.tc.models import BLACKWELL_SM120_E4M3_M16N8K32, HOPPER_E4M3_K32
from verity_pouw.circuit import pc8, plan
from verity_pouw.circuit.anchors import TileAnchors
from verity_pouw.circuit import words as W
from verity_pouw.protocol import DomainError, Tile
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_kw as P
from verity_pouw.schemes.pearl_c_device import DEVICES, SM120_UNPROMOTED

K = 128
CASE = next(c for c in json.loads(V.OUT.read_text())["cases"] if c["case"] == "test-16-v1")
SEED = bytes(range(32))


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

def _case(case: str, shape: tuple[int, int, int], dev: str, index: int = 3, weight: int = 9):
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1", device=DEVICES[dev])
    ctx = V.ctx_of(case, shape, index, weight)
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
    sum of squares, a zero row, a subnormal row, the largest word, and both sides of the noise floor and of liveness (one
    word bisected to its flip; and a row whose third dead entry meets the threshold with equality)."""
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
            "subnormal": [1] * K, "max-unsampled": at(base[0], {1: 0x7F7FFFFF}), "floor-in": floor_in, "floor-out": floor_out,
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
    assert admitted == {"honest-0", "honest-1", "negated", "floor-in", "live-in", "live-below"}


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


def test_shapes_outside_the_circuit_are_refused():
    assert SM120_UNPROMOTED.window == 32
    pc8.Shape(1, 96, 1, tile=(1, 1), dev="sm120-unpromoted").check()
    for shape in (pc8.Shape(1, 96, 1, tile=(1, 1)), pc8.Shape(0, 128, 1, tile=(1, 1)), pc8.Shape(1, 128, 0, tile=(1, 1)),
                  pc8.Shape(1, 1 << 17, 1, tile=(1, 1)), pc8.Shape(2, 128, 2, tile=(4, 1)), pc8.Shape(4, 128, 3, tile=(2, 2)),
                  pc8.Shape(1, 128, 1, tile=(0, 1)), pc8.Shape(1, 128, 1, tile=(1, 1), dev="b200")):
        for build in (pc8.check, pc8.weight, pc8.call):
            with pytest.raises(DomainError):
                build(shape)
