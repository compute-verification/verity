"""Pearl-C4's tile check as a Program (`verity.core.protocols.pouw.circuit.pc4`): its primitives and float forms are the scheme's operations
on the domain the row's admission bit admits, its units evaluate to `schemes.pearl_c4`'s forming, clean-up and checked
words (the pinned ``test-16-nvfp4`` vector), its admission bit is `row_passes` on every row, and its partition computes
every value in exactly one unit; its hiding Programs publish only admission bits and tile digests, their units' ports are
`PORTS`, the verifier derives the anchored inputs (`anchors.TileAnchors`), and `plan.tile_layout` draws a tile with its
rows."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import replace

import pytest

import circuit_fixtures as CF
import pearl_c4_vectors as V
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.program import Program
from verity.ml.circuits.ops.prims import F32ToE2m1Sat, F32ToE4m3Sat, HopperBF16WgmmaDot16
from verity.ml.circuits.ops.scalar import F32Fabs, F32Fmaxf, I32Le
from verity.ml.circuits.silicon import fp32
from verity.ml.randomness import derive
from verity.core.protocols.pouw.circuit import pc4, plan
from verity.core.protocols.pouw.circuit.anchors import TileAnchors, region_bytes
from verity.core.protocols.pouw.circuit import words as W
from verity.core.protocols.pouw.protocol import DomainError
from verity.core.protocols.pouw.schemes import pearl_c4 as F
from verity.core.protocols.pouw.schemes import pearl_kw as P

K = 128
CASE = next(c for c in json.loads(V.OUT.read_text())["cases"] if c["case"] == "test-16-nvfp4")
SEED_B = hashlib.sha256(b"verity/pouw/test-circuit-pc4").digest()


def _finite_words(rng: random.Random, count: int, signed: bool = True) -> list[int]:
    out = []
    while len(out) < count:
        w = rng.getrandbits(32 if signed else 31)
        if (w >> 23) & 0xFF != 0xFF:
            out.append(w)
    return out


def _flat(lines) -> list[int]:
    return [c for ln in lines for c in ln]


# ---- the primitives and the float forms -------------------------------------------------------------------------------

def test_sqrt_is_core_sqrt_rn_with_cudas_specials():
    rng = random.Random(1)
    for w in _finite_words(rng, 2000, signed=False) + [0, 1, 0x007FFFFF, 0x00800000, F.FLOOR, 0x7F7FFFFF]:
        assert W.PouwF32SqrtRn.evaluate(w) == fp32.sqrt(w)
    assert W.PouwF32SqrtRn.evaluate(0x80000000) == 0x80000000
    assert W.PouwF32SqrtRn.evaluate(0x7F800000) == 0x7F800000
    for w in (0xBF800000, 0x80000001, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001):
        assert W.PouwF32SqrtRn.evaluate(w) == 0x7FFFFFFF


def test_e_tilde_is_the_schemes_exact_decode():
    """Every E2M1 code at every a_E byte the hardware reads as a number (bit 7 too, which D-NF never admits), and NaN."""
    for ae in range(256):
        for e in range(16):
            got = W.PouwETilde.evaluate(e, ae)
            if ae & 0x7F == 0x7F:
                assert got == 0x7FFF
            else:
                assert got == F.e_tilde([e], ae)[0], (e, ae)
    assert [W.PouwBf16Neg.evaluate(v) for v in (0, 0x8000, 0x3F80, 0xFFFF)] == [0x8000, 0, 0xBF80, 0x7FFF]


def test_the_casts_and_reductions_are_the_schemes_on_finite_words():
    """`F32ToE2m1Sat` is `e2m1`; `ue4m3(fmaxf(v, 2^-9))` is `max(ue4m3(v), 0x01)` (the block scale's floor); `fmaxf` of
    magnitudes is their integer maximum; and the two signed comparisons are `dnf`."""
    rng = random.Random(2)
    mids = [P.f32_bits(v) for v in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0, 1e30, 2.0 ** -9, 2.0 ** -10)]
    edges = [w + d for w in mids for d in (-1, 0, 1)] + [0, 0x80000000, 1, 0x7F7FFFFF]
    for w in _finite_words(rng, 3000) + edges + [w | 0x80000000 for w in edges]:
        if (w >> 23) & 0xFF != 0xFF:
            assert F32ToE2m1Sat.evaluate(w) == F.e2m1(w), hex(w)
    e4m3_mids = [P.f32_bits((float(P.fp8_to_f32(b)) + float(P.fp8_to_f32(b + 1))) / 2) for b in range(0x7E)]
    for v in _finite_words(rng, 3000, signed=False) + [w + d for w in e4m3_mids for d in (-1, 0, 1)] + [0, 1, 0x3B000000]:
        assert F32ToE4m3Sat.evaluate(F32Fmaxf.evaluate(v, pc4.UE4M3_MIN_WORD)) == max(F.ue4m3(v), F.UE4M3_MIN), hex(v)
    for a, b in zip(_finite_words(rng, 2000), _finite_words(rng, 2000)):
        assert F32Fmaxf.evaluate(F32Fabs.evaluate(a), F32Fabs.evaluate(b)) == max(a & 0x7FFFFFFF, b & 0x7FFFFFFF)
    betas = _finite_words(rng, 3000) + [pc4.DNF_LOW + d for d in (-1, 0, 1)] + [0, 0x80000000, 0x7F7FFFFF, 0x7F800000,
                                                                                0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FFFFFFF]
    for beta in betas:
        assert (I32Le.evaluate(pc4.DNF_LOW, beta) & I32Le.evaluate(beta, pc4.F32_MAX)) == F.dnf(beta), hex(beta)


def test_core_bf16_step_is_the_schemes_peel_step():
    rng = random.Random(3)
    bf16 = lambda: (rng.getrandbits(1) << 15) | (rng.randrange(110, 140) << 7) | rng.getrandbits(7)  # noqa: E731
    for _ in range(200):
        acc = P.f32_bits(rng.uniform(-1e4, 1e4))
        a, b = [bf16() for _ in range(16)], [bf16() for _ in range(16)]
        assert evaluate(HopperBF16WgmmaDot16, acc, a, b) == (F.HOPPER_BF16_M16N8K16.step(acc, a, b),)


# ---- the units against the scheme -------------------------------------------------------------------------------------

def _pinned_ctx():
    return V.scheme("nvfp4", True), V.ctx_of(CASE["case"], tuple(CASE["shape"]), CASE["index"], CASE["weight"])


def _pinned():
    s, ctx = _pinned_ctx()
    m, k, n = CASE["shape"]
    tile = s.tiles(ctx.matmul.shape)[0]
    a, b = V.rows(CASE["case"], "A", m, k), V.rows(CASE["case"], "B", n, k)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, tile)
    fa, fb = F.basis(noise.seed_b, F.ROLE_FA, k), F.basis(noise.seed_b, F.ROLE_FB, k)
    g = _flat(F.gram(F.NVFP4, noise.seed_b, k))
    words = lambda rows: [x >> 16 for row in rows for x in F.parse_row(row, k)]  # noqa: E731
    return pc4.Shape(m, k, n, tile=(4, 8)), ops, words(a), _flat(ops.e_a), words(b), _flat(ops.e_b), _flat(fa), _flat(fb), g


def test_the_programs_evaluate_to_the_pinned_vector():
    """`Pc4Weight` gives B̃'s codes, scale bytes and P_B as the scheme forms them from the anchored G, `Pc4Call` reads them
    and gives the vector's C̃ and U, and `Pc4Check` is the two in one root; every row of the honest tile is admitted.  The
    16 × 16 block is cut into 4 × 8 tiles, so a word placed in the wrong tile or transposed within one changes the digests."""
    shape, ops, a, ea, b, eb, fa, fb, g = _pinned()
    m, k, n = shape.m, shape.k, shape.n
    wout = evaluate(pc4.weight(shape), b, eb, fa, fb, g)
    bc, bs = list(wout[:n * k]), list(wout[n * k:n * k + n * k // 16])
    pb, ok_b = list(wout[n * k + n * k // 16:-n]), list(wout[-n:])
    assert V.sha(bytes(bc)) == CASE["b_codes_sha256"] and V.sha(bytes(bs)) == CASE["b_scales_sha256"]
    assert V.sha(V.halves([pb[64 * j:64 * (j + 1)] for j in range(n)])) == CASE["pb_sha256"]
    assert ok_b == [1] * n
    cout = evaluate(pc4.call(shape), a, ea, fa, fb, bc, bs, pb)
    mn = m * n
    c, u = [cout[i * n:(i + 1) * n] for i in range(m)], [cout[mn + i * n:mn + (i + 1) * n] for i in range(m)]
    assert V.sha(V.words(c)) == CASE["c_sha256"] and V.sha(V.words(u)) == CASE["u_sha256"]
    assert list(cout[2 * mn:]) == [1] * m
    assert evaluate(pc4.check(shape), a, ea, b, eb, fa, fb, g) == tuple(cout) + tuple(ok_b)
    row_a = pc4.templates(k)["row_a"]
    outs = [evaluate(row_a, a[i * k:(i + 1) * k], ea[32 * i:32 * (i + 1)], fa, fb) for i in range(m)]
    assert V.sha(bytes(x for o in outs for x in o[:k])) == CASE["a_codes_sha256"]
    assert V.sha(bytes(x for o in outs for x in o[k:k + k // 16])) == CASE["a_scales_sha256"]
    assert V.sha(V.halves([o[k + k // 16:-1] for o in outs])) == CASE["pa_sha256"]
    assert ops.pa[0][:8] == CASE["pa_0"]


def _scheme_admits(words: list[int]) -> bool:
    try:
        return F.row_passes(F.NVFP4, F.parse_row(F.bf16_row(words), K))
    except P.Reject:
        return False


def _at(words: list[int], updates: dict[int, int]) -> list[int]:
    return [updates.get(t, w) for t, w in enumerate(words)]


MINUS_ZERO_AT = (0, 1, 5, 8, 16, 33, 64, 127)


def _rows() -> dict[str, list[int]]:
    """Honest rows and rows the scheme refuses: non-finite words at sampled (0, 8, …) and unsampled positions, finite
    words whose sampled squares overflow (ρ and β not finite: `pc4Sem_squaresOverflow`), a spike that makes a_E
    subnormal (D-NF), one that only screens its block, a zero row, an all-(−0) row, rows of tiny and of huge words, and
    an honest row with eight −0 words at sampled and unsampled positions."""
    base = [V.row_codes("pc4-circuit", "A", i, K) for i in range(3)]
    spike = lambda v: _at(base[0], {1: P.f32_to_bf16(v)})  # noqa: E731
    return {"honest-0": base[0], "honest-1": base[1], "negated": [w ^ 0x8000 for w in base[2]],
            "inf-unsampled": _at(base[0], {1: 0x7F80}), "minus-inf-sampled": _at(base[0], {0: 0xFF80}),
            "nan-unsampled": _at(base[0], {3: 0x7FC0}), "minus-nan-sampled": _at(base[0], {8: 0xFFC1}),
            "squares-overflow": _at(base[0], {p: 0x7E80 for p in range(0, K, 8)}),
            "spike-2^16": spike(2.0 ** 16), "spike-256": spike(256.0), "max-unsampled": _at(base[0], {1: 0x7F7F}),
            "zero": [0] * K, "minus-zero": [0x8000] * K, "tiny": [0x0001] * K, "huge": [0x7F7F] * K,
            "mixed-zero": _at(base[0], {p: 0x8000 for p in MINUS_ZERO_AT})}


@pytest.mark.parametrize("side", ["row_a", "row_b"])
def test_the_row_units_admission_is_row_passes_and_their_outputs_the_schemes(side):
    """``ok`` is the scheme's admission on every row (`parse_row`'s finiteness, then `row_passes`), and on an admitted row
    the unit's codes, scale bytes and peel row are `form`'s and `peel_factors_a` / `_b`'s.  Nothing is claimed about a
    refused row's other outputs: the scheme rejects it before forming."""
    fa, fb = F.basis(SEED_B, F.ROLE_FA, K), F.basis(SEED_B, F.ROLE_FB, K)
    own, other = (fa, fb) if side == "row_a" else (fb, fa)
    extra = [] if side == "row_a" else [_flat(F.gram(F.NVFP4, SEED_B, K))]
    fn = pc4.templates(K)[side]
    rows = _rows()
    verdicts = {}
    for i, (name, words) in enumerate(rows.items()):
        e = F.line(SEED_B, F.ROLE_EA if side == "row_a" else F.ROLE_EB, i)
        out = evaluate(fn, words, e, _flat(own), _flat(other), *extra)
        verdicts[name] = out[-1]
        assert out[-1] == _scheme_admits(words), name
        if not out[-1]:
            continue
        x = F.parse_row(F.bf16_row(words), K)
        formed = F.form(F.NVFP4, [x], [e], own)
        peel = (F.peel_factors_a if side == "row_a" else F.peel_factors_b)(F.NVFP4, formed, [e], SEED_B, K)[0]
        assert list(out[:K]) == formed.codes[0] and list(out[K:K + K // 16]) == formed.scales[0], name
        assert list(out[K + K // 16:-1]) == peel, name
    assert {n for n, v in verdicts.items() if v} == {"honest-0", "honest-1", "negated", "spike-256", "mixed-zero"}


@pytest.mark.parametrize("side", ["row_a", "row_b"])
def test_a_row_unit_reads_minus_zero_as_zero(side):
    """A row unit's outputs on a row with −0 words (BF16 0x8000) are its outputs with +0 in their place, as the Lean's
    `pc4Sem` reads a row by value (`formRow_table` on `ZeroSign16` words: the table registers each word's high half,
    `rows4`), and the scheme's (the admission test above): an admitted honest row with eight −0 words, and the all-(−0)
    row, refused as the all-zero row is."""
    fa, fb = F.basis(SEED_B, F.ROLE_FA, K), F.basis(SEED_B, F.ROLE_FB, K)
    own, other = (fa, fb) if side == "row_a" else (fb, fa)
    extra = [] if side == "row_a" else [_flat(F.gram(F.NVFP4, SEED_B, K))]
    fn, e = pc4.templates(K)[side], F.line(SEED_B, F.ROLE_EA if side == "row_a" else F.ROLE_EB, 0)
    run = lambda w: evaluate(fn, w, e, _flat(own), _flat(other), *extra)  # noqa: E731
    rows = _rows()
    plus = lambda w: [0 if x == 0x8000 else x for x in w]  # noqa: E731
    assert sum(w == 0x8000 for w in rows["mixed-zero"]) == len(MINUS_ZERO_AT)
    assert run(rows["mixed-zero"]) == run(plus(rows["mixed-zero"])) and run(rows["mixed-zero"])[-1] == 1
    assert run(rows["minus-zero"]) == run(rows["zero"]) and run(rows["minus-zero"])[-1] == 0


# ---- the partition ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hidden", [False, True], ids=["plain", "hidden"])
@pytest.mark.parametrize("side", ["check", "weight", "call"])
def test_the_partition_computes_every_value_in_exactly_one_unit(side, hidden):
    """One unit per row of B and of A, and per tile (G is anchored, so no unit computes it); `validate_unit_cut` certifies
    every gate once with no recompute, and the partition object verifies.  So do the hiding Programs': their rows are
    committed ports, so no unit recomputes another's value, and only the tile changes (its digest)."""
    shape = pc4.Shape(4, 64, 6, tile=(2, 3), hidden=hidden)
    fn = {"check": pc4.check, "weight": pc4.weight, "call": pc4.call}[side](shape)
    prog, q = Program(fn), pc4.query(shape, side)
    us = pc4.units(fn)
    full = {"row_b": shape.n, "row_a": shape.m, "tile": 2 * 2}
    assert Counter(u.role for u in us) == {r: full[r] for r in pc4.ROLES[side]}
    part = PO.evaluate(prog, q)
    tp = pc4.templates(shape.k, shape.tile, hidden)
    assert [t for _k, t, _a, n in part.ranges for _ in range(n)] == [tp[u.role].id for u in us] == [u.template for u in us]
    assert PO.verify(PO.build(prog, q), prog).ok
    _g, _owner, verdict = pc4.unit_cut(fn)
    assert verdict.ok, verdict.codes
    assert verdict.detail["certified"] == verdict.detail["computed_gates"]


# ---- the hiding Programs ----------------------------------------------------------------------------------------------

HIDDEN = pc4.Shape(4, 64, 6, tile=(2, 3), hidden=True)


def test_the_hiding_programs_publish_only_admission_bits_and_tile_digests():
    """Under hiding the weight's Program is `Pc4Weight`, and `Pc4CallHidden` and `Pc4CheckHidden` return each tile's digest
    (`pearl_c.leaf` of the plain call's C̃ and U, by tile row and column) and the admission bits, and no checked word."""
    m, k, n = HIDDEN.m, HIDDEN.k, HIDDEN.n
    a = [w for i in range(m) for w in V.row_codes("pc4-hidden", "A", i, k)]
    b = [w for j in range(n) for w in V.row_codes("pc4-hidden", "B", j, k)]
    fa, fb = _flat(F.basis(SEED_B, F.ROLE_FA, k)), _flat(F.basis(SEED_B, F.ROLE_FB, k))
    ea, eb = (_flat(F.line(SEED_B, role, i) for i in range(c)) for role, c in ((F.ROLE_EA, m), (F.ROLE_EB, n)))
    g = _flat(F.gram(F.NVFP4, SEED_B, k))
    plain = replace(HIDDEN, hidden=False)
    assert pc4.weight(HIDDEN).id == pc4.weight(plain).id
    wout = evaluate(pc4.weight(plain), b, eb, fa, fb, g)
    bc, bs, pb, ok_b = wout[:n * k], wout[n * k:n * k + n * k // 16], wout[n * k + n * k // 16:-n], wout[-n:]
    cout = evaluate(pc4.call(plain), a, ea, fa, fb, bc, bs, pb)
    c, u = ([cout[f * m * n + i * n:f * m * n + (i + 1) * n] for i in range(m)] for f in range(2))
    want = CF.tile_digests(HIDDEN.tile, c, u) + tuple(cout[2 * m * n:])
    assert evaluate(pc4.call(HIDDEN), a, ea, fa, fb, bc, bs, pb) == want
    assert evaluate(pc4.check(HIDDEN), a, ea, b, eb, fa, fb, g) == want + tuple(ok_b)


@pytest.mark.parametrize("role", ["row_b", "row_a", "tile"])
def test_the_ports_are_each_hiding_units_committed_rows(role):
    """`PORTS` lists every parameter and output of each hiding unit once, and each committed row is its schema's word count:
    bf16 input rows, nvfp4-p rows (A′_i ‖ P_A,i, B̃_j ‖ P_B,j), ok rows and the tile digest; the anchored inputs are public
    input regions, each its derivation's width and whole 128-bit words."""
    tm, tn = HIDDEN.tile
    CF.check_ports(pc4.templates(K, HIDDEN.tile, True)[role], pc4.PORTS[role], pc4.DERIVATIONS, K,
                   {"ac": tm, "as": tm, "pa": tm, "bc": tn, "bs": tn, "pb": tn})


def test_the_anchors_are_public_input_regions():
    """`TileAnchors.regions` is a unit's `--public-inputs` content: for a row unit at its row of the block, each parameter
    `PORTS` names as a region, in the unit's parameter order, as its anchored input's u16 words unpacked (an E2M1 code one
    word, an FP32 word of G two, low half first); a tile unit has none, and a row outside the block is refused."""
    shape, _ops, _a, ea, _b, eb, fa, fb, g = _pinned()
    t = TileAnchors.of(*_pinned_ctx(), shape)
    for i in (0, shape.m - 1):
        assert t.regions("row_a", i) == {"e": ea[32 * i:32 * (i + 1)], "fa": fa, "fb": fb}
    for j in (0, shape.n - 1):
        r = t.regions("row_b", j)
        assert list(r) == ["e", "fb", "fa", "g"] and r["e"] == eb[32 * j:32 * (j + 1)] and (r["fb"], r["fa"]) == (fb, fa)
        assert r["g"] == [h for w in g for h in (w & 0xFFFF, w >> 16)]
    assert region_bytes(t.regions("row_a", 1)["e"]) == bytes(b for c in ea[32:64] for b in (c, 0))
    assert t.regions("tile") == {}
    for bad in (None, -1, shape.m, True):
        with pytest.raises(DomainError):
            t.regions("row_a", bad)


def test_the_verifier_derives_the_anchored_inputs():
    """`TileAnchors.of` derives the pinned vector's E_A, E_B, F_A, F_B and G from the scheme's seeds, and gives each Program
    the ones it reads (F_A and F_B both, E_A the call, E_B and G the weight); other inputs, the wrong arity and the wrong
    row ids are refused."""
    shape, _ops, a, ea, b, eb, fa, fb, g = _pinned()
    t = TileAnchors.of(*_pinned_ctx(), shape)
    weight, call = pc4.weight(shape), pc4.call(shape)
    assert t.inputs(weight) == {"eb": eb, "fa": fa, "fb": fb, "g": g}
    assert t.inputs(call) == {"ea": ea, "fa": fa, "fb": fb}
    t.check_inputs(weight, [b, eb, fa, fb, g])
    t.check_inputs(call, [a, ea, fa, fb, [], [], []])
    for bad in ([b, eb, fa, fb, g[:-1] + [g[-1] ^ 1]], [b, eb, fb, fa, g], [b, eb, fa, fb]):
        with pytest.raises(DomainError):
            t.check_inputs(weight, bad)
    with pytest.raises(DomainError):
        t.check_inputs(call, [a, ea[32:] + ea[:32], fa, fb, [], [], []])
    for rows in (t.rows[:-1], (0,) * shape.m, (1 << 32,) + t.rows[1:]):
        with pytest.raises(DomainError):
            TileAnchors(shape, t.seed_a, t.seed_b, rows, t.cols)


def test_the_tile_layout_proves_each_drawn_tile_with_its_rows():
    """`plan.tile_layout`: a tile's closure is itself and its TM `RowA` units, and its ``b_rows`` its TN `RowB` units in the
    weight's Program; a sample proves both, the row units each once (`Plan.proved_b`), opens nothing, and the tile stratum
    has the W_ref given."""
    tm, tn = HIDDEN.tile
    lay = plan.tile_layout(HIDDEN, 1000)
    lay.check()
    us, ws = lay.units, pc4.units(pc4.weight(HIDDEN))
    tiles = [u for u in us if u.role == "tile"]
    assert len(tiles) == 4 and set(lay.closure) == set(lay.b_rows) == {t.index for t in tiles}
    for t in tiles:
        rs, cs = t.where
        assert [(us[i].role, us[i].where) for i in lay.closure[t.index][1:]] == [("row_a", (rs * tm + i,)) for i in range(tm)]
        assert [(ws[j].role, ws[j].where) for j in lay.b_rows[t.index]] == [("row_b", (cs * tn + j,)) for j in range(tn)]
    assert [(s.role, s.work) for s in lay.strata] == [("row_a", 0), ("tile", 4 * 1000)] and lay.tile_work() == 1000
    for law in (plan.CountLaw(3), plan.WorkLaw(2)):
        p = plan.draw(derive(bytes(32), plan.DRAW_DOMAIN, {"test": "pc4"}), lay, law)
        drawn = [u for u in p.drawn if us[u].role == "tile"]
        assert drawn and p.openings == () and p.proved_y == ()
        assert set(p.proved) == set(p.drawn) | {i for t in drawn for i in lay.closure[t]}
        assert p.proved_b == tuple(sorted({j for t in drawn for j in lay.b_rows[t]}))
    assert plan.tile_layout(replace(HIDDEN, hidden=False), 1000).closure == lay.closure


#: `CF.layout_sha` of each layout, pinned: Pearl-C4's tiles are whole, so `tile_layout`'s short tile row (`pc8`'s) never
#: applies to them, at the test shapes and at the scheme's 64 × 64 tile, with the row cut (``row_k``) or without
LAYOUTS = {
    (pc4.Shape(4, 64, 6, tile=(2, 3)), 1000): "7626b8af4961b45fcc15a0a55748f0ff869200ca9488edfb7a929c2202bfe0fd",
    (pc4.Shape(4, 64, 6, tile=(2, 3), hidden=True), 1000): "a2e3a223aa2512b710dec2565d59b55426096a95276b5c96efb0e312239c07e0",
    (pc4.Shape(1, 64, 1, tile=(1, 1)), 1): "0be929ac9e7a8ec589e7f67a6909b52692e96845c50363c48db3fb1d44107992",
    (pc4.Shape(1, 64, 1, tile=(1, 1), hidden=True), 1): "ce457562807539cf6461ac2d342702a73cb81ba23b0dd7efe07182b72c172654",
    (pc4.Shape(16, 64, 16, tile=(4, 8)), 7): "9ede0fdb0436350a5fce48f557f873df6238b1750d68134b14f3c548b261f014",
    (pc4.Shape(64, 64, 64, hidden=True), 4096): "9af149f7bb00e68b3bc89b01cc5cf5de79546d5ebe8f2bcd47657299de59d188",
    (pc4.Shape(128, 64, 192), 4096): "738d302c72a565ccb98e17494dbed82d5652c3a4aebe1e4a70fe67ccb86e7df1",
    (pc4.Shape(128, 64, 192, hidden=True), 4096): "04f82c2ed78e2c034d4ef66bb90b9c7b8ee49f326e02238a8d5c38a9aac8f8b2",
    (pc4.Shape(2, 256, 2, tile=(1, 1), row_k=64), 1): "c2b79019d69d8e8ae9f180288cf28a50d8fde3db667b08da01fe21e9282be9da",
    (pc4.Shape(2, 256, 2, tile=(1, 1), hidden=True, row_k=64), 1):
        "2292dec28f55deadf8617ba5cebea6b7a7457720b5130ff55ddbc92cf950dee6",
    (pc4.Shape(4, 256, 6, tile=(2, 3), row_k=64), 1000): "10f710a8a147066dbd4679f5f577cf7c7d0e4dffd6069c0131af68e4b87d436f",
    (pc4.Shape(4, 256, 6, tile=(2, 3), hidden=True, row_k=64), 1000):
        "6dd9a12615328cd675a363d01ed926338d81b8c1d874fd9b8b1ace6b0f035a7f",
}


@pytest.mark.parametrize("shape, w_ref", list(LAYOUTS),
                         ids=[f"{s.m}x{s.n}-tile{s.tile[0]}x{s.tile[1]}{'-hidden' * s.hidden}"
                              f"{f'-rowk{s.row_k}' * (s.row_k is not None)}" for s, _w in LAYOUTS])
def test_the_tile_layouts_are_pinned(shape, w_ref):
    """Every unit, stratum, closure, B row and W_ref of `plan.tile_layout` on Pearl-C4's shapes is the pinned one."""
    assert CF.layout_sha(plan.tile_layout(shape, w_ref)) == LAYOUTS[shape, w_ref]


def test_shapes_outside_the_circuit_are_refused():
    for shape in (pc4.Shape(1, 96, 1), pc4.Shape(0, 64, 1), pc4.Shape(1, 64, 0), pc4.Shape(2, 64, 2, tile=(4, 1)),
                  pc4.Shape(4, 64, 3, tile=(2, 2)), pc4.Shape(1, 64, 1, tile=(0, 1))):
        for build in (pc4.check, pc4.weight, pc4.call):
            with pytest.raises(DomainError):
                build(shape)
