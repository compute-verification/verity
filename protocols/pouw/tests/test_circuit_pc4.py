"""Pearl-C4's tile check as a Program (`verity_pouw.circuit.pc4`): its primitives and float forms are the scheme's operations
on the domain the row's admission bit admits, its units evaluate to `schemes.pearl_c4`'s forming, clean-up and checked
words (the pinned ``test-16-nvfp4`` vector), its admission bit is `row_passes` on every row, and its partition computes
every value in exactly one unit; its hiding Programs publish only admission bits and commit strings, and only those are
computed twice."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import replace

import pytest

import circuit_fixtures as CF
import pearl_c4_vectors as V
from verity.evaluation import evaluate
from verity.ir import partition_object as PO
from verity.ir.cut import RECOMPUTE, check_cut
from verity.ir.program import Program
from verity.ml.prims import F32ToE2m1Sat, F32ToE4m3Sat, HopperBF16WgmmaDot16
from verity.ml.scalar import F32Fabs, F32Fmaxf, I32Le
from verity.ml.tc import fp32
from verity_pouw.circuit import leaves as L
from verity_pouw.circuit import pc4
from verity_pouw.circuit import words as W
from verity_pouw.protocol import DomainError
from verity_pouw.schemes import pearl_c4 as F
from verity_pouw.schemes import pearl_kw as P

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

def _pinned():
    s, ctx = V.scheme("nvfp4", True), V.ctx_of(CASE["case"], tuple(CASE["shape"]), CASE["index"], CASE["weight"])
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


def _rows() -> dict[str, list[int]]:
    """Honest rows and rows the scheme refuses: non-finite words at sampled (0, 8, …) and unsampled positions, a spike
    that makes a_E subnormal (D-NF), one that only screens its block, a zero row, rows of tiny and of huge words."""
    base = [V.row_codes("pc4-circuit", "A", i, K) for i in range(3)]
    spike = lambda v: _at(base[0], {1: P.f32_to_bf16(v)})  # noqa: E731
    return {"honest-0": base[0], "honest-1": base[1], "negated": [w ^ 0x8000 for w in base[2]],
            "inf-unsampled": _at(base[0], {1: 0x7F80}), "minus-inf-sampled": _at(base[0], {0: 0xFF80}),
            "nan-unsampled": _at(base[0], {3: 0x7FC0}), "minus-nan-sampled": _at(base[0], {8: 0xFFC1}),
            "spike-2^16": spike(2.0 ** 16), "spike-256": spike(256.0), "max-unsampled": _at(base[0], {1: 0x7F7F}),
            "zero": [0] * K, "tiny": [0x0001] * K, "huge": [0x7F7F] * K}


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
    assert {n for n, v in verdicts.items() if v} == {"honest-0", "honest-1", "negated", "spike-256"}


# ---- the partition ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["check", "weight", "call"])
def test_the_partition_computes_every_value_in_exactly_one_unit(side):
    """One unit per row of B and of A, and per tile (G is anchored, so no unit computes it); `validate_unit_cut` certifies
    every gate once with no recompute, and the partition object verifies."""
    shape = pc4.Shape(4, 64, 6, tile=(2, 3))
    fn = {"check": pc4.check, "weight": pc4.weight, "call": pc4.call}[side](shape)
    prog, q = Program(fn), pc4.query(shape, side)
    us = pc4.units(fn)
    full = {"row_b": shape.n, "row_a": shape.m, "tile": 2 * 2}
    assert Counter(u.role for u in us) == {r: full[r] for r in pc4.ROLES[side]}
    part = PO.evaluate(prog, q)
    tp = pc4.templates(shape.k, shape.tile)
    assert [t for _k, t, _a, n in part.ranges for _ in range(n)] == [tp[u.role].id for u in us]
    assert PO.verify(PO.build(prog, q), prog).ok
    _g, _owner, verdict = pc4.unit_cut(fn)
    assert verdict.ok, verdict.codes
    assert verdict.detail["certified"] == verdict.detail["computed_gates"]


# ---- the hiding Programs ----------------------------------------------------------------------------------------------

HIDDEN = pc4.Shape(4, 64, 6, tile=(2, 3), hidden=True)


def test_the_hiding_programs_publish_the_commit_strings_of_every_row_and_tile():
    """`Pc4WeightHidden`, `Pc4CallHidden` and `Pc4CheckHidden` return the admission bits and `leaves.commit` of every row a
    unit reads or writes (A_i and B_j as bf16 rows, A′_i ‖ P_A,i and B̃_j ‖ P_B,j as nvfp4-p rows, each tile's reads equal to
    its row units') and of every tile digest (`pearl_c.leaf` of the plain units' C̃ and U), and no checked word."""
    m, k, n = HIDDEN.m, HIDDEN.k, HIDDEN.n
    a = [w for i in range(m) for w in V.row_codes("pc4-hidden", "A", i, k)]
    b = [w for j in range(n) for w in V.row_codes("pc4-hidden", "B", j, k)]
    fa, fb = _flat(F.basis(SEED_B, F.ROLE_FA, k)), _flat(F.basis(SEED_B, F.ROLE_FB, k))
    ea, eb = (_flat(F.line(SEED_B, role, i) for i in range(c)) for role, c in ((F.ROLE_EA, m), (F.ROLE_EB, n)))
    g = _flat(F.gram(F.NVFP4, SEED_B, k))
    plain = replace(HIDDEN, hidden=False)
    wout = evaluate(pc4.weight(plain), b, eb, fa, fb, g)
    bc, bs, pb, ok_b = wout[:n * k], wout[n * k:n * k + n * k // 16], wout[n * k + n * k // 16:-n], wout[-n:]
    cout = evaluate(pc4.call(plain), a, ea, fa, fb, bc, bs, pb)
    c, u = ([cout[f * m * n + i * n:f * m * n + (i + 1) * n] for i in range(m)] for f in range(2))
    outs = [evaluate(pc4.templates(k)["row_a"], a[i * k:(i + 1) * k], ea[32 * i:32 * (i + 1)], fa, fb) for i in range(m)]
    mids_a = [L.words_nvfp4_p(o[:k], o[k:k + k // 16], o[k + k // 16:-1]) for o in outs]
    mids_b = [L.words_nvfp4_p(bc[j * k:(j + 1) * k], bs[j * k // 16:(j + 1) * k // 16], pb[64 * j:64 * (j + 1)])
              for j in range(n)]
    rows = lambda x, r: [L.words_bf16(x[i * k:(i + 1) * k]) for i in range(r)]   # noqa: E731
    s = CF.salts(m, n, HIDDEN.tile)
    want_call, want_weight = CF.hidden_outputs(HIDDEN.tile, cout[2 * m * n:], rows(a, m), mids_a, ok_b, rows(b, n), mids_b,
                                               c, u, s)
    y = CF.salt_args(s)
    assert evaluate(pc4.weight(HIDDEN), b, eb, fa, fb, g, y["yb"], y["ybm"]) == want_weight
    assert evaluate(pc4.call(HIDDEN), a, ea, fa, fb, bc, bs, pb, y["ya"], y["yam"], y["ybm"], y["yt"]) == want_call
    got = evaluate(pc4.check(HIDDEN), a, ea, b, eb, fa, fb, g, y["ya"], y["yam"], y["yb"], y["ybm"], y["yt"])
    assert got == want_call + want_weight


@pytest.mark.parametrize("side", ["check", "weight", "call"])
def test_the_hiding_partition_computes_only_commit_strings_twice(side):
    """The hiding Programs have the plain ones' units, each its plain unit and its leaves, and the partition object
    verifies.  A row's commit string is computed in its row unit and again in each tile that reads the row (the units link
    by b ‖ c), so `Q_word` v1 refuses the cut for exactly that and v2's report passes it; every value two units compute is a
    leaf's gate."""
    fn = {"check": pc4.check, "weight": pc4.weight, "call": pc4.call}[side](HIDDEN)
    prog, q = Program(fn), pc4.query(HIDDEN, side)
    us = pc4.units(fn)
    full = {"row_b": HIDDEN.n, "row_a": HIDDEN.m, "tile": 2 * 2}
    assert Counter(u.role for u in us) == {r: full[r] for r in pc4.ROLES[side]}
    tp = pc4.templates(HIDDEN.k, HIDDEN.tile, True)
    assert [t for _k, t, _a, n in PO.evaluate(prog, q).ranges for _ in range(n)] == [PO.template_id(tp[u.role]) for u in us]
    assert PO.verify(PO.build(prog, q), prog).ok
    g, owner, v1 = pc4.unit_cut(fn)
    across, outside = CF.recomputed_across(g, owner)
    assert outside == []
    assert v1.codes == (("gate-recomputed",) if side != "weight" else ()) and (across > 0) == (side != "weight")
    v2 = check_cut(g, owner, limits=None, recompute=RECOMPUTE[2])
    assert v2.ok and v2.detail["recomputed_across"]["n"] == across
    assert v2.detail["certified"] == v2.detail["computed_gates"]


def test_shapes_outside_the_circuit_are_refused():
    for shape in (pc4.Shape(1, 96, 1), pc4.Shape(0, 64, 1), pc4.Shape(1, 64, 0), pc4.Shape(2, 64, 2, tile=(4, 1)),
                  pc4.Shape(4, 64, 3, tile=(2, 2)), pc4.Shape(1, 64, 1, tile=(0, 1))):
        for build in (pc4.check, pc4.weight, pc4.call):
            with pytest.raises(DomainError):
                build(shape)
