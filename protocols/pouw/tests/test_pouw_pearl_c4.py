"""``pearl-c-nvfp4-v0`` and ``pearl-c-mxfp4-v0`` (`pearl_c4`): the casts, the noise lines, the noise atom, the clean-up's
algebra, the chain's exactness on the domain (which the design's argument turns on), the accounting, the per-row rules and
debit, the work law through `pearl_c_work`, and the pinned vectors."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

import pearl_c4_vectors as V
from verity.commitments.turboshake import turboshake128
from verity.ml.tc import fp32
from verity_pouw.audit import Epoch, Prover, Workload
from verity_pouw.protocol import DomainError, Matmul, Shape
from verity_pouw.schemes import get
from verity_pouw.schemes import pearl_c4 as F
from verity_pouw.schemes import pearl_c_work as W
from verity_pouw.schemes import pearl_kw as P

K = 128
BEFORE, AFTER = 4, W.Beacon(5, b"beacon round 5")
FMTS = ("nvfp4", "mxfp4")


def rz_f32(v: Fraction) -> int:
    """The FP32 word of v rounded toward zero (gradual underflow)."""
    if v == 0:
        return 0
    a = abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    grid = max(e - 23, -149)
    q = int(a / Fraction(2) ** grid)
    word = ((e + 127) << 23) | (q - (1 << 23)) if e >= -126 else q
    return word | (0x80000000 if v < 0 else 0)


def value(fmt: F.Fmt, codes, scales) -> list[Fraction]:
    s = [Fraction(P.fp8_to_f32(c)) if fmt is F.NVFP4 else Fraction(2) ** (c - 127) for c in scales]
    return [F.e2m1_value(c) * s[t // fmt.block] for t, c in enumerate(codes)]


def test_e2m1_rounds_to_nearest_even_saturates_and_keeps_the_sign():
    cases = {0.0: 0, 0.25: 0, 0.2500001: 1, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6, 5.0001: 7, 6.0: 7,
             1e30: 7, 0.5: 1, 1.5: 3, 3.0: 5}
    for x, code in cases.items():
        assert F.e2m1(P.f32_bits(x)) == code, x
        assert F.e2m1(P.f32_bits(-x)) == code | 8, -x
    assert [F.e2m1_value(c) for c in range(8)] == [Fraction(v) for v in (0, 0.5, 1, 1.5, 2, 3, 4, 6)]


def test_block_scales():
    assert F.ue8m0_up(P.f32_bits(1.0)) == 127 and F.ue8m0_up(P.f32_bits(1.0000001)) == 128
    assert F.ue8m0_up(P.f32_bits(0.75)) == 127 and F.ue8m0_up(0) == 0 and F.ue8m0_up(1) == 0
    assert F.ue8m0_up(0x00400000) == 0 and F.ue8m0_up(0x00400001) == 1 and F.ue8m0_up(3) == 0   # 2^-127 and below
    assert F.ue8m0_up(0x7F7FFFFF) == 254
    assert F.pow2_word(-127) == 0x00400000 and F.pow2_word(0) == 0x3F800000
    # NVFP4: amax/6 to UE4M3 (nearest even, at least 2^-9), and its FP32 reciprocal; MXFP4: rounded up, exact reciprocal
    assert F.block_scale(F.NVFP4, 0) == (F.UE4M3_MIN, P.f32_bits(512.0))
    assert F.block_scale(F.NVFP4, P.f32_bits(6.0)) == (0x38, 0x3F800000)
    assert F.block_scale(F.NVFP4, P.f32_bits(6000.0))[0] == 0x7E
    s, r = F.block_scale(F.MXFP4, P.f32_bits(7.0))
    assert s == 128 and r == P.f32_bits(0.5)


def test_lines_are_turboshake_nibbles():
    seed = bytes(range(32))
    raw = turboshake128(seed + bytes([2]) + (5).to_bytes(4, "little"), F.D_LINE, 16)
    assert F.line(seed, 2, 5) == [x for b in raw for x in (b & 0xF, b >> 4)]
    assert F.line(seed, 1, 5) != F.line(seed, 2, 5) != F.line(seed, 2, 6)


@pytest.mark.parametrize("fmt", FMTS)
def test_the_noise_atom_adds_the_line_product_before_one_rz_rounding(fmt):
    f = F.FORMATS[fmt]
    x = F.parse_row(V.rows("atom", "A", 1, K)[0], K)
    _, _, alpha, _, ae = F.stats(f, x)
    e, lines = F.line(b"s" * 32, F.ROLE_EA, 0), F.basis(b"t" * 32, F.ROLE_FA, K)
    scale = Fraction(P.fp8_to_f32(ae)) / 16
    for t in range(K):
        c = fp32.mul(alpha, x[t])
        noise = sum(F.e2m1_value(a) * scale * F.e2m1_value(b) for a, b in zip(e, lines[t]))
        assert F.noise_atom(c, e, ae, lines[t]) == rz_f32(Fraction(P.f32_of_bits(c)) + noise)


@pytest.mark.parametrize("fmt", FMTS)
def test_the_clean_up_removes_the_noise_up_to_the_peels_bf16_roundings(fmt):
    """U against the exact rational (A′ − N_A)(B̃ − N_B)ᵀ with N = Ẽ·Fᵀ: the peel factors' single BF16 roundings are
    the only difference; C̃ itself is far from it."""
    s, ctx = V.scheme(fmt, True), V.ctx_of("clean-up", (16, K, 16), 1, 2)
    tile = s.tiles(ctx.matmul.shape)[0]
    a, b = V.rows("clean-up", "A", 16, K), V.rows("clean-up", "B", 16, K)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, tile)
    assert s.admissible(ctx, ops)
    got = s.checked(ctx, ops)
    fa, fb = F.basis(noise.seed_b, F.ROLE_FA, K), F.basis(noise.seed_b, F.ROLE_FB, K)

    def clean(side, e_rows, basis, i):
        sc = Fraction(P.fp8_to_f32(side.ae[i])) / 16
        n = [sum(F.e2m1_value(e) * sc * F.e2m1_value(ln[p]) for p, e in enumerate(e_rows[i])) for ln in basis]
        return value(s.fmt, side.codes[i], side.scales[i]), n

    sides_a = [clean(ops.a, ops.e_a, fa, i) for i in range(16)]
    sides_b = [clean(ops.b, ops.e_b, fb, j) for j in range(16)]
    err_u = err_c = Fraction(0)
    for i, (av, na) in enumerate(sides_a):
        for j, (bv, nb) in enumerate(sides_b):
            exact = sum((p - q) * (r - t) for p, q, r, t in zip(av, na, bv, nb))
            size = sum(abs(p * r) for p, r in zip(av, bv))
            du = abs(Fraction(P.f32_of_bits(got.u[i][j])) - exact)
            assert du <= size / 512
            err_u += du
            err_c += abs(Fraction(P.f32_of_bits(got.c[i][j])) - exact)
    assert err_c > 100 * err_u


@pytest.mark.parametrize("fmt", FMTS)
def test_the_chain_is_exact_on_the_domain(fmt):
    """Every C̃ word is the exact dot product rounded once toward zero: nothing the chain drops depends on the order.
    Rounding therefore binds no word to its order, and the design's defense is salt-dependence (``fp4-design.md`` §6)."""
    s, ctx = V.scheme(fmt, True), V.ctx_of("exact", (16, 1024, 16), 1, 2)
    tile = s.tiles(ctx.matmul.shape)[0]
    ops = s.encode(ctx, s.noise(ctx), V.rows("exact", "A", 16, 1024)[:4], V.rows("exact", "B", 16, 1024)[:4],
                   type(tile)(tile.rows[:4], tile.cols[:4]))
    for ca, sa in zip(ops.a.codes, ops.a.scales):
        av = value(s.fmt, ca, sa)
        for cb, sb in zip(ops.b.codes, ops.b.scales):
            exact = sum(p * q for p, q in zip(av, value(s.fmt, cb, sb)))
            assert F.chain(s.fmt, ca, cb, sa, sb) == rz_f32(exact)


def test_accounting_at_8192():
    s = get("pearl-c-nvfp4-v0")
    shape = Shape(8192, 8192, 8192)
    wref, credit = s.wref(shape), s.credit_of(shape)
    assert wref == shape.wmm + 32 * 8192 ** 2 + 256 * 8192 ** 2 + 117 * 8192 ** 2
    assert credit == shape.wmm + 32 * 8192 ** 2 + 256 * 8192 ** 2 + 98 * 8192 ** 2 == wref - 19 * 8192 ** 2
    assert round(wref / shape.wmm, 3) == 1.049
    gamma = 1 - Fraction(399, 400) * (1 - F.CAP) * Fraction(credit, wref)
    assert round(float(gamma) * 100, 3) == 0.719
    assert s.job_cost(8192, 8192) == 117 * 8192 ** 2 + 32 * 8192 ** 2 + 1024 * 8192 + 16 * 8192 * 1024


def test_domain_and_registry():
    s = get("pearl-c-mxfp4-v0")
    assert isinstance(s, F.PearlC4) and s.fmt is F.MXFP4 and s.name == "pearl-c-mxfp4-v0"
    s.check(Matmul(0, Shape(64, 1024, 64), 0))
    for bad in (Shape(64, 1000, 64), Shape(64, 512, 64), Shape(64, 1 << 17, 64), Shape(60, 1024, 64)):
        with pytest.raises(DomainError):
            s.check(Matmul(0, bad, 0))


def spiked(row: bytes, k: int = K, at: tuple[int, ...] = (1,)) -> bytes:
    """Entries at 256 (well past 12ρ) at unsampled positions: each makes its scale block dead."""
    w = [x >> 16 for x in F.parse_row(row, k)]
    for p in at:
        w[p] = P.f32_to_bf16(256.0)
    return F.bf16_row(w)


@pytest.mark.parametrize("fmt", FMTS)
def test_row_rules_and_debit(fmt):
    f = F.FORMATS[fmt]
    rows = [F.parse_row(r, 1024) for r in V.rows("rules", "A", 3, 1024)]
    assert all(F.row_passes(f, x) for x in rows) and not any(any(F.dead_blocks(f, x)) for x in rows)
    assert not F.row_passes(f, [0] * 1024)                      # a_E rounds to zero
    one = F.parse_row(spiked(V.rows("rules", "A", 1, 1024)[0], 1024), 1024)
    two = F.parse_row(spiked(V.rows("rules", "A", 1, 1024)[0], 1024, (1, 2 * f.block + 1)), 1024)
    assert F.row_passes(f, one) == (fmt == "nvfp4")             # one dead block in 64 (nvfp4), in 32 (mxfp4)
    assert sum(F.dead_blocks(f, one)) == 1 and not F.row_passes(f, two)
    e = [F.line(b"\x02" * 32, F.ROLE_EA, i) for i in range(4)]
    a = F.form(f, [one] + rows, e, F.basis(b"\x01" * 32, F.ROLE_FA, 1024))
    b = F.form(f, [one] + rows, e, F.basis(b"\x01" * 32, F.ROLE_FB, 1024))
    clean = F.tile_debit(f, a, b, [1, 2, 3], [1, 2, 3], 4096)
    assert clean.counts["salt_dead_a"] == clean.counts["salt_dead_b"] == 0
    # forming only at positions where every credited B code is zero (their A elements are unneeded), 3 rows x 3 columns'
    # share; MXFP4 chains hold identity atoms (theory §2: about 0.21% on Gaussian inputs), NVFP4's here none
    assert clean.forming == Fraction(F.FORMING_CREDITED * 3 * clean.counts["unneeded_positions"] * 3, 4096)
    assert clean.total == clean.forming + clean.identity and (clean.identity == 0) == (fmt == "nvfp4")
    # the spike saturates its block at code 6 and the block's small codes round to 0 on every draw: salt-dead (x[2] at
    # 3ρ reaches code 0.5 on the extreme draw, so it stays live)
    dead = F.salt_dead(f, one, a.alpha[0], a.ae[0])
    assert dead[1] and sum(dead[:f.block]) >= f.block - 2 and not any(dead[f.block:])
    d = F.tile_debit(f, a, b, [0, 1], [0, 1], 4096)
    assert d.dead_pairs == f.block and d.two_four == 4 * F.two_four_units(dead) and d.identity == d.subgrid == d.windows == 0
    assert d.counts["forming_elements"] >= f.block
    assert d.forming == Fraction(F.FORMING_CREDITED * d.counts["forming_elements"] * 2, 4096)
    assert d.total == d.dead_pairs + d.two_four + d.forming


def test_word_replay_charges_identity_atoms_and_sub_grid_blocks():
    """An atom whose products cancel is an identity atom; after an atom at the largest scales, a block at the smallest
    scale lies below the accumulator's grid (it adds exactly 0), and its atom is then an identity atom too."""
    f = F.NVFP4
    one, minus = [0x2] * 16 + [0] * 48, [0x2] * 8 + [0xA] * 8 + [0] * 48
    a, b = one + one + one, one + minus + one
    assert F.word_replay(f, a, b, [0x38] * 12, [0x38] * 12) == (F.chain(f, a, b, [0x38] * 12, [0x38] * 12), 1, 0)
    top, low = [0x7] * 64, [0x2] * 16 + [0] * 48
    word, identity, subgrid = F.word_replay(f, top + low, top + low, [0x7E] * 4 + [0x01] * 4, [0x7E] * 4 + [0x01] * 4)
    assert (identity, subgrid) == (1, 4) and word == F.chain(f, top, top, [0x7E] * 4, [0x7E] * 4)
    assert F.two_four_windows([0x1, 0x9, 0, 0] * 32 + [0x1] * 128) == [True, False]


def lifecycle(s: F.PearlC4, real=20):
    """32 x 128 x 16 on 16 x 16 tiles: A's rows 20 … 31 are zero filler and row 3 has a dead block; B passes."""
    rows_a = V.rows("work-life", "A", real, K) + [F.bf16_row([0] * K)] * (32 - real)
    rows_a[3] = spiked(rows_a[3])
    rows_b = V.rows("work-life", "B", 16, K)
    work = Workload((Matmul(3, Shape(32, K, 16), 9),))
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.start(work, prover.weight_roots, b"beacon")
    return epoch, prover, prover.prove(epoch, {3: rows_a}), rows_a, rows_b


def test_the_work_law_audits_pearl_c4():
    s = V.scheme("nvfp4", True)
    epoch, prover, tr, rows_a, rows_b = lifecycle(s)
    splits = {9: W.Split(K)}
    records = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits, real_rows={3: 20})
    assert records == [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), ())]
    out = W.audit(s, epoch, prover, tr, splits, records, BEFORE, AFTER, 8)
    assert out.audit.accepted and out.job == s.job_cost(16, K)
    assert out.wref == W.cell_wref(s, records[0]) * 19 * 16
    hides = W.audit(s, epoch, prover, tr, splits, [W.UnitRecord(epoch.workload.matmuls[0], 20, K)], BEFORE, AFTER, 8)
    assert not hides.audit.accepted
    assert any("A row 3 is credited but fails" in v.reason for v in hides.audit.verdicts if not v.ok)


def test_pinned_vectors_fast():
    pinned = json.loads(V.OUT.read_text())
    cases = [c for c in V.CASES if c[5]]
    assert [c["case"] for c in pinned["cases"] if c["test_instance"]] == [c[0] for c in cases]
    for case, want in zip(cases, (c for c in pinned["cases"] if c["test_instance"])):
        assert V.case_vector(*case) == want


@pytest.mark.slow
def test_pinned_vectors_production_tile():
    pinned = json.loads(V.OUT.read_text())
    for case, want in zip(V.CASES, pinned["cases"]):
        if not case[5]:
            assert V.case_vector(*case) == want
