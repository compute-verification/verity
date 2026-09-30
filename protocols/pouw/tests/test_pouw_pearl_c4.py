"""``pearl-c-nvfp4-v0`` and ``pearl-c-mxfp4-v0`` (`pearl_c4`): the casts, the noise lines, the noise atom, the clean-up's
algebra, the chain's exactness on the domain (which the design's argument turns on), the accounting, the per-row rules and
debit, the work law through `pearl_c_work`, and the pinned vectors."""

from __future__ import annotations

import base64
import json
import pickle
from fractions import Fraction

import pytest

import pearl_c4_vectors as V
from verity.commitments.turboshake import turboshake128
from verity.ml.tc import fp32
from verity_pouw.audit import Epoch, Prover, Workload
from verity_pouw.protocol import DomainError, Matmul, Shape
from verity_pouw.schemes import get
from verity_pouw.schemes import pearl_c4 as F
from verity_pouw.schemes import pearl_c4_replay as R
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


def spiked(row: bytes, k: int = K, at: tuple[int, ...] = (1,), value: float = 256.0) -> bytes:
    """Entries at `value` at unsampled positions: at 256 (well past 9.5ρ) each screens its scale block; at 2^16 the row's
    a_E is subnormal (D-NF)."""
    w = [x >> 16 for x in F.parse_row(row, k)]
    for p in at:
        w[p] = P.f32_to_bf16(value)
    return F.bf16_row(w)


@pytest.mark.parametrize("fmt", FMTS)
def test_dead_screen_is_90_rho_squared(fmt):
    """ρ = 1 (every 8th position at 1.0): a block whose |max| is 9.4375 is live, 9.5 (just past √90) and 10 (live under the
    earlier 10ρ) screened."""
    f = F.FORMATS[fmt]

    def row(v):
        x = [P.f32_bits(1.0) if p % 8 == 0 else 0 for p in range(1024)]
        x[1] = P.f32_bits(v)
        return x
    assert F.DEAD_RHO_SQ == 90 and ("dead_rho_sq", 90) in F.PearlC4(fmt).certificate.parameters
    assert [F.dead_blocks(f, row(v))[0] for v in (9.4375, 9.5, 10.0)] == [False, True, True]
    assert not any(F.dead_blocks(f, row(10.0))[1:])


@pytest.mark.parametrize("fmt", FMTS)
def test_row_rules_and_debit(fmt):
    f = F.FORMATS[fmt]
    rows = [F.parse_row(r, 1024) for r in V.rows("rules", "A", 3, 1024)]
    assert all(F.row_passes(f, x) for x in rows) and not any(any(F.dead_blocks(f, x)) for x in rows)
    assert not F.row_passes(f, [0] * 1024)                      # a_E rounds to zero
    one = F.parse_row(spiked(V.rows("rules", "A", 1, 1024)[0], 1024), 1024)
    two = F.parse_row(spiked(V.rows("rules", "A", 1, 1024)[0], 1024, (1, 2 * f.block + 1)), 1024)
    # screened blocks no longer fail the row (the one-in-64 cap is gone): the debit charges them
    assert sum(F.dead_blocks(f, one)) == 1 and sum(F.dead_blocks(f, two)) == 2 and F.row_passes(f, one) and F.row_passes(f, two)
    assert not F.row_passes(f, F.parse_row(spiked(V.rows("rules", "A", 1, 1024)[0], 1024, value=2.0 ** 16), 1024))
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
    unneeded = [all(b.codes[j][t] & 7 == 0 for j in (0, 1)) for t in range(1024)]
    witness = sum(p or u for i in (0, 1) for p, u in zip(F.salt_dead(f, a.x[i], a.alpha[i], a.ae[i]), unneeded))
    assert d.counts["screened_blocks_a"] == d.counts["screened_blocks_b"] == 1 and d.counts["forming_elements"] == witness
    assert d.forming == Fraction(F.FORMING_CREDITED * witness * 2, 4096)
    assert d.total == d.dead_pairs + d.two_four + d.forming


def screened_tile(blocks: int, b_blocks: int = 0):
    """A 4 x 4 credited tile of a 4 x 1024 x 1024 NVFP4 unit whose A row 0 has `blocks` screened blocks and B row 0 its
    first `b_blocks` of them, each by one element at 9.75ρ, which leaves the block's other codes salt-live: (scheme,
    record, tile, A rows, B rows, ops, x_0)."""
    s, k = V.scheme("nvfp4", False), 1024
    ctx = V.ctx_of("screen", (4, k, 1024), 0, 0)
    a_rows, b_rows = V.rows("screen", "A", 4, k), V.rows("screen", "B", 4, k)
    for rows, count in ((a_rows, blocks), (b_rows, b_blocks)):
        rho = float(F._exact_rho_sq(F.parse_row(rows[0], k))) ** 0.5
        rows[0] = spiked(rows[0], k, tuple(16 * b + 1 for b in range(count)), 9.75 * rho)
    tile = type(s.tiles(Shape(64, k, 64))[0])(range(4), range(4))
    ops = s.encode(ctx, s.noise(ctx), a_rows, b_rows, tile)
    return s, W.UnitRecord(ctx.matmul, 4, k), tile, a_rows, b_rows, ops, F.parse_row(a_rows[0], k)


def witness_elements(s, ops) -> int:
    """Item 4's elements over the 4 x 4 tile: the witness's salt-dead A elements and the unneeded positions' A elements."""
    unneeded = [all(ops.b.codes[j][t] & 7 == 0 for j in range(4)) for t in range(1024)]
    return sum(p or u for i in range(4)
               for p, u in zip(F.salt_dead(s.fmt, ops.a.x[i], ops.a.alpha[i], ops.a.ae[i]), unneeded))


@pytest.mark.parametrize("blocks", (2, 16))
def test_a_row_the_old_cap_rejected_is_admitted_and_debited_on_the_witness(blocks):
    """Screened blocks past D-SS's retired one-in-64 cap, on A alone: the row is credited, item 4 charges only the
    witness's salt-dead elements (not every element of a screened block), no block pair is screened, and the tile stays
    under ρ (server.md 11:32Z)."""
    s, record, tile, a_rows, b_rows, ops, x = screened_tile(blocks)
    assert sum(F.dead_blocks(s.fmt, x)) == blocks > (len(x) // s.fmt.block) // 64 and F.row_passes(s.fmt, x)
    assert sum(F.salt_dead(s.fmt, x, ops.a.alpha[0], ops.a.ae[0])[:blocks * s.fmt.block]) <= blocks
    assert W.check_opened(s, record, tile, a_rows, b_rows, ops) is None
    d = s.last_debit
    assert d.counts["screened_blocks_a"] == blocks and d.dead_pairs == 0
    assert d.counts["forming_elements"] == witness_elements(s, ops) < blocks * s.fmt.block
    assert d.forming == Fraction(F.FORMING_CREDITED * witness_elements(s, ops) * 4, 1024)
    assert 0 < d.total <= s.tile_cap(record, ops, range(4), range(4))[1]


def test_blocks_both_operands_screen_push_the_tile_past_rho_by_item_3():
    """Four blocks screened on A row 0 and B row 0 alike: 4 pairs on word (0, 0), 16 units each, pass the tile's cap
    (16 cells' share of creditOf at ρ = 1/400, about 56 units), so the tile is rejected; the same four on A alone pass."""
    s, record, tile, a_rows, b_rows, ops, x = screened_tile(4, 4)
    reason = W.check_opened(s, record, tile, a_rows, b_rows, ops)
    assert reason is not None and reason.startswith("the replayed debit") and "exceeds the cap" in reason
    d, cap = s.last_debit, s.tile_cap(record, ops, range(4), range(4))[1]
    assert d.counts["dead_pairs"] == 4 and d.dead_pairs == 4 * s.fmt.block and d.total > cap
    assert d.counts["forming_elements"] == witness_elements(s, ops)
    s, record, tile, a_rows, b_rows, ops, x = screened_tile(4)
    assert W.check_opened(s, record, tile, a_rows, b_rows, ops) is None and s.last_debit.dead_pairs == 0


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


def test_scale_bytes_have_one_encoding_and_no_zero():
    """sm_120 reads a UE4M3 scale from its low 7 bits (the assessor's run of all 256 bytes), so every replayed step does too;
    D-SB rejects bit 7 (a second encoding of the same C̃), a zero scale (a noise block's products zeroed) and 0x7F (NaN), and
    honest forming writes none: its bytes over an amax sweep from 0 to FP32's largest stay in [0x01, 0x7E]."""
    f = F.NVFP4
    assert all(F.scale_dyadic(f, 0x80 | b) == F.scale_dyadic(f, b) for b in range(0x7F))
    codes, other = list(range(16)) * 8, [(3 * c + 5) % 16 for c in range(128)]
    scales = [0x38, 0x01, 0x7E, 0x30, 0x40, 0x10, 0x07, 0x39]
    high = [0x80 | s for s in scales]
    assert F.chain(f, codes, other, high, scales) == F.chain(f, codes, other, scales, scales) != 0
    assert F.word_replay(f, codes, other, scales, high) == F.word_replay(f, codes, other, scales, scales)
    assert F.scale_bytes_ok(f, range(0x01, 0x7F))
    assert not any(F.scale_bytes_ok(f, [b]) for b in [0x00, 0x7F, *range(0x80, 0x100)])
    assert F.scale_bytes_ok(F.MXFP4, range(0xFF)) and not F.scale_bytes_ok(F.MXFP4, [0xFF])
    sweep = [0, 1, 0x00400000, *range(0x00800000, 0x7F800000, 0x00100001), 0x7F7FFFFF]
    assert {F.block_scale(f, w)[0] for w in sweep} <= set(range(0x01, 0x7F))
    assert F.scale_bytes_ok(F.MXFP4, [F.block_scale(F.MXFP4, w)[0] for w in sweep])
    counts = R.device_scales(f, bytes([0x38, 0x00, 0x80, 0xB8, 0x7F, 0x01]))
    assert counts == {"bytes": 6, "outside_range": 4, "min": 0, "max": 0xB8, "bit7": 2, "zero": 1, "nan": 1}


def scales_side() -> F.Formed:
    return F.form(F.NVFP4, [F.parse_row(r, K) for r in V.rows("scales", "A", 4, K)],
                  [F.line(b"e" * 32, 0, i) for i in range(4)], F.basis(b"f" * 32, F.ROLE_FA, K))


@pytest.mark.parametrize("edit", [lambda s: s.scales[2].__setitem__(1, 0x80 | s.scales[2][1]),
                                  lambda s: s.ae.__setitem__(1, 0x80 | s.ae[1]),
                                  lambda s: s.scales[2].__setitem__(1, 0x00),
                                  lambda s: s.scales[2].__setitem__(1, 0x7F),
                                  lambda s: s.ae.__setitem__(1, 0x7F)],
                         ids=["scale-bit-7", "a_E-bit-7", "scale-0x00", "scale-0x7F", "a_E-0x7F"])
def test_d_sb_rejects_a_scale_byte_outside_0x01_to_0x7E(edit):
    side = scales_side()
    F.check_scale_bytes(F.NVFP4, side)
    edit(side)
    with pytest.raises(P.Reject):
        F.check_scale_bytes(F.NVFP4, side)


def test_d_sb_leaves_a_zero_rows_a_e_to_d_nf():
    """A zero row (decode's filler) forms a_E = 0x00 and block scales 0x01: D-SB admits the side, D-NF excludes the row."""
    side = F.form(F.NVFP4, [[0] * K], [F.line(b"e" * 32, 0, 0)], F.basis(b"f" * 32, F.ROLE_FA, K))
    assert side.ae == [0] and set(side.scales[0]) == {F.UE4M3_MIN}
    F.check_scale_bytes(F.NVFP4, side)
    assert not F.row_passes(F.NVFP4, [0] * K)


@pytest.mark.parametrize("word", [0x7FC0, 0x7F80, 0xFF80], ids=["nan", "inf", "-inf"])
def test_d_sb_rejects_a_non_finite_input_word(word):
    """A NaN or ±inf activation word: the row is malformed on the tile rule, and the verifier forms no operands from it."""
    s, ctx = V.scheme("nvfp4", True), V.ctx_of("finite", (16, K, 16), 1, 2)
    tile = s.tiles(ctx.matmul.shape)[0]
    a, b = V.rows("finite", "A", 16, K), V.rows("finite", "B", 16, K)
    w = [x >> 16 for x in F.parse_row(a[5], K)]
    w[7] = word
    a[5] = F.bf16_row(w)
    with pytest.raises(P.Reject):
        F.parse_row(a[5], K)
    assert s.encode(ctx, s.noise(ctx), a, b, tile) is None
    record = W.UnitRecord(ctx.matmul, 16, K)
    assert W.check_opened(s, record, tile, a, b, None) == "A row 5 is malformed: non-finite FP32 word"


def lifecycle(s: F.PearlC4, real=20):
    """32 x 128 x 16 on 16 x 16 tiles: A's rows 20 … 31 are zero filler and row 3 fails D-NF; B passes."""
    rows_a = V.rows("work-life", "A", real, K) + [F.bf16_row([0] * K)] * (32 - real)
    rows_a[3] = spiked(rows_a[3], value=2.0 ** 16)
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


def device_dump(path, s: F.PearlC4, swap_leaves=False, flip_root_a=False) -> None:
    """`lifecycle`'s run as a device would dump it (its roots and leaves), optionally with its first two leaves swapped
    or A's root altered."""
    epoch, prover, tr, rows_a, rows_b = lifecycle(s)
    leaves = [prover.digest(i) for i in range(2)]
    root_a = tr.activation_roots[0]
    call = {"scheme": s.name, "shape": [32, K, 16], "index": 3, "weight": 9, "real_rows": 20, "epoch_beacon": b"beacon".hex()}
    R.write_dump(path, call, b"".join(rows_a), b"".join(rows_b), bytes([root_a[0] ^ 1]) + root_a[1:] if flip_root_a else root_a,
                 prover.weight_roots[9], b"".join(leaves[::-1] if swap_leaves else leaves))


def test_replay_accepts_the_device_transcript(tmp_path):
    s = V.scheme("nvfp4", True)
    device_dump(tmp_path, s)
    (tmp_path / "scales_b.bin").write_bytes(bytes([0x38, 0x01, 0x7E]))
    t = json.loads(json.dumps(R.prove(tmp_path, "r-test", 8, scheme=s)))
    assert list(t["device_scales"]) == ["b"] and t["device_scales"]["b"]["outside_range"] == 0
    assert t["records"] == [{"real_rows": 20, "k_kept": K, "excluded_a": [3], "excluded_b": []}]
    assert t["host_roots_match"] == {"a": True, "b": True} and [o["tile"] for o in t["openings"]] == [0, 1]
    rep = R.verify(t, scheme=s)
    assert rep["accepted"] and rep["opened"] == 9 and rep["reason"] == ""
    first = rep["tiles"][0]
    assert first["credited"] in ([15, 16], [4, 16]) and Fraction(first["debit"]["total"]) <= Fraction(first["cap"])
    assert pickle.loads(pickle.dumps(s)).fmt is s.fmt and R.verify(t, workers=2, scheme=s)["tiles"] == rep["tiles"]
    assert R.accept_line(rep, "r-test:t.json", "00", "abc").startswith(f"ACCEPT {R.REPORT} {s.name} r-test:t.json sha256:00 ")


def test_replay_rejects(tmp_path):
    """A wrong leaf, a root the rows don't have, a tampered row, a beacon not the commitment's, hidden exclusions, and a
    missing opening."""
    s = V.scheme("nvfp4", True)
    for name, kw, why in (("swap", {"swap_leaves": True}, "checked values"),
                          ("root", {"flip_root_a": True}, "activation opening")):
        device_dump(tmp_path / name, s, **kw)
        rep = R.verify(R.prove(tmp_path / name, "r-test", 8, scheme=s), scheme=s)
        assert not rep["accepted"] and rep["reason"] == why
    device_dump(tmp_path / "ok", s)
    good = R.prove(tmp_path / "ok", "r-test", 8, scheme=s)

    def bad(edit):
        t = json.loads(json.dumps(good))
        edit(t)
        return R.verify(t, scheme=s)

    def tamper(t):
        row = bytearray(base64.b64decode(t["openings"][0]["a"][0]["value"]))
        row[0] ^= 1
        t["openings"][0]["a"][0]["value"] = base64.b64encode(bytes(row)).decode()

    def hide(t):
        t["records"][0]["excluded_a"] = []
        unit = Matmul(3, Shape(32, K, 16), 9)
        epoch = Epoch.start(Workload((unit,)), {9: bytes.fromhex(t["weight_roots"]["9"])}, b"beacon")
        rec = W.UnitRecord(unit, 20, K)
        beacon = R.draw_beacon(bytes.fromhex(t["transcript"]["root"]),
                               W.records_root(epoch.salt, epoch.weights_root, {9: W.Split(K)}, [rec]))
        t["draw"]["value"] = beacon.value.hex()

    assert bad(tamper)["reason"] == "activation opening"
    assert bad(lambda t: t["draw"].update(value="00" * 32))["reason"].startswith("the draws' beacon is not")
    assert "A row 3 is credited but fails" in bad(hide)["reason"]
    assert bad(lambda t: t["openings"].pop(0))["reason"] == "'the transcript does not open tile 0'"


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
