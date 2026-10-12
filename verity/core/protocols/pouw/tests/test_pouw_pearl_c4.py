"""``pearl-c-nvfp4-v0`` (`pearl_c4`), with MXFP4 as its contrast line: the casts, the noise lines, the noise atom, the clean-up's
algebra, the chain's exactness on the domain (which the design's argument turns on), the accounting, the per-row rules and
debit, the work law through `pearl_c_work`, and the pinned vectors."""

from __future__ import annotations

import base64
import copy
import hashlib
import itertools
import json
import math
import pickle
import random
import sys
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

import epoch_coins as E

import pearl_c4_vectors as V
from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon import fp32
from verity.core.protocols.pouw.audit import Epoch, Prover, Workload
from verity.core.protocols.pouw.protocol import Context, DomainError, Matmul, Shape
from verity.core.protocols.pouw.schemes import get
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c4 as F
from verity.core.protocols.pouw.schemes import pearl_c_work as W
from verity.core.protocols.pouw.schemes import pearl_kw as P

# the device-run replay is a benchmark's (benchmarks/pouw/pearl_c4), imported by name so that its forkserver workers find it
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "benchmarks" / "pouw" / "pearl_c4"))
import replay as R  # noqa: E402

K = 128
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


def test_nvfp4_blocks_keep_the_sign_of_minus_zero():
    """`fp32.mul(−0, r)` is −0, so a −0 word's code is 0x8 and a +0 word's 0x0. The Lean's `nvf4Block` pins the same
    block (`DeviceFp4Gamma.nvf4Block_signedZero`)."""
    d = [0x80000000, P.f32_bits(1.0)] + [0] * 13 + [0x80000000]
    assert F.quantize(F.NVFP4, d) == ([8, 7] + [0] * 13 + [8], [0x23])


def test_lines_are_sha512_nibbles_four_to_a_block():
    """Line t is quarter t mod 4 of SHA-512 over the padded "pearl-c4/v2/line" block, the seed, the role and ⌊t/4⌋ u32 LE,
    low nibble first; the pinned bytes are line 5 of role 2 under the seed 0, 1, ..., 31."""
    seed = bytes(range(32))
    block = hashlib.sha512(b"pearl-c4/v2/line\0".ljust(128, b"\0") + seed + bytes([2]) + (1).to_bytes(4, "little")).digest()
    raw = bytes.fromhex("ccba50373990aeaffa1ad59bbf454387")
    assert block[16:32] == raw and F.line_block(seed, 2, 1) == block
    assert F.line(seed, 2, 5) == [x for b in raw for x in (b & 0xF, b >> 4)]
    assert [F.line(seed, 2, t) for t in range(4, 8)] == [[x for b in block[16 * q:16 * q + 16] for x in (b & 0xF, b >> 4)]
                                                         for q in range(4)]
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
    Rounding therefore binds no word to its order, and the design's defense is salt-dependence."""
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
    """Pearl-C4 claims NVFP4 only (Daniel, 15:34Z): MXFP4 stays constructible as the contrast line, unregistered and with a
    certificate that claims nothing."""
    s = get("pearl-c-nvfp4-v0")
    assert isinstance(s, F.PearlC4) and s.fmt is F.NVFP4 and not s.contrast and "tt-out/fp4-sm120" in s.certificate.claims
    with pytest.raises(KeyError):
        get("pearl-c-mxfp4-v0")
    mx = F.PearlC4("mxfp4")
    assert mx.contrast and mx.name == "pearl-c-mxfp4-v0" and mx.certificate.claims == () and mx.certificate.gamma0 is None
    s.check(Matmul(0, Shape(64, 1024, 128), 0))
    s.check(Matmul(0, Shape(64, 1024, 1 << 18), 0))
    # n = 64, an MQA unit of one 64-wide KV head, would need B-OVF's 5.7% and is not claimed
    for bad in (Shape(64, 1000, 128), Shape(64, 512, 128), Shape(64, 1 << 17, 128), Shape(60, 1024, 128),
                Shape(64, 1024, 1 << 19), Shape(64, 1024, 64)):
        with pytest.raises(DomainError):
            s.check(Matmul(0, bad, 0))
    assert s.n_min == F.B_OVF_MIN_N == 128 and ("n_min", 128) in s.certificate.parameters
    V.scheme("nvfp4", True).check(Matmul(0, Shape(16, 128, 16), 0))


def test_vo_rotation_and_head_interleave_register_inside_the_8_block_rotation_only():
    """Daniel's 5:52 PM PDT ruling: the keyed V/O rotation plus the head interleave register inside the keyed 8-block
    rotation, and the same transforms on an unrotated checkpoint are refused, curated or not."""
    s = get("pearl-c-nvfp4-v0")
    vo = (F.VO_ROTATION, F.HEAD_INTERLEAVE)
    for curated in (False, True):
        s.check_registration((F.ROTATION_8, *vo), curated)
        for unrotated in (vo, vo[:1], vo[1:]):
            with pytest.raises(DomainError, match="never on an unrotated checkpoint"):
                s.check_registration(unrotated, curated)


def test_registered_weights_take_the_8_block_rotation_or_the_curated_list():
    """The keyed 8-block rotation is the registered-weights rule, and an unrotated checkpoint registers only from the curated
    list (Daniel, 1:36 PM PDT).  Only the adopted stacks pass: V/O and the interleave together and after the rotation."""
    s = get("pearl-c-nvfp4-v0")
    s.check_registration((F.ROTATION_8,))
    s.check_registration((), curated=True)
    with pytest.raises(DomainError, match="curated list"):
        s.check_registration(())
    for bad in ((F.ROTATION_8, F.VO_ROTATION), (F.ROTATION_8, F.HEAD_INTERLEAVE),
                (F.ROTATION_8, F.HEAD_INTERLEAVE, F.VO_ROTATION), (F.VO_ROTATION, F.ROTATION_8, F.HEAD_INTERLEAVE),
                (F.ROTATION_8, F.ROTATION_8), (F.ROTATION_8, "an-unnamed-transform"), (F.GAMMA_FOLD, F.ROTATION_8),
                (F.ROTATION_8, F.GAMMA_FOLD), (F.GAMMA_FOLD, F.GAMMA_FOLD), ("an-unnamed-transform",)):
        with pytest.raises(DomainError, match="not an adopted stack"):
            s.check_registration(bad, curated=True)


def test_a_checkpoint_that_needs_gamma_folded_registers_only_with_it():
    """70B registrations fold γ (the cast moves Llama-3.1-70B's layer 0 update by 158% unfolded).  The rotation folds γ,
    so its stacks pass as they are; an unrotated checkpoint declares `GAMMA_FOLD`, and is still curated only.  A folded
    but unrotated checkpoint takes no V/O or interleave."""
    s = get("pearl-c-nvfp4-v0")
    vo = (F.VO_ROTATION, F.HEAD_INTERLEAVE)
    for stack in F.REGISTRATION_STACKS:
        s.check_registration(stack, needs_gamma_fold=True)
    s.check_registration((F.GAMMA_FOLD,), curated=True, needs_gamma_fold=True)
    s.check_registration((F.GAMMA_FOLD,), curated=True)
    with pytest.raises(DomainError, match="only with γ folded"):
        s.check_registration((), curated=True, needs_gamma_fold=True)
    with pytest.raises(DomainError, match="curated list"):
        s.check_registration((F.GAMMA_FOLD,), needs_gamma_fold=True)
    for curated in (False, True):
        with pytest.raises(DomainError, match="never on an unrotated checkpoint"):
            s.check_registration((F.GAMMA_FOLD, *vo), curated, needs_gamma_fold=True)


def spiked(row: bytes, k: int = K, at: tuple[int, ...] = (1,), value: float = 256.0) -> bytes:
    """Entries at `value` at unsampled positions: at 256 (well past 9.5ρ) each screens its scale block; at 2^16 the row's
    a_E is subnormal (D-NF)."""
    w = [x >> 16 for x in F.parse_row(row, k)]
    for p in at:
        w[p] = P.f32_to_bf16(value)
    return F.bf16_row(w)


def test_dnf_rejects_a_negative_beta_whose_byte_would_be_zero_noise():
    """D-NF with bc-824e54a2's two fixes: 0 ≤ β, and a_E's E4M3 byte v < 128.  At β = −1/4096 the cast gives 0x80, which
    a check of 8 ≤ v alone admits and sm_120 reads from its low 7 bits as a zero scale, so the row's noise would be zero.
    A larger negative β (−1.0, byte 0xB8) would read as +1.0; D-NF rejects both.  The pinned stats never make β negative,
    and every honest row's verdict is unchanged."""
    neg = P.f32_bits(-1 / 4096)
    assert P.f32_to_fp8(P.f32_of_bits(neg)) == 0x80 and F.scale_dyadic(F.NVFP4, 0x80)[0] == 0
    assert not F.dnf(neg) and not F.dnf(P.f32_bits(-1.0)) and not F.dnf(P.f32_bits(-0.0))
    assert F.dnf(P.f32_bits(2.0 ** -6)) and F.dnf(P.f32_bits(448.0)) and F.dnf(P.f32_bits(1e30))
    assert not F.dnf(P.f32_bits(2.0 ** -7)) and not F.dnf(P.f32_bits(0.0))
    # a NaN β with its sign clear (the hardware's cast gives 0x7F) and ±inf: the accepted bytes are Lean's, 0x08–0x7E
    for word in (0x7FC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0xFFC00000):
        assert not F.dnf(word)
    finite = [v for v in range(256) if v & 0x7F != 0x7F]
    assert [v for v in finite if F.dnf(P.f32_bits(P.fp8_to_f32(v)))] == list(range(0x08, 0x7F))
    for fmt in FMTS:
        f = F.FORMATS[fmt]
        rows = [F.parse_row(r, 1024) for r in V.rows("dnf", "A", 8, 1024)] + [[0] * 1024]
        rows.append(F.parse_row(spiked(V.rows("dnf", "A", 1, 1024)[0], 1024, value=2.0 ** 16), 1024))
        for x in rows:
            beta, ae = F.stats(f, x)[3:]
            assert not beta >> 31
            assert F.row_passes(f, x) == (ae >= F.UE4M3_NORMAL)


@pytest.mark.parametrize("fmt", FMTS)
def test_row_passes_asks_finite_widened_bf16_words(fmt):
    """`row_passes` refuses a row with a word whose low 16 bits are not zero (`0x3F800001`, 1 + 2⁻²³, finite and not a
    widened BF16 word) and a row with a non-finite word, which `parse_row` never writes, as the Lean's
    `Fp4Sem.RowOK` and `OperandOK` refuse them (`rowOK4_notBf16`, `rowOK4_nonFinite`); the honest row passes."""
    f = F.FORMATS[fmt]
    x = F.parse_row(V.rows("rules", "A", 1, 1024)[0], 1024)
    assert F.row_passes(f, x)
    assert not F.row_passes(f, [0x3F800001 if p == 1 else w for p, w in enumerate(x)])
    assert not F.row_passes(f, [0x7F800000 if p == 1 else w for p, w in enumerate(x)])


def tolerant_kernel(scheme, ctx, noise, a, b, tiles):
    """`reference_kernel`, with a zero digest for a tile the scheme does not encode, so a weight with a refused row
    still commits."""
    out = []
    for t in tiles:
        ops = scheme.encode(ctx, noise, [a[i] for i in t.rows], [b[j] for j in t.cols], t)
        if ops is None:
            out.append((bytes(32), [[0.0] * len(t.cols) for _ in t.rows]))
            continue
        c = scheme.checked(ctx, ops)
        out.append((scheme.digest(ctx, noise, c), scheme.useful(ctx, noise, ops, c)))
    return out


@pytest.mark.parametrize("clause", [None, "finite", "admit"])
def test_check_tile_refuses_a_weight_row_outside_the_rules(clause):
    """`check_tile` refuses a tile one of whose weight rows fails a clause of `OperandOK`, as Lean's `TileCheck.Accepts`
    asks each of the tile's weight rows to pass the protocol's `weightOK` (`Fp4Sem.RowOK`). One row per clause: a
    non-finite word (Lean's `rowOK4_nonFinite`), and every 8th word `2¹²⁶`, which `RowAdmit4` refuses since β is not
    finite (`pc4Sem_squaresOverflow`). The word clause's widened-BF16 half holds of every opened row, which is BF16 words
    (`parse_row`); `test_row_passes_asks_finite_widened_bf16_words` checks it on `row_passes`. The tile beside it, whose
    weight rows pass, is accepted, and so are both on the honest weight."""
    from verity.core.protocols.pouw.audit import Verifier

    s = V.scheme("nvfp4", True)
    rows_a, rows_b = V.rows("weight-rule", "A", 16, K), V.rows("weight-rule", "B", 32, K)
    w = [x >> 16 for x in F.parse_row(rows_b[0], K)]
    if clause == "finite":
        rows_b[0] = F.bf16_row([0x7F80 if t == 1 else v for t, v in enumerate(w)])
    elif clause == "admit":
        rows_b[0] = F.bf16_row([0x7E80 if t % 8 == 0 else v for t, v in enumerate(w)])
    prover = Prover(s, {9: rows_b}, tolerant_kernel)
    epoch = Epoch.unverified(Workload((Matmul(3, Shape(16, K, 32), 9),)), prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    v = Verifier(s, epoch)
    assert [0 in r.tile.cols for r in v.refs] == [True, False]
    verdicts = [v.check_tile(tr, prover.open(epoch, r.id)) for r in v.refs]
    assert [(x.ok, x.reason) for x in verdicts] == [(clause is None, "" if clause is None else "inadmissible inputs"),
                                                    (True, "")]


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
    clean = F.tile_debit(f, a, b, [1, 2, 3], [1, 2, 3], 4096, e, e)
    assert clean.counts["salt_dead_a"] == clean.counts["salt_dead_b"] == 0 and clean.int8 == 0
    # F1′: NVFP4's scales move with the salt, so a Gaussian window's saving survives κ only on rare draws (a trace, under
    # 10^-4 units against a cap of 29.5); MXFP4's UE8M0 scales don't, so its windows keep it (about 6% of this tile's cap
    # with B̃'s overfit factor; MXFP4 is contrast only)
    assert 0 <= clean.split < (Fraction(1, 10 ** 4) if fmt == "nvfp4" else 3)
    # forming only at positions where every credited B code is zero (their A elements are unneeded), 3 rows x 3 columns'
    # share; MXFP4 chains hold identity atoms on some draws (about 0.21% of atoms on Gaussian inputs), NVFP4's here none
    assert clean.forming == Fraction(F.FORMING_CREDITED * 3 * clean.counts["unneeded_positions"] * 3, 4096)
    assert clean.total == clean.forming + clean.identity + clean.split and (fmt != "nvfp4" or clean.identity == 0)
    # the spike saturates its block at code 6 and the block's small codes round to 0 on every draw: salt-dead (x[2] at
    # 3ρ reaches code 0.5 on the extreme draw, so it stays live)
    dead = F.salt_dead(f, one, a.alpha[0], a.ae[0])
    assert dead[1] and sum(dead[:f.block]) >= f.block - 2 and not any(dead[f.block:])
    d = F.tile_debit(f, a, b, [0, 1], [0, 1], 4096, e, e)
    dead_b = F.salt_dead(f, b.x[0], b.alpha[0], b.ae[0])
    assert d.counts["screened_block_pairs"] == 1 and d.identity == d.subgrid == d.windows == 0
    assert 0 < d.dead_pairs == d.counts["dead_pairs"] == sum(p and q for p, q in zip(dead, dead_b)) <= f.block
    unneeded = [all(b.codes[j][t] & 7 == 0 for j in (0, 1)) for t in range(1024)]
    witness = sum(p or u for i in (0, 1) for p, u in zip(F.salt_dead(f, a.x[i], a.alpha[i], a.ae[i]), unneeded))
    assert d.counts["screened_blocks_a"] == d.counts["screened_blocks_b"] == 1 and d.counts["forming_elements"] == witness
    assert d.forming == Fraction(F.FORMING_CREDITED * witness * 2, 4096)
    assert d.total == d.dead_pairs + d.split + d.int8 + d.forming


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


@pytest.mark.parametrize("blocks", (2, 4))
def test_a_row_the_old_cap_rejected_is_admitted_and_debited_on_the_witness(blocks):
    """Screened blocks past D-SS's retired one-in-64 cap, on A alone: the row is credited, item 4 charges only the
    witness's salt-dead and unneeded elements (not every element of a screened block), no block pair
    is screened, and the tile stays under ρ."""
    s, record, tile, a_rows, b_rows, ops, x = screened_tile(blocks)
    assert sum(F.dead_blocks(s.fmt, x)) == blocks > (len(x) // s.fmt.block) // 64 and F.row_passes(s.fmt, x)
    assert sum(F.salt_dead(s.fmt, x, ops.a.alpha[0], ops.a.ae[0])[:blocks * s.fmt.block]) <= blocks
    assert W.check_opened(s, record, tile, a_rows, b_rows, ops) is None
    d = s.last_debit
    assert d.counts["screened_blocks_a"] == blocks and d.dead_pairs == 0
    assert d.counts["forming_elements"] == witness_elements(s, ops) < blocks * s.fmt.block
    assert d.counts["screened_only_elements"] > 0
    assert d.forming == Fraction(F.FORMING_CREDITED * witness_elements(s, ops) * 4, 1024)
    assert 0 < d.total <= s.tile_cap(record, ops, range(4), range(4))[1]


def test_a_row_whose_screened_blocks_pass_r1_is_rejected_by_f1_prime():
    """Sixteen screened, otherwise salt-live blocks on A alone: item 4 on the witness is far under ρ, and F1′ charges the
    blocks' near-certain codes (next to a 9.75ρ max they change rarely, σ' ≈ 0.15 E2M1 units), which puts the row's own
    F1′ past R1's bound, so declared credited it fails the tile."""
    s, record, tile, a_rows, b_rows, ops, x = screened_tile(16)
    assert F.row_passes(s.fmt, x) and sum(F.salt_dead(s.fmt, x, ops.a.alpha[0], ops.a.ae[0])[:16 * s.fmt.block]) <= 16
    reason = W.check_opened(s, record, tile, a_rows, b_rows, ops)
    assert reason is not None and "R1's bound" in reason
    assert s.row_debit_ratio(record, ops, "A", 0) > 1
    cap = s.tile_cap(record, ops, range(4), range(4))[1]
    d = s.last_debit
    assert d.counts["screened_blocks_a"] == 16 and d.dead_pairs == 0 and d.total > cap
    assert d.counts["forming_elements"] == witness_elements(s, ops) and d.forming + d.int8 + d.identity + d.subgrid < cap
    assert d.split > cap


def dead_rows(rows: int, k: int = 1024) -> list[bytes]:
    """Rows salt-dead at every position: ρ from 1.0 at every 8th position, and one element per block at 256ρ, so each
    block's scale sits on its spike and every other code rounds to 0 at every draw."""
    x = [P.f32_bits(1.0) if p % 8 == 0 else 0 for p in range(k)]
    for b in range(0, k, 16):
        x[b + 1] = P.f32_bits(256.0)
    return [F.bf16_row([w >> 16 for w in x])] * rows


def dead_tile(a_dead: bool, b_dead: bool):
    """A 2 x 2 NVFP4 tile of a 2 x 1024 x 1024 unit, each side all salt-dead rows or Gaussian ones: (scheme, record, ops)."""
    s, k = V.scheme("nvfp4", False), 1024
    ctx = V.ctx_of("dead", (2, k, 1024), 0, 0)
    a_rows = dead_rows(2) if a_dead else V.rows("dead", "A", 2, k)
    b_rows = dead_rows(2) if b_dead else V.rows("dead", "B", 2, k)
    tile = type(s.tiles(Shape(64, k, 64))[0])(range(2), range(2))
    return s, W.UnitRecord(ctx.matmul, 2, k), s.encode(ctx, s.noise(ctx), a_rows, b_rows, tile)


def test_item_3_charges_per_element_where_both_codes_are_salt_dead():
    """Item 3 per element (the assessor's B, 17:25Z): one MAC per word and position whose A and B codes are both
    salt-dead.  Where both sides are dead everywhere every word is charged k, each atom its full 64; a dead side against
    a live one (dead zeros included) charges nothing; and it never exceeds the block form."""
    s, record, ops = dead_tile(True, True)
    assert all(all(F.salt_dead(s.fmt, x, al, ae)) for side in (ops.a, ops.b) for x, al, ae in zip(side.x, side.alpha, side.ae))
    s.tile_cap(record, ops, range(2), range(2))
    d = s.last_debit
    assert d.dead_pairs == 4 * 1024 == 4 * (1024 // F.ATOM) * F.ATOM
    assert d.dead_pairs <= s.fmt.block * d.counts["screened_block_pairs"]
    for a_dead, b_dead in ((True, False), (False, True)):
        s, record, ops = dead_tile(a_dead, b_dead)
        s.tile_cap(record, ops, range(2), range(2))
        dead = ops.a if a_dead else ops.b
        assert any(c & 7 == 0 for c in dead.codes[0]) and s.last_debit.dead_pairs == 0


@pytest.mark.parametrize("a_blocks, b_blocks", ((4, 4), (16, 16), (2, 0)))
def test_item_3_never_charges_more_than_its_block_form(a_blocks, b_blocks):
    """On spiked tiles (each spike at 9.75ρ salt-dead, the rest of its block live) and a Gaussian control: the per-element
    count is at most `block` per co-screened block pair (`saltDead4_screened`), and on four co-screened blocks it is their
    four spikes, 4 units, where the block form charged 64 and failed the tile's cap (about 56 units)."""
    s, record, tile, a_rows, b_rows, ops, x = screened_tile(a_blocks, b_blocks)
    s.tile_cap(record, ops, range(4), range(4))
    d = s.last_debit
    assert d.counts["screened_block_pairs"] == min(a_blocks, b_blocks)
    assert d.dead_pairs <= s.fmt.block * d.counts["screened_block_pairs"]
    if (a_blocks, b_blocks) == (4, 4):
        assert d.dead_pairs == 4 and W.check_opened(s, record, tile, a_rows, b_rows, ops) is None
        assert s.fmt.block * d.counts["screened_block_pairs"] > s.tile_cap(record, ops, range(4), range(4))[1]


def test_the_provers_debit_is_the_verifiers():
    """Items 3 and 4 read the witness from the tile's own forming on both sides, so a prover forming the committed rows
    from the unit's seeds gets the verifier's debit exactly; over the credited rows only, the spiked row's dead pairs
    drop out."""
    s, record, tile, a_rows, b_rows, ops, x = screened_tile(4, 4)
    ctx, k = V.ctx_of("screen", (4, 1024, 1024), 0, 0), 1024
    noise = s.noise(ctx)
    a, e_a = s._side(a_rows, tile.rows, F.ROLE_EA, F.ROLE_FA, noise.seed_a, noise.seed_b, k)
    b, e_b = s._side(b_rows, tile.cols, F.ROLE_EB, F.ROLE_FB, noise.seed_b, noise.seed_b, k)
    mine = F.tile_debit(s.fmt, a, b, range(4), range(4), 1024, e_a, e_b, 4)
    s.tile_cap(record, ops, range(4), range(4))
    assert mine == s.last_debit and mine.dead_pairs == 4
    s.tile_cap(record, ops, [1, 2, 3], range(4))
    assert s.last_debit.dead_pairs == 0 and s.last_debit.counts["words"] == 12


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


def test_d24_is_the_hardware_pair_rule():
    """NVFP4's sparse MMA keeps 2 of each 8-chunk's 4 pairs: a window counts when every chunk has at most 2 nonzero pairs,
    whatever its 4-groups hold (a sign-only code, 0x8, is zero)."""
    assert F.two_four_windows([0x3, 0x5, 0x6, 0x7, 0, 0, 0x8, 0] * 16) == [True]
    assert F.two_four_windows([0x1, 0, 0x2, 0, 0x3, 0, 0, 0] * 16) == [False]
    assert F.two_four_windows([0x1] * 4 + [0] * 4 + [0x1, 0, 0x2, 0, 0x3, 0, 0, 0] + [0] * 112) == [False]
    assert F.two_four_windows([0] * 128 + [0x1, 0x9] * 64) == [True, False]


def _row(values) -> list[int]:
    return F.parse_row(F.bf16_row([P.f32_to_bf16(v) for v in values]), len(values))


_COLS = random.Random(7)
_BLOCK_SIGN = [_COLS.choice((-1, 1)) for _ in range(4096 // 16)]
_CELL = [_COLS.choice((0.0, 0.5, 1.0, 1.5)) for _ in range(4096)]
_SIGN = [_COLS.choice((-1, 1)) for _ in range(4096)]
_SAMPLE = [1.0 if _COLS.random() < 0.275 else 0.5 for _ in range(4096)]


def spike_row(i: int, k: int = 1024, s: float = 8.9, top: float | None = None) -> list[int]:
    """The assessor's base-split rows (fp4-base-split-break.md): every 8th position at ±1 (ρ = 1), 14 spikes per block at
    s·ρ off ρ's positions, signed by a row sign times a block sign.  `top` puts one entry in the last block (screened)
    that sets α so the spikes' block max centres UE4M3's byte 384: every scale then sits at one byte."""
    rng = random.Random(i)
    rs = 1 if i % 2 else -1
    v = [rng.choice((-1.0, 1.0)) if t % 8 == 0 else s * rs * _BLOCK_SIGN[t // 16] for t in range(k)]
    if top is not None:
        v[k - 15] = top
    return _row(v)


def narrow_cell_row(i: int, k: int = 1024, s: float = 8.9) -> list[int]:
    """One max per block at s·ρ, and every other code, ρ's samples included, at the centre of a narrow E2M1 cell (0,
    ±0.5, ±1 or ±1.5 of s/6), in a rank-one pattern: no code is near-deterministic at ε = 5%, yet the split pays."""
    u, rs = s / 6, 1 if i % 2 else -1
    v = [rs * _SIGN[t] * (_SAMPLE[t] if t % 8 == 0 else _CELL[t]) * u for t in range(k)]
    for b in range(k // 16):
        v[16 * b + 1] = s * rs
    return _row(v)


def formed(f: F.Fmt, rows, role_e: int, role_f: int, k: int):
    e = [F.line(b"\x03" * 32, role_e, i) for i in range(len(rows))]
    return F.form(f, rows, e, F.basis(b"\x04" * 32, role_f, k)), e


@pytest.mark.parametrize("family", [spike_row, narrow_cell_row], ids=["spike", "narrow-cell"])
def test_f1_prime_charges_the_base_split_rows_past_the_cap(family):
    """F1′: the assessor's spike rows, which the support witness reads as salt-live (every other item 0),
    and the narrow-cell rows, which escape a count of near-deterministic codes, are both charged far past the cap, while
    the Gaussian side carries nothing."""
    f, k, n = F.NVFP4, 1024, 4096
    rows = [family(i) for i in range(4)]
    assert all(F.row_passes(f, x) and not any(F.dead_blocks(f, x)) for x in rows)
    a, e_a = formed(f, rows, F.ROLE_EA, F.ROLE_FA, k)
    b, e_b = formed(f, [F.parse_row(r, k) for r in V.rows("rules", "B", 4, k)], F.ROLE_EB, F.ROLE_FB, k)
    d = F.tile_debit(f, a, b, range(4), range(4), n, e_a, e_b)
    cap = F.CAP * Fraction(F.PearlC4().credit_of(Shape(1, k, n)), n) * 16
    assert d.split > 50 * cap
    gauss = [F.change_probabilities(f, b.x[j], b.alpha[j], e_b[j], b.ae[j], [False] * k, b.scales[j]) for j in range(4)]
    assert sum(F.split_units(f, p, modes, b.scales[j]) for j, (p, modes) in enumerate(gauss)) < Fraction(1, 10 ** 4)
    if family is spike_row:
        assert d.counts["salt_dead_a"] == 0


@pytest.mark.parametrize("family", [spike_row, narrow_cell_row, None], ids=["spike", "narrow-cell", "gaussian"])
def test_c_charges_at_least_the_first_reading(family):
    """C (`change_probabilities`, the rule: certified modal byte, 4-lookup lower bound) charges at least the reading it
    bounds (`change_probabilities_exact`: every cell, the modal byte over the ±6σ bytes) on every row, with no block
    uncertified on these rows."""
    f, k = F.NVFP4, 1024
    rows = [family(i) for i in range(3)] if family else [F.parse_row(r, k) for r in V.rows("rules", "A", 3, k)]
    a, e = formed(f, rows, F.ROLE_EA, F.ROLE_FA, k)
    for i in range(3):
        dead = F.salt_dead(f, a.x[i], a.alpha[i], a.ae[i])
        p, modes = F.change_probabilities(f, a.x[i], a.alpha[i], e[i], a.ae[i], dead, a.scales[i])
        q, exact = F.change_probabilities_exact(f, a.x[i], a.alpha[i], e[i], a.ae[i], dead)
        assert None not in modes and all(x <= y + 1e-15 for x, y in zip(p, q))
        assert F.split_units(f, p, modes, a.scales[i]) >= F.split_units(f, q, exact, a.scales[i]) * (1 - 1e-12)


def test_b_tilde_rows_are_charged_at_the_overfit_factor():
    """B̃'s F1′ is charged at F1_B_OVERFIT = 10 times its expectation per window, capped at the window's MACs (a layout
    chosen after the salt for B̃'s residual is shared by all n weight rows and gains up to 6.8× on narrow weights);
    A′'s at 1×.  R1 reads B rows the same way."""
    f, k, n = F.NVFP4, 1024, 4096
    assert F.F1_B_OVERFIT == 10
    a, e_a = formed(f, [F.parse_row(r, k) for r in V.rows("rules", "A", 2, k)], F.ROLE_EA, F.ROLE_FA, k)
    b, e_b = formed(f, [narrow_cell_row(i) for i in range(2)], F.ROLE_EB, F.ROLE_FB, k)
    d = F.tile_debit(f, a, b, range(2), range(2), n, e_a, e_b)
    per_b = [F.row_f1(f, b, e_b, j, F.salt_dead(f, b.x[j], b.alpha[j], b.ae[j]), F.F1_B_OVERFIT)[0] for j in range(2)]
    once = [F.row_f1(f, b, e_b, j, F.salt_dead(f, b.x[j], b.alpha[j], b.ae[j]))[0] for j in range(2)]
    per_a = [F.row_f1(f, a, e_a, i, F.salt_dead(f, a.x[i], a.alpha[i], a.ae[i]))[0] for i in range(2)]
    assert d.split == 2 * sum(per_a) + 2 * sum(per_b)
    assert all(once[j] < per_b[j] <= min(10 * once[j], k) for j in range(2))


def test_f1_prime_takes_the_cheaters_layout_inside_a_block():
    """A block keeps its scale under any reordering, so reliable codes one per aligned pair (0 and ½ alternating) are
    paired together: every chunk keeps two unchanged pairs and the 2:4 pass always fits (S_w = ½).  All reliable: the
    capped sum, 1; all changing half the time: nothing."""
    f = F.NVFP4
    assert F.split_saving(f, [0.0, 0.5] * 64) == pytest.approx(0.5)
    assert F.split_saving(f, [0.0] * 128) == 1.0
    assert F.split_saving(f, [0.5] * 128) < 1e-4


def test_f1_prime_puts_the_correction_inside_the_expectation():
    """κ inside (the assessor's FP4 condition 1, 15:18Z): a cheater routes each row after the salt, so F1′ charges
    E[max(0, R − κ)] per route, never less than max(0, E[R] − κ), and κ = 0 leaves S_w as it was."""
    f = F.NVFP4
    p = [0.0] * 16 + [0.45] * 112
    base = F.split_saving(f, p)
    for kappa in (0.02, 0.066, 0.125):
        inside = F.split_saving(f, p, kappa)
        assert inside <= base and inside >= max(0.0, base - 2 * kappa)
    rare = [0.0, 0.5] * 64
    assert F.split_saving(f, rare, 0.125) > 0 and F.split_saving(f, [0.3] * 128, 0.0) == pytest.approx(F.split_saving(f, [0.3] * 128))


def test_f1_prime_reads_the_exact_noise_law():
    """p_t under the exact law given E (the assessor's FP4 condition 2, 15:18Z): the pmf of Σ (2E_r)(2F_r) has the
    Gaussian's variance exactly, sums to 1, and is symmetric; a lone E code of 6 gives F's own 16-point law scaled
    by 12."""
    e = tuple(F.line(b"\x05" * 32, F.ROLE_EA, 0))
    lo, pmf = F.noise_law(e)
    assert sum(pmf) == pytest.approx(1.0) and lo == -(len(pmf) - 1) // 2
    assert pmf == pytest.approx(pmf[::-1])
    var = sum(q * (lo + i) ** 2 for i, q in enumerate(pmf)) / 16
    assert var == pytest.approx(8.5625 * sum(float(F.e2m1_value(c)) ** 2 for c in e))
    lo, pmf = F.noise_law((0x7,) + (0,) * 31)
    assert lo == -144 and pmf[0] == pytest.approx(1 / 16) and pmf[144] == pytest.approx(2 / 16)


def two_draw_row(k: int = 128) -> tuple[list[int], tuple[int, ...], list[int]]:
    """A row, an E line with codes at two positions and 0 elsewhere (so an element's noise takes one of 256 equally likely
    values, by its F line's codes there), and those positions."""
    x = F.parse_row(V.rows("law", "A", 1, k)[0], k)
    e = [0] * F.R
    e[3], e[17] = 0x6, 0xB
    return x, tuple(e), [3, 17]


def atom_words(c: int, e, ae: int, at) -> list[int]:
    """The noise atom's D on each of the 256 pairs of F codes at `at`, every other F code 0."""
    out = []
    for f1 in range(16):
        for f2 in range(16):
            f = [0] * F.R
            f[at[0]], f[at[1]] = f1, f2
            out.append(F.noise_atom(c, e, ae, f))
    return out


@pytest.mark.parametrize("fmt", FMTS)
def test_change_probabilities_match_the_noise_atom_line_by_line(fmt):
    """The law against the noise atom itself: on `two_draw_row`'s E line each element's D is one of 256 equally likely
    words, so its p_t is enumerable.  `change_probabilities_exact`' p_t at the block's modal byte is 1 − the most common
    code value's share of the atom's words there, to 1e-12 (the law leaves out the atom's one RZ rounding, which moves no
    word across a cell edge here), and 0 on salt-dead codes."""
    f = F.FORMATS[fmt]
    x, e, at = two_draw_row()
    _, _, alpha, _, ae = F.stats(f, x)
    dead = F.salt_dead(f, x, alpha, ae)
    p, modes = F.change_probabilities_exact(f, x, alpha, e, ae, dead)
    for t in range(0, len(x), 3):
        byte = modes[t // f.block]
        r = fp32.div(F.ONE, F.ue4m3_word(byte)) if f is F.NVFP4 else F.pow2_word(127 - byte)
        codes = Counter(F.e2m1_value(F.e2m1(fp32.mul(d, r))) for d in atom_words(fp32.mul(alpha, x[t]), e, ae, at))
        assert p[t] == (0.0 if dead[t] else pytest.approx(1 - max(codes.values()) / 256, abs=1e-12))


def test_the_modal_byte_is_the_realised_bytes_mode():
    """`modal_byte` against 600 formings of two NVFP4 blocks under one E line, every element's F line drawn afresh: the
    most frequent realised byte."""
    f, rng = F.NVFP4, random.Random(3)
    x = F.parse_row(V.rows("mode", "A", 1, 1024)[0], 1024)
    _, _, alpha, _, ae = F.stats(f, x)
    e = tuple(F.line(b"mode" * 8, F.ROLE_EA, 0))
    cdf, sigma = F._below_law(e, F._scale_value(F.NVFP4, ae) / 64), F.noise_sigma(e, ae)
    for b0 in (0, 16 * 9):
        cs = [fp32.mul(alpha, w) for w in x[b0:b0 + 16]]
        seen = Counter(F.block_scale(f, max(F.noise_atom(c, e, ae, [rng.randrange(16) for _ in range(F.R)]) & 0x7FFFFFFF
                                            for c in cs))[0] for _ in range(600))
        assert seen.most_common(1)[0][0] == F.modal_byte(f, [F.P.f32_of_bits(c) for c in cs], cdf, sigma)


def r1_tile():
    """A 4 x 4 credited tile of a 4 x 1024 x 4096 NVFP4 unit: A row 0 the assessor's spike row, rows 1–3 and B Gaussian."""
    s, k = V.scheme("nvfp4", False), 1024
    ctx = V.ctx_of("r1", (4, k, 4096), 0, 0)
    a_rows = [F.bf16_row([x >> 16 for x in spike_row(0, k)])] + V.rows("r1", "A", 3, k)
    b_rows = V.rows("r1", "B", 4, k)
    tile = type(s.tiles(Shape(64, k, 64))[0])(range(4), range(4))
    ops = s.encode(ctx, s.noise(ctx), a_rows, b_rows, tile)
    return s, ctx, tile, a_rows, b_rows, ops


def test_r1_uncredits_a_row_whose_own_debit_passes_its_cell_cap():
    """R1 (the assessor's FP4 condition 4, 15:18Z): the spike row's own F1′ passes ρ·(its cell credit), so it must be
    declared excluded; declared so, the tile passes, since the Gaussian rows' debit is far under the cap.  Credited, it
    fails the tile; an excluded Gaussian row passes the per-row rules and R1, so its exclusion fails.  The honest
    prover's declaration finds exactly row 0."""
    s, ctx, tile, a_rows, b_rows, ops = r1_tile()
    k = ctx.matmul.shape.k
    assert all(F.row_passes(s.fmt, F.parse_row(r, k)) for r in a_rows + b_rows)
    assert s.row_over_debit(W.UnitRecord(ctx.matmul, 4, k), ops, "A", 0)
    assert not any(s.row_over_debit(W.UnitRecord(ctx.matmul, 4, k), ops, side, p) for side in "AB" for p in (1, 2, 3))
    assert W.check_opened(s, W.UnitRecord(ctx.matmul, 4, k, (0,)), tile, a_rows, b_rows, ops) is None
    reason = W.check_opened(s, W.UnitRecord(ctx.matmul, 4, k), tile, a_rows, b_rows, ops)
    assert reason is not None and reason.startswith("A row 0 is credited") and "R1's bound" in reason
    reason = W.check_opened(s, W.UnitRecord(ctx.matmul, 4, k, (0, 1)), tile, a_rows, b_rows, ops)
    assert reason is not None and reason.startswith("A row 1 is excluded but passes the per-row rules")
    assert s.r1_rows(ctx, a_rows, b_rows, k) == ({0}, set())


def test_r1_allows_either_declaration_inside_its_tolerance_band():
    """R1's tolerance band (the assessor's condition 3, 16:10Z): a row within `R1_TOLERANCE` of its bound may be declared
    credited or excluded, so implementations that differ in the last bits agree on the audit."""
    s, ctx, tile, a_rows, b_rows, ops = r1_tile()
    k = ctx.matmul.shape.k

    class Near(F.PearlC4):
        def row_debit_ratio(self, record, ops, side, pos):
            return {("A", 0): Fraction(10), ("A", 1): 1 + Fraction(1, 10 ** 10)}.get((side, pos), Fraction(0))

    near = Near("nvfp4")
    assert near.r1_tolerance == F.R1_TOLERANCE > Fraction(1, 10 ** 10)
    for excluded in ((0,), (0, 1)):
        reason = W.check_opened(near, W.UnitRecord(ctx.matmul, 4, k, excluded), tile, a_rows, b_rows, ops)
        assert reason is None or ("R1" not in reason and "per-row rules" not in reason)


def test_b_ovf_discounts_credit_by_its_pinned_table():
    """B-OVF (the assessor's B, 18:45Z; condition 7): creditOf less ⌈β(n_b)·mnk⌉, β the table
    measured on each bucket's R1 boundary (`beta_final.py`), n_b the largest tabled n ≤ the unit's, non-increasing, 0 from
    4,096 up."""
    assert F.B_OVF == ((128, Fraction(414, 10_000)), (256, Fraction(223, 10_000)), (512, Fraction(119, 10_000)),
                       (1024, Fraction(50, 10_000)), (2048, Fraction(18, 10_000)), (4096, Fraction(0)))
    got = {n: F.b_ovf(n) for n in (128, 192, 256, 512, 896, 1024, 2047, 2048, 3584, 4096, 1 << 18)}
    assert got == {128: Fraction(414, 10_000), 192: Fraction(414, 10_000), 256: Fraction(223, 10_000),
                   512: Fraction(119, 10_000), 896: Fraction(119, 10_000), 1024: Fraction(50, 10_000),
                   2047: Fraction(50, 10_000), 2048: Fraction(18, 10_000), 3584: Fraction(18, 10_000), 4096: 0,
                   1 << 18: 0}
    assert [b for _, b in F.B_OVF] == sorted((b for _, b in F.B_OVF), reverse=True)
    s = F.PearlC4()
    for m, k, n in ((64, 1024, 128), (2048, 3584, 512), (64, 8192, 896), (1, 1024, 1024), (64, 1024, 4096)):
        shape = Shape(m, k, n)
        plain = s.wref(shape) - F.FORMING_UNCREDITED * m * k
        assert s.credit_of(shape) == plain - math.ceil(F.b_ovf(n) * m * n * k)
        assert (s.credit_of(shape) < plain) == (n < 4096)
    assert ("b_ovf", tuple((nb, str(b)) for nb, b in F.B_OVF)) in s.certificate.parameters


def test_r1_and_the_tile_cap_read_the_discounted_credit(monkeypatch):
    """R1's per-row bound and the tile cap are ρ times `credit_of`'s share, so B-OVF's discount tightens both: on a
    unit of n = 1,024 (β = 0.50%) the cap and the bound fall by exactly the credit's ratio to the undiscounted one."""
    s, record, tile, a_rows, b_rows, ops, x = screened_tile(4, 4)
    k, n = record.k_kept, record.matmul.shape.n
    disc = s.credit_of(Shape(1, k, n))
    cap = s.tile_cap(record, ops, range(4), range(4))[1]
    ratio = s.row_debit_ratio(record, ops, "B", 0)
    assert cap == F.CAP * Fraction(disc, n) * 16
    monkeypatch.setattr(F, "B_OVF", ((128, Fraction(0)),))
    plain = s.credit_of(Shape(1, k, n))
    assert plain - disc == math.ceil(Fraction(50, 10_000) * n * k) > 0
    assert s.tile_cap(record, ops, range(4), range(4))[1] == cap * Fraction(plain, disc)
    assert s.row_debit_ratio(record, ops, "B", 0) == ratio * Fraction(disc, plain)


def test_a_drawn_tile_evaluates_each_rows_f1_once():
    """R1 on every row and the debit over the credited words read one evaluation of each row's F1′ per encoded tile (F1′
    is most of the verifier's per-tile cost), and the debit is the uncached one; other operands start afresh."""
    s, ctx, tile, a_rows, b_rows, ops = r1_tile()
    k, shape = ctx.matmul.shape.k, ctx.matmul.shape
    calls: list[tuple[int, int]] = []

    def counted(fmt, side, e_rows, pos, dead, overfit=1):
        calls.append((id(side), pos))
        return F.row_f1(fmt, side, e_rows, pos, dead, overfit)

    s.f1 = counted
    record = W.UnitRecord(ctx.matmul, 4, k, (0,))
    assert W.check_opened(s, record, tile, a_rows, b_rows, ops) is None
    assert sorted(calls) == sorted((id(side), p) for side in (ops.a, ops.b) for p in range(4))
    s.tile_cap(record, ops, range(4), range(4))
    assert len(calls) == 8
    assert s.last_debit == F.tile_debit(s.fmt, ops.a, ops.b, range(4), range(4), shape.n, ops.e_a, ops.e_b, shape.m)
    s.tile_cap(record, copy.copy(ops), [1], [0])
    assert len(calls) == 10


def strassen_c_L(m: int, k: int, n: int) -> float:
    """The c_L F2 pinned before the catalogue (bc-d9842080, 13:35Z; the assessor's FP4 conditions, 15:18Z):
    Strassen–Winograd at W1's int8 leaf of 2 slots, depth capped by log2(k/16) alone."""
    return min((2 * (7 / 8) ** L + 4.25 * (1 / m + 1 / n) * ((7 / 4) ** L - 1) / 0.75
                for L in range(1, (k // 16).bit_length())), default=2.0)


def test_f2_depth_cost_is_the_pinned_catalogue_minimum():
    """c_L is the fast-matmul catalogue's minimum at its lower-bound add count, one add per new form (the assessor's
    20:00Z ruling): 0.661 / 0.579 / 0.507 / 0.408 at 8k–64k³, where Strassen's
    formula gave 0.808 / 0.707 / 0.619 / 0.542.  At the domain's corner (m = 2^24, k = 2^16, n = 2^18, the claimed n
    cap) 0.270 against Strassen's 0.421; rectangular shapes read their own entries; F2 fires only past f_A + f_B = 1.915
    at 8k³, far from honest tiles' 0.13–0.30."""
    f = F.NVFP4
    assert F.INT8_LEAF_SLOTS == 2
    cubes = [F.int8_depth_cost(f, s, s, s) for s in (8192, 16384, 32768, 65536)]
    assert cubes == [0.6612, 0.579, 0.5069, 0.4083] and cubes == pytest.approx([0.661, 0.579, 0.507, 0.408], abs=1e-3)
    assert [strassen_c_L(s, s, s) for s in (8192, 16384, 32768, 65536)] == pytest.approx([0.808, 0.707, 0.619, 0.542],
                                                                                          abs=6e-4)
    assert 2 - (1 - cubes[0]) / F.INT8_PATCH_SLOTS == pytest.approx(1.915, abs=1e-3)
    corner = F.int8_depth_cost(f, 1 << 24, 1 << 16, 1 << 18)
    assert corner == 0.2697 and strassen_c_L(1 << 24, 1 << 16, 1 << 18) == pytest.approx(0.421, abs=6e-4)
    assert F.int8_depth_cost(f, 1 << 16, 8192, 1024) == F.int8_depth_cost(f, 1024, 8192, 1 << 16) == 0.8081
    assert F.int8_depth_cost(f, 1 << 24, 8192, 128) == 1.1302 > 1         # a 128-wide weight: no route pays
    assert F.int8_depth_cost(f, 1 << 24, 16, 1 << 18) == 2.0              # one block of k: no level
    assert F.int8_depth_cost(F.MXFP4, 1 << 24, 1 << 16, 1 << 18) == F.int8_depth_cost(f, 1 << 24, 1 << 15, 1 << 18)
    for beyond in ((1 << 25, 8192, 8192), (8192, 8192, 1 << 19), (8192, 1 << 17, 8192)):
        with pytest.raises(ValueError):
            F.int8_depth_cost(f, *beyond)


def test_f2_c_l_table_takes_the_lower_value_between_its_points():
    """The pinned table never lies above Strassen's formula and falls in each of m, k and n, so a shape between tabled
    points reads the next tabled point up in each dimension, whose c_L is the lower one."""
    from verity.core.protocols.pouw.schemes import pearl_c4_c_L as CL
    for (i, m), (j, k), (l, n) in itertools.product(enumerate(CL.MS), enumerate(CL.KS), enumerate(CL.NS)):
        v = CL.C_L[i][j][l]
        assert v / 10_000 <= strassen_c_L(m, k, n) + 1e-12
        assert all(nxt <= v for nxt in (CL.C_L[i + 1][j][l] if i + 1 < len(CL.MS) else v,
                                        CL.C_L[i][j + 1][l] if j + 1 < len(CL.KS) else v,
                                        CL.C_L[i][j][l + 1] if l + 1 < len(CL.NS) else v))
    f = F.NVFP4
    for shape, up in (((3000, 5000, 700), (8192, 8192, 1024)), ((20_000, 9000, 100), (32768, 16384, 128)),
                      ((64, 128, 16), (128, 128, 128))):
        assert F.int8_depth_cost(f, *shape) == F.int8_depth_cost(f, *up) <= strassen_c_L(*shape)


@pytest.mark.parametrize("fmt", FMTS)
def test_f2_reads_flatness_per_row_and_per_block_column_class(fmt):
    """F2 on the tile's realised scale bytes: flat at one byte, the route's whole 1 − c_L; flat per row (row factors of
    two) or per block-column class (two classes a byte apart, each a sub-GEMM over half of k), the same, where the tile's
    modal share (0.5) alone would read nothing; bytes spread as honest ones are, nothing."""
    f, m, n, k = F.FORMATS[fmt], 1 << 24, 1 << 18, 8192
    nb, base = k // f.block, 0x50 if fmt == "nvfp4" else 120
    full, half = 1 - F.int8_depth_cost(f, m, k, n), 1 - F.int8_depth_cost(f, m, k // 2, n)
    flat = [[base] * nb for _ in range(8)]
    assert F.int8_saving(f, flat, flat, m, n) == pytest.approx(full) and full > 0
    shifted = [[base + (8 if fmt == "nvfp4" else 1) * (r % 2)] * nb for r in range(8)]
    assert F.int8_saving(f, shifted, flat, m, n) == pytest.approx(full)
    two = [[base + c % 2 for c in range(nb)] for _ in range(8)]
    assert F.int8_saving(f, two, two, m, n) == pytest.approx(half) and half > 0
    rng = random.Random(1)
    honest = [[base + rng.randrange(6) for _ in range(nb)] for _ in range(8)]
    assert F.int8_saving(f, honest, honest, m, n) == 0


def test_f2_charges_a_formed_tile_whose_scales_sit_at_one_byte():
    """The flattest rows (fp4-int8-route.md 12:00Z): spike rows with α tuned so every block max centres byte 384, on both
    sides.  Their realised scales sit at one byte, so F2 charges the masked int8 route's saving on every chain MAC."""
    f, k, m, n = F.NVFP4, 4096, 1 << 24, 1 << 18
    rows = [spike_row(i, k, top=9.88) for i in range(4)]
    a, e_a = formed(f, rows, F.ROLE_EA, F.ROLE_FA, k)
    b, e_b = formed(f, rows, F.ROLE_EB, F.ROLE_FB, k)
    d = F.tile_debit(f, a, b, range(4), range(4), n, e_a, e_b, m)
    save = F.int8_saving(f, a.scales, b.scales, m, n)
    assert save > 0 and d.int8 == Fraction(save) * 16 * k


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


#: The four rows the decode's 28-row `nvf4Specials` table adds (bc-5a715b19's review): (accumulator, a, b, sa, sb),
#: codes and bytes packed little-endian, as the Lean `Vec`.  Each is a NaN accumulator with one scale byte that sets
#: bit 7.
NAN_BIT7_SPECIALS = (
    (0xFFACB8FF, 0xD012B2233DC6403271C58AC78DFF95B32950DA366B5B32A97D33B2DECE69A903,
     0x932E07AB6589BD2774DC10686F6F47C3F71883659810A8DEEA135E394A8BA32E, 0x28DB5712, 0x4F2C4D74),
    (0x7F8163AC, 0x8C723EEE523093DAF5F775033AA7D281B9FB908BBBE1EF1F462490252D508333,
     0x357A39B4614523259FAFD800A9D93ABA65E1E668F3DD38A4AD9D9E120FCA114E, 0x384E2E0F, 0x578E342B),
    (0xFF9F8CAE, 0x4BC3E89FB7E108A21C2DDF27D95A2F8F9792EC8E966FFC2CD96ADEDA5AAF81D9,
     0xFC1D14F7D95984BAD926475BE192F56E9577D5A73A914EE07411125B44C4EC14, 0x3551142B, 0x74E13924),
    (0x7FFEAA77, 0xC1E2F0693F03C513B5FCA39B949062349E45A9E1FC2EA0E0E25E1D9F6D905825,
     0x2311785E72C70F5C98D86F50B5CB5856F42807A0191C155683D3ABAFE7F2C6F4, 0x5CF25B71, 0x3B501375),
)


@pytest.mark.parametrize("row", NAN_BIT7_SPECIALS, ids=lambda r: f"{r[0]:08X}")
def test_nan_accumulator_specials_read_their_scales_as_the_hardware(row):
    """The decode's 28-row specials table: a NaN accumulator whose scale bytes set bit 7.  Read as sm_120 reads them
    (low 7 bits: `hw_scales`, and the pinned model's own decode since GPU 4's replay, r20260930-161935-83a8), the bytes
    are valid scales, so the vector is outside the model only through its accumulator (the step raises, Lean's `none`;
    the card writes 0x7FFFFFFF).  D-SB keeps such bytes off every formed side."""
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4 as atom
    acc, a, b, sa, sb = row
    codes_a, codes_b = [(a >> 4 * i) & 15 for i in range(64)], [(b >> 4 * i) & 15 for i in range(64)]
    sa, sb = [(sa >> 8 * i) & 255 for i in range(4)], [(sb >> 8 * i) & 255 for i in range(4)]
    assert (acc >> 23) & 0xFF == 0xFF and acc & 0x7FFFFF and any(s & 0x80 for s in sa + sb)
    assert atom.group_terms(codes_a, codes_b, sa, sb) == atom.group_terms(codes_a, codes_b, F.hw_scales(F.NVFP4, sa),
                                                                          F.hw_scales(F.NVFP4, sb))
    with pytest.raises(InvalidArtifact):
        atom.step_scaled(acc, codes_a, codes_b, F.hw_scales(F.NVFP4, sa), F.hw_scales(F.NVFP4, sb))
    assert not F.scale_bytes_ok(F.NVFP4, sa + sb)


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


def lifecycle(s: F.PearlC4, real=20, salt_key=None):
    """32 x 128 x 16 on 16 x 16 tiles: A's rows 20 … 31 are zero filler and row 3 fails D-NF; B passes."""
    rows_a = V.rows("work-life", "A", real, K) + [F.bf16_row([0] * K)] * (32 - real)
    rows_a[3] = spiked(rows_a[3], value=2.0 ** 16)
    rows_b = V.rows("work-life", "B", 16, K)
    work = Workload((Matmul(3, Shape(32, K, 16), 9),))
    prover = Prover(s, {9: rows_b}, salt_key=salt_key)
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon", E.COINS)
    return epoch, prover, prover.prove(epoch, {3: rows_a}), rows_a, rows_b


def test_the_work_law_audits_pearl_c4():
    s = V.scheme("nvfp4", True)
    epoch, prover, tr, rows_a, rows_b = lifecycle(s)
    splits = {9: W.Split(K)}
    records = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits, real_rows={3: 20})
    assert records == [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), ())]
    out = W.audit(s, epoch, prover, tr, splits, records, E.challenger(), 8)
    assert out.audit.accepted and out.job == s.job_cost(16, K)
    assert out.wref == W.cell_wref(s, records[0]) * 19 * 16
    hides = W.audit(s, epoch, prover, tr, splits, [W.UnitRecord(epoch.workload.matmuls[0], 20, K)], E.challenger(), 8)
    assert not hides.audit.accepted
    assert any("A row 3 is credited but fails" in v.reason for v in hides.audit.verdicts if not v.ok)


#: -h3's row seeds under Pearl-C4's seed-A label: (unit, row) -> seed, at salt bytes 0 … 31, root_B bytes 32 … 63 and the
#: leaf SHA-512("pearl-c row leaf")
ROW_SEEDS = {(7, 3): "f0f7f0ecbc5b5aa0a0146c655ff68f7d0f798280672ffbb41d64741ae7b504b2",
             (7, 200): "85ee536b3f107c5489f96ba8c5c61a7eb914f9186256057ad2aa930d4d33c079",
             (2 ** 32 - 1, 3): "84f5b70a105d94cda2b2882d177db0a53ac7dd12010327044d1bbe0b0f87243d",
             (2 ** 32 - 1, 200): "44bb2a7603202bc1e7357e7657aebfb04a935e8ee4c78fa14ac5aad3f3b5fcb9"}


def _h(label: bytes, message: bytes) -> bytes:
    return hashlib.sha512((label + b"\0").ljust(128, b"\0") + message).digest()[:32]


def test_h3_row_seeds_are_pearl_cs_layout_under_pearl_c4s_label():
    """-h3's row seed is the pinned bytes: `pearl_c.row_seed`'s message under `pearl-c4/v1/seed-A`, the one-row unit's
    seed_A (`unit_seed_a`, which `noise` keys a unit by) at index unit · 2^32 + row, so never Pearl-C's seed; a unit or
    row of 2^32 or more is refused.  Both seeds are the first 32 bytes of SHA-512 after the label's zero-padded 128-byte
    block."""
    salt, root_b, leaf = bytes(range(32)), bytes(range(32, 64)), hashlib.sha512(b"pearl-c row leaf").digest()
    for (unit, row), seed in ROW_SEEDS.items():
        got = F.row_seed(salt, leaf, root_b, unit, row)
        assert got.hex() == seed and got == F.unit_seed_a(salt, leaf, root_b, unit << 32 | row) != C.row_seed(salt, leaf, root_b, unit, row)
        assert got == _h(b"pearl-c4/v1/seed-A", salt + leaf + root_b + (unit << 32 | row).to_bytes(8, "little"))
    ctx = V.ctx_of("seeds", (16, K, 16), 3, 9)
    noise = F.PearlC4().noise(ctx)
    assert noise.seed_a == _h(F.LABELS["seed-A"], ctx.salt + ctx.root_a + ctx.root_b + (3).to_bytes(8, "little"))
    assert noise.seed_b == _h(b"pearl-c4/v1/seed-B", ctx.salt + ctx.root_b)
    for unit, row in ((2 ** 32, 0), (0, 2 ** 32), (-1, 0), (0, -1)):
        with pytest.raises(ValueError, match="must each be below 2"):
            F.row_seed(salt, leaf, root_b, unit, row)


def test_h3_draws_each_rows_e_a_from_its_own_digest():
    """`PearlC4(hashing="h3")` is `pearl_c`'s -h3: hm96-sha512 trees, the SHA-512 leaf and tickets, and row i's E_A is line
    0 of `row_seed` over row i's SHA-512 digest (`pearl_c.row_digest`), so it moves with that row's bytes and position
    alone and not with root_A, which keys only the tickets.  B's side and seed_B are h0's."""
    from verity.core.protocols.pouw.serving import HASH_FORMATS, SchemeId, hash_format

    s, h0 = V.scheme("nvfp4", True, "h3"), V.scheme("nvfp4", True)
    assert (s.name, s.commitment_hash) == ("pearl-c-nvfp4-v0-h3", "hm96-sha512")
    assert SchemeId.parse(s.name).hashing == "h3" and hash_format(s) is HASH_FORMATS["h3"] and W.row_cap(s)
    assert hash_format(h0) is HASH_FORMATS["h0"] and not W.row_cap(h0) and h0.commitment_hash == "sha256"
    assert dict(s.certificate.parameters)["hashing"] == "h3" and "hashing" not in dict(h0.certificate.parameters)
    assert all("cr/sha-512" in x.certificate.claims and not any("blake3" in c for c in x.certificate.claims) for x in (s, h0))
    for bad in ("h1", "h2", "h3s"):
        with pytest.raises(ValueError):
            F.PearlC4(hashing=bad)
    ctx = V.ctx_of("h3-rows", (16, K, 16), 3, 9, wide=True)
    a, b = V.rows("h3-rows", "A", 16, K), V.rows("h3-rows", "B", 16, K)
    tile, noise = s.tiles(ctx.matmul.shape)[0], s.noise(ctx)
    assert noise == h0.noise(ctx)
    seeds = s.row_seeds(ctx, a, tile.rows)
    assert seeds == [F.row_seed(ctx.salt, C.row_digest(a[i]), ctx.root_b, 3, i) for i in range(16)] and len(set(seeds)) == 16
    ops, ops0 = s.encode(ctx, noise, a, b, tile), h0.encode(ctx, noise, a, b, tile)
    assert ops.e_a == [F.line(x, F.ROLE_EA, 0) for x in seeds] != ops0.e_a
    assert ops.e_b == ops0.e_b and ops.b.codes == ops0.b.codes and ops.pb == ops0.pb
    checked = s.checked(ctx, ops)
    assert s.digest(ctx, noise, checked) == C.leaf(checked, "h3") != C.leaf(checked)
    assert s.tickets(noise, checked) == C.PearlC(hashing="h3").tickets(noise, checked) == h0.tickets(noise, checked)
    other = list(a)
    other[5] = a[6]
    e = s.encode(ctx, noise, other, b, tile).e_a
    assert [i for i in range(16) if e[i] != ops.e_a[i]] == [5] and e[5] != ops.e_a[6]
    moved = Context(ctx.salt, ctx.matmul, bytes(64), ctx.root_b)
    assert s.encode(moved, s.noise(moved), a, b, tile).e_a == ops.e_a and s.noise(moved).seed_a != noise.seed_a


def test_the_work_law_audits_pearl_c4_under_h3():
    """The work law on `PearlC4(hashing="h3")`: R1's declaration forms each A row under its own seed, the audit accepts,
    and h0's verifier refuses the transcript's openings (another tree hash).  Each row's seed reads the hiding roots, so
    whether a row passes -h3's row cap changes with the salts: under this salt key none is over it, and the honest
    prover gives up nothing (`volunteer`)."""
    s = V.scheme("nvfp4", True, "h3")
    epoch, prover, tr, rows_a, rows_b = lifecycle(s, salt_key=bytes(32))
    splits = {9: W.Split(K)}
    records = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits, real_rows={3: 20}, transcript=tr)
    assert records == [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), ())]
    assert W.volunteer(s, epoch, tr, records, {3: rows_a}, {9: rows_b}) == ({}, {})
    assert W.audit(s, epoch, prover, tr, splits, records, E.challenger(), 8).audit.accepted
    h0 = W.audit(V.scheme("nvfp4", True), epoch, prover, tr, splits, records, E.challenger(), 8)
    assert not h0.audit.accepted and {v.reason for v in h0.audit.verdicts if not v.ok} == {"activation opening"}


def test_v_ex_a_voluntary_row_that_passes_the_rules_is_accepted():
    """V-EX (the assessor's B at 17:32Z): a row given up before the draws may pass the per-row rules and R1,
    earns nothing, opens no exclusion tile, and is committed in `records_root`; declared excluded instead, it fails.  A row
    the rules exclude stays excluded."""
    s = V.scheme("nvfp4", True)
    epoch, prover, tr, rows_a, rows_b = lifecycle(s)
    splits = {9: W.Split(K)}
    plain = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits, real_rows={3: 20})
    records = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits, real_rows={3: 20}, voluntary_a={3: [3, 5]},
                        voluntary_b={9: [2]})
    assert records == [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), (), (5,), (2,))]
    assert W.records_root(epoch.salt, epoch.weights_root, splits, records) != \
        W.records_root(epoch.salt, epoch.weights_root, splits, plain)
    assert W.exclusion_tiles(s, records) == W.exclusion_tiles(s, plain)
    out = W.audit(s, epoch, prover, tr, splits, records, E.challenger(), 8)
    assert out.audit.accepted and out.wref == W.cell_wref(s, records[0]) * 18 * 15
    as_excluded = [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3, 5), ())]
    reasons = [v.reason for v in W.audit(s, epoch, prover, tr, splits, as_excluded, E.challenger(), 8).audit.verdicts]
    assert any("A row 5 is excluded but passes" in (r or "") for r in reasons)


def test_v_ex_leaves_r1_and_the_cap_on_the_credited_rows():
    """A voluntary row doesn't shelter the others: a credited row over R1's bound still fails its tile, and so does a
    tile whose credited rows' debit passes the cap.  The spike row itself may be given up instead of excluded."""
    s, ctx, tile, a_rows, b_rows, ops = r1_tile()
    k = ctx.matmul.shape.k
    reason = W.check_opened(s, W.UnitRecord(ctx.matmul, 4, k, (), (), (1,)), tile, a_rows, b_rows, ops)
    assert reason is not None and reason.startswith("A row 0 is credited") and "R1's bound" in reason
    assert W.check_opened(s, W.UnitRecord(ctx.matmul, 4, k, (), (), (0,)), tile, a_rows, b_rows, ops) is None

    class NoR1(F.PearlC4):
        def row_debit_ratio(self, record, ops, side, pos):
            return Fraction(0)

    s = NoR1("nvfp4")
    ctx = V.ctx_of("dead", (2, 1024, 1024), 0, 0)
    a_rows, b_rows = dead_rows(2), dead_rows(2)
    tile = type(s.tiles(Shape(64, 1024, 64))[0])(range(2), range(2))
    ops = s.encode(ctx, s.noise(ctx), a_rows, b_rows, tile)
    reason = W.check_opened(s, W.UnitRecord(ctx.matmul, 2, 1024, (), (), (0,)), tile, a_rows, b_rows, ops)
    assert reason is not None and reason.startswith("the replayed debit") and "exceeds the cap" in reason


def test_v_ex_past_its_count_cap_fails_the_audit():
    """At most ⌈m/64⌉ voluntary A rows per unit and ⌈n/64⌉ voluntary B rows per weight: one each on 32 x 128 x 16."""
    s = V.scheme("nvfp4", True)
    epoch, prover, tr, rows_a, rows_b = lifecycle(s)
    splits, u = {9: W.Split(K)}, epoch.workload.matmuls[0]
    assert (W.voluntary_cap(32), W.voluntary_cap(16), W.voluntary_cap(65)) == (1, 1, 2)
    for rec in (W.UnitRecord(u, 20, K, (3,), (), (5, 6)), W.UnitRecord(u, 20, K, (3,), (), (), (1, 2))):
        out = W.audit(s, epoch, prover, tr, splits, [rec], E.challenger(), 8)
        assert not out.audit.accepted and "more voluntary rows than V-EX allows" in out.audit.verdicts[0].reason
    with pytest.raises(ValueError):
        W.UnitRecord(u, 20, K, (3,), (), (3,))


def test_volunteer_gives_up_the_row_that_puts_a_tile_over_the_cap():
    """`volunteer`, the honest prover's remedy: on a scheme whose debit puts tile position 7 at 20 units per word against a
    cap of 1 per credited word, it gives up A row 7 (not a column, which leaves the tile over), and the audit then
    accepts; without it the tile fails."""

    class Heavy(F.PearlC4):
        def tile_cap(self, record, ops, rows, cols):
            return Fraction(20 * len(cols) * (7 in rows)), Fraction(len(rows) * len(cols))

    s = Heavy("nvfp4", k_min=128, tile=(16, 16), n_min=16)
    epoch, prover, tr, rows_a, rows_b = lifecycle(s)
    splits = {9: W.Split(K)}
    records = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits, real_rows={3: 20})
    va, vb = W.volunteer(s, epoch, tr, records, {3: rows_a}, {9: rows_b})
    assert (va, vb) == ({3: [7]}, {})
    fixed = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits, real_rows={3: 20}, voluntary_a=va, voluntary_b=vb)
    assert W.audit(s, epoch, prover, tr, splits, fixed, E.challenger(), 8).audit.accepted
    tile = s.tiles(epoch.workload.matmuls[0].shape)[0]
    ctx = Context(epoch.salt, epoch.workload.matmuls[0], tr.activation_roots[0], epoch.weight_roots[9])
    ops = s.encode(ctx, s.noise(ctx), rows_a[:16], rows_b, tile)
    assert "exceeds the cap" in W.check_opened(s, records[0], tile, rows_a[:16], rows_b, ops)


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
    t = json.loads(json.dumps(R.prove(tmp_path, "r-test", 16, scheme=s)))
    assert list(t["device_scales"]) == ["b"] and t["device_scales"]["b"]["outside_range"] == 0
    assert t["records"] == [{"real_rows": 20, "k_kept": K, "excluded_a": [3], "excluded_b": []}]
    assert t["host_roots_match"] == {"a": True, "b": True} and [o["tile"] for o in t["openings"]] == [0, 1]
    rep = R.verify(t, scheme=s)
    assert rep["accepted"] and rep["opened"] == 17 and rep["reason"] == ""
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
        epoch = Epoch.unverified(Workload((unit,)), {9: bytes.fromhex(t["weight_roots"]["9"])}, b"beacon")
        rec = W.UnitRecord(unit, 20, K)
        beacon = R.draw_beacon(bytes.fromhex(t["transcript"]["root"]),
                               W.records_root(epoch.salt, epoch.weights_root, {9: W.Split(K)}, [rec]))
        t["draw"]["value"] = beacon.value.hex()

    assert bad(tamper)["reason"] == "activation opening"
    assert bad(lambda t: t["draw"].update(value="00" * 32))["reason"].startswith("the draws' beacon is not")
    assert "A row 3 is credited but fails" in bad(hide)["reason"]
    assert bad(lambda t: t["openings"].pop(0))["reason"] == "'the transcript does not open tile 0'"


def test_replay_declares_r1_rows_and_its_workers_give_the_serial_transcript(tmp_path):
    """The replay's prove declares R1's rows as the honest prover does (`pearl_c_work.declare` on the transcript): A row 5,
    the assessor's spike row, passes the per-row rules but its own F1′ passes R1's bound, so prove excludes it and the
    verifier accepts.  Credited, as prove declared before R1 (r20261001-134930-22d2's m64-n512-k2048 unit, B row 386), it
    rejects.  prove maps R1 row by row over its workers, and two workers give the one worker's transcript."""
    s = V.scheme("nvfp4", True)
    rows_a = V.rows("work-life", "A", 20, K) + [F.bf16_row([0] * K)] * 12
    rows_a[5] = F.bf16_row([x >> 16 for x in spike_row(0, K)])
    rows_b = V.rows("work-life", "B", 16, K)
    unit = Matmul(3, Shape(32, K, 16), 9)
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.unverified(Workload((unit,)), prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    assert F.row_passes(s.fmt, F.parse_row(rows_a[5], K))
    honest = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, {9: W.Split(K)}, real_rows={3: 20}, transcript=tr)
    assert honest == [W.UnitRecord(unit, 20, K, (5,), ())]
    call = {"scheme": s.name, "shape": [32, K, 16], "index": 3, "weight": 9, "real_rows": 20, "epoch_beacon": b"beacon".hex()}
    R.write_dump(tmp_path, call, b"".join(rows_a), b"".join(rows_b), tr.activation_roots[0], prover.weight_roots[9],
                 b"".join(prover.digest(i) for i in range(2)))
    serial = R.prove(tmp_path, "r-test", 12, workers=1, scheme=s)
    assert serial["records"] == [{"real_rows": 20, "k_kept": K, "excluded_a": [5], "excluded_b": []}]
    assert [o["tile"] for o in serial["openings"]] == [0, 1]
    assert R.prove(tmp_path, "r-test", 12, workers=2, scheme=s) == serial
    rep = R.verify(json.loads(json.dumps(serial)), scheme=s)
    assert rep["accepted"] and rep["reason"] == ""

    credited = json.loads(json.dumps(serial))
    credited["records"][0]["excluded_a"] = []
    digest = W.records_root(epoch.salt, epoch.weights_root, {9: W.Split(K)}, [W.UnitRecord(unit, 20, K)])
    credited["draw"]["value"] = R.draw_beacon(bytes.fromhex(serial["transcript"]["root"]), digest).value.hex()
    rep = R.verify(credited, scheme=s)
    assert not rep["accepted"] and rep["reason"].startswith("A row 5 is credited but its own debit passes R1's bound")


def stale_b(s: F.PearlC4, salt: bytes) -> F.PearlC4:
    """`s` with B̃'s noise keyed by `salt` in place of the job's: a weight forming reused from another epoch, or one
    keyed by no salt."""
    cheat = copy.copy(s)

    def noise(self, ctx):
        return C.Noise(F.PearlC4.noise(self, ctx).seed_a, F.seed_b(salt, ctx.root_b))

    cheat.noise = types.MethodType(noise, cheat)
    return cheat


def test_b_tilde_is_keyed_by_this_jobs_salt(tmp_path):
    """D-SK: B̃'s noise is keyed by the job's salt (seed-B = keyed(salt ‖ root_b)), so B̃ exists only once the salt
    does, and the salt-free base split is two-sided.  The verifier forms B̃ itself from that seed, so a prover whose B̃
    was formed under another epoch's salt, or under none, fails the checked values of a drawn tile, in the work law and
    in the device replay."""
    s = V.scheme("nvfp4", True)
    epoch, prover, tr, rows_a, rows_b = lifecycle(s)
    splits = {9: W.Split(K)}
    records = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits, real_rows={3: 20})
    assert W.audit(s, epoch, prover, tr, splits, records, E.challenger(), 8).audit.accepted
    for n, salt in enumerate((b"another epoch's salt", b"")):
        cheat = stale_b(s, salt)
        assert F.seed_b(salt, prover.weight_roots[9]) != F.seed_b(epoch.salt, prover.weight_roots[9])
        liar = Prover(cheat, {9: rows_b})
        out = W.audit(s, epoch, liar, liar.prove(epoch, {3: rows_a}), splits, records, E.challenger(), 8)
        assert not out.audit.accepted
        assert {v.reason for v in out.audit.verdicts if not v.ok} == {"checked values"}
        device_dump(tmp_path / str(n), cheat)
        rep = R.verify(R.prove(tmp_path / str(n), "r-test", 8, scheme=s), scheme=s)
        assert not rep["accepted"] and rep["reason"] == "checked values"


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
