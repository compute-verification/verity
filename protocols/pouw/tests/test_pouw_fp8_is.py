"""``fp8-is-h100-*``: the pinned vectors, the core against independent implementations, D-2's exact algebra, the two
refutations of C-2's forming reproduced bit for bit (negative controls), and the lifecycle on a test instance."""

from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import fp8_is_vectors as FV
from verity.ml.tc.term import decode_e4m3
from verity.ml.tc.total_fp8 import tc_dot_total_e4m3
from verity_pouw.audit import Epoch, Prover, Sampled, Verifier, Workload, reference_kernel
from verity_pouw.protocol import Context, DomainError, Matmul, PoUWScheme, Shape, Tile
from verity_pouw.schemes import fp8_is as F
from verity_pouw.schemes import get

PINNED = json.loads((Path(__file__).with_name("vectors") / "fp8_is.json").read_text())


def value(code: int) -> Fraction:
    sign, e, m = decode_e4m3(code)
    v = Fraction(m) * Fraction(2) ** e
    return -v if sign else v


def ctx_of(shape, salt: bytes = b"salt", index: int = 3, weight: int = 5) -> Context:
    return Context(hashlib.sha256(salt).digest(), Matmul(index, Shape(*shape), weight),
                   hashlib.sha256(b"root_a").digest(), hashlib.sha256(b"root_b").digest())


def codes(rng: random.Random, k: int, cap: int) -> list[int]:
    return [(rng.getrandbits(1) << 7) | rng.randint(0, cap) for _ in range(k)]


def test_vectors_are_pinned():
    assert FV.vectors() == PINNED


def test_registry_and_certificates():
    for name, layer in (("fp8-is-h100-d2-draft", F.D2Draft), ("fp8-is-h100-fpncp-v0", F.FpNcpV0)):
        s = get(name)
        assert isinstance(s, PoUWScheme) and isinstance(s.forming, F.Forming) and isinstance(s.forming, layer)
        assert s.name == name and s.certificate.gamma0 is None and s.certificate.gamma is None
        (f_a, f_b), (c_a, c_b) = s.forming.units, s.forming.credited
        assert s.fadd == 32 and s.prices == "round8"
        assert s.certificate.omega == Fraction(6 * 8192 - 32 + f_a + f_b, 6 * 8192 - 32 + c_a + c_b)
        sh = Shape(8192, 8192, 16384)
        assert s.wref(sh) == 6 * sh.wmm - 32 * sh.m * sh.n + f_a * sh.m * sh.k + f_b * sh.n * sh.k
    assert get("fp8-is-h100-d2-draft").forming.units == (336, 336)
    assert get("fp8-is-h100-d2-draft").forming.credited == (272, 272)
    assert get("fp8-is-h100-fpncp-v0").forming.units == (384, 384)


def test_domain():
    s = get("fp8-is-h100-d2-draft")
    s.check(Matmul(0, Shape(8192, 8192, 8192), 1))
    for bad in (Shape(4096, 8192, 8192), Shape(8192, 8192, 8200), Shape(8192, 8192, 8196), Shape(8200, 8192, 8192), Shape(8192, 8208, 8192), Shape(8192, 1 << 17, 8192)):
        with pytest.raises(DomainError):
            s.check(Matmul(0, bad, 1))


def test_core_against_independent_implementations():
    """I_τ is the total FP8 step (`tc_dot_total_e4m3`) from +0; R_τ (τ >= 2) is numpy's float32 running sum of the I_τ,
    and R_1 = I_1 is not a checked word."""
    rng = random.Random("core")
    as_f32 = lambda w: np.array([w], dtype=np.uint32).view(np.float32)[0]
    for _ in range(20):
        x, y = codes(rng, 32 * 12, 0x7E), codes(rng, 32 * 12, 0x7E)
        atoms, running = F.cell_words(x, y)
        assert len(atoms) == len(running) == 12 and running[0] is None
        assert atoms == [tc_dot_total_e4m3(F.ATOM, 0, x[32 * t:32 * t + 32], y[32 * t:32 * t + 32]) for t in range(12)]
        r = as_f32(atoms[0])
        for i, rt in zip(atoms[1:], running[1:]):
            r = np.float32(r + as_f32(i))
            assert rt == int(np.array([r], dtype=np.float32).view(np.uint32)[0])


def test_d2_exact_algebra_is_the_product():
    """Before any rounding, D-2's transforms (αH on the left, βHᵀ on the right, the NCP blocks with E2 = E1·P and
    F2 = −P·F1, block 3 negated) multiply to A·Bᵀ exactly: αβ·32 = 1 and the cross terms cancel."""
    layer, k = F.D2Draft(), 128
    pi, inv = F.ncp_perm(k, layer.P_SEED)
    rng = random.Random("algebra")
    a, b = codes(rng, k, layer.a_max), codes(rng, k, layer.b_max)
    ea, eb = rng.randbytes(k), rng.randbytes(k)

    def transformed(row, stream, left):
        scale = Fraction(1, 8) if left else Fraction(1, 4)
        out = []
        for parts, negated in layer.parts(row, stream, pi, inv, left):
            for g in range(0, k, 32):
                for c in range(32):
                    out.append(scale * sum((-1 if F.hadamard_negative(l, c) != n else 1) * value(p[g + l])
                                           for p, n in zip(parts, negated) for l in range(32)))
        return out

    u, v = transformed(a, ea, True), transformed(b, eb, False)
    assert sum(p * q for p, q in zip(u, v)) == sum(value(p) * value(q) for p, q in zip(a, b))


def test_d2_useful_output_tracks_the_product():
    s, k = F.FP8IS(F.D2Draft(), mnk_min=16, tile=(2, 2)), 256
    rng = random.Random("useful")
    a = [codes(rng, k, s.forming.a_max) for _ in range(2)]
    b = [codes(rng, k, s.forming.b_max) for _ in range(2)]
    ctx, tile = ctx_of((64, k, 16)), Tile((0, 1), (0, 1))
    ops = s.encode(ctx, s.noise(ctx), [bytes(r) for r in a], [bytes(r) for r in b], tile)
    got = s.useful(ctx, None, ops, s.checked(ctx, ops))
    for i in range(2):
        for j in range(2):
            want = sum(value(p) * value(q) for p, q in zip(a[i], b[j]))
            scale = sum(abs(value(p) * value(q)) for p, q in zip(a[i], b[j]))
            assert abs(Fraction(got[i][j]) - want) < scale / 8


def test_d2_input_domain_and_trace_binding():
    s, k = F.FP8IS(F.D2Draft(), mnk_min=16, tile=(1, 1)), 128
    ctx, tile = ctx_of((64, k, 16)), Tile((0,), (0,))
    good_a, good_b = bytes([0x68] * k), bytes([0x60] * k)
    assert s.encode(ctx, s.noise(ctx), [bytes([0x69] * k)], [good_b], tile) is None      # |A| > 64 would saturate
    assert s.encode(ctx, s.noise(ctx), [good_a], [bytes([0x61] * k)], tile) is None      # |B| > 32 would saturate
    assert s.encode(ctx, s.noise(ctx), [bytes([0x7F] * k)], [good_b], tile) is None      # NaN
    ops = s.encode(ctx, s.noise(ctx), [good_a], [good_b], tile)
    assert all(c & 0x7F < 0x7E for c in ops.x[0] + ops.y[0])                             # no transform saturates at the cap
    checked = s.checked(ctx, ops)
    assert len(ops.trace_x[0]) == len(ops.trace_y[0]) == 7 * k * 4 + 3 * k
    forged = F.Checked(bytes([checked.trace[0] ^ 1]) + checked.trace[1:], checked.cells)
    assert s.digest(ctx, None, forged) != s.digest(ctx, None, checked)


def one_spike(k: int, index: int, code: int) -> list[int]:
    row = [0] * k
    row[index] = code
    return row


def spike_slice_words(layer, k: int, spike_a: int, spike_b: int, salts: int):
    """For the one-spike input (track D, X-D1-1), each block's atom over the spike's slice, and its live products."""
    index, out = 100, []
    for s in range(salts):
        ctx = ctx_of((64, k, 16), salt=bytes([s]))
        st = layer.form(ctx, layer.noise(ctx), [one_spike(k, index, spike_a)], [one_spike(k, index, spike_b)], Tile((0,), (0,)))
        sl = [index // 32 + blk * (k // 32) for blk in range(3)]
        x, y = st.x[0], st.y[0]
        out.append(tuple((F.ATOM.step(0, x[32 * t:32 * t + 32], y[32 * t:32 * t + 32]),
                          F.live_products(x[32 * t:32 * t + 32], y[32 * t:32 * t + 32])) for t in sl))
    return out


def test_fpncp_v0_one_spike_is_salt_independent():
    """Negative control (X-D1-1): under C-2's forming as posted, a legal one-spike row and column make every block's
    spike-slice atom one live product, the same word for every salt (track D's replay words)."""
    runs = spike_slice_words(F.FpNcpV0(), 4096, 0x7E, 0x7E, salts=3)
    assert len(set(runs)) == 1
    assert runs[0] == ((0x48440000, 1), (0x48440000, 1), (0xC8440000, 1))


def test_fpncp_v0_rounds_row_scaled_noise_away():
    """Negative control (X-C2-B1): one 448 entry and 14 slices of 36 make σ = 1.71 below the half-cell 2 at 36, so blocks
    1 and 2 of those slices have the same codes for every salt, and so do their atoms."""
    layer, k = F.FpNcpV0(), 4096
    row = [0x7E] + [0x61] * (32 * 14 - 1) + [0] * (k - 32 * 14)
    stacks = []
    for s in range(3):
        ctx = ctx_of((64, k, 16), salt=bytes([s]))
        stacks.append(layer.form(ctx, layer.noise(ctx), [row], [row], Tile((0,), (0,))))
    for blk in (0, 1):
        spans = {tuple(st.x[0][blk * k + 32:blk * k + 32 * 14]) for st in stacks}
        assert len(spans) == 1 and set(next(iter(spans))) == {0x61}
    assert len({tuple(st.x[0][2 * k + 32:2 * k + 32 * 14]) for st in stacks}) > 1                  # block 3: 2σ survives


def test_d2_draft_keeps_the_spike_slices_dense_and_salted():
    runs = spike_slice_words(F.D2Draft(), 1024, 0x68, 0x60, salts=2)
    assert all(live > 8 for run in runs for _, live in run)
    assert runs[0][0][0] != runs[1][0][0]


def test_lifecycle_on_a_test_instance():
    s = F.FP8IS(F.FpNcpV0(), mnk_min=16, tile=(64, 32))
    rng = random.Random("lifecycle")
    work = Workload((Matmul(0, Shape(64, 128, 64), 7),))
    weights = {7: [bytes(codes(rng, 128, 0x7E)) for _ in range(64)]}
    acts = {0: [bytes(codes(rng, 128, 0x7E)) for _ in range(64)]}

    def corrupt(scheme, ctx, noise, a, b, tiles):
        out = reference_kernel(scheme, ctx, noise, a, b, tiles)
        out[1] = (bytes(32), out[1][1])
        return out

    for kernel, ok in ((reference_kernel, True), (corrupt, False)):
        prover = Prover(s, weights, kernel)
        epoch = Epoch.start(work, prover.weight_roots, b"beacon")
        tr = prover.prove(epoch, acts)
        v = Verifier(s, epoch)
        assert [v.check_tile(tr, prover.open(epoch, t.id)).ok for t in v.refs] == [True, ok]
        if ok:
            audit = v.audit(prover, tr, Sampled(2), b"source")
            assert audit.accepted and audit.profile.work == s.wref(Shape(64, 128, 64))


class SeededNcp(F.FpNcpV0):
    """A test layer: fpncp-v0 with each cell's R_0 a salt word (a small power of two, so the seed is removed exactly)."""

    name, seeded = "fpncp-v0-seeded", True

    def form(self, ctx, noise, a, b, tile):
        st = super().form(ctx, noise, a, b, tile)
        seeds = [[0x3F80_0000 | (F.xof("test/seed", 1, ctx.salt, i, j)[0] << 23 & 0x8000_0000) for j in tile.cols] for i in tile.rows]
        return F.Stack(st.x, st.y, st.trace_x, st.trace_y, seeds)


def test_a_seeded_layer_checks_r1():
    """Unseeded, R_1 = I_1 is dropped (`c2FirstRunCopy`); a seeded layer checks R_1 = add(R_0, I_1), pays its FADD in wref,
    and its useful output removes the seed."""
    rng = random.Random("seeded")
    k, tile = 128, Tile((0, 1), (0, 1))
    a, b = [bytes(codes(rng, k, 0x7E)) for _ in range(2)], [bytes(codes(rng, k, 0x7E)) for _ in range(2)]
    ctx = ctx_of((64, k, 64))
    plain, seeded = F.FP8IS(F.FpNcpV0(), mnk_min=16), F.FP8IS(SeededNcp(), mnk_min=16)
    p_ops, s_ops = plain.encode(ctx, plain.noise(ctx), a, b, tile), seeded.encode(ctx, seeded.noise(ctx), a, b, tile)
    p_chk, s_chk = plain.checked(ctx, p_ops), seeded.checked(ctx, s_ops)
    for i in range(2):
        for j in range(2):
            (pa, pr), (sa, sr) = p_chk.cells[i][j], s_chk.cells[i][j]
            assert pa == sa and pr[0] is None and sr[0] == F.fp32.add(s_ops.seeds[i][j], sa[0]) != sa[0]
            assert None not in pr[1:] and None not in sr
    for got, want in zip(sum(seeded.useful(ctx, None, s_ops, s_chk), []), sum(plain.useful(ctx, None, p_ops, p_chk), [])):
        assert abs(got - want) <= 1 + abs(want) / 2 ** 20
    sh = Shape(64, 128, 64)
    assert seeded.wref(sh) - plain.wref(sh) == 32 * sh.m * sh.n

    class Mislabelled(SeededNcp):
        seeded = False

    liar = F.FP8IS(Mislabelled(), mnk_min=16)
    with pytest.raises(DomainError):
        liar.encode(ctx, liar.noise(ctx), a, b, tile)


# ---- d3-v0: repaired C-2 --------------------------------------------------------------------------------------------

def f16_value(w: int) -> Fraction:
    neg, m, e = F.fp16.decode(w)
    v = Fraction(m) * Fraction(2) ** e
    return -v if neg else v


def test_d3_registry_price_and_domain():
    s = get("fp8-is-h100-d3-v0")
    assert isinstance(s.forming, F.D3) and not s.forming.seeded
    assert s.forming.units == (372, 372) and s.forming.measured_units == Fraction(9893, 20)
    assert dict(s.certificate.parameters)["trace_units"] == 372
    claimed = F.FP8IS(F.D3(form_units=192))
    assert claimed.certificate.omega == Fraction(6 * 8192 - 32 + 384, 6 * 8192 - 32)
    s.check(Matmul(0, Shape(8192, 8192, 16384), 1))


def test_d3_exact_algebra_is_the_scaled_product():
    """Before any rounding: with the scheme's own f16 noise E1 (partner-max, per-entry) and F1 (column-keyed),
    E2 = E1·P, F2 = −P·F1 and exact E3, F3, the three blocks multiply to (αa)·(βb) exactly."""
    layer, k = F.D3(), 128
    pi, inv = F.ncp_perm(k, layer.P_SEED)
    rng = random.Random("d3-algebra")
    a, b = codes(rng, k, 0x7E), codes(rng, k, 0x7E)
    ctx = ctx_of((64, k, 64))

    def noise(row, stream, left):
        h = [F.fp16.from_e4m3(c) for c in row]
        alpha, sigma = layer.scale(h)
        y = [f16_value(F.fp16.mul(alpha, w)) for w in h]
        st = [F.fp16.max_(F.fp16.mul(alpha, w) & 0x7C00, F.F16_2_POW_M6) for w in h]
        partner = inv if left else pi
        s = [F.fp16.max_(st[l], st[partner[l]]) for l in range(k)]
        two_sigma, c = F.fp16.mul(sigma, 0x4000), F.fp16.mul(sigma, 0xC200)
        e = []
        for l in range(k):
            w = int.from_bytes(stream[4 * l:4 * l + 4], "little")
            su = F.fp16.fma(two_sigma, F.F16_ONE | (w & 0x3FF), c)
            e.append(f16_value(F.fp16.fma(s[l], F.fp16.fma(F.F16_TWO_INV4, F.F16_ONE | ((w >> 16) & 0x3FF), F.F16_MINUS_3_8), su)))
            assert f16_value(s[l]) >= 8 * Fraction(1, 512)          # s·v/8 spans at least one E4M3 cell
        return y, e

    y, e1 = noise(a, F.xof("verity/pouw/fp8-is/d3/E1", 4 * k, ctx.salt, 3, ctx.root_a, 5, 0), True)
    z, f1 = noise(b, F.xof("verity/pouw/fp8-is/d3/F1", 4 * k, ctx.salt, 5, 0), False)
    e2 = [e1[pi[l]] for l in range(k)]
    f2 = [-f1[inv[l]] for l in range(k)]
    blocks = ((e1, f1, 1), (e2, f2, 1), ([p + q for p, q in zip(e1, e2)], [p + q for p, q in zip(f1, f2)], -1))
    total = sum(sign * (y[l] + e[l]) * (z[l] + f[l]) for e, f, sign in blocks for l in range(k))
    assert total == sum(p * q for p, q in zip(y, z))


def test_d3_repairs_both_refuted_inputs():
    """X-C2-B1's absorbed-noise row and X-D1-1's one spike, which `fpncp-v0` leaves salt-independent, give salt-dependent
    codes and words under d3-v0."""
    layer, k = F.D3(), 4096
    row = [0x7E] + [0x61] * (32 * 14 - 1) + [0] * (k - 32 * 14)
    stacks = [layer.form(c, layer.noise(c), [row], [row], Tile((0,), (0,)))
              for c in (ctx_of((64, k, 64), salt=bytes([s])) for s in range(3))]
    for blk in (0, 1):
        assert len({tuple(st.x[0][blk * k + 32:blk * k + 32 * 14]) for st in stacks}) == 3
    runs = spike_slice_words(layer, k, 0x7E, 0x7E, salts=3)
    assert len({tuple(w for w, _ in run) for run in runs}) == 3


def test_d3_identical_weight_columns_are_distinct():
    """X-D2-5: weight noise is keyed per column, so two identical committed weight rows give different formed columns."""
    layer, k = F.D3(), 128
    rng = random.Random("d3-columns")
    w, act = codes(rng, k, 0x7E), codes(rng, k, 0x7E)
    ctx = ctx_of((64, k, 64))
    st = layer.form(ctx, layer.noise(ctx), [act], [w, w], Tile((0,), (0, 1)))
    assert st.y[0] != st.y[1]
    assert F.cell_words(st.x[0], st.y[0]) != F.cell_words(st.x[0], st.y[1])


def test_d3_useful_output_tracks_the_product():
    s, k = F.FP8IS(F.D3(), mnk_min=16, tile=(2, 2)), 256
    rng = random.Random("d3-useful")
    a, b = [codes(rng, k, 0x6E) for _ in range(2)], [codes(rng, k, 0x6E) for _ in range(2)]
    ctx, tile = ctx_of((64, k, 64)), Tile((0, 1), (0, 1))
    ops = s.encode(ctx, s.noise(ctx), [bytes(r) for r in a], [bytes(r) for r in b], tile)
    got = s.useful(ctx, None, ops, s.checked(ctx, ops))
    for i in range(2):
        for j in range(2):
            scale = f16_value(s.forming.scale([F.fp16.from_e4m3(c) for c in a[i]])[0]) * \
                    f16_value(s.forming.scale([F.fp16.from_e4m3(c) for c in b[j]])[0])
            want = sum(value(p) * value(q) for p, q in zip(a[i], b[j]))
            bound = sum(abs(value(p) * value(q)) for p, q in zip(a[i], b[j]))
            assert abs(Fraction(got[i][j]) / scale - want) < bound / 8


# ---- d3m-v0: the lead (D-3 with the max law, on X_q) -------------------------------------------------------------------

def test_d3m_variants_prices_and_gamma():
    """Both weight-copy variants exist; their prices are schedule M4's at Round 8 (426 + a 17/64 chain share per activation,
    290 or 322 per weight), and their certificates give the strike's γ at 16,384³ with r = 1."""
    copy, decode = get("fp8-is-h100-d3m-v0"), get("fp8-is-h100-d3m-v0-decode")
    assert copy.forming.units == (Fraction(27281, 64), 290) and decode.forming.units == (Fraction(27281, 64), 322)
    assert copy.forming.tag == decode.forming.tag == "d3m-v0"
    for weight_copy, want in ((True, "0.972"), (False, "1.004")):
        s = F.FP8IS(F.D3m(weight_copy), mnk_min=16384)
        gamma = 1 - Fraction(399, 400) / s.certificate.omega
        assert f"{float(100 * gamma):.3f}" == want
    r7 = F.D3m(True, prices="round7")
    assert r7.units[0] > 450 and F.FP8IS(r7, prices="round7").fadd == Fraction(67, 2)
    sh = Shape(16384, 16384, 16384)
    assert copy.wref(sh) == -(-(sh.m * sh.n * (32 * 1536 + 32 * 1535) + Fraction(27281, 64) * sh.m * sh.k
                              + 290 * sh.n * sh.k) // 1)


def d3m_noise(layer, row, stream, left, pi, inv):
    """Recompute e (exact values) and the amplitude A8/8 per entry from the layer's own f16 steps."""
    h = [F.fp16.from_e4m3(c) for c in row]
    s8 = layer.sigma8(h)
    c8 = [w & 0x7C00 for w in h]
    partner = inv if left else pi
    amp, e = [], []
    for l in range(len(row)):
        a8 = F.fp16.max_(F.fp16.max_(c8[l], c8[partner[l]]), s8)
        t = F.F16_ONE | (int.from_bytes(stream[2 * l:2 * l + 2], "little") & 0x3FF)
        amp.append(f16_value(a8) / 8)
        e.append(f16_value(F.fp16.mul(a8, F.fp16.fma(F.F16_TWO_INV4, t, F.F16_MINUS_3_8))))
    return [value(c) for c in row], e, amp


def e4m3_cell(v: Fraction) -> Fraction:
    a = abs(v)
    if a < Fraction(1, 64):
        return Fraction(1, 512)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    return Fraction(2) ** (e - 3)


def test_d3m_exact_algebra_and_amplitude():
    """Before any rounding the three blocks multiply to X_q·W_qᵀ exactly, and every entry's amplitude covers one E4M3 cell
    of itself and of its NCP partner, on the cast's grid (X-17d-3)."""
    layer, k = F.D3m(), 128
    pi, inv = F.ncp_perm(k, layer.P_SEED)
    rng = random.Random("d3m-algebra")
    a, b = codes(rng, k, 0x7E), codes(rng, k, 0x7E)
    ctx = ctx_of((64, k, 64))
    x, e1, amp_a = d3m_noise(layer, a, F.xof("verity/pouw/fp8-is/d3m/E1", 2 * k, ctx.salt, 3, ctx.root_a, 5, 0), True, pi, inv)
    y, f1, amp_b = d3m_noise(layer, b, F.xof("verity/pouw/fp8-is/d3m/F1", 2 * k, ctx.salt, 3, 5, 0), False, pi, inv)
    for l in range(k):
        assert amp_a[l] >= max(e4m3_cell(x[l]), e4m3_cell(x[inv[l]])) and amp_b[l] >= max(e4m3_cell(y[l]), e4m3_cell(y[pi[l]]))
    e2 = [e1[pi[l]] for l in range(k)]
    f2 = [-f1[inv[l]] for l in range(k)]
    blocks = ((e1, f1, 1), (e2, f2, 1), ([p + q for p, q in zip(e1, e2)], [p + q for p, q in zip(f1, f2)], -1))
    assert sum(sign * (x[l] + e[l]) * (y[l] + f[l]) for e, f, sign in blocks for l in range(k)) == sum(p * q for p, q in zip(x, y))


def test_d3m_repairs_the_refuted_inputs_and_x_d3m_4():
    """X-C2-B1's absorbed row, X-D1-1's spike and X-D3m-4's tiny column (one code 2^−9, the rest zero, as a per-tensor
    scale would give) all give salt-dependent codes: the max law and the σ floor leave no free word."""
    layer, k = F.D3m(), 4096
    absorbed = [0x7E] + [0x61] * (32 * 14 - 1) + [0] * (k - 32 * 14)
    tiny = [0x01] + [0] * (k - 1)
    for row in (absorbed, tiny):
        stacks = [layer.form(c, layer.noise(c), [row], [row], Tile((0,), (0,)))
                  for c in (ctx_of((64, k, 64), salt=bytes([s])) for s in range(3))]
        for blk in range(3):
            assert len({tuple(st.x[0][blk * k + 32:blk * k + 32 * 14]) for st in stacks}) == 3
            assert len({tuple(st.y[0][blk * k + 32:blk * k + 32 * 14]) for st in stacks}) == 3
    runs = spike_slice_words(layer, k, 0x7E, 0x7E, salts=3)
    assert len({tuple(w for w, _ in run) for run in runs}) > 1


def test_d3m_zero_rows_and_saturation():
    """An all-zero row gets σ8 = max(f16(224/√k), 2^−6) and a near-zero useful output; a row of 448s saturates at ±448 and
    never produces a NaN code."""
    s, k = F.FP8IS(F.D3m(), mnk_min=16, tile=(2, 1)), 1024
    zero, full = bytes(k), bytes([0x7E] * k)
    rng = random.Random("d3m-zero")
    w = bytes(codes(rng, k, 0x7E))
    ctx = ctx_of((64, k, 64))
    assert s.forming.sigma8([0] * k) == F.fp16.round_to_f16(False, 7, 0)          # 224/√1024 = 7
    ops = s.encode(ctx, s.noise(ctx), [zero, full], [w], Tile((0, 1), (0,)))
    assert len(set(ops.x[0][:k])) > 1                                             # zero entries carry noise
    assert all(c & 0x7F != 0x7F for c in ops.x[1]) and (0x7E in ops.x[1] or 0xFE in ops.x[1])
    useful = s.useful(ctx, None, ops, s.checked(ctx, ops))
    scale = sum(abs(value(c)) * 7 for c in w)
    assert abs(useful[0][0]) < scale / 8


def test_d3m_weight_noise_is_per_unit():
    """r = 1 (X-D3-3): two units that share a weight form different weight columns."""
    layer, k = F.D3m(), 128
    rng = random.Random("d3m-units")
    w, act = codes(rng, k, 0x7E), codes(rng, k, 0x7E)
    one, two = ctx_of((64, k, 64)), Context(ctx_of((64, k, 64)).salt, Matmul(4, Shape(64, k, 64), 5),
                                              ctx_of((64, k, 64)).root_a, ctx_of((64, k, 64)).root_b)
    assert layer.form(one, layer.noise(one), [act], [w], Tile((0,), (0,))).y != \
        layer.form(two, layer.noise(two), [act], [w], Tile((0,), (0,))).y


def test_d3m_weight_copy_variants_share_the_checked_function():
    rng = random.Random("d3m-copy")
    k, tile = 128, Tile((0, 1), (0, 1))
    a, b = [bytes(codes(rng, k, 0x7E)) for _ in range(2)], [bytes(codes(rng, k, 0x7E)) for _ in range(2)]
    ctx = ctx_of((64, k, 64))
    digests = []
    for name in ("fp8-is-h100-d3m-v0", "fp8-is-h100-d3m-v0-decode"):
        s = F.FP8IS(F.FORMINGS[name.removeprefix("fp8-is-h100-")](), mnk_min=16)
        ops = s.encode(ctx, s.noise(ctx), a, b, tile)
        digests.append(s.digest(ctx, None, s.checked(ctx, ops)))
    assert digests[0] == digests[1]


def test_d3m_useful_output_tracks_the_product():
    s, k = F.FP8IS(F.D3m(), mnk_min=16, tile=(2, 2)), 512
    rng = random.Random("d3m-useful")
    a, b = [codes(rng, k, 0x6E) for _ in range(2)], [codes(rng, k, 0x6E) for _ in range(2)]
    ctx, tile = ctx_of((64, k, 64)), Tile((0, 1), (0, 1))
    ops = s.encode(ctx, s.noise(ctx), [bytes(r) for r in a], [bytes(r) for r in b], tile)
    got = s.useful(ctx, None, ops, s.checked(ctx, ops))
    for i in range(2):
        for j in range(2):
            want = sum(value(p) * value(q) for p, q in zip(a[i], b[j]))
            bound = sum(abs(value(p) * value(q)) for p, q in zip(a[i], b[j]))
            assert abs(Fraction(got[i][j]) - want) < bound / 8
