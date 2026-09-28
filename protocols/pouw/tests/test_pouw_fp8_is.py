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
        assert s.prices == "round9" and s.fadd == Fraction(3211, 100)
        t = 3 * 8192 // 32
        assert s.certificate.omega == Fraction(32 * t + s.fadd * (t - 1) + f_a + f_b, 32 * (2 * t - 1) + c_a + c_b)
        sh = Shape(8192, 8192, 16384)
        t = 3 * sh.k // 32
        assert s.wref(sh) == -(-(sh.m * sh.n * (32 * t + s.fadd * (t - 1)) + f_a * sh.m * sh.k + f_b * sh.n * sh.k) // 1)
        old = get(name, prices="round8")
        assert old.fadd == 32 and old.wref(sh) == 6 * sh.wmm - 32 * sh.m * sh.n + f_a * sh.m * sh.k + f_b * sh.n * sh.k
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
        a, b = layer.parse(bytes(one_spike(k, index, spike_a)), k, True), layer.parse(bytes(one_spike(k, index, spike_b)), k, False)
        st = layer.form(ctx, layer.noise(ctx), [a], [b], Tile((0,), (0,)))
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
    plain, seeded = F.FP8IS(F.FpNcpV0(), mnk_min=16, prices="round8"), F.FP8IS(SeededNcp(), mnk_min=16, prices="round8")
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
    claimed = F.FP8IS(F.D3(form_units=192), prices="round8")
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


# ---- the D-3 family: d3s-v0 (the lead law) and d3m-v0, on X_q or Y16 -------------------------------------------------

def y16_row(rng: random.Random, k: int) -> bytes:
    """A Y16 row: f16 words of magnitude at most 448 with full 10-bit mantissas."""
    words = []
    for _ in range(k):
        w = (rng.getrandbits(1) << 15) | (rng.randint(1, 23) << 10) | rng.getrandbits(10)
        words.append(min(w & 0x7FFF, F.F16_448) | (w & 0x8000))
    return b"".join(w.to_bytes(2, "little") for w in words)


def e4m3_cell(v: Fraction) -> Fraction:
    a = abs(v)
    if a < Fraction(1, 64):
        return Fraction(1, 512)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    return Fraction(2) ** (e - 3)


CELLS = [(law, route, copy) for law in ("d3s", "d3m") for route in ("xq", "y16") for copy in (True, False)]


def cell_name(law: str, route: str, copy: bool, sums: str = "f32") -> str:
    return f"{law}-v0{'-y16' if route == 'y16' else ''}{'-f16' if sums == 'f16' else ''}{'' if copy else '-decode'}"


def gamma_at(s, n: int) -> Fraction:
    return 1 - Fraction(399, 400) / F.FP8IS(s.forming, mnk_min=n, prices=s.prices).certificate.omega


@pytest.mark.parametrize("law,route,copy", CELLS)
def test_d3_family_round8_reproduces_the_strike(law, route, copy):
    """At Round 8 each cell's price is schedule M4's (D-3s drops the cell mask), and its certificate gives Phase 17f's γ
    at 16,384³ with r = 1 (within the 0.27-unit chain share the red team does not count)."""
    s = get(f"fp8-is-h100-{cell_name(law, route, copy)}", prices="round8")
    assert s.forming.law == law and s.forming.route == route and s.forming.weight_copy == copy
    assert s.forming.prices == s.prices == "round8"
    act = Fraction(27281, 64) - (32 if law == "d3s" else 0)
    assert s.forming.units == (act, 290 if copy else 322)
    want = {("d3s", True): "0.940", ("d3s", False): "0.972", ("d3m", True): "0.972", ("d3m", False): "1.004"}[law, copy]
    assert f"{float(100 * gamma_at(s, 16384)):.3f}" == want
    assert s.forming.tag == cell_name(law, route, copy).removesuffix("-decode")


def round9_units(law: str, route: str, copy: bool, sums: str, sass: bool) -> tuple[Fraction, Fraction]:
    """Round 9's per-element trace from §14's figures (not from `PRICES`): FADD 32.11, half2 36, HMNMX2 64, LOP3 64,
    F2FP 64, HMUL2 a·K 32.17, HADD2.F32 64, the four-code unpack 48 per code, the f16 pack 32 per element; the f16 -> e4m3
    cast 64 per code (Round 8, F2FP + PRMT), or 32 with ``sass`` (ptxas 12.9's PRMT-free pair), when the amplitude's two
    maxima are also one VHMNMX (32 per element)."""
    fadd, half, lop = Fraction(3211, 100), Fraction(18), Fraction(32)
    noise = lop + 3 * half                                    # field, centre, scale, s = e + e∘P
    if sums == "f32":
        blocks, widen = 3 * (64 + fadd) + 3 * 32, Fraction(64)
    else:
        blocks, widen = 3 * half + 3 * (32 if sass else 64), Fraction(0)
    act = ((48 if route == "xq" else 32) + Fraction(3217, 200) + half + (64 + fadd) / 256
           + (32 if law == "d3m" else 0) + (32 if sass else 64) + widen + noise + blocks)
    weight = noise + blocks + (0 if copy else 48 + widen)
    return act, weight


R9_CELLS = [(law, route, copy, sums) for law in ("d3s", "d3m") for route in ("xq", "y16") for copy in (True, False)
            for sums in ("f32", "f16")]


@pytest.mark.parametrize("law,route,copy,sums", R9_CELLS)
def test_d3_family_round9_prices(law, route, copy, sums):
    """Every cell is registered with its Round 9 trace, measured (the default) and with ptxas 12.9's cast and max."""
    for prices, sass in (("round9", False), ("round9-sass", True)):
        s = get(f"fp8-is-h100-{cell_name(law, route, copy, sums)}", prices=prices)
        assert s.forming.sums == sums and s.forming.prices == s.prices == prices
        assert s.forming.units == round9_units(law, route, copy, sums, sass)
    assert get(f"fp8-is-h100-{cell_name(law, route, copy, sums)}").prices == "round9"


#: γ in % at N³ with r = 1 and FADD at 32.11, every checked word credited at 32 (the certificate's Ω)
ROUND9_GAMMA = {
    ("d3s", "y16", True, "f32"): ("1.556", "0.992", "0.707"),
    ("d3s", "y16", False, "f32"): ("1.667", "1.048", "0.735"),
    ("d3s", "xq", True, "f32"): ("1.572", "1.000", "0.711"),
    ("d3s", "xq", False, "f32"): ("1.683", "1.056", "0.739"),
    ("d3s", "y16", True, "f16"): ("1.218", "0.821", "0.622"),
    ("d3s", "y16", False, "f16"): ("1.266", "0.845", "0.634"),
    ("d3s", "xq", True, "f16"): ("1.234", "0.829", "0.626"),
    ("d3s", "xq", False, "f16"): ("1.282", "0.853", "0.638"),
}


@pytest.mark.parametrize("cell", sorted(ROUND9_GAMMA))
def test_round9_gamma_at_16384_32768_65536(cell):
    """Nothing clears at 16,384³; D-4 (Track D: D-3s with 32,768 <= m, k, n <= 65,536, ``mnk_min=32768``) clears at
    32,768³ only on Y16 with the copy under f32 sums (by 0.8%), and in every f16-sums cell."""
    s = get(f"fp8-is-h100-{cell_name(*cell)}")
    assert tuple(f"{float(100 * gamma_at(s, n)):.3f}" for n in (16384, 32768, 65536)) == ROUND9_GAMMA[cell]


def test_round9_sass_best_cell_at_16384():
    """With ptxas 12.9's cast and max, D-3s on Y16 with f16 sums and the copy clears 16,384³ at FADD 32.11 by 0.5%."""
    s = get("fp8-is-h100-d3s-v0-y16-f16", prices="round9-sass")
    assert f"{float(100 * gamma_at(s, 16384)):.3f}" == "0.995"
    assert f"{float(100 * gamma_at(s, 32768)):.3f}" == "0.709"


def test_one_price_row_prices_forming_and_core():
    assert get("fp8-is-h100-d3s-v0-y16", prices="round8").forming.prices == "round8"
    assert F.FP8IS(F.D3s(prices="round8")).prices == "round8"
    with pytest.raises(ValueError):
        F.FP8IS(F.D3s(prices="round8"), prices="round9")


def test_d4_domain_is_selected_by_mnk_min():
    s = get("fp8-is-h100-d3s-v0-y16", mnk_min=32768)
    for ok in (Shape(32768, 32768, 32768), Shape(65536, 65536, 65536), Shape(32768, 65536, 49152)):
        s.check(Matmul(0, ok, 1))
    for bad in (Shape(16384, 32768, 32768), Shape(32768, 32768, 1 << 17), Shape(32768, 32768, 32800)):
        with pytest.raises(DomainError):
            s.check(Matmul(0, bad, 1))
    assert dict(s.certificate.parameters)["mnk_min"] == 32768


def test_f16_sums_round_once_in_f16_then_cast():
    """With f16 sums each code is cvt.rn.satfinite.e4m3x2.f16x2 of HADD2(h, e_b); it differs from the f32-sums code only
    by double rounding, and a 448 row still saturates at ±448."""
    k = 128
    rng = random.Random("f16-sums")
    f32, f16 = F.D3s(route="xq"), F.D3s(route="xq", sums="f16")
    pi, inv = F.ncp_perm(k, f32.P_SEED)
    stream = F.xof("test/f16-sums", 2 * k)
    for left in (True, False):
        h = f32.parse(bytes(codes(rng, k, 0x7E)), k, left)
        u, v = f32.side(h, stream, pi, inv, left), f16.side(h, stream, pi, inv, left)
        diff = [l for l in range(3 * k) if u[l] != v[l]]
        assert len(diff) < 3 * k // 16
        assert all(abs(value(u[l]) - value(v[l])) <= max(e4m3_cell(value(u[l])), e4m3_cell(value(v[l]))) for l in diff)
    h = [F.fp16.round_to_f16(False, 7, 6)] * k                                    # 448
    ctx = ctx_of((64, k, 64))
    st = f16.form(ctx, f16.noise(ctx), [h], [h], Tile((0,), (0,)))
    assert all(c & 0x7F != 0x7F for c in st.x[0] + st.y[0]) and 0x7E in st.x[0] + st.y[0]


def noise_of(layer, h, stream, left, pi, inv):
    amp = layer.amplitudes(h, inv if left else pi)
    e = []
    for l in range(len(h)):
        t = F.F16_ONE | (int.from_bytes(stream[2 * l:2 * l + 2], "little") & 0x3FF)
        e.append(f16_value(F.fp16.mul(amp[l], F.fp16.fma(F.F16_TWO_INV4, t, F.F16_MINUS_3_8))))
    return [f16_value(w) for w in h], e, [f16_value(a) / 8 for a in amp]


@pytest.mark.parametrize("law,route", [(law, route) for law in ("d3s", "d3m") for route in ("xq", "y16")])
def test_d3_family_exact_algebra_and_amplitude(law, route):
    """Before any rounding the three blocks multiply to h·wᵀ exactly, and every amplitude covers one E4M3 cell of the
    entry and of its NCP partner, on the cast's grid (X-17d-3)."""
    layer, k = F.D3Family(law, route), 128
    pi, inv = F.ncp_perm(k, layer.P_SEED)
    rng = random.Random(f"{law}-{route}-algebra")
    a = layer.parse(y16_row(rng, k) if route == "y16" else bytes(codes(rng, k, 0x7E)), k, True)
    b = layer.parse(bytes(codes(rng, k, 0x7E)), k, False)
    ctx = ctx_of((64, k, 64))
    x, e1, amp_a = noise_of(layer, a, F.xof(f"verity/pouw/fp8-is/{layer.tag}/E1", 2 * k, ctx.salt, 3, ctx.root_a, 5, 0),
                            True, pi, inv)
    y, f1, amp_b = noise_of(layer, b, F.xof(f"verity/pouw/fp8-is/{layer.tag}/F1", 2 * k, ctx.salt, 3, 5, 0), False, pi, inv)
    for l in range(k):
        assert amp_a[l] >= max(e4m3_cell(x[l]), e4m3_cell(x[inv[l]])) and amp_b[l] >= max(e4m3_cell(y[l]), e4m3_cell(y[pi[l]]))
    e2 = [e1[pi[l]] for l in range(k)]
    f2 = [-f1[inv[l]] for l in range(k)]
    blocks = ((e1, f1, 1), (e2, f2, 1), ([p + q for p, q in zip(e1, e2)], [p + q for p, q in zip(f1, f2)], -1))
    assert sum(sign * (x[l] + e[l]) * (y[l] + f[l]) for e, f, sign in blocks for l in range(k)) == sum(p * q for p, q in zip(x, y))


@pytest.mark.parametrize("law", ["d3s", "d3m"])
def test_d3_family_repairs_the_refuted_inputs(law):
    """X-C2-B1's absorbed row, X-D3m-4's tiny column (one code 2^−9, the rest zero) and X-D1-1's spike all give
    salt-dependent codes: no free word."""
    layer, k = F.D3Family(law), 4096
    for row in ([0x7E] + [0x61] * (32 * 14 - 1) + [0] * (k - 32 * 14), [0x01] + [0] * (k - 1)):
        h = layer.parse(bytes(row), k, True)
        stacks = [layer.form(c, layer.noise(c), [h], [h], Tile((0,), (0,)))
                  for c in (ctx_of((64, k, 64), salt=bytes([s])) for s in range(3))]
        for blk in range(3):
            assert len({tuple(st.x[0][blk * k + 32:blk * k + 32 * 14]) for st in stacks}) == 3
            assert len({tuple(st.y[0][blk * k + 32:blk * k + 32 * 14]) for st in stacks}) == 3
    runs = spike_slice_words(layer, k, 0x7E, 0x7E, salts=3)
    assert len({tuple(w for w, _ in run) for run in runs}) > 1


def test_d3s_zero_row_floor_and_saturation():
    """D-3s's floor rms′ = max(rms, 2^−6) gives an all-zero row noise of one subnormal cell and a near-zero output; a row of
    448s saturates at ±448 and never produces a NaN code."""
    s, k = F.FP8IS(F.D3s(), mnk_min=16, tile=(2, 1)), 1024
    rng = random.Random("d3s-zero")
    w = bytes(codes(rng, k, 0x7E))
    ctx = ctx_of((64, k, 64))
    assert s.forming.floor([0] * k) == F.F16_2_POW_M6
    ops = s.encode(ctx, s.noise(ctx), [bytes(k), bytes([0x7E] * k)], [w], Tile((0, 1), (0,)))
    assert set(ops.x[0][:k]) - {0x00, 0x80} and set(ops.x[0][:k]) <= {0x00, 0x80, 0x01, 0x81}
    assert all(c & 0x7F != 0x7F for c in ops.x[1]) and (0x7E in ops.x[1] or 0xFE in ops.x[1])
    useful = s.useful(ctx, None, ops, s.checked(ctx, ops))
    assert abs(useful[0][0]) < sum(abs(value(c)) for c in w) / 64


def test_y16_rows_are_parsed_and_bounded():
    layer, k = F.D3s(route="y16"), 64
    ok = F.fp16.round_to_f16(False, 7, 6)                                       # 448
    assert layer.parse(ok.to_bytes(2, "little") * k, k, True) == [ok] * k
    for bad in (F.fp16.round_to_f16(False, 449, 0), 0x7C00, 0x7E00):            # above 448, +inf, NaN
        assert layer.parse(bad.to_bytes(2, "little") * k, k, True) is None
    assert layer.parse(bytes(k), k, True) is None                               # a Y16 row is 2k bytes
    assert layer.parse(bytes(k), k, False) == [0] * k                           # weights stay E4M3 codes


@pytest.mark.parametrize("law", ["d3s", "d3m"])
def test_d3_family_weight_noise_is_per_unit(law):
    """r = 1 (X-D3-3): two units that share a weight form different weight columns."""
    layer, k = F.D3Family(law), 128
    rng = random.Random("d3-units")
    w, act = layer.parse(bytes(codes(rng, k, 0x7E)), k, False), layer.parse(bytes(codes(rng, k, 0x7E)), k, True)
    one = ctx_of((64, k, 64))
    two = Context(one.salt, Matmul(4, Shape(64, k, 64), 5), one.root_a, one.root_b)
    assert layer.form(one, layer.noise(one), [act], [w], Tile((0,), (0,))).y != \
        layer.form(two, layer.noise(two), [act], [w], Tile((0,), (0,))).y


@pytest.mark.parametrize("law,route", [(law, route) for law in ("d3s", "d3m") for route in ("xq", "y16")])
def test_d3_family_weight_copy_variants_share_the_checked_function(law, route):
    rng = random.Random("d3-copy")
    k, tile = 128, Tile((0, 1), (0, 1))
    a = [y16_row(rng, k) if route == "y16" else bytes(codes(rng, k, 0x7E)) for _ in range(2)]
    b = [bytes(codes(rng, k, 0x7E)) for _ in range(2)]
    ctx = ctx_of((64, k, 64))
    digests = []
    for sums in ("f32", "f16"):
        for copy in (True, False):
            s = F.FP8IS(F.D3Family(law, route, copy, sums=sums), mnk_min=16)
            ops = s.encode(ctx, s.noise(ctx), a, b, tile)
            digests.append(s.digest(ctx, None, s.checked(ctx, ops)))
    assert digests[0] == digests[1] != digests[2] == digests[3]


@pytest.mark.parametrize("law,route,sums", [(law, route, sums) for law in ("d3s", "d3m") for route in ("xq", "y16")
                                            for sums in ("f32", "f16")])
def test_d3_family_useful_output_tracks_the_product(law, route, sums):
    s, k = F.FP8IS(F.D3Family(law, route, sums=sums), mnk_min=16, tile=(2, 2)), 512
    rng = random.Random(f"{law}-{route}-useful")
    a = [y16_row(rng, k) if route == "y16" else bytes(codes(rng, k, 0x6E)) for _ in range(2)]
    b = [bytes(codes(rng, k, 0x6E)) for _ in range(2)]
    ctx, tile = ctx_of((64, k, 64)), Tile((0, 1), (0, 1))
    ops = s.encode(ctx, s.noise(ctx), a, b, tile)
    got = s.useful(ctx, None, ops, s.checked(ctx, ops))
    xs = [[f16_value(w) for w in s.forming.parse(r, k, True)] for r in a]
    ys = [[value(c) for c in r] for r in b]
    for i in range(2):
        for j in range(2):
            want = sum(p * q for p, q in zip(xs[i], ys[j]))
            bound = sum(abs(p * q) for p, q in zip(xs[i], ys[j]))
            assert abs(Fraction(got[i][j]) - want) < bound / 8
