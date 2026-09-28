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
        assert s.credit == 6
        assert s.certificate.omega == Fraction(6 * 8192 - 32 + f_a + f_b, 6 * 8192 - 32 + c_a + c_b)
        sh = Shape(8192, 8192, 16384)
        assert s.wref(sh) == 6 * sh.wmm - 32 * sh.m * sh.n + f_a * sh.m * sh.k + f_b * sh.n * sh.k
    assert get("fp8-is-h100-d2-draft").forming.units == (336, 336)
    assert get("fp8-is-h100-d2-draft").forming.credited == (272, 272)
    assert get("fp8-is-h100-fpncp-v0").forming.units == (384, 384)


def test_domain():
    s = get("fp8-is-h100-d2-draft")
    s.check(Matmul(0, Shape(8192, 8192, 8192), 1))
    for bad in (Shape(4096, 8192, 8192), Shape(8192, 8192, 8196), Shape(8200, 8192, 8192), Shape(8192, 8208, 8192), Shape(8192, 1 << 17, 8192)):
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
        assert len(atoms) == 12 and len(running) == 11
        assert atoms == [tc_dot_total_e4m3(F.ATOM, 0, x[32 * t:32 * t + 32], y[32 * t:32 * t + 32]) for t in range(12)]
        r = as_f32(atoms[0])
        for i, rt in zip(atoms[1:], running):
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
    s = F.FP8IS(F.D2Draft(), mnk_min=16, tile=(64, 8))
    rng = random.Random("lifecycle")
    work = Workload((Matmul(0, Shape(64, 128, 16), 7),))
    weights = {7: [bytes(codes(rng, 128, s.forming.b_max)) for _ in range(16)]}
    acts = {0: [bytes(codes(rng, 128, s.forming.a_max)) for _ in range(64)]}

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
            assert audit.accepted and audit.profile.work == s.wref(Shape(64, 128, 16))
