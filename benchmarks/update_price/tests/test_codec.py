"""The codecs are exact: every encoding decodes to the target's bit patterns, and malformed streams are refused."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codec as C

FMTS = ("fp32", "bf16")


def lcg(n: int, seed: int = 1) -> np.ndarray:
    """Deterministic uint32 words independent of numpy's generators (64-bit LCG, high halves)."""
    out = np.empty(n, dtype=np.uint64)
    x = np.uint64(seed)
    a, c = np.uint64(6364136223846793005), np.uint64(1442695040888963407)
    with np.errstate(over="ignore"):
        for i in range(n):
            x = a * x + c
            out[i] = x >> np.uint64(32)
    return out.astype(np.uint32)


def weights(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """θ-like values and a heavy-tailed update of them, float32, from IEEE basic operations only (so the same bits on
    every platform): θ a sum of four uniforms, the change a uniform times another's cube."""
    u = lcg(6 * n, seed).astype(np.float64) / 2**31 - 1.0
    u = u.reshape(6, n)
    x0 = (0.01 * (u[0] + u[1] + u[2] + u[3])).astype(np.float32)
    a = np.abs(u[5])
    xn = (x0 + 1e-3 * u[4] * (a * a * a)).astype(np.float32)
    return x0, xn


def test_keys_are_the_float_order_and_invert():
    for fmt in FMTS:
        width = 32 if fmt == "fp32" else 16
        dt = C.FORMATS[fmt][0]
        special = np.array([0, 1 << (width - 1), 1, (1 << (width - 1)) | 1, (0x7F7FFFFF if width == 32 else 0x7F7F),
                            (0xFF7FFFFF if width == 32 else 0xFF7F)], dtype=np.uint64).astype(dt)
        bits = np.concatenate([special, (lcg(4000, 3) >> np.uint32(32 - width)).astype(dt)])
        vals = C.from_bits(bits, fmt)
        bits = bits[np.isfinite(vals)]
        k = C.keys(bits, fmt)
        assert np.array_equal(C.from_keys(k, fmt), bits)
        order = np.argsort(k, kind="stable")
        assert np.all(np.diff(C.from_bits(bits, fmt)[order].astype(np.float64)) >= 0)
        assert C.keys(np.array([1 << (width - 1)], dt), fmt)[0] == -1  # -0 sits one below +0


def test_bf16_rounds_to_nearest_even():
    one = np.float32(1.0)
    x = np.array([one, one + np.float32(2**-8), one + np.float32(3 * 2**-8), np.float32(-2.5)], np.float32)
    assert C.to_bits(x, "bf16").tolist() == [0x3F80, 0x3F80, 0x3F82, 0xC020]


def test_symbols_round_trip_over_the_whole_range():
    d = np.concatenate([np.arange(-70, 71), np.array([2**31, -(2**31), 2**32 - 1, -(2**32 - 1), 12345, -98765])])
    sym, raw, w = C.symbolize(d)
    assert sym.max() < C.ALPHABET and np.array_equal(C.widths(sym), w)
    assert np.array_equal(C.desymbolize(sym, raw), d)


@pytest.mark.parametrize("n", [0, 1, 7, C.LANE_MIN * 3 + 5])
def test_rans_round_trips_with_mixed_contexts(n):
    words = lcg(n, 11)
    ctx = (words % 3).astype(np.int64)
    sym = np.where(ctx == 0, 5, (words >> np.uint32(8)) % (40 if n > 100 else 3)).astype(np.int64)
    freq = np.stack([C.quantize(np.bincount(sym[ctx == c], minlength=C.ALPHABET)) if np.any(ctx == c)
                     else C.quantize(np.eye(C.ALPHABET, dtype=np.int64)[0]) for c in range(3)])
    states, w = C.rans_encode(sym, ctx, freq)
    assert np.array_equal(C.rans_decode(states, w, ctx, freq), sym)


def test_raw_bits_round_trip():
    v = lcg(3000, 5).astype(np.int64)
    w = (v % 33).astype(np.int64)
    raw = v & ((np.int64(1) << w) - 1)
    assert np.array_equal(C.unpack_raw(C.pack_raw(raw, w), w), raw)


@pytest.mark.parametrize("fmt", FMTS)
def test_ulp_codec_is_exact_and_pinned(fmt):
    x0, xn = weights(20000, 7)
    r0, t = C.to_bits(x0, fmt), C.to_bits(xn, fmt)
    g = (np.arange(t.size) // 7000).astype(np.int64)
    enc = C.ulp_encode(t, r0, g, fmt)
    assert np.array_equal(C.ulp_decode(enc, r0, g, fmt), t)
    assert abs(8 * len(enc) - C.ideal_bits(t, r0, g, fmt)) < 0.02 * 8 * len(enc) + 2000
    assert 8 * len(enc) < (32 if fmt == "fp32" else 16) * t.size
    pins = {"fp32": "277e5e9f", "bf16": "d42014e1"}
    assert hashlib.sha256(enc).hexdigest()[:8] == pins[fmt]


@pytest.mark.parametrize("fmt", FMTS)
def test_ulp_codec_on_tiny_and_unchanged_inputs(fmt):
    for n in (0, 1, 5):
        x0, xn = weights(max(n, 1), 9)
        x0, xn = x0[:n], xn[:n]
        r0, t = C.to_bits(x0, fmt), C.to_bits(xn, fmt)
        g = np.zeros(n, np.int64)
        assert np.array_equal(C.ulp_decode(C.ulp_encode(t, r0, g, fmt), r0, g, fmt), t)
    x0, _ = weights(10000, 2)
    r0 = C.to_bits(x0, fmt)
    g = np.zeros(r0.size, np.int64)
    enc = C.ulp_encode(r0, r0, g, fmt)
    assert np.array_equal(C.ulp_decode(enc, r0, g, fmt), r0) and len(enc) < 100


def test_ulp_decode_refuses_a_damaged_stream():
    x0, xn = weights(5000, 4)
    r0, t = C.to_bits(x0, "fp32"), C.to_bits(xn, "fp32")
    g = np.zeros(t.size, np.int64)
    enc = C.ulp_encode(t, r0, g, "fp32")
    with pytest.raises(C.CodecError, match="holds 5000 values"):
        C.ulp_decode(enc, r0[:-1], g[:-1], "fp32")
    with pytest.raises(C.CodecError):
        C.ulp_decode(enc[: len(enc) // 2], r0, g, "fp32")


@pytest.mark.parametrize("fmt", FMTS)
def test_lowrank_codec_is_exact_and_finds_a_planted_rank(fmt):
    m, n = 96, 64
    x0, noise = weights(m * n, 13)
    u = (lcg(m * 3 + n * 3, 17).astype(np.float64) / 2**32 - 0.5)
    planted = u[: m * 3].reshape(m, 3) @ u[m * 3:].reshape(n, 3).T
    w0 = x0.reshape(m, n)
    wn = (w0 + 4e-3 * planted + 1e-6 * (noise.reshape(m, n) - w0)).astype(np.float32)
    r0, t = C.to_bits(w0, fmt).reshape(-1), C.to_bits(wn, fmt).reshape(-1)
    delta = C.from_bits(t, fmt).astype(np.float64).reshape(m, n) - C.from_bits(r0, fmt).astype(np.float64).reshape(m, n)
    sizes = {}
    for rank in (0, 3, 8):
        enc = C.lowrank_encode(t, r0, C.factorize(delta, rank, 10), fmt)
        assert np.array_equal(C.lowrank_decode(enc, r0, fmt), t)
        sizes[rank] = len(enc)
    assert sizes[3] < sizes[0]


@pytest.mark.parametrize("fmt", FMTS)
def test_affine_codec_is_exact_and_checks_its_bases(fmt):
    x0, xn = weights(30000, 21)
    r0, t = C.to_bits(x0, fmt), C.to_bits(xn, fmt)
    g = np.zeros(t.size, np.int64)
    half = 0.5 * (xn.astype(np.float64) - x0.astype(np.float64))
    bases = [C.from_bits(r0, fmt).astype(np.float64), half]
    coef = C.fit_affine(t, r0, bases, fmt)
    enc = C.affine_encode(t, r0, bases, coef, g, fmt)
    assert np.array_equal(C.affine_decode(enc, r0, bases, g, fmt), t)
    assert len(enc) < len(C.ulp_encode(t, r0, g, fmt)) / 4
    with pytest.raises(C.CodecError, match="2 coefficients for 1 bases"):
        C.affine_decode(enc, r0, bases[:1], g, fmt)


def first_adam_step(n: int, seed: int, lr: float) -> tuple[np.ndarray, np.ndarray]:
    """θ and θ after one AdamW step: every weight moves by ±lr (a few by nothing) after a weight decay of 1e-5·lr."""
    x0, _ = weights(n, seed)
    s = (lcg(n, seed + 1) % 7).astype(np.float64)
    s = np.where(s == 0, 0.0, np.where(s % 2 == 1, 1.0, -1.0))
    return x0, (x0.astype(np.float64) * (1 - 1e-5 * lr) - lr * s).astype(np.float32)


@pytest.mark.parametrize("fmt", FMTS)
def test_lattice_codec_is_exact_and_pays_for_the_steps_not_their_low_bits(fmt):
    lr = 2e-6 if fmt == "fp32" else 1e-3
    x0, xn = first_adam_step(30000, 23, lr)
    r0, t = C.to_bits(x0, fmt), C.to_bits(xn, fmt)
    g = (np.arange(t.size) // 10000).astype(np.int64)
    bases = [C.from_bits(r0, fmt).astype(np.float64)]
    fit = C.fit_lattice(t, r0, bases, C.fit_affine(t, r0, bases, fmt), g, fmt)
    assert fit is not None
    coef, step = fit
    base = C.affine64(r0, bases, coef, fmt)
    assert min(abs(step / (lr * c) - 1) for c in (0.25, 0.5, 1, 2, 4)) < 0.05
    enc = C.lattice_encode(t, r0, bases, coef, step, g, fmt)
    assert np.array_equal(C.lattice_decode(enc, r0, bases, g, fmt), t)
    assert abs(8 * len(enc) - C.lattice_ideal(t, r0, base, step, g, fmt)) < 0.02 * 8 * len(enc) + 4000
    assert abs(C.lattice_slot_bits(t, r0, base, step, g, fmt).sum() - C.lattice_ideal(t, r0, base, step, g, fmt)) < 4000
    assert len(enc) < len(C.ulp_encode(t, r0, g, fmt)) / (3 if fmt == "fp32" else 1.5)
    pins = {"fp32": "d2f15321", "bf16": "869e0d28"}
    assert hashlib.sha256(enc).hexdigest()[:8] == pins[fmt]
    with pytest.raises(C.CodecError, match="1 coefficients for 2 bases"):
        C.lattice_decode(enc, r0, bases * 2, g, fmt)


def test_lattice_base_codec_takes_a_step_per_slot_and_refuses_bad_ones():
    x0, xn = first_adam_step(12000, 29, 2e-6)
    r0, t = C.to_bits(x0, "fp32"), C.to_bits(xn, "fp32")
    g = (np.arange(t.size) // 6000).astype(np.int64)
    base = C.from_bits(r0, "fp32").astype(np.float64)
    steps = np.where(g == 0, 2e-6, 1e-6).astype(np.float32)
    enc = C.lattice_encode_base(t, r0, base, steps, g, "fp32")
    assert np.array_equal(C.lattice_decode_base(enc, r0, base, steps, g, "fp32"), t)
    assert C.fit_lattice(r0, r0, [base], np.zeros(1, np.float32), g, "fp32") is None
    for bad in (0.0, -1e-6, np.inf):
        with pytest.raises(C.CodecError, match="positive and finite"):
            C.lattice_encode_base(t, r0, base, bad, g, "fp32")
    with pytest.raises(C.CodecError, match="33-bit"):
        C.lattice_steps(t, base - 1.0, 1e-12, "fp32")


def test_general_purpose_baselines_round_trip():
    x0, xn = weights(4000, 6)
    r0, t = C.to_bits(x0, "fp32"), C.to_bits(xn, "fp32")
    import lzma

    assert np.array_equal(C.unplanes(lzma.decompress(C.xz(C.xor_planes(t, r0))), np.uint32), t ^ r0)
