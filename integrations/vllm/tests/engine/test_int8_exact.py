"""`engine.int8_exact`: the torch-composed plain arm equals the NumPy reference of egemm_ref.py's plain path element for element.

Inputs are random bf16 activations and |W| <= 63 weights, plus the edges: |x| at amax (Xq = +-127), ties at .5 under inv_s = 1
(rint to even), amax = 0, rows with |W| = 63, products past 2^24 (RN32(Y) rounds), and M on both sides of the padding row
count.  The scales are checked against exact rational division."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.engine import int8_exact as E  # noqa: E402

from tests.engine import int8_reference as R  # noqa: E402


def _torch(x_bits: np.ndarray, w: np.ndarray, s_w: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x_bits.astype(np.int16)).view(torch.bfloat16)
    y = E.plain_linear(x, torch.from_numpy(w.astype(np.int8)), torch.from_numpy(s_w.astype(np.float32)))
    assert y.dtype == torch.bfloat16
    return y.view(torch.int16).numpy().view(np.uint16)


def _bits(x: np.ndarray) -> np.ndarray:
    """float32 values that are bf16 exactly -> their bf16 bits."""
    b = R.f32_to_bf16(x)
    assert np.array_equal(R.bf16_to_f32(b), x.astype(np.float32))
    return b


def _weights(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    w = rng.integers(-E.W_MAX, E.W_MAX + 1, size=(n, k)).astype(np.int8)
    w[np.arange(n), rng.integers(0, k, size=n)] = rng.choice([-E.W_MAX, E.W_MAX], size=n)    # every row reaches |W| = 63
    s_w = (rng.random(n, dtype=np.float32) * np.float32(0.02) + np.float32(1e-4)).astype(np.float32)
    return w, s_w


def _random_x(rng: np.random.Generator, m: int, k: int, scale: float) -> np.ndarray:
    return R.f32_to_bf16(rng.standard_normal((m, k), dtype=np.float32) * np.float32(scale))


def _check(x_bits: np.ndarray, w: np.ndarray, s_w: np.ndarray) -> dict:
    ref = R.plain(x_bits, w, s_w)
    got = _torch(x_bits, w, s_w)
    bad = np.argwhere(got != ref["y"])
    assert bad.size == 0, f"{len(bad)} words differ, first at {bad[0].tolist()}: torch {got[tuple(bad[0])]:#06x} ref {ref['y'][tuple(bad[0])]:#06x}"
    x = torch.from_numpy(x_bits.astype(np.int16)).view(torch.bfloat16)
    inv_s, s_x = E.plain_scales(x)
    assert (inv_s.item(), s_x.item()) == (float(ref["inv_s"]), float(ref["s_x"]))
    assert np.array_equal(E.quantize_plain(x, inv_s).numpy().astype(np.int64), ref["xq"])
    return ref


@pytest.mark.parametrize("m", [1, 5, 16, 17, 40])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 300.0])
def test_random_inputs_equal_the_reference(m, scale):
    rng = np.random.default_rng(1000 * m + int(scale * 7))
    w, s_w = _weights(rng, 64, 256)
    _check(_random_x(rng, m, 256, scale), w, s_w)


def test_reference_rows_keep_the_whole_calls_scale():
    rng = np.random.default_rng(11)
    w, s_w = _weights(rng, 64, 256)
    x = _random_x(rng, 30, 256, 2.0)
    full, rows = R.plain(x, w, s_w), np.array([0, 7, 29])
    part = R.plain(x, w, s_w, rows=rows)
    assert (part["inv_s"], part["s_x"]) == (full["inv_s"], full["s_x"])
    assert all(np.array_equal(part[k], full[k][rows]) for k in ("xq", "y32", "y"))


def test_ties_round_to_even_under_inv_s_one():
    """amax = 127 exactly makes inv_s = 1, so x * inv_s = x and every .5 is a tie for rint."""
    row = np.array([127.0, 0.5, 1.5, 2.5, -2.5, -0.5, 126.5, -126.5] * 4, dtype=np.float32)
    x = _bits(np.stack([row, -row]))
    rng = np.random.default_rng(7)
    w, s_w = _weights(rng, 16, 32)
    ref = _check(x, w, s_w)
    assert ref["inv_s"] == 1.0 and ref["s_x"] == 1.0
    assert ref["xq"][0, :8].tolist() == [127, 0, 2, 2, -2, 0, 126, -126]


def test_amax_quantizes_to_127_for_every_amax():
    """|x| = amax lands on +-127 whatever the rounding of 127 / amax, over every bf16 exponent of a mantissa sweep."""
    rng = np.random.default_rng(11)
    w, s_w = _weights(rng, 8, 64)
    for e in range(-20, 21, 4):
        for mant in (1.0, 1.0078125, 1.3359375, 1.9921875):
            a = np.float32(mant * 2.0 ** e)
            x = R.f32_to_bf16(rng.uniform(-1, 1, size=(3, 64)).astype(np.float32) * a)
            x[0, 3], x[2, 60] = _bits(np.array([a, -a], dtype=np.float32))
            ref = _check(x, w, s_w)
            assert ref["xq"][0, 3] == 127 and ref["xq"][2, 60] == -127


def test_scales_are_one_correctly_rounded_division():
    amaxes = R.bf16_to_f32(np.arange(0x0780, 0x7F80, 97, dtype=np.uint16))          # positive normal bf16 values >= 2^-112
    for a in amaxes:
        x = torch.tensor([[float(a), -float(a) / 2]], dtype=torch.float32).to(torch.bfloat16)
        inv_s, s_x = (t.item() for t in E.plain_scales(x))
        for got, exact in ((inv_s, Fraction(127) / Fraction(float(a))), (s_x, Fraction(float(a)) / 127)):
            near = np.float32(float(exact))                                          # double rounding is exact here: f64 holds f32's
            assert got == float(near)                                               # tie cases with room to spare (53 >= 2 * 24 + 2)


@pytest.mark.parametrize("amax", [0.0, 2.0 ** -126, 2.0 ** -121, 2.0 ** -120])
def test_zero_and_tiny_activations(amax):
    """Below 2^-120 (0 and where 127 / amax overflows) both scales are 1 and every Xq is 0; at 2^-120 the division is finite."""
    rng = np.random.default_rng(3)
    w, s_w = _weights(rng, 16, 64)
    x = np.zeros((4, 64), dtype=np.float32)
    x[1, 7], x[2, 9] = amax, -amax / 2
    ref = _check(_bits(x), w, s_w)
    if amax < 2.0 ** -120:
        assert ref["inv_s"] == ref["s_x"] == 1.0 and not ref["y"].any()
    else:
        assert np.isfinite(ref["inv_s"]) and ref["xq"][1, 7] == 127


def test_products_past_two_to_the_24_round_once():
    k = 4096
    x = _bits(np.full((2, k), 3.0, dtype=np.float32))                               # Xq = 127 everywhere
    w = np.full((8, k), E.W_MAX, dtype=np.int8)
    w[1:, -1] = np.arange(62, 55, -1)                                              # Y = 127 * (63 k - d), odd d: not a float32
    s_w = np.full(8, 1e-6, dtype=np.float32)
    ref = _check(x, w, s_w)
    y = ref["y32"][0]
    assert y.max() > 2 ** 24 and (y.astype(np.float32).astype(np.int64) != y).any()


def test_dequantize_rounds_the_scale_product_first():
    """y = RN_bf16(RN32(RN32(Y) * RN32(s_x * s_w))): on words where (Y * s_x) * s_w rounds to another bf16 (searched for: such
    words sit within a float32 ulp of a bf16 tie), torch gives the stated order's word."""
    rng = np.random.default_rng(13)
    n = 1 << 22
    y = rng.integers(-(2 ** 30), 2 ** 30, size=n)
    s_x = rng.uniform(1e-4, 1e-1, size=n).astype(np.float32)
    s_w = rng.uniform(1e-5, 1e-2, size=n).astype(np.float32)
    yf = y.astype(np.float32)
    stated, other = R.f32_to_bf16(yf * (s_x * s_w)), R.f32_to_bf16((yf * s_x) * s_w)
    hits = np.flatnonzero(stated != other)[:32]
    assert len(hits) >= 8
    for i in hits:
        got = E.dequantize(torch.tensor([[y[i]]], dtype=torch.int32), torch.tensor(s_x[i]), torch.tensor([s_w[i]]))
        assert got.view(torch.int16).item() & 0xFFFF == stated[i] != other[i]


def test_padding_keeps_every_real_row():
    rng = np.random.default_rng(5)
    w = torch.from_numpy(rng.integers(-63, 64, size=(24, 32)).astype(np.int8))
    for m in (1, 2, E.MIN_ROWS - 1, E.MIN_ROWS, 33):
        xq = torch.from_numpy(rng.integers(-127, 128, size=(m, 32)).astype(np.int8))
        got = E.int_mm(xq, w)
        assert got.shape == (m, 24) and got.dtype == torch.int32
        assert torch.equal(got.long(), xq.long() @ w.long().t())


def test_empty_call_and_leading_dims():
    w, s_w = torch.zeros(8, 16, dtype=torch.int8), torch.ones(8)
    assert E.plain_linear(torch.zeros(0, 16, dtype=torch.bfloat16), w, s_w).shape == (0, 8)
    rng = np.random.default_rng(9)
    wn, sn = _weights(rng, 8, 16)
    x = _random_x(rng, 6, 16, 1.0)
    flat = _torch(x, wn, sn)
    x3 = torch.from_numpy(x.astype(np.int16)).view(torch.bfloat16).reshape(2, 3, 16)
    y3 = E.plain_linear(x3, torch.from_numpy(wn), torch.from_numpy(sn))
    assert np.array_equal(y3.reshape(6, 8).view(torch.int16).numpy().view(np.uint16), flat)


def test_applied_record_reads_every_linear_method():
    LinearMethodBase = type("LinearMethodBase", (), {})
    Int8 = type("Int8PlainLinearMethod", (LinearMethodBase,), {"int8_arm": "plain"})
    Unquantized = type("UnquantizedLinearMethod", (LinearMethodBase,), {})
    q = Int8()
    q.calls = 3

    class Mod:
        def __init__(self, qm=None):
            self.quant_method = qm

    class Model:
        def named_modules(self):
            return iter([("", Mod()), ("a.qkv", Mod(q)), ("a.o", Mod(Unquantized())), ("emb", Mod(object()))])

    assert E.applied_record(Model()) == {"a.qkv": {"method": "Int8PlainLinearMethod", "arm": "plain", "calls": 3},
                                         "a.o": {"method": "UnquantizedLinearMethod", "arm": None, "calls": None}}
