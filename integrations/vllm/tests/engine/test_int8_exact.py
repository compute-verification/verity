"""`engine.int8_exact` (with `engine.int8_checkpoint`'s weights) equals the perplexity bench's plain `int8` linear step,
`benchmarks/ppl/steps.py`'s `int8_linear` run on the CPU (`tests/engine/ppl_oracle.py`), word for word.

Inputs: random bf16 activations and weights at several magnitudes, and the edges: zero rows and rows under steps.py's 1e-30
scale floor, ties at .5 under s = 1, |x| at the row's max (+-127), constant weight rows (|W| = 127 everywhere), products past
2^24 (RN32(Y) rounds), M on both sides of the padding row count, and a row's value against the same row served alone (a row
does not depend on its batch).  Two discriminating checks: the other dequantization order and the reciprocal scale both
differ from steps.py somewhere on the inputs used here."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.engine import int8_checkpoint as I  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402

from tests.engine import ppl_oracle as O  # noqa: E402


def _bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16)


def _weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The int8 checkpoint's (W, s_w [N]) of a bf16 weight, by the recipe's NumPy code."""
    w32 = w.float().numpy()
    s = I.channel_scales(w32)
    return torch.from_numpy(I.quantize(w32, s)), torch.from_numpy(s[:, 0])


def _words(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy()


def _check(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wq, sw = _weights(w)
    oq, osw = O.int8_weights(w)
    assert torch.equal(wq, oq) and torch.equal(sw, osw)
    got, want = E.plain_linear(x, wq, sw), O.int8_linear(x, w)
    bad = np.argwhere(_words(got) != _words(want))
    assert bad.size == 0, f"{len(bad)} words differ, first at {bad[0].tolist()}"
    return got


def _random(rng: np.random.Generator, m: int, n: int, k: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = _bf16(rng.standard_normal((m, k), dtype=np.float32) * np.float32(scale))
    w = _bf16(rng.standard_normal((n, k), dtype=np.float32) * np.float32(0.05))
    return x, w


@pytest.mark.parametrize("m", [1, 5, 16, 17, 40])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 300.0])
def test_random_inputs_equal_steps_py(m, scale):
    rng = np.random.default_rng(1000 * m + int(scale * 7))
    _check(*_random(rng, m, 64, 256, scale))


def test_zero_and_floored_rows():
    rng = np.random.default_rng(3)
    x, w = _random(rng, 6, 32, 64, 1.0)
    x[0] = 0
    x[1] = _bf16(rng.standard_normal(64, dtype=np.float32) * np.float32(1e-31))    # max |x| under the 1e-30 floor
    x[2] = _bf16(rng.standard_normal(64, dtype=np.float32) * np.float32(1e-29))    # just over it
    x[3, :] = 0
    x[3, 5] = _bf16(np.array([2.0 ** -133], dtype=np.float32))                     # the smallest bf16 subnormal
    w[4] = 0                                                                         # a zero weight row
    y = _check(x, w)
    assert not y[0].any() and not y[:, 4].any()


def test_ties_round_to_even_under_unit_scale():
    """max |x| = 127 makes s = 1, so x / s = x and every .5 is a tie."""
    row = np.array([127.0, 0.5, 1.5, 2.5, -2.5, -0.5, 126.5, -126.5] * 4, dtype=np.float32)
    rng = np.random.default_rng(7)
    _check(_bf16(np.stack([row, -row])), _random(rng, 1, 16, 32, 1.0)[1])


def test_row_max_quantizes_to_127_at_every_exponent():
    rng = np.random.default_rng(8)
    rows = []
    for e in range(-30, 30, 3):
        r = rng.uniform(-1, 1, 32).astype(np.float32) * np.float32(2.0 ** e)
        r[rng.integers(32)] = np.float32(2.0 ** e) * (1 if e % 2 else -1)
        rows.append(r)
    x = _bf16(np.stack(rows))
    xq, _ = E.quantize_rows(x)
    assert (xq.abs().amax(dim=1) == 127).all()
    _check(x, _random(rng, 1, 16, 32, 1.0)[1])


def test_products_past_two_to_the_24_round_once():
    """Constant weight rows quantize to 127 everywhere; with x = 127 s everywhere |Y| = 127 * 127 * K > 2^24."""
    k = 2048
    rng = np.random.default_rng(4)
    w = _bf16(np.full((16, k), 0.01, dtype=np.float32) * rng.uniform(0.5, 2, (16, 1)).astype(np.float32))
    x = _bf16(np.full((3, k), 3.0, dtype=np.float32) * np.array([[1.0], [-1.0], [0.37]], dtype=np.float32))
    wq, _ = _weights(w)
    assert (wq == 127).all()
    xq, _ = E.quantize_rows(x)
    assert int(E.int_mm(xq, wq).abs().max()) > 2 ** 24
    _check(x, w)


def test_dequantization_is_steps_py_order_on_four_million_products():
    """steps.int8_linear's scaling of an integer product (the step with its chain returning a given int32 Y), on 2048 x 2048
    words: equal to `dequantize`, and the other order, Y * RN32(s_a s_w), differs from it on some of them."""
    rng = np.random.default_rng(12)
    x, w = _random(rng, 2048, 2048, 16, 1.0)
    y = torch.from_numpy(rng.integers(-2 ** 31 + 1, 2 ** 31, size=(2048, 2048)))
    _, sa = E.quantize_rows(x)
    sw = _weights(w)[1]
    got = E.dequantize(y.to(torch.int32), sa, sw)
    assert torch.equal(_words_t(got), _words_t(O.int8_linear_scaled(x, w, y)))
    other = (y.to(torch.float32) * (sa[:, None] * sw[None, :])).to(torch.bfloat16)
    assert (_words_t(other) != _words_t(got)).any()


def test_the_activation_scale_is_a_division():
    rng = np.random.default_rng(11)
    x, _ = _random(rng, 256, 1, 256, 1.0)
    xq, sa = E.quantize_rows(x)
    recip = torch.round(x.float() * (1 / sa)[:, None]).clamp(-127, 127).to(torch.int8)
    assert not torch.equal(recip, xq)


@pytest.mark.parametrize("m", [3, 17, 40])
def test_a_row_does_not_depend_on_its_batch(m):
    rng = np.random.default_rng(20 + m)
    x, w = _random(rng, m, 48, 128, 2.0)
    x[1] *= 1000                                                     # one loud row
    wq, sw = _weights(w)
    batched = E.plain_linear(x, wq, sw)
    for i in range(m):
        assert torch.equal(_words_t(E.plain_linear(x[i:i + 1], wq, sw)), _words_t(batched[i:i + 1]))
        assert torch.equal(_words_t(O.int8_linear(x[i:i + 1], w)), _words_t(batched[i:i + 1]))


def _words_t(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def test_padding_keeps_every_real_row():
    rng = np.random.default_rng(5)
    w = torch.from_numpy(rng.integers(-127, 128, size=(24, 32)).astype(np.int8))
    for m in (1, 2, E.MIN_ROWS - 1, E.MIN_ROWS, 33):
        xq = torch.from_numpy(rng.integers(-127, 128, size=(m, 32)).astype(np.int8))
        got = E.int_mm(xq, w)
        assert got.shape == (m, 24) and got.dtype == torch.int32
        assert torch.equal(got.long(), xq.long() @ w.long().t())


def test_empty_call_and_leading_dims():
    w, s_w = torch.zeros(8, 16, dtype=torch.int8), torch.ones(8)
    assert E.plain_linear(torch.zeros(0, 16, dtype=torch.bfloat16), w, s_w).shape == (0, 8)
    rng = np.random.default_rng(9)
    x, wb = _random(rng, 6, 8, 16, 1.0)
    wq, sw = _weights(wb)
    flat = E.plain_linear(x, wq, sw)
    assert torch.equal(E.plain_linear(x.reshape(2, 3, 16), wq, sw).reshape(6, 8), flat)


def test_applied_record_reads_every_linear_method():
    LinearMethodBase = type("LinearMethodBase", (), {})
    Int8 = type("Int8PlainLinearMethod", (LinearMethodBase,), {"int8_arm": "plain"})
    Unquantized = type("UnquantizedLinearMethod", (LinearMethodBase,), {})
    Experts = type("Int8ExpertsMethod", (type("FusedMoEMethodBase", (), {}),), {"int8_arm": "plain"})
    q, x = Int8(), Experts()
    q.calls, x.calls = 3, 2

    class Mod:
        def __init__(self, qm=None):
            self.quant_method = qm

    class Model:
        def named_modules(self):
            return iter([("", Mod()), ("a.qkv", Mod(q)), ("a.o", Mod(Unquantized())), ("emb", Mod(object())),
                         ("m.experts.routed_experts", Mod(x))])

    assert E.applied_record(Model()) == {"a.qkv": {"method": "Int8PlainLinearMethod", "arm": "plain", "calls": 3},
                                         "a.o": {"method": "UnquantizedLinearMethod", "arm": None, "calls": None},
                                         "m.experts.routed_experts": {"method": "Int8ExpertsMethod", "arm": "plain", "calls": 2}}
