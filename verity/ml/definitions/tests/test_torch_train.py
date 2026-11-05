"""`verity.ml.definitions.registry.torch_train`: the training-step Definitions equal a torch CPU reference composed in the CUDA
kernels' order (each f32 op one IEEE rounding, as the GPU's ``.rn`` instructions; `Reduce.cuh`'s sum emulated lane by lane;
``expf`` the scalar `moe_twins.nv_expf_bits`, the same MUFU.EX2 table the Definitions bind), and the RMSNorm backward equals
autograd through HF `LlamaRMSNorm`'s forward on inputs where every f32 op is exact. The weight's gradient equals sm_120's words
(`fixtures/torch_train_wgrad_sm120.npz`) on inputs whose bf16 result shows the order of the f32 sum. NaN results are compared as
the GPU's (every bf16 NaN 0x7FFF)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean import mufu as BMU
from verity.ml.definitions.numerics.moe_twins import nv_expf_bits
from verity.ml.definitions.registry import torch_train as TT

torch = pytest.importorskip("torch")

U32 = np.uint32
BF16_NAN = 0x7FFF
BF16_SPECIALS = np.array([0, 0x8000, 1, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x3F00, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0,
                          0xFFC0, 0x7F81, 0x7FFF, 0x4120, 0xC120, 0x42B0, 0xC2B0, 0x42C8, 0xC2C8], U32)
EX2 = bind(BMU.MufuEx2Ftz)


def _f(u) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u, U32).view(np.float32))


def _bf(u) -> torch.Tensor:
    return _f(np.asarray(u, U32) << 16)


def _bits(t: torch.Tensor) -> np.ndarray:
    """bf16(t) as the GPU's cvt.rn.bf16.f32 writes it: round to nearest even, every NaN 0x7FFF."""
    u = t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).astype(U32)
    return np.where(torch.isnan(t).numpy(), U32(BF16_NAN), u)


def _bf16_words(rng, n: int, scale: float = 1.0, raw: float = 0.1, special: float = 0.05) -> np.ndarray:
    """bf16 words: normal values times 2^-6 .. 2^6 times ``scale``, with fractions ``raw`` of raw 16-bit words and ``special``
    of `BF16_SPECIALS`."""
    normal = _bits(torch.from_numpy((rng.standard_normal(n) * scale * np.exp2(rng.integers(-6, 7, n))).astype(np.float32)))
    pick = rng.random(n)
    out = np.where(pick < raw, rng.integers(0, 1 << 16, n).astype(U32), normal)
    return np.where(pick > 1 - special, BF16_SPECIALS[rng.integers(0, len(BF16_SPECIALS), n)], out).astype(U32)


# --- Transpose -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("R,C", [(3, 5), (128, 128)])
def test_transpose_is_the_tile_transpose(R: int, C: int) -> None:
    rng = np.random.default_rng(R * C)
    x = rng.integers(0, 1 << 16, (6, R, C)).astype(U32)
    got = BS.evaluate_words(bind(TT.Transpose, R=R, C=C), x)
    assert np.array_equal(got, x.transpose(0, 2, 1))


def test_transpose_has_no_gate() -> None:
    assert not any(BS.compile(bind(TT.Transpose, R=4, C=8)).counts.values())


# --- RMSNorm backward ------------------------------------------------------------------------------------------------------

def test_reduce_lanes_follows_set_reduce_config() -> None:
    """`Reduce.cuh`'s block width for the training shapes, and the launches the Definition doesn't build."""
    assert TT.reduce_lanes(576, 8 * 512) == 32
    assert TT.reduce_lanes(576, 16) == 32
    assert TT.reduce_lanes(160, 4096) == 32
    assert TT.reduce_lanes(128, 4096) == 32
    assert TT.reduce_lanes(256, 16) == 32
    for H, rows in ((576, 1), (576, 2), (256, 2), (100, 4096), (130, 4096), (8192, 4096)):
        with pytest.raises(ValueError):
            TT.reduce_lanes(H, rows)


def reduce_ref(p: torch.Tensor, lanes: int) -> torch.Tensor:
    """`Reduce.cuh`'s vectorized f32 sum of each row of p at ``lanes`` threads: lane l's four running sums from +0 over the float4s
    l, l + lanes, ..., combined ((s0 + s1) + s2) + s3, then the shfl_down tree at offsets lanes / 2 .. 1."""
    n, H = p.shape
    v = p.reshape(n, H // 4, 4)
    sums = []
    for lane in range(lanes):
        acc = torch.zeros(n, 4, dtype=torch.float32)
        for j in range(lane, H // 4, lanes):
            acc = acc + v[:, j, :]
        s = acc[:, 0]
        for i in range(1, 4):
            s = s + acc[:, i]
        sums.append(s)
    off = lanes // 2
    while off:
        sums = [sums[i] + sums[i + off] for i in range(off)]
        off //= 2
    return sums[0]


def ref_rms_backward(dy, x, w, r, lanes: int) -> np.ndarray:
    """dx of `TT.RmsNormTorchBackward` on torch CPU f32 tensors, op by op (the module docstring's chain)."""
    D, X, Wt, R = _bf(dy), _bf(x), _bf(w), _f(r)[:, None]
    H = X.shape[1]
    g = (D * Wt).to(torch.bfloat16).to(torch.float32)
    ga = g * R
    gr = reduce_ref(g * X, lanes)[:, None]
    r3 = (R * R) * R
    gv = (gr * torch.tensor(-0.5, dtype=torch.float32)) * r3
    gp = gv * torch.tensor(np.float32(1.0 / H))
    gb = gp * (X * torch.tensor(2.0, dtype=torch.float32))
    return _bits(ga + gb)


def _rms_rows(rng, n: int, H: int):
    """n rows (dy, x, w, r): activations and gradients at training's scales, r the f32 rsqrt of the row's mean square plus 1e-5
    (a raw or special word in some rows), some rows scaled by 2^-60 .. 2^60, and a few raw and special words throughout."""
    scale = np.where(rng.random(n) < 0.2, np.exp2(rng.integers(-60, 61, n)), 1.0)[:, None]
    x = _bf16_words(rng, n * H, raw=0.002, special=0.002).reshape(n, H)
    x = np.where(scale != 1.0, _bits(_bf(x) * torch.from_numpy(scale.astype(np.float32))), x).astype(U32)
    dy = _bf16_words(rng, n * H, 1e-3, raw=0.002, special=0.002).reshape(n, H)
    w = _bf16_words(rng, n * H, 0.05, raw=0.002, special=0.002).reshape(n, H)
    w = _bits(_bf(w) + 1.0)
    X = _bf(x)
    r = torch.rsqrt((X * X).mean(-1) + 1e-5).numpy().view(U32).copy()
    odd = rng.random(n)
    r = np.where(odd < 0.05, rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32), r)
    r = np.where(odd > 0.97, np.array([0, 0x7F800000, 0x7FC00000, 1], U32)[rng.integers(0, 4, n)], r)
    return dy, x, w, r.astype(U32)


def _check_rms(H: int, lanes: int, dy, x, w, r, want) -> None:
    got = BS.evaluate_words(bind(TT.RmsNormTorchBackward, H=H, LANES=lanes), dy, x, w, r)
    bad = np.argwhere(got != want)
    assert not bad.size, (f"{len(bad)} of {got.size} differ, first row {bad[0][0]} col {bad[0][1]}: "
                          f"{got[tuple(bad[0])]:#06x} != {want[tuple(bad[0])]:#06x}")


@pytest.mark.slow
@pytest.mark.parametrize("H,lanes", [(160, 32), (128, 4)])
def test_rms_backward_matches_reference(H: int, lanes: int) -> None:
    """Random rows against the op-by-op reference. H = 160 at 32 lanes (a real launch: lanes 0-7 read two float4s, the others
    one) and H = 128 at 4 lanes (eight float4s a lane) cover both loops of the sum, which an H a multiple of 4 * LANES at
    LANES = 32 leaves one-deep; tracing each Definition takes about 90 s."""
    rng = np.random.default_rng(H + lanes)
    dy, x, w, r = _rms_rows(rng, 192, H)
    _check_rms(H, lanes, dy, x, w, r, ref_rms_backward(dy, x, w, r, lanes))


def _exact_rows(rng, n: int, H: int):
    """bf16 rows whose every f32 product and sum in the backward is exact: x = ±m 2^e (m in 1, 3, 5, 7; e in -3 .. 0), w and dy
    ±m 2^e (m in 1, 3; e in -2 .. 0 and -3 .. 0), so each g x is a multiple of 2^-8 below 64 and a row's sum stays within f32's
    24 bits in any order."""
    def words(ms, e_lo, e_hi, shape):
        m = np.asarray(ms)[rng.integers(0, len(ms), shape)]
        v = m * np.exp2(rng.integers(e_lo, e_hi + 1, shape)) * rng.choice([-1.0, 1.0], shape)
        return _bits(torch.from_numpy(v.astype(np.float32)))
    return words((1, 3), -3, 0, (n, H)), words((1, 3, 5, 7), -3, 0, (n, H)), words((1, 3), -2, 0, (H,))


def autograd_rms_dx(dy, x, w, eps: float = 1e-5):
    """(dx, r) from autograd through HF `LlamaRMSNorm.forward` on CPU: bf16 x and weight, the saved rsqrt r."""
    xx = _bf(x).to(torch.bfloat16).requires_grad_()
    weight = _bf(w).to(torch.bfloat16)
    h = xx.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    r = torch.rsqrt(variance + eps)
    out = weight * (h * r).to(torch.bfloat16)
    out.backward(_bf(dy).to(torch.bfloat16))
    return _bits(xx.grad.to(torch.float32)), r[:, 0].detach().numpy().view(U32).copy()


def test_rms_reference_is_autograd_on_exact_rows() -> None:
    """The reference's chain is the one autograd builds (mul, `_to_copy`, rsqrt, mean, pow 2 and their backward formulas), at a
    power-of-two H where CPU's divide by H and CUDA's multiply by f32(1 / H) agree and on rows where the sums are exact."""
    rng = np.random.default_rng(5)
    H = 512
    dy, x, w = _exact_rows(rng, 64, H)
    dx, r = autograd_rms_dx(dy, x, np.tile(w, (64, 1)))
    assert np.array_equal(ref_rms_backward(dy, x, np.tile(w, (64, 1)), r, TT.reduce_lanes(H, 64)), dx)


@pytest.mark.slow
def test_rms_backward_is_autograd_on_exact_rows() -> None:
    """The Definition itself against autograd at H = 128, end to end on exact rows (the reference test's chain, without the
    reference in between)."""
    rng = np.random.default_rng(6)
    H = 128
    dy, x, w = _exact_rows(rng, 48, H)
    w = np.tile(w, (48, 1))
    dx, r = autograd_rms_dx(dy, x, w)
    _check_rms(H, 4, dy, x, w, r, dx)


def test_rms_backward_refuses_launches_it_does_not_build() -> None:
    for H, lanes in ((100, 32), (130, 32), (128, 64), (128, 12)):
        with pytest.raises(ValueError):
            bind(TT.RmsNormTorchBackward, H=H, LANES=lanes).signature()


# --- RMSNorm weight gradient -------------------------------------------------------------------------------------------------

#: w.grad's 576 bf16 words on sm_120 (RTX PRO 6000, torch 2.14.0) for ``(w * hb).backward(dy)`` on `cancel_inputs(M)`, by M,
#: from a GPU probe
WGRAD_SM120 = np.load(Path(__file__).parent / "fixtures" / "torch_train_wgrad_sm120.npz")
#: sm_120's grid over CTAs a column at H = 576: 188 multiprocessors times 1536 / 128 blocks, over 5 blocks of columns
SM120_GRID_CTAS = 452


def cancel_inputs(M: int, H: int = 576) -> tuple[np.ndarray, np.ndarray]:
    """(dy, hb) bf16 words (M, H) where the f32 order of a column's sum shows in its bf16 result: in each column a random half of
    the rows are pairs of products +2^e and -2^e (e in 2 .. 10, dy = ±2^a, hb = 2^(e - a)), which cancel exactly, and the rest
    small products ±2^-12 (1 + j/128)(1 + k/128), which each partial sum absorbs in part. The probe's own copy made the fixture."""
    rng = np.random.default_rng([117, M])
    dy, hb = np.empty((M, H), U32), np.empty((M, H), U32)
    nbig = (M // 2) & ~1
    for c in range(H):
        perm = rng.permutation(M)
        e = rng.integers(2, 11, nbig // 2)
        a = rng.integers(-4, 3, nbig // 2)
        sgn = rng.integers(0, 2, nbig // 2).astype(U32) << 15
        plus, minus = perm[0:nbig:2], perm[1:nbig:2]
        dy[plus, c] = sgn | ((a + 127).astype(U32) << 7)
        dy[minus, c] = (sgn ^ 0x8000) | ((a + 127).astype(U32) << 7)
        hb[plus, c] = hb[minus, c] = (e - a + 127).astype(U32) << 7
        rest = perm[nbig:]
        s = rng.integers(0, 2, rest.size).astype(U32) << 15
        dy[rest, c] = s | U32(127 - 12) << 7 | rng.integers(0, 128, rest.size).astype(U32)
        hb[rest, c] = U32(127) << 7 | rng.integers(0, 128, rest.size).astype(U32)
    return dy, hb


def column_sum_ref(p: torch.Tensor, ctas: int) -> torch.Tensor:
    """`Reduce.cuh`'s f32 sum of the columns of p (M, H) at ``ctas`` CTAs a column, torch CPU f32 ops: thread y of CTA c sums
    rows y + 4c + i stride + 4 stride j into accumulator i (stride 4 ctas), ((a0 + a1) + a2) + a3, then (y0 + y2) + (y1 + y3);
    the last CTA's thread y adds partials y, y + 4, .. to +0 and reduces again."""
    M, H = p.shape
    stride, zero = 4 * ctas, torch.zeros(H, dtype=torch.float32)

    def block_y(ys):
        return (ys[0] + ys[2]) + (ys[1] + ys[3])

    parts = []
    for c in range(ctas):
        ys = []
        for y in range(4):
            idx, acc = y + 4 * c, [zero] * 4
            while idx + 3 * stride < M:
                acc = [a + p[idx + i * stride] for i, a in enumerate(acc)]
                idx += 4 * stride
            for i in range(4):
                if idx >= M:
                    break
                acc[i] = acc[i] + p[idx]
                idx += stride
            ys.append(((acc[0] + acc[1]) + acc[2]) + acc[3])
        parts.append(block_y(ys))
    if ctas == 1:
        return parts[0]
    return block_y([sum((parts[j] for j in range(y, ctas, 4)), zero) for y in range(4)])


def ref_weight_grad(dy, hb, ctas: int) -> np.ndarray:
    return _bits(column_sum_ref(_bf(_bits(_bf(dy) * _bf(hb))), ctas))


def test_column_ctas_follows_set_reduce_config() -> None:
    """The CTAs a column for H = 576: one below 1,024 tokens, ceil(M / 64) from there while that is below sm_120's grid; and the
    launches the Definition doesn't build."""
    sizes = (64, 100, 256, 1020, 1021, 1024, 1100, 4096, 16384, 28928)
    assert [TT.column_ctas(M, 576) for M in sizes] == [1, 1, 1, 1, 16, 16, 18, 64, 256, 452]
    for M, H in ((28929, 576), (32768, 576), (63, 576), (4096, 100), (4096, 124), (4096, 578)):
        with pytest.raises(ValueError):
            TT.column_ctas(M, H)


def test_column_sum_reference_is_sm120() -> None:
    """The reference's order is the GPU's: on `cancel_inputs` it gives sm_120's words at every M of the fixture (one CTA a
    column, the CTA split with a tail, and at 32,768 tokens the device's grid of 452 CTAs), where a serial sum misses most."""
    for key in WGRAD_SM120.files:
        M = int(key[1:])
        dy, hb = cancel_inputs(M)
        want = WGRAD_SM120[key].astype(U32)
        ctas = SM120_GRID_CTAS if M == 32768 else TT.column_ctas(M, 576)
        got = ref_weight_grad(dy, hb, ctas)
        assert np.array_equal(got, want), (M, np.flatnonzero(got != want)[:5])
        serial = np.cumsum(_bf(_bits(_bf(dy) * _bf(hb))).numpy(), axis=0, dtype=np.float32)[-1]
        assert (_bits(torch.from_numpy(serial.copy())) != want).sum() > 500, M


def _check_wgrad(M: int, dy, hb, want) -> None:
    got = BS.evaluate_words(bind(TT.RmsNormTorchWeightGrad, M=M, H=576), dy.T, hb.T).reshape(-1)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"M = {M}: {bad.size} of {got.size} differ, first column {bad[0]}: {got[bad[0]]:#06x} != {want[bad[0]]:#06x}"


@pytest.mark.slow
@pytest.mark.parametrize("M", [64, 100])
def test_weight_grad_is_sm120_one_cta(M: int) -> None:
    """The Definition gives sm_120's words on all 576 columns at one CTA a column, 64 rows (four rounds of four accumulators)
    and 100 (with the tail loop)."""
    dy, hb = cancel_inputs(M)
    _check_wgrad(M, dy, hb, WGRAD_SM120[f"M{M}"].astype(U32))


@pytest.mark.slow
def test_weight_grad_is_sm120_across_ctas() -> None:
    """The Definition at 1,100 tokens, 18 CTAs a column with a tail, on sm_120's words: the cross-CTA sum, which no 1-CTA size
    reaches; tracing it takes about a minute."""
    dy, hb = cancel_inputs(1100)
    _check_wgrad(1100, dy, hb, WGRAD_SM120["M1100"].astype(U32))


@pytest.mark.slow
def test_weight_grad_on_random_and_special_words() -> None:
    """Gradient-scale words with raw and special ones (NaN, infinities, subnormals) against the reference at 64 rows."""
    rng = np.random.default_rng(41)
    dy = _bf16_words(rng, 64 * 96, 1e-3, raw=0.05, special=0.05).reshape(64, 96)
    hb = _bf16_words(rng, 64 * 96, 1.0, raw=0.05, special=0.05).reshape(64, 96)
    _check_wgrad(64, dy, hb, ref_weight_grad(dy, hb, 1))


def test_weight_grad_reference_is_autograd_on_exact_rows() -> None:
    """The reference's chain is the one CPU autograd builds for HF's ``weight * hb`` (MulBackward0's bf16 product, `sum_to_size`
    to the weight, the bf16 result) on rows where every f32 sum is exact in any order."""
    rng = np.random.default_rng(43)
    M, H = 256, 576
    dy, x, _ = _exact_rows(rng, M, H)
    w = torch.ones(H, dtype=torch.bfloat16, requires_grad=True)
    (w * _bf(x).to(torch.bfloat16)[None]).backward(_bf(dy).to(torch.bfloat16)[None])
    assert np.array_equal(ref_weight_grad(dy, x, TT.column_ctas(M, H)), _bits(w.grad.to(torch.float32)))


def test_weight_grad_refuses_launches_it_does_not_build() -> None:
    for M, H in ((32, 576), (32768, 576), (256, 100)):
        with pytest.raises(ValueError):
            bind(TT.RmsNormTorchWeightGrad, M=M, H=H).signature()


# --- SiLU ------------------------------------------------------------------------------------------------------------------

def _expf_neg(x) -> np.ndarray:
    """expf(-f32(x)) for bf16 words x, `nv_expf_bits` once per distinct word."""
    u, inv = np.unique(np.asarray(x, U32), return_inverse=True)
    e = np.array([nv_expf_bits(int(v) << 16 ^ 0x80000000) for v in u], U32)
    return e[inv.reshape(-1)].reshape(np.shape(x))


def ref_silu(x) -> np.ndarray:
    X = _bf(x)
    return _bits(X / (torch.ones_like(X) + _f(_expf_neg(x))))


def ref_silu_backward(dy, x, contract: bool) -> np.ndarray:
    X, one = _bf(x), torch.ones(np.shape(x), dtype=torch.float32)
    s = one / (one + _f(_expf_neg(x)))
    oms = one - s
    if contract:
        flat = [W.f32_fma_bits(int(a), int(b), 0x3F800000) for a, b in zip(X.numpy().view(U32).ravel(), oms.numpy().view(U32).ravel())]
        t = _f(np.array(flat, U32).reshape(np.shape(x)))
    else:
        t = one + X * oms
    return _bits((_bf(dy) * s) * t)


ALL_BF16 = np.arange(1 << 16, dtype=U32)


@pytest.mark.slow
def test_silu_forward_on_every_bf16_word() -> None:
    """`SiluTorch_v1` at N = 4 on all 65,536 bf16 words."""
    x = ALL_BF16.reshape(-1, 4)
    got = BS.evaluate_words(bind(TT.SiluTorch, N=4, EX2=EX2), x)
    want = ref_silu(x)
    bad = np.argwhere(got != want)
    assert not bad.size, f"{len(bad)} differ, first x {x[tuple(bad[0])]:#06x}: {got[tuple(bad[0])]:#06x} != {want[tuple(bad[0])]:#06x}"


@pytest.mark.slow
@pytest.mark.parametrize("contract", [0, 1])
def test_silu_backward_on_every_bf16_x(contract: int) -> None:
    """`SiluTorchBackward_v1` at N = 4 on all 65,536 bf16 x, each with a random dy (gradient scales, raw and special words)."""
    rng = np.random.default_rng(31 + contract)
    x = ALL_BF16.reshape(-1, 4)
    dy = _bf16_words(rng, x.size, 1e-3).reshape(x.shape)
    got = BS.evaluate_words(bind(TT.SiluTorchBackward, N=4, EX2=EX2, CONTRACT=contract), dy, x)
    want = ref_silu_backward(dy, x, bool(contract))
    bad = np.argwhere(got != want)
    assert not bad.size, (f"{len(bad)} differ, first x {x[tuple(bad[0])]:#06x} dy {dy[tuple(bad[0])]:#06x}: "
                          f"{got[tuple(bad[0])]:#06x} != {want[tuple(bad[0])]:#06x}")


def test_silu_on_random_words() -> None:
    """The quick tier's SiLU check: both Definitions at N = 1 on random and special words, each contraction."""
    rng = np.random.default_rng(37)
    x = _bf16_words(rng, 1024, 4.0, raw=0.2, special=0.1)[:, None]
    dy = _bf16_words(rng, 1024, 1e-3)[:, None]
    assert np.array_equal(BS.evaluate_words(bind(TT.SiluTorch, N=1, EX2=EX2), x), ref_silu(x))
    for contract in (0, 1):
        got = BS.evaluate_words(bind(TT.SiluTorchBackward, N=1, EX2=EX2, CONTRACT=contract), dy, x)
        assert np.array_equal(got, ref_silu_backward(dy, x, bool(contract))), contract


def test_silu_refuses_an_ex2_that_is_not_one() -> None:
    for f in (TT.SiluTorch, TT.SiluTorchBackward):
        with pytest.raises(ValueError, match="EX2"):
            bind(f, EX2=bind(TT.Transpose, R=1, C=2))
