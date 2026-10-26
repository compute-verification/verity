"""`registry.inductor` (Inductor's fused Triton kernels and FA3's split decode, as vLLM serves them compiled) and its numpy twins
`kernels.inductor_rows`: each twin is its Definition word for word on random rows and the edge words (NaN, infinities, signed
zeros, subnormals, squares that overflow), at small statics (rows of one or more warps, and of fewer lanes), at Llama-3.2-1B's
and at Qwen3's q/k head norm (cc 9.0's and cc 12.0's launches); a launch layout the PTX was not read for is refused; the split
head is softmax attention within bf16 rounding, and one split is `AttentionHead_v4`."""
from __future__ import annotations

__models__ = ("unsloth/Llama-3.2-1B",)

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.definitions.numerics import inductor_rows as TW
from verity.ml.definitions.registry import hopper
from verity.ml.definitions.registry import inductor as I
from verity.ml.definitions.registry.fa3_check_inf import AttentionHeadV4

U16 = np.uint16
FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
#: zeros, the smallest and largest subnormals, the smallest normal, 1, the largest finite, the infinities and a NaN
EDGE = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x3F80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0], U16)
LLAMA = {"N": 2048, "EPS": 1e-5, "XBLOCK": 2, "R0_BLOCK": 2048, "NUM_WARPS": 16}


def _bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape).astype(np.float32) * np.float32(scale)).view(np.uint32) >> 16).astype(U16)


def _rows(rng, N: int) -> np.ndarray:
    """Seven rows: plain, one NaN, one infinity, squares that overflow, subnormals, zeros (signed), edge words throughout."""
    r = _bf16(rng, (7, N))
    r[1, N // 3] = 0x7FC0
    r[2, N // 2] = 0xFF80
    r[3] = _bf16(rng, N, 1e30)
    r[4] = rng.integers(0, 0x80, N) | (rng.integers(0, 2, N) << 15)
    r[5] = rng.integers(0, 2, N) << 15
    r[6] = EDGE[rng.integers(0, EDGE.size, N)]
    return r.astype(U16)


def _ev(defn, *args) -> np.ndarray:
    return np.asarray(evaluate(defn, *(np.asarray(a).reshape(-1).tolist() for a in args)), dtype=np.uint32)


LAYOUTS = [
    {"N": 256, "EPS": 1e-6, "XBLOCK": 1, "R0_BLOCK": 256, "NUM_WARPS": 1},     # one warp, one chunk
    {"N": 384, "EPS": 1e-5, "XBLOCK": 2, "R0_BLOCK": 256, "NUM_WARPS": 2},     # a partial second chunk
    {"N": 1024, "EPS": 1e-6, "XBLOCK": 1, "R0_BLOCK": 512, "NUM_WARPS": 2},    # two warps a row, two chunks
    {"N": 1024, "EPS": 1e-6, "XBLOCK": 1, "R0_BLOCK": 512, "NUM_WARPS": 4},    # four columns a thread, four warps, two chunks
    {"N": 192, "EPS": 1e-6, "XBLOCK": 2, "R0_BLOCK": 64, "NUM_WARPS": 2},      # two columns a thread, three chunks
    {"N": 320, "EPS": 1e-5, "XBLOCK": 4, "R0_BLOCK": 128, "NUM_WARPS": 2},     # sixteen lanes a row, a partial third chunk
    LLAMA,
]
#: Qwen3's per-head q/k norm (D = 128) at its combo kernel's autotune candidates: one chunk of four columns a thread (the winner
#: it served at cc 9.0), two chunks of two, and cc 12.0's winner: two chunks of eight columns over eight lanes a row (four rows a
#: warp)
QK_HEAD = [{"D": 128, "EPS": 1e-6, "XBLOCK": 2, "R0_BLOCK": 128, "NUM_WARPS": 2},
           {"D": 128, "EPS": 1e-6, "XBLOCK": 2, "R0_BLOCK": 64, "NUM_WARPS": 2},
           {"D": 128, "EPS": 1e-6, "XBLOCK": 64, "R0_BLOCK": 64, "NUM_WARPS": 16}]


@pytest.mark.parametrize("S", [pytest.param(S, marks=pytest.mark.slow) if S is LLAMA else S for S in LAYOUTS], ids=lambda S: f"N{S['N']}-R{S['R0_BLOCK']}-W{S['NUM_WARPS']}")
def test_the_norm_twins_are_the_definitions(S: dict) -> None:
    rng = np.random.default_rng(S["N"])
    N = S["N"]
    x, res, r2 = (_rows(rng, N) for _ in range(3))
    w = _bf16(rng, N)
    for i in range(x.shape[0]):
        normed = TW.norm_rows(TW.residual_rows(x[i:i + 1]), w, **S)[0]
        assert np.array_equal(_ev(bind(I.InductorRMSNorm, **S), x[i], w), normed), i
        normed = TW.norm_rows(TW.residual_rows(x[i:i + 1], res[i:i + 1]), w, **S)[0]
        assert np.array_equal(_ev(bind(I.InductorAddRMSNorm, **S), x[i], res[i], w), normed), i
        r = TW.residual_rows(x[i:i + 1], res[i:i + 1], r2[i:i + 1])
        normed = TW.norm_rows(r, w, **S)[0]
        assert np.array_equal(_ev(bind(I.InductorAdd2RMSNormFinal, **S), x[i], res[i], r2[i], w), normed), i
        if S is LLAMA:
            got = _ev(bind(I.InductorAdd2RMSNorm, **S), x[i], res[i], r2[i], w)
            assert np.array_equal(got, np.concatenate([TW.DR.f2fp_bf16(r[0]), normed])), i


def test_the_norm_is_an_rms_norm() -> None:
    """Within two bf16 ulps of float64's x / sqrt(mean(x^2) + eps) * w on plain rows."""
    rng = np.random.default_rng(3)
    x, res, w = _bf16(rng, (64, 2048)), _bf16(rng, (64, 2048)), _bf16(rng, 2048)
    r = TW.residual_rows(x, res)
    got = (TW.norm_rows(r, w, **LLAMA).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    rf = r.view(np.float32).astype(np.float64)
    want = rf / np.sqrt((rf * rf).mean(axis=1, keepdims=True) + 1e-5) * (w.astype(np.uint32) << 16).view(np.float32)
    assert np.all(np.abs(got - want) <= 2 * 2.0 ** -7 * np.abs(want) + 1e-30)


@pytest.mark.parametrize("bad", [{"XBLOCK": 3}, {"XBLOCK": 64}, {"R0_BLOCK": 4096}, {"R0_BLOCK": 256}, {"R0_BLOCK": 128},
                                 {"R0_BLOCK": 1536}, {"NUM_WARPS": 12}, {"N": 2044}, {"EPS": 0.0}, {"EPS": 1e-45}])
def test_a_layout_not_read_from_ptx_is_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        I.norm_layout(**{**LLAMA, **bad})


@pytest.mark.parametrize("S", QK_HEAD, ids=lambda S: f"X{S['XBLOCK']}-R{S['R0_BLOCK']}")
def test_the_qk_norm_rope_twin_is_the_definition(S: dict) -> None:
    rng = np.random.default_rng([S["XBLOCK"], S["R0_BLOCK"]])
    D = S["D"]
    x = _rows(rng, D)[:6].reshape(3, 2, D)
    w, cs = _bf16(rng, D), _bf16(rng, (3, D))
    cs[2, ::16] = EDGE[:8]
    got = TW.qk_norm_rope_rows(x, w, cs, NHEADS=2, **S)
    fn = bind(I.InductorQKNormRoPE, NHEADS=2, **S)
    for i in range(3):
        assert np.array_equal(_ev(fn, x[i], w, cs[i]), got[i].reshape(-1)), i


def test_the_qk_norm_rope_is_a_norm_then_a_rotation() -> None:
    """Within two bf16 ulps of float64's RMSNorm of each head then its NeoX rotation, on plain rows."""
    rng = np.random.default_rng(8)
    D, R = 128, 64
    x, w, cs = _bf16(rng, (16, 4, D)), _bf16(rng, D), _bf16(rng, (16, D))
    val = lambda a: (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    xf = val(x)
    y = xf / np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + 1e-6) * val(w)
    c, s = val(cs)[:, None, :R], val(cs)[:, None, R:]
    want = np.concatenate([y[..., :R] * c - y[..., R:] * s, y[..., R:] * c + y[..., :R] * s], axis=-1)
    got = val(TW.qk_norm_rope_rows(x, w, cs, NHEADS=4, **QK_HEAD[0]))
    assert np.all(np.abs(got - want) <= 2 * 2.0 ** -7 * np.abs(want) + 2.0 ** -12)


def _silu_words(rng) -> tuple[np.ndarray, np.ndarray]:
    """Every exponent at its smallest, middle and largest mantissas, both signs, and random words; u random with edge words."""
    e = np.arange(256, dtype=U16) << 7
    g = np.concatenate([e | m for m in (0, 0x40, 0x7F)] + [rng.integers(0, 1 << 16, 1024).astype(U16)])
    g = np.concatenate([g, g | 0x8000])
    u = _bf16(rng, g.size)
    u[::7] = EDGE[rng.integers(0, EDGE.size, u[::7].size)]
    return g, u


def test_the_silu_twin_is_the_definition() -> None:
    g, u = _silu_words(np.random.default_rng(11))
    want = np.array([evaluate(I.InductorSiluMulElem, int(a), int(b))[0] for a, b in zip(g, u)], np.uint32)
    assert np.array_equal(TW.silu_mul_rows(np.stack([g, u], axis=1))[:, 0], want)
    gu = np.concatenate([g[:8], u[:8]])
    assert np.array_equal(_ev(bind(I.InductorSiluMul, I=8), gu), TW.silu_mul_rows(gu))


def test_the_silu_is_silu() -> None:
    """Within one bf16 ulp of float64's g / (1 + e^-g) * u on every finite g with |g| < 80 (u = 1)."""
    g = np.arange(1 << 16, dtype=U16)
    gf = (g.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    keep = np.isfinite(gf) & (np.abs(gf) < 80)
    g, gf = g[keep], gf[keep]
    got = TW.silu_mul_rows(np.stack([g, np.full_like(g, 0x3F80)], axis=1))[:, 0]
    got = (got.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    want = gf / (1 + np.exp(-gf))
    assert np.all(np.abs(got - want) <= 2.0 ** -8 * np.abs(want) + 1e-38)


def test_the_rope_twin_is_the_definition() -> None:
    rng = np.random.default_rng(5)
    x, cs = _bf16(rng, (3, 16)), _bf16(rng, 16)
    x[1, ::3] = EDGE[: x[1, ::3].size]
    x[2, 2], cs[8 + 2] = 0x7F80, 0x0000                                    # x[2] * sin[2] = inf * 0: a NaN, stored 0x7FFF
    got = _ev(bind(I.InductorRoPE, NHEADS=3, D=16), x, cs)
    assert np.array_equal(got, TW.rope_rows(x, cs).reshape(-1))
    assert 0x7FFF in got and 0x7FC0 not in got


SPLITS = [(50, 1), (50, 2), (50, 4), (80, 3), (80, 4), (33, 3)]


@pytest.mark.parametrize("T,NS", SPLITS)
def test_the_split_head_twin_is_the_definition(T: int, NS: int) -> None:
    rng = np.random.default_rng(T * 8 + NS)
    D, BN = 16, 16
    q, k, v = _bf16(rng, (2, D)), _bf16(rng, (2, T, D), 2.0), _bf16(rng, (2, T, D))
    # row 1's last block scores -inf from finite words: q > 0 scaled by 2^20 and that block's k < 0 by 2^120, so each of its
    # products overflows
    lo = (T - 1) // BN * BN
    q[1] = (q[1] & 0x7FFF) + (20 << 7)
    k[1, lo:] = ((k[1, lo:] & 0x7FFF) + (120 << 7)) | 0x8000
    for mf in sorted({(T - 1) // BN, 0}):
        head = bind(I.Fa3AttentionHeadSplit, T=T, NS=NS, D=D, BN=BN, MASKED_FROM=mf, **FA3)
        out, nf = TW.fa3_head_split_rows(q, k, v, T=T, NS=NS, BN=BN, masked_from=mf)
        assert not nf.any()
        for i in range(2):
            assert np.array_equal(_ev(head, q[i], k[i], v[i]), out[i]), (i, mf)


def test_one_split_is_attention_head_v4() -> None:
    rng = np.random.default_rng(2)
    T, D, BN = 40, 16, 16
    q, k, v = _bf16(rng, D), _bf16(rng, (T, D)), _bf16(rng, (T, D))
    one = bind(I.Fa3AttentionHeadSplit, T=T, NS=1, D=D, BN=BN, MASKED_FROM=2, **FA3)
    assert np.array_equal(_ev(one, q, k, v), _ev(bind(AttentionHeadV4, T=T, D=D, BN=BN, MASKED_FROM=2, **FA3), q, k, v))


def test_the_gqa_split_twin_is_the_definition() -> None:
    rng = np.random.default_rng(4)
    T, NH, KVH, D, BN, NS = 70, 4, 2, 16, 16, 3
    q, kc, vc = _bf16(rng, (NH, D)), _bf16(rng, (T, KVH, D)), _bf16(rng, (T, KVH, D))
    fn = bind(I.Fa3AttentionSplit, T=T, NS=NS, NH=NH, KVH=KVH, D=D, BN=BN, MASKED_FROM=(T - 1) // BN, **FA3)
    out, nf = TW.fa3_attention_rows(q, kc, vc, NS=NS, BN=BN, masked_from=(T - 1) // BN)
    assert not nf.any() and np.array_equal(_ev(fn, q, kc, vc), out.reshape(-1))


def test_the_llama_decode_split_is_attention() -> None:
    """Llama-3.2-1B's decode tile (D = 64, BN = 192) at T = 1000 in 1 to 32 splits: within 2 bf16 ulps and 2^-10 (P's bf16
    rounding, against each split's own max) of float64's softmax attention (`Fa3AttentionHeadSplit_v1` is the twin at this size:
    the tests above)."""
    rng = np.random.default_rng(9)
    T, NH, KVH, D, BN = 1000, 32, 8, 64, 192
    q, kc, vc = _bf16(rng, (NH, D)), _bf16(rng, (T, KVH, D)), _bf16(rng, (T, KVH, D))
    val = lambda a: (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    kf, vf = (np.transpose(val(a), (1, 0, 2))[np.arange(NH) // (NH // KVH)] for a in (kc, vc))
    s = np.einsum("hd,htd->ht", val(q), kf) * D ** -0.5
    p = np.exp(s - s.max(axis=1, keepdims=True))
    want = np.einsum("ht,htd->hd", p / p.sum(axis=1, keepdims=True), vf)
    for NS in (1, 2, 6, 32):
        out, nf = TW.fa3_attention_rows(q, kc, vc, NS=NS, BN=BN, masked_from=(T - 1) // BN)
        assert not nf.any()
        assert np.all(np.abs(val(out) - want) <= 2 * 2.0 ** -7 * np.abs(want) + 2.0 ** -10), NS
