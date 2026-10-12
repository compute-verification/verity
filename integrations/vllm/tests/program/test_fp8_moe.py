"""The block-scaled FP8 GEMMs GLM-5.3-FP8 runs on Hopper: `fp8_moe` (the Triton fused-MoE expert GEMM, `fused_moe_kernel` at
BLOCK_SIZE_K 32) and its Boolean versions (`boolean_fp8_moe`), and `fp8`'s blockwise CUTLASS GEMM over a ragged N (GLM-5.3's
`fused_qkv_a_proj`, N = 2624 over 21 weight-scale rows).

* the plain-integer reference (`verity.ml.circuits.evaluation.evaluate`) against a model written here from the kernels: core's k32 step
  (`HopperE4m3QgmmaDot32`), FMUL and FFMA each rounded once from the exact rational, and `cvt.rn.bf16.f32`;
* the step's pinned vectors (`verity/ml/circuits/silicon/captures/vectors/fp8-hopper-wgmma-k{32,1536}.json`, model replays) through both GEMMs'
  coordinates;
* operands on which the per-tile promotion, SWAP_AB and CUTLASS's per-block promotion give different words;
* the numpy twins in `derived_rows` and `fp8_moe_rows` that the H200 capture compares the kernels with, against the Definitions;
* `SiluMulFp8BlockQuant_v1` (`silu_and_mul_per_block_quant` between the launches) against a model of the kernel's f32 steps, its NaN,
  overflow and zero groups, and the bf16 `SiluMul_v2` + `Fp8GroupQuant_v1` it is not;
* the Boolean versions against their word views on random and special words, the switch, and pinned counts and digests.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.evaluation.batch import type_shape
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.ops.prims import F32ToE4m3Sat
from verity.ml.circuits.ops.prims import HopperE4m3QgmmaDot32 as DOT
from verity_vllm.program import boolean as SW
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import fp8_moe_rows as MR
from verity_vllm.program.registry import act as A
from verity_vllm.program.registry import boolean_fp8 as BF
from verity_vllm.program.registry import boolean_fp8_moe as BM
from verity_vllm.program.registry import fp8 as F
from verity_vllm.program.registry import fp8_moe as M
from verity_vllm.program.kernels.moe_twins import nv_expf_bits
from tests.program.test_boolean_fp8 import BF16_SPECIALS, _flat, _inputs

VECTORS = Path(__file__).resolve().parents[4] / "verity" / "ml" / "circuits" / "silicon" / "captures" / "vectors"
SIGN, ONE = 1 << 31, 0x3F800000


# --- the model -------------------------------------------------------------------------------------------------------------------

def _q(w: int) -> Fraction:
    """A finite f32 word's value."""
    e, m = (w >> 23) & 0xFF, w & 0x7FFFFF
    assert e != 0xFF, hex(w)
    v = Fraction(m if e == 0 else m | 1 << 23) * Fraction(2) ** (max(e, 1) - 150)
    return -v if w & SIGN else v


def _rn(v: Fraction, zero: int) -> int:
    """``v`` rounded to f32, to nearest with ties to even, past the largest finite to infinity; an exact zero is the word ``zero``."""
    if v == 0:
        return zero
    s, v = (SIGN if v < 0 else 0), abs(v)
    e = v.numerator.bit_length() - v.denominator.bit_length()
    if Fraction(2) ** e > v:
        e -= 1
    e = max(e, -126)
    n, r = divmod(v / Fraction(2) ** (e - 23), 1)
    n += r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1)
    return s | min(((e + 126) << 23) + int(n), 0x7F800000)


def _fmul(a: int, b: int) -> int:
    return _rn(_q(a) * _q(b), (a ^ b) & SIGN)


def _ffma(a: int, b: int, c: int) -> int:
    p, z = _q(a) * _q(b), _q(c)
    both = p == 0 and z == 0 and (a ^ b) & SIGN == c & SIGN
    return _rn(p + z, c & SIGN if both else 0)


def _bf16(w: int) -> int:
    assert (w >> 23) & 0xFF != 0xFF or not w & 0x7FFFFF, hex(w)
    return ((w + 0x7FFF + ((w >> 16) & 1)) >> 16) & 0xFFFF


def _dot(acc: int, x: list[int], w: list[int], lo: int, hi: int) -> int:
    for k in range(lo, hi, 32):
        acc = DOT.evaluate(acc, *x[k:k + 32], *w[k:k + 32])
    return acc


def moe_model(x, sx, w, sw, K: int, G: int, KB: int, swap: bool, g: int | None = None) -> int:
    """`fused_moe_kernel`'s block-wise branch (fused_moe.py:542-552, 593-603): per tile k, `dot` from +0 over KB elements, then
    `acc = FFMA(FMUL(dot, a_scale), b_scale, acc)` (scales swapped under SWAP_AB); the routed weight's FMUL; the bf16 cast."""
    acc = 0
    for k in range(K // KB):
        d = _dot(0, x, w, k * KB, (k + 1) * KB)
        a, b = sx[k * KB // G], sw[k * KB // G]
        acc = _ffma(_fmul(d, b if swap else a), a if swap else b, acc)
    return _bf16(acc if g is None else _fmul(acc, g))


def cutlass_model(x, sx, w, sw, K: int, G: int) -> int:
    """The blockwise CUTLASS GEMM's coordinate: per 128-K block, `temp` from +0 over the block, `acc = FFMA(temp, FMUL(sx, sw), acc)`."""
    acc = 0
    for kb in range(K // G):
        acc = _ffma(_dot(0, x, w, kb * G, (kb + 1) * G), _fmul(sx[kb], sw[kb]), acc)
    return _bf16(acc)


def _operands(rng: np.random.Generator, K: int, G: int, rows: int = 1):
    """Finite e4m3 rows and positive f32 scales near the served ones (activation amax / 448, checkpoint weight scales)."""
    def e4m3(*shape):
        b = rng.integers(0, 256, shape)
        b[(b & 0x7F) == 0x7F] ^= 1
        return b.tolist()

    def f32(lo, hi, *shape):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32).view(np.uint32).tolist()

    return e4m3(K), f32(1e-3, 1e-1, K // G), e4m3(rows, K), f32(1e-4, 1e-2, -(-rows // G), K // G)


# --- the plain-integer reference against the model ------------------------------------------------------------------------------

@pytest.mark.parametrize("K,G,KB,SWAP", [(256, 128, 32, False), (256, 128, 32, True), (256, 128, 64, False), (256, 128, 128, True),
                                         (128, 64, 32, False)])
def test_coordinate_matches_the_kernel_model(K: int, G: int, KB: int, SWAP: bool) -> None:
    rng = np.random.default_rng(K * 1000 + G + KB + SWAP)
    coord = bind(M.MoeExpertFp8BlockCoordinate, K=K, G=G, KB=KB, SWAP=SWAP)
    weighted = bind(M.MoeExpertFp8BlockCoordinateW, K=K, G=G, KB=KB, SWAP=SWAP)
    for _ in range(12):
        x, sx, (w,), (sw,) = _operands(rng, K, G)
        g = int(np.float32(rng.uniform(0.01, 1.0)).view(np.uint32))
        assert evaluate(coord, x, sx, w, sw) == (moe_model(x, sx, w, sw, K, G, KB, SWAP),)
        assert evaluate(weighted, x, sx, g, w, sw) == (moe_model(x, sx, w, sw, K, G, KB, SWAP, g),)


@pytest.mark.parametrize("W", [False, True])
def test_gemm_rows_read_their_weight_block_scales(W: bool) -> None:
    """The (token, slot) GEMM: column n is the coordinate over the expert's row n and the scale row n // 128, the last weight block
    partial (N = 200: 128 + 72)."""
    K, N, G = 256, 200, 128
    x, sx, w, ws = _operands(np.random.default_rng(7 + W), K, G, N)
    g = int(np.float32(0.375).view(np.uint32))
    fam = M.MoeExpertFp8BlockGemmW if W else M.MoeExpertFp8BlockGemm
    args = (x, sx, g) if W else (x, sx)
    got = evaluate(bind(fam, K=K, N=N, G=G, KB=32, SWAP=False), *args, sum(w, []), sum(ws, []))
    assert got == tuple(moe_model(x, sx, w[n], ws[n // G], K, G, 32, False, g if W else None) for n in range(N))


@pytest.mark.parametrize("N", [200, 72])
def test_cutlass_block_gemm_over_a_ragged_n(N: int) -> None:
    """`ScaledMmFp8Block_v1` over N = 200 (a whole block and 72 rows) and N = 72 (a partial block only) is the CUTLASS coordinate
    per column, and its shared-scale restatement computes the same words."""
    K, G = 256, 128
    x, sx, w, ws = _operands(np.random.default_rng(N), K, G, N)
    want = tuple(cutlass_model(x, sx, w[n], ws[n // G], K, G) for n in range(N))
    for fam in (F.ScaledMmFp8Block, F.ScaledMmFp8BlockSharedScale):
        assert evaluate(bind(fam, K=K, N=N, G=G), x, sx, sum(w, []), sum(ws, [])) == want, fam.name


def test_glm53_fused_qkv_a_proj_has_21_scale_rows() -> None:
    """q_a_proj (2048) + kv_a_proj_with_mqa (576) = 2624 rows read cdiv(2624, 128) = 21 weight-scale rows."""
    for fam in (F.ScaledMmFp8Block, F.ScaledMmFp8BlockSharedScale, M.MoeExpertFp8BlockGemm):
        statics = {"KB": 32, "SWAP": False} if fam is M.MoeExpertFp8BlockGemm else {}
        params, _ = bind(fam, K=6144, N=2048 + 576, G=128, **statics).signature()
        assert type_shape(dict(params)["ws"]) == (21, 48), fam.name


# --- the k32 step's pinned vectors ----------------------------------------------------------------------------------------------

def _vector(k: int) -> tuple[list[int], list[int], list[int], int]:
    d = json.loads((VECTORS / f"fp8-hopper-wgmma-k{k}.json").read_text())
    h = lambda s: int(s, 16)  # noqa: E731
    return [h(a) for a in d["a"]], [h(b) for b in d["b"]], [h(s["c_out"]) for s in d["steps"]], h(d["y"])


@pytest.mark.parametrize("jx,jw", [(0, 0), (-7, 3), (5, -12)])
def test_k32_vector(jx: int, jw: int) -> None:
    """One tile, power-of-two scales: both GEMMs' coordinates are the vector's accumulator times 2^(jx+jw), cast to bf16."""
    a, b, _, y = _vector(32)
    sx, sw = [ONE + (jx << 23)], [ONE + (jw << 23)]
    want = _bf16(_rn(_q(y) * Fraction(2) ** (jx + jw), 0))
    for swap in (False, True):
        assert evaluate(bind(M.MoeExpertFp8BlockCoordinate, K=32, G=32, KB=32, SWAP=swap), a, sx, b, sw) == (want,)
    assert evaluate(bind(F.ScaledMmFp8BlockCoordinate, K=32, G=32), a, sx, b, sw) == (want,)
    g = ONE - (1 << 23)
    assert evaluate(bind(M.MoeExpertFp8BlockCoordinateW, K=32, G=32, KB=32, SWAP=False), a, sx, g, b, sw) == (
        _bf16(_rn(_q(y) * Fraction(2) ** (jx + jw - 1), 0)),)


@pytest.mark.parametrize("K", [32, 64, 128, 1536])
def test_k1536_vector_prefixes(K: int) -> None:
    """A single tile over the vector's first K elements (KB = G = K, unit scales) is the step chain the vector pins:
    `bf16(steps[K/32 - 1])`, the whole vector's `y` at K = 1536, in both GEMMs."""
    a, b, steps, y = _vector(1536)
    assert steps[-1] == y
    want = (_bf16(steps[K // 32 - 1]),)
    assert evaluate(bind(M.MoeExpertFp8BlockCoordinate, K=K, G=K, KB=K, SWAP=False), a[:K], [ONE], b[:K], [ONE]) == want
    assert evaluate(bind(F.ScaledMmFp8BlockCoordinate, K=K, G=K), a[:K], [ONE], b[:K], [ONE]) == want


# --- what tells the promotions apart ----------------------------------------------------------------------------------------------

def _tie_then(x32: int, w32: int) -> tuple[list[int], list[int]]:
    """Tile 0 sums to 1 + 2^-8 (1 * 1 + 2^-4 * 2^-4), a bf16 tie; tile 1 is the one product x32 * w32."""
    x, w = [0] * 64, [0] * 64
    x[0] = w[0] = 0x38
    x[1] = w[1] = 0x18
    x[32], w[32] = x32, w32
    return x, w


def test_per_tile_promotion_is_not_cutlass_per_block() -> None:
    """Tile 1 adds 2^-18 (2^-9 * 2^-9).  CUTLASS chains both tiles in the step, whose accumulator drops it: 1 + 2^-8, the tie, rounds
    to even, 0x3F80.  The Triton kernel at BLOCK_SIZE_K 32 promotes each tile through an f32 FFMA, which keeps it: 0x3F81.  At
    BLOCK_SIZE_K 64 the tile is one chain again, 0x3F80: the promotion follows the launch's BLOCK_SIZE_K."""
    x, w = _tie_then(0x01, 0x01)
    assert DOT.evaluate(DOT.evaluate(0, *x[:32], *w[:32]), *x[32:], *w[32:]) == 0x3F808000
    assert evaluate(bind(F.ScaledMmFp8BlockCoordinate, K=64, G=64), x, [ONE], w, [ONE]) == (0x3F80,)
    assert evaluate(bind(M.MoeExpertFp8BlockCoordinate, K=64, G=64, KB=32, SWAP=False), x, [ONE], w, [ONE]) == (0x3F81,)
    assert evaluate(bind(M.MoeExpertFp8BlockCoordinate, K=64, G=64, KB=64, SWAP=False), x, [ONE], w, [ONE]) == (0x3F80,)


def test_swap_ab_moves_the_rounding() -> None:
    """Tile 1 (scale block 1: sx = s = 2^-6 RN(1/3), sw = 1) is d = 3 * 2^-18, and d * s = 2^-24 (1 + 2^-25).  Without SWAP_AB,
    FMUL(d, s) rounds it to 2^-24 and the FFMA lands on the f32 tie 1 + 2^-8 + 2^-24, which rounds to even, then to bf16 0x3F80; with
    SWAP_AB, FFMA(FMUL(d, 1), s, acc) rounds once, past the tie, to 0x3F81."""
    x, w = _tie_then(0x03, 0x01)
    s = 0x3BAAAAAB
    assert _fmul(DOT.evaluate(0, *x[32:], *w[32:]), s) == 0x33800000
    for swap, want in ((False, 0x3F80), (True, 0x3F81)):
        fn = bind(M.MoeExpertFp8BlockCoordinate, K=64, G=32, KB=32, SWAP=swap)
        assert evaluate(fn, x, [ONE, s], w, [ONE, ONE]) == (want,)


def test_statics_the_kernel_cannot_launch_are_refused() -> None:
    for K, G, KB in ((256, 128, 48), (256, 64, 128), (192, 128, 32)):
        with pytest.raises(ValueError, match="fused_moe_kernel block FP8 needs"):
            _ = bind(M.MoeExpertFp8BlockCoordinate, K=K, G=G, KB=KB, SWAP=False).body
    with pytest.raises(ValueError, match="fused_moe_kernel block FP8 needs"):
        MR.moe_expert_fp8_block_row(np.zeros(256, np.uint8), np.zeros(2, np.uint32), np.zeros((1, 256), np.uint8),
                                    np.zeros((1, 2), np.uint32), KB=48)


# --- the vectorized twins the H200 capture compares with -------------------------------------------------------------------------

def _wide(rng: np.random.Generator, K: int, G: int, N: int):
    """`_operands` with a subnormal activation scale and weight-scale rows near 2^-146 .. 2^126 (finite): columns whose words are
    subnormal, and columns that overflow (sooner under SWAP_AB, whose FMUL takes the weight scale)."""
    x, _, w, _ = _operands(rng, K, G, N)
    R = -(-N // G)
    sx = np.exp2(rng.uniform(-12, -4, K // G))
    sx[1] = 2.0 ** -140
    ws = np.exp2(np.array([-146, -136, -128, 0, 100, 116, 126])[np.arange(R) % 7, None] + rng.uniform(-1, 1, (R, K // G)))
    return x, sx.astype(np.float32).view(np.uint32).tolist(), w, ws.astype(np.float32).view(np.uint32).tolist()


def _twin_agrees(want: tuple, got: np.ndarray, nonfinite: np.ndarray) -> None:
    """The twin's words are the Definition's on every lane it does not flag, and it flags only lanes whose word is not finite."""
    got = got.tolist()
    for n, (a, b) in enumerate(zip(want, got)):
        assert nonfinite[n] or a == b, (n, hex(a), hex(b))
        assert not nonfinite[n] or (a & 0x7F80) == 0x7F80, (n, hex(a))
    assert len(want) == len(got)


@pytest.mark.parametrize("KB,SWAP,W,wide", [(32, False, False, False), (32, True, True, False), (64, False, True, False),
                                             (128, True, False, False), (32, False, True, True), (32, True, False, True)])
def test_the_moe_twin_is_the_gemm(KB: int, SWAP: bool, W: bool, wide: bool) -> None:
    """`fp8_moe_rows.moe_expert_fp8_block_row` against `MoeExpertFp8BlockGemm(W)_v1` over a ragged N (200 = 128 + 72, or 6 * 32 + 8
    for the wide scales)."""
    K, N, G = 256, 200, 32 if wide else 128
    rng = np.random.default_rng(KB + 2 * SWAP + 4 * W + 8 * wide)
    x, sx, w, ws = (_wide if wide else _operands)(rng, K, G, N)
    g = int(np.float32(rng.uniform(0.01, 1.0)).view(np.uint32))
    fam = M.MoeExpertFp8BlockGemmW if W else M.MoeExpertFp8BlockGemm
    want = evaluate(bind(fam, K=K, N=N, G=G, KB=KB, SWAP=SWAP), *((x, sx, g) if W else (x, sx)), sum(w, []), sum(ws, []))
    got, nonfinite = MR.moe_expert_fp8_block_row(np.array(x), np.array(sx), np.array(w), np.array(ws), G=G, KB=KB, swap=SWAP,
                                                 g=g if W else None)
    _twin_agrees(want, got, nonfinite)
    assert nonfinite.any() and not nonfinite.all() if wide else not nonfinite.any()
    assert not wide or any(0 < v & 0x7FFF < 0x80 for v in want)


@pytest.mark.parametrize("N,wide", [(200, False), (72, False), (200, True)])
def test_the_cutlass_twin_is_the_gemm_over_a_ragged_n(N: int, wide: bool) -> None:
    """`derived_rows.scaled_mm_fp8_block_row` against `ScaledMmFp8Block_v1` with cdiv(N, G) weight-scale rows."""
    K, G = 256, 32 if wide else 128
    rng = np.random.default_rng(N + wide)
    x, sx, w, ws = (_wide if wide else _operands)(rng, K, G, N)
    want = evaluate(bind(F.ScaledMmFp8Block, K=K, N=N, G=G), x, sx, sum(w, []), sum(ws, []))
    got, nonfinite = DR.scaled_mm_fp8_block_row(np.array(x), np.array(sx), np.array(w), np.array(ws), G=G)
    _twin_agrees(want, got, nonfinite)
    assert nonfinite.any() and not nonfinite.all() if wide else not nonfinite.any()
    assert not wide or any(0 < v & 0x7FFF < 0x80 for v in want)


def test_the_twins_keep_the_promotions_apart() -> None:
    """The twins on the operands that tell the promotions apart give the Definitions' words."""
    x, w = (np.array(v, np.uint8) for v in _tie_then(0x01, 0x01))
    one = np.array([ONE], np.uint32)
    assert DR.scaled_mm_fp8_block_row(x, one, w[None], one[None], G=64)[0].tolist() == [0x3F80]
    for KB, want in ((32, 0x3F81), (64, 0x3F80)):
        assert MR.moe_expert_fp8_block_row(x, one, w[None], one[None], G=64, KB=KB)[0].tolist() == [want]
    x, w = (np.array(v, np.uint8) for v in _tie_then(0x03, 0x01))
    sx, sw = np.array([ONE, 0x3BAAAAAB], np.uint32), np.array([[ONE, ONE]], np.uint32)
    for swap, want in ((False, 0x3F80), (True, 0x3F81)):
        assert MR.moe_expert_fp8_block_row(x, sx, w[None], sw, G=32, KB=32, swap=swap)[0].tolist() == [want]


# --- silu_and_mul_per_block_quant -------------------------------------------------------------------------------------------------

F448 = 0x43E00000


def act_result_model(gate: int, up: int) -> int:
    """fused_silu_mul_block_quant.cu:54-60 on finite words whose exponential is finite: e = expf(-g) (the math library's sequence),
    then RN(1 + e), RN(1 / that), RN(g * sig) and RN(that * u), each rounded once from the exact rational."""
    g, u = gate << 16, up << 16
    e = nv_expf_bits(g ^ SIGN)
    sig = _rn(1 / _q(_rn(1 + _q(e), 0)), 0)
    return _fmul(_fmul(g, sig), u)


def act_group_model(gate: list[int], up: list[int]) -> list[int]:
    """One thread block on finite results: scale = fmaxf(RN(max |r| / 448), 1 / (448 * 512)), q = e4m3(clamp(RN(r / scale), +-448))."""
    r = [act_result_model(g, u) for g, u in zip(gate, up)]
    s = _rn(max(abs(_q(x)) for x in r) / 448, 0)
    s = s if _q(s) > _q(F.MIN_SCALE_BITS) else F.MIN_SCALE_BITS
    def quant(x: int) -> int:
        v = _q(x) / _q(s)
        return F32ToE4m3Sat.evaluate(_rn(max(-448, min(448, v)), x & SIGN))
    return [quant(x) for x in r] + [s]


def _bf16_rows(rng: np.random.Generator, rows: int, n: int, lo: float = -4, hi: float = 4) -> np.ndarray:
    s = np.exp2(rng.uniform(lo, hi, (rows, 1))).astype(np.float32)
    return ((rng.standard_normal((rows, n)).astype(np.float32) * s).view(np.uint32) >> 16).astype(np.uint16)


def test_act_result_matches_the_kernel_model() -> None:
    """Gates from -88 to 88 (expf(-g) from 2^-127 to 2^127), bf16 subnormals and zeros, ups at every scale."""
    rng = np.random.default_rng(11)
    gates = np.concatenate([(np.float32(v) * np.ones(1, np.float32)).view(np.uint32) >> 16 for v in np.linspace(-88, 88, 241)]).tolist()
    gates += [0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080] + _bf16_rows(rng, 1, 200, -8, 6)[0].tolist()
    ups = _bf16_rows(rng, 1, len(gates), -20, 20)[0].tolist()
    fn = bind(M.SiluMulFp8BlockQuantResult)
    for g, u in zip(gates, ups):
        assert evaluate(fn, g, u) == (act_result_model(g, u),), (hex(g), hex(u))


def test_act_groups_match_the_kernel_model() -> None:
    rng = np.random.default_rng(12)
    I, G = 32, 16
    fn = bind(M.SiluMulFp8BlockQuant, I=I, G=G)
    for p, gu in enumerate(_bf16_rows(rng, 6, 2 * I, -6, 4)):
        gu = gu.tolist()
        if p == 5:
            gu[I:I + G] = [0x0001] * G                       # a group whose results are subnormal: the floor scale
        want = []
        for g in range(I // G):
            want.append(act_group_model(gu[g * G:(g + 1) * G], gu[I + g * G:I + (g + 1) * G]))
        got = evaluate(fn, gu)
        assert list(got) == sum((w[:-1] for w in want), []) + [w[-1] for w in want]
    assert want[0][-1] == F.MIN_SCALE_BITS


def test_act_nan_overflow_and_zero_words() -> None:
    """A NaN result quantises to 448 and leaves the scale to the others; an all-NaN group takes the floor scale; expf(-g) overflowing
    at g = -100 makes the result -0 (g * 0) and at g = -inf NaN (-inf * 0); a zero group is the floor scale and signed zeros."""
    grp = bind(M.SiluMulFp8BlockQuantGroup, G=4)
    up = [0x3F80, 0x4000, 0xC040, 0x3F00]
    gate = [0x3F80, 0x4100, 0xC0A0, 0x4000]
    nan = evaluate(grp, [0x7FC0] + gate[1:], up)
    zero = evaluate(grp, [0x0000] + gate[1:], up)
    assert nan[0] == 0x7E and nan[1:] == zero[1:] and zero[4] != F.MIN_SCALE_BITS
    assert evaluate(grp, [0x7FC0, 0xFFC1, 0x7F81, 0x7FC0], up) == (0x7E,) * 4 + (F.MIN_SCALE_BITS,)
    assert evaluate(bind(M.SiluMulFp8BlockQuantResult), 0xC2C8, 0x3F80) == (SIGN,)
    assert (evaluate(bind(M.SiluMulFp8BlockQuantResult), 0xFF80, 0x3F80)[0] & 0x7FFFFFFF) > 0x7F800000
    assert evaluate(grp, [0x0000, 0x8000, 0x0000, 0x8000], up) == (0x00, 0x80, 0x80, 0x80, F.MIN_SCALE_BITS)


def test_act_is_not_bf16_silu_then_group_quant() -> None:
    """The fused kernel keeps `result` in f32; `SiluMul_v2` (silu_and_mul's bf16 output) then `Fp8GroupQuant_v1` rounds it to bf16
    first, and the two give different words on ordinary rows."""
    rng = np.random.default_rng(13)
    I, G = 64, 32
    differ = 0
    for gu in _bf16_rows(rng, 4, 2 * I):
        fused = evaluate(bind(M.SiluMulFp8BlockQuant, I=I, G=G), gu.tolist())
        two = evaluate(bind(F.Fp8GroupQuant, K=I, G=G), list(evaluate(bind(A.SiluMul, I=I), gu.tolist())))
        differ += fused != two
    assert differ == 4


def test_act_statics_the_kernel_cannot_launch_are_refused() -> None:
    for I, G in ((192, 96), (200, 128), (64, 128)):
        with pytest.raises(ValueError, match="silu_and_mul_per_block_quant needs"):
            _ = bind(M.SiluMulFp8BlockQuant, I=I, G=G).body
    with pytest.raises(ValueError, match="silu_and_mul_per_block_quant needs"):
        MR.silu_mul_fp8_block_quant_rows(np.zeros((1, 400), np.uint16), G=128)


@pytest.mark.parametrize("I,G,specials", [(64, 16, False), (64, 16, True), (256, 128, True)])
def test_the_act_twin_is_the_definition(I: int, G: int, specials: bool) -> None:
    """`fp8_moe_rows.silu_mul_fp8_block_quant_rows` against `SiluMulFp8BlockQuant_v1` row by row, on rows with the bf16 special words
    (zeros, subnormals, infinities, NaNs) planted at random places when `specials`."""
    rng = np.random.default_rng(I + specials)
    gu = _bf16_rows(rng, 6 if I < 256 else 2, 2 * I, -8, 10)
    if specials:
        m = rng.random(gu.shape) < 0.08
        gu[m] = rng.choice(BF16_SPECIALS, int(m.sum()))
        gu[0, :G] = 0x7FC0
    q, s = MR.silu_mul_fp8_block_quant_rows(gu, G=G)
    fn = bind(M.SiluMulFp8BlockQuant, I=I, G=G)
    for p in range(gu.shape[0]):
        assert list(evaluate(fn, gu[p].tolist())) == q[p].tolist() + s[p].tolist(), p


# --- on bits ---------------------------------------------------------------------------------------------------------------------

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MoeExpertFp8BlockCoordinate_v2{K=256,G=128,KB=32,SWAP=False}":
        (109778, 197810, 11175, "94332fbb6c435fd7a8344438ac9402c3021410860b04032cad9c253d621ddf11897c17dbee1b6640a600295bfe8324c81077ee22459a581778d3e1f96d283071"),
    "MoeExpertFp8BlockCoordinateW_v2{K=256,G=128,KB=32,SWAP=False}":
        (112208, 202460, 11285, "c31feda4eca0e41c22447406df556f76638ddb553ec548b8e56c6a76e9efb09d135f63ee56caff5dc513aa936b5d5046e26a3eb0f91eae4be66c0347b8b74502"),
    "MoeExpertFp8BlockGemm_v2{K=32,N=40,G=32,KB=32,SWAP=False}":
        (551200, 992480, 56400, "02bf354d3d15fd0343efdc9434e277df290446a412dc26ea0b8e8d9d6449d19b24560c22fdf9a10b2862ab020c4c4fe542a81bce55f183ef391efdd7b9e10c00"),
    "ScaledMmFp8Block_v2{K=32,N=40,G=32}":
        (551200, 992480, 56400, "7dc8e98a0274e6b1dd7a6b88c7b60488ad3eef4c3a0376704007526ed5b709d58c56e2a6a8feb20be84dd97c0b82190f5e8d9545e579bddc9a0983d4e186282a"),
    "ScaledMmFp8BlockSharedScale_v2{K=32,N=40,G=32}":
        (458860, 815780, 52220, "60abef94ce2a00604d9ee7dd5052e678fa4f9c0eac967867d9dc04d409c293eda985414ec9e0ad3648bff480a186ad6f194506ce22def0a1a913aceec6eb4b53"),
    "SiluMulFp8BlockQuantResult_v2":
        (19561, 38091, 1468, "6712f4b4c28cb6b216780434b20f11d82695f118f34388ea79535d8436790b91cf25e4d129bf7d57b39fb6e0e62b5cafec3be7ad65372823c9f0062226cb8bb9"),
    "SiluMulFp8BlockQuant_v2{I=8,G=4}":
        (191042, 377416, 16984, "b962d6fdb1581ff9e7ae32f9ce57a2463eca87644a16a1fe94143c546dcfcaf351046c037e6e44f452441379edb2e809ff41679ddc4f7f68de59165592a33041"),
}
PINNED = [bind(BM.MoeExpertFp8BlockCoordinateV2, K=256, G=128, KB=32, SWAP=False),
          bind(BM.MoeExpertFp8BlockCoordinateWV2, K=256, G=128, KB=32, SWAP=False),
          bind(BM.MoeExpertFp8BlockGemmV2, K=32, N=40, G=32, KB=32, SWAP=False),
          bind(BF.ScaledMmFp8BlockV2, K=32, N=40, G=32), bind(BF.ScaledMmFp8BlockSharedScaleV2, K=32, N=40, G=32),
          bind(BM.SiluMulFp8BlockQuantResultV2), bind(BM.SiluMulFp8BlockQuantV2, I=8, G=4)]

#: two tiles over two scale blocks, two tiles in one block, two steps in one tile, SWAP_AB; the GEMMs and the ragged CUTLASS GEMM
#: over a whole and a partial weight block, and over a partial block alone
AGREEMENT = [bind(BM.MoeExpertFp8BlockCoordinateV2, K=64, G=32, KB=32, SWAP=False),
             bind(BM.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=32, SWAP=True),
             bind(BM.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=64, SWAP=False),
             bind(BM.MoeExpertFp8BlockCoordinateWV2, K=64, G=32, KB=32, SWAP=True),
             bind(BM.MoeExpertFp8BlockGemmV2, K=32, N=40, G=32, KB=32, SWAP=False),
             bind(BM.MoeExpertFp8BlockGemmWV2, K=32, N=40, G=32, KB=32, SWAP=True),
             bind(BF.ScaledMmFp8BlockV2, K=32, N=40, G=32), bind(BF.ScaledMmFp8BlockSharedScaleV2, K=32, N=40, G=32),
             bind(BF.ScaledMmFp8BlockSharedScaleV2, K=32, N=8, G=32),
             bind(BM.SiluMulFp8BlockQuantResultV2), bind(BM.SiluMulFp8BlockQuantGroupV2, G=4), bind(BM.SiluMulFp8BlockQuantV2, I=8, G=4)]


@pytest.mark.parametrize("target", PINNED, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    assert BL.is_boolean(target)
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, target.bindings)
    n = 64 if BS.compile(target).counts["and"] > 200_000 else 400
    inputs = _inputs(word, n, sum(map(ord, target.id)))
    got, ref = _flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_the_switch_maps_its_word_view_to_it(target) -> None:
    assert SW.boolean_version(TR.word_of(target)).id == target.id
