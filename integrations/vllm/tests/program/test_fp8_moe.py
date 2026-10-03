"""The block-scaled FP8 GEMMs GLM-5.3-FP8 runs on Hopper: `fp8_moe` (the Triton fused-MoE expert GEMM, `fused_moe_kernel` at
BLOCK_SIZE_K 32) and its Boolean versions (`boolean_fp8_moe`), and `fp8`'s blockwise CUTLASS GEMM over a ragged N (GLM-5.3's
`fused_qkv_a_proj`, N = 2624 over 21 weight-scale rows).

* the plain-integer reference (`verity.evaluation.evaluate`) against a model written here from the kernels: core's k32 step
  (`HopperE4m3QgmmaDot32`), FMUL and FFMA each rounded once from the exact rational, and `cvt.rn.bf16.f32`;
* the step's pinned vectors (`fixtures/tc/vectors/fp8-hopper-wgmma-k{32,1536}.json`, model replays) through both GEMMs'
  coordinates;
* operands on which the per-tile promotion, SWAP_AB and CUTLASS's per-block promotion give different words;
* the numpy twins in `derived_rows` that the H200 capture compares the kernels with, against the Definitions;
* the Boolean versions against their word views on random and special words, the switch, and pinned counts and digests.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation import evaluate
from verity.evaluation.batch import type_shape
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import bind
from verity.ml.boolean import trace as TR
from verity.ml.prims import HopperE4m3QgmmaDot32 as DOT
from verity_vllm.program import boolean as SW
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import boolean_fp8 as BF
from verity_vllm.program.registry import boolean_fp8_moe as BM
from verity_vllm.program.registry import fp8 as F
from verity_vllm.program.registry import fp8_moe as M
from tests.program.test_boolean_fp8 import _flat, _inputs

VECTORS = Path(__file__).resolve().parents[4] / "fixtures" / "tc" / "vectors"
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
        DR.moe_expert_fp8_block_row(np.zeros(256, np.uint8), np.zeros(2, np.uint32), np.zeros((1, 256), np.uint8),
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
    """`derived_rows.moe_expert_fp8_block_row` against `MoeExpertFp8BlockGemm(W)_v1` over a ragged N (200 = 128 + 72, or 6 * 32 + 8
    for the wide scales)."""
    K, N, G = 256, 200, 32 if wide else 128
    rng = np.random.default_rng(KB + 2 * SWAP + 4 * W + 8 * wide)
    x, sx, w, ws = (_wide if wide else _operands)(rng, K, G, N)
    g = int(np.float32(rng.uniform(0.01, 1.0)).view(np.uint32))
    fam = M.MoeExpertFp8BlockGemmW if W else M.MoeExpertFp8BlockGemm
    want = evaluate(bind(fam, K=K, N=N, G=G, KB=KB, SWAP=SWAP), *((x, sx, g) if W else (x, sx)), sum(w, []), sum(ws, []))
    got, nonfinite = DR.moe_expert_fp8_block_row(np.array(x), np.array(sx), np.array(w), np.array(ws), G=G, KB=KB, swap=SWAP,
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
        assert DR.moe_expert_fp8_block_row(x, one, w[None], one[None], G=64, KB=KB)[0].tolist() == [want]
    x, w = (np.array(v, np.uint8) for v in _tie_then(0x03, 0x01))
    sx, sw = np.array([ONE, 0x3BAAAAAB], np.uint32), np.array([[ONE, ONE]], np.uint32)
    for swap, want in ((False, 0x3F80), (True, 0x3F81)):
        assert DR.moe_expert_fp8_block_row(x, sx, w[None], sw, G=32, KB=32, swap=swap)[0].tolist() == [want]


# --- on bits ---------------------------------------------------------------------------------------------------------------------

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MoeExpertFp8BlockCoordinate_v2{K=256,G=128,KB=32,SWAP=False}":
        (109778, 197810, 11175, "35f87679820e9c4408a1f8d3dc97b2bf0e0e070f7e785aab95b3bdeb4151d56a"),
    "MoeExpertFp8BlockCoordinateW_v2{K=256,G=128,KB=32,SWAP=False}":
        (112208, 202460, 11285, "a339a0d6d91e03c1f72091bc0ba7aa5ae5f9420a5a28b675f2104e498f249811"),
    "MoeExpertFp8BlockGemm_v2{K=32,N=40,G=32,KB=32,SWAP=False}":
        (551200, 992480, 56400, "237c564cdc35ad094647ca77dc67a0b9f038e135c4a8bc5d106f5aa0dfd38648"),
    "ScaledMmFp8Block_v2{K=32,N=40,G=32}":
        (551200, 992480, 56400, "4712c394556bdf70301a0b131d8875349a98e017362da895011442a2b5f12ce8"),
    "ScaledMmFp8BlockSharedScale_v2{K=32,N=40,G=32}":
        (458860, 815780, 52220, "2d1026d10adfb618f5085d1c72600dc8bd2c2d9cf9f5f380c87c0bfdf0b20186"),
}
PINNED = [bind(BM.MoeExpertFp8BlockCoordinateV2, K=256, G=128, KB=32, SWAP=False),
          bind(BM.MoeExpertFp8BlockCoordinateWV2, K=256, G=128, KB=32, SWAP=False),
          bind(BM.MoeExpertFp8BlockGemmV2, K=32, N=40, G=32, KB=32, SWAP=False),
          bind(BF.ScaledMmFp8BlockV2, K=32, N=40, G=32), bind(BF.ScaledMmFp8BlockSharedScaleV2, K=32, N=40, G=32)]

#: two tiles over two scale blocks, two tiles in one block, two steps in one tile, SWAP_AB; the GEMMs and the ragged CUTLASS GEMM
#: over a whole and a partial weight block, and over a partial block alone
AGREEMENT = [bind(BM.MoeExpertFp8BlockCoordinateV2, K=64, G=32, KB=32, SWAP=False),
             bind(BM.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=32, SWAP=True),
             bind(BM.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=64, SWAP=False),
             bind(BM.MoeExpertFp8BlockCoordinateWV2, K=64, G=32, KB=32, SWAP=True),
             bind(BM.MoeExpertFp8BlockGemmV2, K=32, N=40, G=32, KB=32, SWAP=False),
             bind(BM.MoeExpertFp8BlockGemmWV2, K=32, N=40, G=32, KB=32, SWAP=True),
             bind(BF.ScaledMmFp8BlockV2, K=32, N=40, G=32), bind(BF.ScaledMmFp8BlockSharedScaleV2, K=32, N=40, G=32),
             bind(BF.ScaledMmFp8BlockSharedScaleV2, K=32, N=8, G=32)]


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
