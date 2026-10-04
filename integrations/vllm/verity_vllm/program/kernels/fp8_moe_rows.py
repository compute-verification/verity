"""The fused-MoE expert GEMM's twin (`registry.fp8_moe`: `MoeExpertFp8BlockCoordinate(W)_v1`), vectorised over one (token, expert)
pair's output columns with `derived_rows`'s k32 e4m3 step, and the twin of the activation between the two launches
(`SiluMulFp8BlockQuant_v1`) over whole rows.  The scalar Definitions are the reference; tests/program/test_fp8_moe.py checks these
twins against them word for word, and the H200 capture (tests/properties/glm53_block_fp8_capture_gpu.py) checks the kernels' words
against them."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from verity_vllm.program.kernels.derived_rows import f2fp_bf16, fma_rn, mul_rn, tc_dot32_e4m3
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.fp8 import MIN_SCALE_BITS, np_bf16_to_f32, np_f32_to_e4m3_sat, weight_blocks
from verity_vllm.program.registry.fp8_moe import check_act_statics, check_statics
from verity_vllm.program.registry.moe import nv_expf_bits

F32_ONE = 0x3F800000


def moe_expert_fp8_block_row(xq: np.ndarray, sx: np.ndarray, w: np.ndarray, ws: np.ndarray, *, G: int = 128, KB: int = 32,
                             swap: bool = False, g: int | None = None, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """`MoeExpertFp8BlockCoordinate_v1{K,G,KB,SWAP}` (g None) or `MoeExpertFp8BlockCoordinateW_v1` (g the pair's f32 topk weight
    word) on ONE (token, expert) pair of `fused_moe_kernel`: xq (K,) e4m3 bytes, sx (K/G,) f32 words, w (N, K) the expert's e4m3
    bytes, ws (cdiv(N, G), K/G) f32 words -> (N bf16 words, nonfinite mask).  Per output column n (`fp8_moe.triton_block_acc`):
    for each K tile k ascending, `d` = KB/32 `tc_dot32_e4m3` steps from +0, then `acc` = FFMA(FMUL(d, sx[kb]), ws[n // G][kb], acc)
    from +0 (the two scales swapped under `swap`), kb = k*KB // G; then FMUL(acc, g) when g is given, and `cvt.rn.bf16.f32`."""
    xq = np.asarray(xq, dtype=np.uint8).reshape(-1)
    K = xq.size
    check_statics(K, G, KB)
    w = np.asarray(w, dtype=np.uint8).reshape(-1, K)
    N = w.shape[0]
    sx = np.asarray(sx, dtype=np.uint32).reshape(K // G)
    ws = np.asarray(ws, dtype=np.uint32).reshape(weight_blocks(N, G), K // G)
    out = np.empty(N, dtype=np.uint16)
    nonfin = np.zeros(N, dtype=bool)
    for n0 in range(0, N, chunk):
        n1 = min(N, n0 + chunk)
        wc = w[n0:n1]
        rows = np.arange(n0, n1) // G
        acc = np.zeros(n1 - n0, dtype=np.uint32)
        nf = np.zeros(n1 - n0, dtype=bool)
        for k in range(K // KB):
            d = np.zeros(n1 - n0, dtype=np.uint32)
            for st in range(KB // 32):
                lo = k * KB + 32 * st
                d, f = tc_dot32_e4m3(d, np.broadcast_to(xq[lo:lo + 32], (n1 - n0, 32)), wc[:, lo:lo + 32])
                nf |= f
            kb = k * KB // G
            a, b = np.broadcast_to(sx[kb], (n1 - n0,)), ws[rows, kb]
            first, second = (b, a) if swap else (a, b)
            acc = fma_rn(mul_rn(d, first), second, acc)
        if g is not None:
            acc = mul_rn(acc, np.broadcast_to(np.uint32(g), (n1 - n0,)))
        out[n0:n1] = f2fp_bf16(acc)
        nonfin[n0:n1] = nf | ((acc & 0x7F800000) == 0x7F800000)
    return out, nonfin


@lru_cache(maxsize=1)
def silu_gate_table() -> np.ndarray:
    """`g * (1.0f / (1.0f + expf(-g)))` as an f32 word for each of the 2^16 bf16 gate words: `SiluMulFp8BlockQuantResult_v1` up to
    its last product, a function of the gate alone, through the scalar `nv_expf_bits` and the word primitives."""
    out = np.empty(1 << 16, dtype=np.uint32)
    for w in range(1 << 16):
        g = w << 16
        sig = P.F32Div.evaluate(F32_ONE, P.F32Add.evaluate(F32_ONE, nv_expf_bits(g ^ 0x80000000)))
        out[w] = P.F32Mul.evaluate(g, sig)
    return out


def silu_mul_fp8_block_quant_rows(gu: np.ndarray, *, G: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """`SiluMulFp8BlockQuant_v1{I,G}` on rows gu (rows, 2I) bf16 words [gate || up] -> (q (rows, I) e4m3 bytes, scale (rows, I/G) f32
    words): `result` = `silu_gate_table()[gate] * f32(up)`, each group's scale `fmaxf(max|result| / 448, 1 / (448 * 512))` (an all-NaN
    group's max is NaN, and its scale the floor), `q` = e4m3(fmaxf(-448, fminf(result / scale, 448)))."""
    gu = np.asarray(gu, dtype=np.uint16)
    gu = gu.reshape(-1, gu.shape[-1])
    I = gu.shape[1] // 2
    check_act_statics(I, G)
    with np.errstate(all="ignore"):
        r = (silu_gate_table()[gu[:, :I]].view(np.float32) * np_bf16_to_f32(gu[:, I:])).reshape(-1, I // G, G)
        s = np.fmax(np.fmax.reduce(np.abs(r), axis=2) / np.float32(448.0), np.uint32(MIN_SCALE_BITS).view(np.float32))
        v = np.fmax(np.float32(-448.0), np.fmin(r / s[..., None], np.float32(448.0)))
    return np_f32_to_e4m3_sat(v).reshape(-1, I), s.astype(np.float32).view(np.uint32)
