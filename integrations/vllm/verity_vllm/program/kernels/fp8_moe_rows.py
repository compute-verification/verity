"""The fused-MoE expert GEMM's twin (`registry.fp8_moe`: `MoeExpertFp8BlockCoordinate(W)_v1`), vectorised over one (token, expert)
pair's output columns with `derived_rows`'s k32 e4m3 step.  The scalar Definitions are the reference; tests/program/test_fp8_moe.py
checks this twin against them word for word, and the H200 capture (tests/properties/glm53_block_fp8_capture_gpu.py) checks the
kernel's words against it."""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels.derived_rows import f2fp_bf16, fma_rn, mul_rn, tc_dot32_e4m3
from verity_vllm.program.registry.fp8 import weight_blocks
from verity_vllm.program.registry.fp8_moe import check_statics


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
