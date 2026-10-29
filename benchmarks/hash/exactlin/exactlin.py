"""The int8-exact linear fused: `quant_rows`, one Triton kernel bit for bit `int8_exact.quantize_rows`, and `scaled_mm`, one
Triton GEMM whose epilogue is `int8_exact.dequantize`, so the exact linear is two launches a call (stock vLLM int8's count)
instead of `int8_ops`' roughly fourteen.  Both write into caller-owned buffers, allocate nothing and are graph-capturable
once compiled (`warm`).

For x [M, K] bf16, W [N, K] int8 and s_w [N] float32 (`engine.int8_exact`'s statement):

    a[i]   = max_k |f32(x[i, k])|                                  exact (a max)
    s_a[i] = RN32(max(a[i], RN32(1e-30)) / 127)                    div.rn.f32
    X[i,k] = clamp(rint(RN32(f32(x[i, k]) / s_a[i])), -127, 127)   div.rn.f32, then cvt.rni (round half to even)
    Y      = X W^T in int32                                        tl.dot, s8 x s8 -> s32, exact for K <= MAX_K
    y[i,n] = RN_bf16(RN32(RN32(f32(Y[i,n]) * s_a[i]) * s_w[n]))    cvt.rn.f32.s32, two mul.rn, cvt.rn.bf16

Triton's `/` on float32 is div.full.f32 (approximate) and its libdevice calls are flush-to-zero by default, so the kernels
use tl.div_rn, an inline cvt.rni, enable_reflect_ftz=False and enable_fp_fusion=False; `ptx_check` reads the compiled PTX
for exactly these instructions.  The GEMM is vLLM's `triton_scaled_mm` (the stock arm's kernel) on the pointer path, without
bias, with the same epilogue order (activation scale first); its config per architecture, shape and M is `CONFIGS`
(ksweep.py's gated winners), else vLLM's tiles.  A split-K config adds int32 partials (exact in any order) into a
self-zeroing `Workspace`, and the last split applies the epilogue.  A TMA config runs `exactlin_scaled_mm_tma`, the same
product and epilogue with X and W read through host tensor descriptors (kernel parameters, so a graph captures them;
out-of-bounds boxes read zeros), optionally warp-specialized.  The product needs no row
padding: a row of X W^T reads only that row, so M < 17 is just a masked tile (int8_exact pads only because torch._int_mm
refuses M <= 16).  Non-finite activations are outside the statement and not gated.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

OPTS = {"enable_fp_fusion": False, "enable_reflect_ftz": False}


@triton.jit
def _rni_s32(v):
    return tl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _membar_gl(v):
    return tl.inline_asm_elementwise("membar.gl;\n\tmov.b32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def exactlin_quant_rows(x_ptr, q_ptr, s_ptr, K, stride_x, stride_q, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.max(tl.abs(x), axis=0)
    s = tl.div_rn(tl.maximum(a, 1e-30), 127.0)
    # x / s == (x * 2^e) / (s * 2^e) with s * 2^e in [1, 2): s is normal, so both products are exact (x * 2^e unless it is
    # below 2^-126, where the quotient is too and X is 0 either way), and the rounded quotient is the same.  A |x * 2^e|
    # under 0.25 has |quotient| < 0.25, so X = 0 without dividing; the divisions left never leave div.rn's fast path.
    sb = s.to(tl.int32, bitcast=True)
    s1 = ((sb & 0x7FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    p = ((254 - (sb >> 23)) << 23).to(tl.float32, bitcast=True)
    xs = x * p
    small = tl.abs(xs) < 0.25
    q = _rni_s32(tl.div_rn(tl.where(small, 1.0, xs), tl.broadcast_to(s1, (BLOCK_K,))))
    q = tl.where(small, 0, q)
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + row * stride_q + offs, q, mask=mask)
    tl.store(s_ptr + row, s)


@triton.jit
def _epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm):
    sa = tl.load(sa_ptr + offs_m, mask=mask_m, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
    y = acc.to(tl.float32) * sa[:, None]
    y = y * sb[None, :]
    tl.store(c_ptr + stride_cm * offs_m[:, None] + offs_n[None, :], y.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def exactlin_scaled_mm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, ws_ptr, cnt_ptr, M, N, K, stride_am, stride_bn, stride_cm,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    nk = tl.cdiv(K, BLOCK_K)
    per = tl.cdiv(nk, SPLIT_K)
    k_lo = tl.program_id(1) * per
    k_hi = tl.minimum(k_lo + per, nk)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + stride_am * offs_m[:, None] + (k_lo * BLOCK_K + offs_k)[None, :]
    b_ptrs = b_ptr + (k_lo * BLOCK_K + offs_k)[:, None] + stride_bn * offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(k_lo, k_hi):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    if SPLIT_K == 1:
        _epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm)
    else:
        # Integer partial sums add exactly in any order.  Each split adds its tile into the zeroed int32 workspace, and the
        # split that counts last reads the sum back while zeroing it, resets the tile's counter and applies the epilogue
        # once, so the workspace and counters are zero again for the next call (and the next graph replay).
        mask = mask_m[:, None] & mask_n[None, :]
        ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
        _membar_gl(pid)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
        if done == SPLIT_K - 1:
            _membar_gl(pid)
            tot = tl.atomic_xchg(ws_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32), mask=mask, sem="relaxed")
            tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed")
            _epilogue(tot, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm)


@triton.jit
def exactlin_scaled_mm_tma(a_desc, b_desc, sa_ptr, sb_ptr, c_ptr, M, N, K, stride_cm, BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WARP_SPECIALIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    off_m = (pid // num_pid_n) * BLOCK_M
    off_n = (pid % num_pid_n) * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    # the descriptors' out-of-bounds boxes read zeros, so the tail tiles need no masks
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([off_m, k * BLOCK_K])
        b = b_desc.load([off_n, k * BLOCK_K])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.int32)
    offs_m = off_m + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = off_n + tl.arange(0, BLOCK_N).to(tl.int64)
    _epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, offs_m < M, offs_n < N, stride_cm)


def quant_warps(k: int) -> int:
    return 16


def tiles(m: int, n: int) -> tuple[int, int, int]:
    """vLLM triton_scaled_mm's heuristic (the stock arm's tiles)."""
    p = max(32, triton.next_power_of_2(m))
    if p <= 32:
        return (64, 64, 256) if n < 8192 else (64, 128, 256)
    if p <= 64:
        return 64, 64, 256
    if p <= 128:
        return 64, 128, 128
    return 128, 128, 128


class Cfg(tuple):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, SPLIT_K, TMA); `Cfg.parse("32x64x256:w4:s3:k2")`.  TMA 1 is
    `exactlin_scaled_mm_tma` (host tensor descriptors, SPLIT_K 1), 2 the same warp-specialized, 0 the pointer kernel."""

    def __new__(cls, bm, bn, bk, warps=4, stages=3, split=1, tma=0):
        return super().__new__(cls, (bm, bn, bk, warps, stages, split, tma))

    @classmethod
    def parse(cls, text: str) -> Cfg:
        dims, *rest = text.split(":")
        kw = {"w": 4, "s": 3, "k": 1, "t": 0}
        for r in rest:
            kw[r[0]] = int(r[1:])
        return cls(*map(int, dims.split("x")), kw["w"], kw["s"], kw["k"], kw["t"])

    def __str__(self) -> str:
        bm, bn, bk, w, s, k, t = self
        return f"{bm}x{bn}x{bk}:w{w}:s{s}:k{k}" + (f":t{t}" if t else "")


def default_cfg(m: int, n: int, k: int) -> Cfg:
    return Cfg(*tiles(m, n))


# Per architecture and (N, K): (m_max, config) rows, ksweep.py's winners where they beat vLLM's tiles (best.json).
CONFIGS: dict[str, dict[tuple[int, int], list[tuple[int, str]]]] = {
    "sm_120": {  # RTX PRO 6000: swept at M <= 64, then M 256 to 8192 (pointer and TMA)
        (6144, 4096): [(1, "16x64x128:w4:s5:k1"), (16, "16x32x128:w4:s4:k1"), (32, "32x32x128:w4:s4:k1"),
                       (64, "32x64x128:w4:s4:k1"), (256, "128x64x128:w4:s3:k1:t1"), (2048, "128x128x64:w4:s3:k1:t1"),
                       (8192, "128x128x64:w4:s3:k1:t1")],
        (4096, 4096): [(1, "16x32x256:w4:s5:k1"), (16, "16x16x256:w4:s3:k1"), (32, "32x32x256:w4:s4:k1"),
                       (64, "32x64x256:w8:s4:k1"), (256, "128x64x128:w4:s4:k1"), (2048, "128x128x64:w4:s3:k1:t1"),
                       (8192, "128x128x64:w4:s3:k1:t1")],
        (28672, 4096): [(1, "16x32x128:w4:s4:k1"), (16, "16x32x128:w4:s4:k1"), (32, "32x32x256:w4:s2:k1"),
                        (64, "64x64x128:w4:s3:k1"), (256, "128x64x64:w4:s3:k1:t1"), (2048, "128x128x64:w4:s4:k1:t1"),
                        (8192, "128x128x64:w4:s3:k1:t1")],
        (4096, 14336): [(1, "16x32x512:w4:s4:k1"), (16, "16x32x512:w4:s4:k1"), (32, "32x32x512:w8:s4:k1"),
                        (64, "32x64x512:w8:s3:k1"), (256, "128x64x128:w4:s5:k1"), (2048, "128x128x64:w4:s3:k1:t1"),
                        (8192, "128x128x64:w4:s3:k1:t1")],
        (24576, 4096): [(1, "16x32x128:w4:s4:k1"), (16, "16x32x128:w4:s3:k1"), (32, "32x32x256:w4:s2:k1"),
                        (64, "32x64x128:w8:s2:k1"), (256, "128x128x64:w4:s4:k1:t1"), (2048, "128x128x64:w4:s3:k1:t1"),
                        (8192, "128x128x64:w4:s3:k1:t1")],
        (4096, 12288): [(1, "16x32x512:w4:s5:k1"), (16, "16x32x512:w4:s4:k1"), (32, "32x32x512:w4:s4:k1"),
                        (64, "32x64x512:w8:s3:k1"), (256, "128x64x128:w4:s5:k1"), (2048, "128x128x64:w4:s3:k1:t1"),
                        (8192, "128x128x64:w4:s3:k1:t1")],
    },
    "sm_90": {  # H200: swept at M <= 32, then M 64 to 8192 (pointer and TMA)
        (6144, 4096): [(1, "16x64x512:w4:s4:k1"), (16, "16x64x512:w4:s4:k1"), (32, "32x32x512:w4:s3:k1"),
                       (64, "64x32x256:w4:s4:k1:t1"), (256, "128x128x256:w4:s3:k1:t1"),
                       (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (4096, 4096): [(1, "16x32x512:w4:s4:k1"), (16, "16x32x512:w4:s4:k1"), (32, "32x32x256:w4:s5:k1"),
                       (64, "64x16x256:w4:s5:k1:t1"), (256, "128x64x256:w4:s4:k1:t1"),
                       (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (28672, 4096): [(1, "16x32x256:w4:s3:k1"), (16, "16x32x128:w4:s3:k1"), (32, "32x128x256:w4:s3:k1"),
                        (64, "64x128x128:w8:s4:k1:t1"), (256, "128x128x128:w4:s3:k1:t1"),
                        (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (4096, 14336): [(1, "16x64x256:w4:s3:k4"), (16, "16x32x512:w4:s3:k2"), (32, "32x32x512:w4:s5:k1"),
                        (64, "64x32x256:w4:s5:k1:t2"), (256, "128x64x256:w4:s4:k1:t1"),
                        (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (24576, 4096): [(1, "16x64x256:w4:s3:k1"), (16, "16x64x256:w4:s3:k1"), (32, "32x64x256:w4:s3:k1"),
                        (64, "64x64x128:w4:s4:k1:t1"), (256, "128x64x128:w4:s3:k1:t1"),
                        (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (4096, 12288): [(1, "16x32x256:w4:s3:k4"), (16, "16x32x512:w4:s4:k2"), (32, "32x32x512:w4:s5:k1"),
                        (64, "64x32x256:w4:s5:k1:t2"), (256, "64x128x256:w8:s4:k1:t2"),
                        (2048, "128x128x128:w8:s3:k1:t1"), (8192, "128x128x128:w8:s3:k1:t1")],
    },
    "sm_100": {  # B200: swept at M <= 64, then M 256 to 8192 (pointer and TMA)
        (6144, 4096): [(1, "16x64x512:w4:s4:k1"), (16, "16x64x512:w4:s4:k1"), (32, "32x16x256:w4:s4:k1"),
                       (64, "64x64x512:w4:s4:k1"), (256, "64x64x256:w4:s3:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                       (8192, "128x64x128:w4:s3:k1:t1")],
        (4096, 4096): [(1, "16x32x512:w4:s4:k1"), (16, "16x32x512:w4:s4:k1"), (32, "32x32x512:w4:s4:k1"),
                       (64, "64x32x512:w4:s4:k1"), (256, "64x64x256:w4:s3:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                       (8192, "128x128x128:w4:s3:k1:t1")],
        (28672, 4096): [(1, "16x64x256:w4:s3:k1"), (16, "16x64x256:w4:s3:k1"), (32, "32x128x256:w4:s3:k1"),
                        (64, "64x128x256:w4:s3:k1"), (256, "128x64x128:w4:s4:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                        (8192, "128x128x128:w4:s4:k1:t1")],
        (4096, 14336): [(1, "16x64x256:w4:s5:k4"), (16, "16x64x512:w4:s3:k4"), (32, "32x64x256:w4:s4:k4"),
                        (64, "64x64x512:w4:s4:k2"), (256, "64x64x256:w4:s4:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                        (8192, "128x128x128:w4:s4:k1:t1")],
        (24576, 4096): [(1, "16x64x256:w4:s4:k1"), (16, "16x64x256:w4:s4:k1"), (32, "32x64x256:w4:s3:k1"),
                        (64, "64x64x256:w4:s3:k1"), (256, "64x128x128:w4:s4:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                        (8192, "128x128x128:w4:s3:k1:t1")],
        (4096, 12288): [(1, "16x64x256:w4:s5:k4"), (16, "16x64x512:w4:s3:k4"), (32, "32x64x256:w4:s5:k4"),
                        (64, "64x32x512:w4:s4:k1"), (256, "64x64x256:w4:s3:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                        (8192, "128x128x128:w4:s3:k1:t1")],
    },
}


@functools.cache
def arch(index: int | None = None) -> str:
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm_{major}{minor}"


def pick(m: int, n: int, k: int, sm: str | None = None) -> Cfg:
    """The GEMM config for one call on `sm` (default: the current device's): CONFIGS' first (m_max, cfg) for (n, k)
    with m <= m_max, else vLLM's tiles."""
    for m_max, text in CONFIGS.get(sm or arch(torch.cuda.current_device()), {}).get((n, k), ()):
        if m <= m_max:
            return Cfg.parse(text)
    return default_cfg(m, n, k)


class Workspace:
    """The int32 partial sums and per-tile counters a split-K launch needs, zero between calls; one per device, sized
    once, so a call allocates nothing."""

    def __init__(self, device, words: int = 64 * 32768, tiles: int = 1 << 16):
        self.ws = torch.zeros(words, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(tiles, dtype=torch.int32, device=device)


def quant_rows(x: torch.Tensor, xq: torch.Tensor, s: torch.Tensor, num_warps: int | None = None):
    """x [M, K] bf16 (unit stride along K) -> xq [M, K] int8 and s [M] (or [M, 1]) float32, in place."""
    m, k = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1, (x.dtype, x.stride())
    assert xq.dtype == torch.int8 and xq.shape == x.shape and xq.stride(1) == 1
    assert s.dtype == torch.float32 and s.numel() == m and s.is_contiguous()
    return exactlin_quant_rows[(m,)](x, xq, s, k, x.stride(0), xq.stride(0), BLOCK_K=triton.next_power_of_2(k),
                              num_warps=num_warps or quant_warps(k), **OPTS)


def scaled_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
              cfg: Cfg | None = None, work: Workspace | None = None):
    """xq [M, K] int8, s_a [M] f32, w [N, K] int8 (row-major), s_w [N] or [N, 1] f32 -> y [M, N] bf16, in place.  A
    split-K `cfg` needs `work`."""
    m, k = xq.shape
    n = w.shape[0]
    assert xq.dtype == torch.int8 and w.dtype == torch.int8 and w.shape[1] == k and xq.stride(1) == 1 and w.stride(1) == 1
    assert s_a.dtype == torch.float32 and s_a.numel() == m and s_a.is_contiguous()
    assert s_w.dtype == torch.float32 and s_w.numel() == n and s_w.is_contiguous()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.stride(1) == 1
    bm, bn, bk, warps, stages, split, tma = cfg or default_cfg(m, n, k)
    tiles_mn = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    if tma:
        from triton.tools.tensor_descriptor import TensorDescriptor
        assert split == 1, cfg
        a_desc = TensorDescriptor.from_tensor(xq, [bm, bk])
        b_desc = TensorDescriptor.from_tensor(w, [bn, bk])
        return exactlin_scaled_mm_tma[(tiles_mn,)](a_desc, b_desc, s_a, s_w, y, m, n, k, y.stride(0), BLOCK_M=bm,
                                                   BLOCK_N=bn, BLOCK_K=bk, WARP_SPECIALIZE=tma == 2, num_warps=warps,
                                                   num_stages=stages, **OPTS)
    if split > 1:
        assert work is not None and m * n <= work.ws.numel() and tiles_mn <= work.cnt.numel(), (m, n, tiles_mn)
        ws, cnt = work.ws, work.cnt
    else:
        ws, cnt = y, y
    return exactlin_scaled_mm[(tiles_mn, split)](xq, w, s_a, s_w, y, ws, cnt, m, n, k, xq.stride(0), w.stride(0), y.stride(0),
                                          BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, num_warps=warps,
                                          num_stages=stages, **OPTS)


def linear(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, xq: torch.Tensor, s_a: torch.Tensor, y: torch.Tensor):
    """The exact linear into caller-owned (xq, s_a, y)."""
    quant_rows(x, xq, s_a)
    scaled_mm(xq, s_a, w, s_w, y)


def ptx_check(compiled) -> dict:
    """What the PTX of a compiled kernel (a launch's return value) says about its float semantics."""
    ptx = compiled.asm["ptx"]
    name = compiled.metadata.name
    doc = {"kernel": name, "ftz": ptx.count(".ftz"), "div_rn_f32": ptx.count("div.rn.f32"),
           "div_approx_or_full": ptx.count("div.approx") + ptx.count("div.full"), "cvt_rni_s32_f32": ptx.count("cvt.rni.s32.f32"),
           "fma_f32": sum(ptx.count(f) for f in ("fma.rn.f32", "fma.rn.ftz.f32", "mad.f32", "mad.rn.f32")),
           "cvt_rn_f32_s32": ptx.count("cvt.rn.f32.s32"), "cvt_rn_bf16": ptx.count("cvt.rn.bf16"),
           # mma.sync's .s32.s8.s8.s32 (sm_80, sm_120), wgmma's .s32.s8.s8 (sm_90), tcgen05's kind::i8 (sm_100)
           "mma_s8": ptx.count(".s8.s8") + ptx.count("kind::i8"),
           "global_scratch": int(getattr(compiled.metadata, "global_scratch_size", 0) or 0),
           "atom_global": ptx.count("atom.global"), "membar_gl": ptx.count("membar.gl")}
    if name.startswith("exactlin_quant_rows"):
        doc["ok"] = (doc["ftz"] == 0 and doc["div_rn_f32"] >= 2 and doc["div_approx_or_full"] == 0
                     and doc["cvt_rni_s32_f32"] >= 1 and doc["fma_f32"] == 0 and doc["global_scratch"] == 0)
    else:
        doc["ok"] = (doc["ftz"] == 0 and doc["div_approx_or_full"] == 0 and doc["fma_f32"] == 0 and doc["cvt_rn_f32_s32"] >= 1
                     and doc["cvt_rn_bf16"] >= 1 and doc["mma_s8"] >= 1 and doc["global_scratch"] == 0
                     and (doc["atom_global"] == 0 or doc["membar_gl"] >= 2))
    return doc


_WORK: dict[int, Workspace] = {}


def row_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`verity_int8::row_quant`'s CUDA body: (xq, s_a) of x [M, K] bf16, allocating only them."""
    if x.stride(1) != 1:
        x = x.contiguous()
    xq = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    if x.shape[0]:
        quant_rows(x, xq, s)
    return xq, s


def product(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    """`verity_int8::scaled_mm`'s CUDA body: y [M, N] bf16, allocating only it.  A split-K config uses the device's
    workspace, made at the device's first split-K call; a capture that would make it runs the config unsplit instead
    (the same values), since a workspace made inside a capture would live in the graph's pool."""
    m, n, k = xq.shape[0], w.shape[0], w.shape[1]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    if m:
        dev = xq.device.index
        cfg, work = pick(m, n, k, arch(dev)), None
        if cfg[5] > 1:
            if dev not in _WORK and not torch.cuda.is_current_stream_capturing():
                _WORK[dev] = Workspace(xq.device)
            work = _WORK.get(dev)
            if work is None:
                cfg = Cfg(*cfg[:5], 1, cfg[6])
        scaled_mm(xq, s_a, w, s_w.reshape(-1), y, cfg, work)
    return y


def install() -> None:
    """Serve `verity_int8::row_quant` and `verity_int8::scaled_mm` on CUDA from these kernels (`register_kernel`):
    `row_quant` and `product`, each allocating only its own results, as the ops' default bodies do."""
    try:
        from verity_vllm.engine import int8_ops
    except ImportError as e:
        raise SystemExit(f"exactlin.install: {e}; install integrations/vllm (verity_vllm.engine.int8_ops)") from e

    int8_ops.row_quant.register_kernel("cuda")(row_quant)
    int8_ops.scaled_mm.register_kernel("cuda")(product)
