"""Poseidon2 / BabyBear as one fused CUDA kernel (cupy ``RawModule``): one thread per permutation, the state in
registers, Montgomery arithmetic (R = 2^32, Plonky3's ``monty_reduce`` with MU = P^-1 mod 2^32 = 0x88000001).

Same permutation as ``poseidon2_ref`` (bit-exact; the round constants and the internal diagonal are embedded in
Montgomery form at module build time).  Kernel ``p2_level``: thread ``(k, x)`` reads the 16 elements
``in[(16 k + i) * N + x]`` (two consecutive 8-element blocks of column ``x``), permutes (zero-padded to the width)
and writes ``out[(8 k + j) * N + x]`` -- the column tree of ``poseidon2_torch.leaf_tree`` without any transposes,
coalesced across ``x``.  With ``N = 1`` it is a Merkle level over ``(count, 8)`` digests.

This is the number that matters for "Poseidon2 on the GPU"; the torch version is the tensor-core MDS experiment.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:            # pragma: no cover
    cp = None

from .poseidon2_ref import INTERNAL_DIAG, P, Poseidon2, instance

DIGEST = 8


def to_monty(x: int) -> int:
    return (x << 32) % P


def _c_array(name: str, values, dims: str) -> str:
    return f"__constant__ u32 {name}{dims} = {{" + ",".join(f"{v}u" for v in values) + "};\n"


def source(p: Poseidon2) -> str:
    W = p.width
    rc_ext = [to_monty(c) for r in p.ext_init + p.ext_final for c in r]
    rc_int = [to_monty(c) for c in p.internal]
    diag = [to_monty(v) for v in INTERNAL_DIAG[W]]
    consts = (_c_array("RC_EXT", rc_ext, f"[{8 * W}]") + _c_array("RC_INT", rc_int, f"[{p.rp}]")
              + _c_array("DIAG", diag, f"[{W}]"))
    return "typedef unsigned int u32;\ntypedef unsigned long long u64;\n" + consts + r"""
#define P 0x78000001u
#define MU 0x88000001u
#define R2 1172168163u      // 2^64 mod P: fmul(x, R2) = x * 2^32 mod P (into Montgomery form)
#define W """ + str(W) + r"""
#define RP """ + str(p.rp) + r"""

__device__ __forceinline__ u32 monty_reduce(u64 x) {
    u32 t = (u32)x * MU;
    u64 u = (u64)t * (u64)P;
    u64 xs = x - u;
    u32 hi = (u32)(xs >> 32);
    return (x < u) ? hi + P : hi;
}
__device__ __forceinline__ u32 fmul(u32 a, u32 b) { return monty_reduce((u64)a * (u64)b); }
__device__ __forceinline__ u32 fadd(u32 a, u32 b) { u32 s = a + b; return (s >= P) ? s - P : s; }
__device__ __forceinline__ u32 fdbl(u32 a) { return fadd(a, a); }
__device__ __forceinline__ u32 sbox7(u32 x) { u32 x2 = fmul(x, x); u32 x3 = fmul(x2, x); u32 x4 = fmul(x2, x2); return fmul(x3, x4); }

__device__ __forceinline__ void mds_external(u32* s) {
    #pragma unroll
    for (int b = 0; b < W; b += 4) {
        u32 x0 = s[b], x1 = s[b+1], x2 = s[b+2], x3 = s[b+3];
        u32 t01 = fadd(x0, x1), t23 = fadd(x2, x3);
        u32 t0123 = fadd(t01, t23);
        u32 t01123 = fadd(t0123, x1), t01233 = fadd(t0123, x3);
        s[b+3] = fadd(t01233, fdbl(x0));
        s[b+1] = fadd(t01123, fdbl(x2));
        s[b]   = fadd(t01123, t01);
        s[b+2] = fadd(t01233, t23);
    }
    u32 sum[4] = {0u, 0u, 0u, 0u};
    #pragma unroll
    for (int b = 0; b < W; b += 4) { sum[0] = fadd(sum[0], s[b]); sum[1] = fadd(sum[1], s[b+1]); sum[2] = fadd(sum[2], s[b+2]); sum[3] = fadd(sum[3], s[b+3]); }
    #pragma unroll
    for (int i = 0; i < W; ++i) s[i] = fadd(s[i], sum[i & 3]);
}

__device__ __forceinline__ void mds_internal(u32* s) {
    u32 total = 0u;
    #pragma unroll
    for (int i = 0; i < W; ++i) total = fadd(total, s[i]);
    #pragma unroll
    for (int i = 0; i < W; ++i) s[i] = fadd(total, fmul(s[i], DIAG[i]));
}

__device__ __forceinline__ void permute(u32* s) {
    mds_external(s);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int i = 0; i < W; ++i) s[i] = sbox7(fadd(s[i], RC_EXT[r * W + i]));
        mds_external(s);
    }
    #pragma unroll 1
    for (int r = 0; r < RP; ++r) {
        s[0] = sbox7(fadd(s[0], RC_INT[r]));
        mds_internal(s);
    }
    #pragma unroll
    for (int r = 4; r < 8; ++r) {
        #pragma unroll
        for (int i = 0; i < W; ++i) s[i] = sbox7(fadd(s[i], RC_EXT[r * W + i]));
        mds_external(s);
    }
}

// thread (k, x): state = in[(16k + i) * ld + x], i < 16 (canonical), zero-padded; out[(8k + j) * N + x] = lanes 0..7.
extern "C" __global__
void p2_level(const u32* __restrict__ in, int npairs, int N, long long ld, u32* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)npairs * N) return;
    int k = (int)(idx / N), x = (int)(idx % N);
    const u32* src = in + (long long)(16 * k) * ld + x;
    u32 s[W];
    #pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = fmul(src[(long long)i * ld], R2);                  // to Montgomery form
    #pragma unroll
    for (int i = 16; i < W; ++i) s[i] = 0u;
    permute(s);
    u32* dst = out + (long long)(8 * k) * N + x;
    #pragma unroll
    for (int j = 0; j < 8; ++j) dst[(long long)j * N] = monty_reduce((u64)s[j]);        // from Montgomery form
}

// plain batched permutation for tests: in/out (S, W) canonical, row-major.
extern "C" __global__
void p2_permute(const u32* __restrict__ in, int S, u32* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S) return;
    u32 s[W];
    #pragma unroll
    for (int i = 0; i < W; ++i) s[i] = fmul(in[idx * W + i], R2);
    permute(s);
    #pragma unroll
    for (int i = 0; i < W; ++i) out[idx * W + i] = monty_reduce((u64)s[i]);
}
"""


class Poseidon2Cuda:
    def __init__(self, params: Poseidon2 | None = None, block: int = 128) -> None:
        self.p = params or instance(16)
        self.W = self.p.width
        self.block = block
        self.mod = cp.RawModule(code=source(self.p), options=("-std=c++17",))
        self._level = self.mod.get_function("p2_level")
        self._perm = self.mod.get_function("p2_permute")

    def permute(self, states) -> cp.ndarray:
        s = cp.ascontiguousarray(cp.asarray(states).astype(cp.uint32))
        S = s.shape[0]
        out = cp.empty_like(s)
        self._perm(((S + self.block - 1) // self.block,), (self.block,), (s, np.int32(S), out))
        return out

    def level(self, m, npairs: int, N: int, out=None, ld: int | None = None):
        """``m``: u32 buffer (row stride ``ld`` words, default ``N``) whose first ``16 * npairs`` rows of ``N`` words
        are consumed; returns ``(8 npairs, N)`` contiguous."""
        if out is None:
            out = cp.empty((8 * npairs, N), dtype=cp.uint32)
        total = npairs * N
        self._level(((total + self.block - 1) // self.block,), (self.block,),
                    (m, np.int32(npairs), np.int32(N), np.int64(N if ld is None else ld), out))
        return out

    def leaf_tree(self, m, col_block: int | None = None, block_elements: int = 1 << 28) -> cp.ndarray:
        """Column leaves ``(N, 8)`` of an ``(R, N)`` u32 matrix (same tree as ``poseidon2_torch.leaf_tree``),
        processed in column blocks of about ``block_elements`` matrix elements (default 1 GB) so the level
        buffers stay small (the level-0 output is half the block)."""
        m = cp.asarray(m)
        if m.dtype != cp.uint32:
            m = m.view(cp.uint32)
        R, N = m.shape
        if col_block is None:
            col_block = max(1, min(N, block_elements // max(1, R)))
        out = cp.empty((N, 8), dtype=cp.uint32)
        for c0 in range(0, N, col_block):
            c1 = min(N, c0 + col_block)
            out[c0:c1] = self._leaf_tree_block(m, c0, c1)
        return out

    def _leaf_tree_block(self, m, c0: int, c1: int):
        R, N = m.shape
        nc = c1 - c0
        pad = -R % DIGEST
        count = (R + pad) // DIGEST
        if pad or count == 1:
            sub = cp.zeros(((count + (count == 1)) * DIGEST, nc), dtype=cp.uint32)
            sub[:R] = m[:, c0:c1]
            count = sub.shape[0] // DIGEST
            level, ld = sub, nc
        else:
            level, ld = m[:, c0:c1], N
        while count > 1:
            npairs = count // 2
            nout = (count + 1) // 2
            nxt = cp.empty((8 * nout, nc), dtype=cp.uint32)
            self.level(level, npairs, nc, out=nxt[: 8 * npairs], ld=ld)
            if count % 2:
                nxt[8 * npairs:] = level[16 * npairs:]
            level, ld, count = nxt, nc, nout
        return level.T.copy()                                  # (nc, 8)

    def merkle_level(self, level) -> cp.ndarray:
        """``(count, 8)`` -> ``((count + 1) // 2, 8)``, odd last promoted."""
        count = level.shape[0]
        npairs = count // 2
        out = self.level(cp.ascontiguousarray(level[: 2 * npairs]).reshape(-1), npairs, 1).reshape(npairs, 8)
        if count % 2:
            out = cp.concatenate((out, level[-1:]), axis=0)
        return out
