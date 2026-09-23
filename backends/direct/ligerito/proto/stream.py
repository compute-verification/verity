"""Streaming round-1 commit: the codeword U (C, n) is encoded in groups of ``g`` columns (g a multiple of 16) into one reusable
(g, n) buffer, and every group advances the Blake3 leaf state of all n positions by g/16 blocks -- a leaf is Blake3 of the C words
at one position (C <= 256: a single chunk, counter 0, CHUNK_START on block 0, CHUNK_END | ROOT on the last), so the digests
equal ``hash_gpu.merkle.leaves(U)`` bit for bit while only g/C of the codeword is ever resident.  The opened rows are recomputed
after the query coin as RS evaluations of the message columns (``device.contract_geom``)."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from ._ligero import merkle as ligero_merkle

try:
    import cupy as cp
except Exception:  # noqa: BLE001
    cp = None

_SRC = r"""
extern "C" __global__ void __launch_bounds__(256)
leaf_absorb(const u32* __restrict__ U, long long ld, int nblk, long long n, u32* __restrict__ cv_io, int first, int last)
{
    const long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    u32 cv[8];
    if (first) {
        iv(cv);
    } else {
        const uint4* p = reinterpret_cast<const uint4*>(cv_io + s * 8);
        const uint4 a = p[0], b = p[1];
        cv[0] = a.x; cv[1] = a.y; cv[2] = a.z; cv[3] = a.w; cv[4] = b.x; cv[5] = b.y; cv[6] = b.z; cv[7] = b.w;
    }
    u32 block[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) block[j] = __ldg(U + (long long)j * ld + s);
    for (int b = 0; b < nblk; ++b) {
        u32 nxt[16];
        if (b + 1 < nblk) {                             /* the next block's loads in flight while this one compresses */
            #pragma unroll
            for (int j = 0; j < 16; ++j) nxt[j] = __ldg(U + (long long)((b + 1) * 16 + j) * ld + s);
        }
        u32 flags = 0u;
        if (first && b == 0) flags |= CHUNK_START;
        if (last && b == nblk - 1) flags |= CHUNK_END | ROOT;
        compress(cv, block, 0ull, 64u, flags);
        #pragma unroll
        for (int j = 0; j < 16; ++j) block[j] = nxt[j];
    }
    uint4* q = reinterpret_cast<uint4*>(cv_io + s * 8);
    q[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
    q[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
}
"""

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        # Ligero's compress (constexpr message schedule, block in registers; the shared hash_gpu one spills it to local memory)
        from ._ligero import commit_gpu
        _MOD = cp.RawModule(code=commit_gpu._SRC + _SRC, options=("-std=c++17", "-DPREFETCH=1"))
    return _MOD


def leaf_absorb(Ug: torch.Tensor, lv, first: bool, last: bool) -> None:
    """Advance the (n, 8) uint32 leaf states ``lv`` by the (g, n) int32 group ``Ug`` (g/16 Blake3 blocks)."""
    g, n = Ug.shape
    assert g % 16 == 0 and Ug.stride(1) == 1
    ts = torch.cuda.current_stream(Ug.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cs = cp.cuda.ExternalStream(ts.cuda_stream)
    with cs:
        _module().get_function("leaf_absorb")(((n + 255) // 256,), (256,),
                                              (np.uint64(Ug.data_ptr()), np.int64(Ug.stride(0)), np.int32(g // 16), np.int64(n),
                                               lv, np.int32(int(first)), np.int32(int(last))))


def commit_streamed(M: torch.Tensor, enc, group: int, commit_gpu, pad: torch.Tensor | None = None):
    """Round-1 commit of M (C, R) without materialising U: -> (root bytes, tree).  ``commit_gpu`` = Ligero's module (tree levels);
    ``pad`` (C, t): ZK padding coefficients per column."""
    C, R = M.shape
    assert C % group == 0 and group % 16 == 0 and C <= 256
    n = enc.n
    lv = cp.empty((n, 8), dtype=cp.uint32)
    buf = torch.empty((group, n), dtype=torch.int32, device=M.device)
    for q0 in range(0, C, group):
        enc.encode(M[q0:q0 + group], out=buf, pad=None if pad is None else pad[q0:q0 + group])
        leaf_absorb(buf, lv, first=(q0 == 0), last=(q0 + group == C))
    del buf
    levels = [lv] + (commit_gpu.tree_levels(lv) if n > 1 else [])
    tree = commit_gpu.Tree(C, n, levels)
    return ligero_merkle._gpu_merkle.digest_bytes(tree.root, "blake3"), tree
