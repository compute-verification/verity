"""The ``veritor/protocol/merkle/frame/v3`` indexed trees of :mod:`verity.commitments` on the GPU (cupy ``RawKernel``),
byte-identical to ``CommitmentDomain.leaf`` / ``.node`` / the pad leaf, plus the row digests whose outputs are the
values of row leaves: keyed BLAKE3 (``blake3-keyed/row/v2``, ``leaf/blake3.py``) and prefixed SHA-256 of row bytes.

Framing (``verity.commitments.merkle._hash``): ``SHA-256(FRAME || u32be(len tag) || tag || (u64be(len p) || p)*)``;
leaf parts ``domain_id, uint(rank), uint(position), schema, value``, node parts ``domain_id, uint(level), uint(index),
left, right``, pad parts ``domain_id, uint(rank)`` for the ranks ``[count, width)``; ``uint`` is the minimal big-endian
encoding (at least one byte).  One thread per hash.  The constant head of every message (frame, tag, domain id; the
level of a node launch) is hashed once on the host: its whole 64-byte blocks as a midstate, the rest as a byte tail
the kernel feeds before the per-thread parts.

Trees are built level by level on the device into one buffer (``Tree.dev``, all levels, leaves first); the host copy
(``Tree.levels``) is made once, eagerly for small trees and on first use for large ones (openings index it).
"""
from __future__ import annotations

import struct

import numpy as np

from . import blake3_cuda, sha256_cuda

try:
    import cupy as cp
except ImportError:            # pragma: no cover - laptop
    cp = None

FRAME = b"veritor/protocol/merkle/frame/v3\0"
LEAF, NODE, PAD = b"leaf", b"node", b"pad"

# words of the values: raw bytes, u16 words (host / device little-endian) as 2 big-endian bytes, u32 words as 4
VALUE_BYTES, VALUE_U16, VALUE_U32 = 0, 1, 2

_SRC = sha256_cuda.SRC.split('extern "C"')[0] + r"""
typedef unsigned char u8;

struct Sha { u32 st[8]; u32 w[16]; int n; u64 done; };

__device__ __forceinline__ void sha_init(Sha& s, const u32* mid, u64 done) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) s.st[j] = mid[j];
    #pragma unroll
    for (int j = 0; j < 16; ++j) s.w[j] = 0u;
    s.n = 0; s.done = done;
}

__device__ __forceinline__ void sha_put(Sha& s, u32 b) {
    s.w[s.n >> 2] |= (b & 0xffu) << (24 - 8 * (s.n & 3));
    if (++s.n == 64) {
        sha256_compress(s.st, s.w);
        #pragma unroll
        for (int j = 0; j < 16; ++j) s.w[j] = 0u;
        s.n = 0; s.done += 64;
    }
}

__device__ __forceinline__ void sha_bytes(Sha& s, const u8* p, int n) { for (int i = 0; i < n; ++i) sha_put(s, p[i]); }

__device__ __forceinline__ void sha_be(Sha& s, u64 v, int len) { for (int k = len - 1; k >= 0; --k) sha_put(s, (u32)(v >> (8 * k))); }

__device__ __forceinline__ void sha_uint_part(Sha& s, u64 v) {           // u64be(len) || minimal big-endian v
    int len = 1;
    while (len < 8 && (v >> (8 * len))) ++len;
    sha_be(s, (u64)len, 8);
    sha_be(s, v, len);
}

__device__ __forceinline__ void sha_final(Sha& s, u8* out) {
    u64 bits = (s.done + (u64)s.n) * 8ull;
    sha_put(s, 0x80u);
    while (s.n != 56) sha_put(s, 0u);
    sha_be(s, bits, 8);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        u32 h = s.st[j];
        out[4*j] = (u8)(h >> 24); out[4*j+1] = (u8)(h >> 16); out[4*j+2] = (u8)(h >> 8); out[4*j+3] = (u8)h;
    }
}

// leaf i: head || uint_part(start + i) twice (rank == position) || suffix (schema part, u64be(len value)) || value i
extern "C" __global__
void fv3_leaf(const u32* __restrict__ mid, u64 done, const u8* __restrict__ tail, int tail_len,
              const u8* __restrict__ suffix, int suffix_len, const u8* __restrict__ values, int vmode, int vlen,
              long long start, long long n, u8* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Sha s; sha_init(s, mid, done);
    sha_bytes(s, tail, tail_len);
    u64 r = (u64)(start + i);
    sha_uint_part(s, r);
    sha_uint_part(s, r);
    sha_bytes(s, suffix, suffix_len);
    if (vmode == 1) {
        sha_be(s, ((const unsigned short*)values)[i], 2);
    } else if (vmode == 2) {
        sha_be(s, ((const u32*)values)[i], 4);
    } else {
        sha_bytes(s, values + i * (long long)vlen, vlen);
    }
    sha_final(s, out + i * 32);
}

// pad i: head || uint_part(start + i)
extern "C" __global__
void fv3_pad(const u32* __restrict__ mid, u64 done, const u8* __restrict__ tail, int tail_len, long long start, long long n,
             u8* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Sha s; sha_init(s, mid, done);
    sha_bytes(s, tail, tail_len);
    sha_uint_part(s, (u64)(start + i));
    sha_final(s, out + i * 32);
}

// node i of a level: head (with uint_part(level)) || uint_part(i) || u64be(32) || child 2i || u64be(32) || child 2i+1
extern "C" __global__
void fv3_node(const u32* __restrict__ mid, u64 done, const u8* __restrict__ tail, int tail_len, const u8* __restrict__ in,
              long long n, u8* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Sha s; sha_init(s, mid, done);
    sha_bytes(s, tail, tail_len);
    sha_uint_part(s, (u64)i);
    sha_be(s, 32ull, 8);
    sha_bytes(s, in + i * 64, 32);
    sha_be(s, 32ull, 8);
    sha_bytes(s, in + i * 64 + 32, 32);
    sha_final(s, out + i * 32);
}

// row i: SHA-256(prefix || row bytes); the head / tail carry the prefix, rows are nbytes each
extern "C" __global__
void sha256_rows(const u32* __restrict__ mid, u64 done, const u8* __restrict__ tail, int tail_len, const u8* __restrict__ rows,
                 long long nbytes, long long n, u8* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Sha s; sha_init(s, mid, done);
    sha_bytes(s, tail, tail_len);
    const u8* p = rows + i * nbytes;
    long long k = 0;
    while (s.n != 0 && k < nbytes) sha_put(s, p[k++]);
    if ((((unsigned long long)(p + k)) & 3) == 0) {
        for (; k + 64 <= nbytes; k += 64) {                     // whole blocks straight into the schedule
            const u32* q = (const u32*)(p + k);
            #pragma unroll
            for (int j = 0; j < 16; ++j) s.w[j] = __byte_perm(q[j], 0, 0x0123);
            sha256_compress(s.st, s.w);
            #pragma unroll
            for (int j = 0; j < 16; ++j) s.w[j] = 0u;
            s.done += 64;
        }
    }
    for (; k < nbytes; ++k) sha_put(s, p[k]);
    sha_final(s, out + i * 32);
}
"""

_B3_SRC = blake3_cuda.SRC.split('// thread (x, y)')[0] + r"""
#define KEYED_HASH 16u

// thread (row r, chunk c): the keyed chunk chaining value of rows (N, nbytes) bytes (nbytes a multiple of 64);
// block_out (N, nb, 8) optional: the chaining value after every block (the chain witness of leaf/blake3.py)
extern "C" __global__
void b3k_chunks(const unsigned char* __restrict__ rows, long long nbytes, int nch, const u32* __restrict__ key, long long n,
                u32* __restrict__ cvs, u32* __restrict__ block_out)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nch) return;
    long long r = t / nch; int c = (int)(t % nch);
    long long lo = (long long)c * 1024, hi = lo + 1024 < nbytes ? lo + 1024 : nbytes;
    int nblk = (int)((hi - lo) / 64);
    u32 cv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv[j] = key[j];
    const uint4* p = (const uint4*)(rows + r * nbytes + lo);
    long long nb = nbytes / 64;
    for (int b = 0; b < nblk; ++b) {
        u32 m[16];
        #pragma unroll
        for (int q = 0; q < 4; ++q) { uint4 v = p[4 * b + q]; m[4*q] = v.x; m[4*q+1] = v.y; m[4*q+2] = v.z; m[4*q+3] = v.w; }
        u32 fl = KEYED_HASH | (b == 0 ? CHUNK_START : 0u) | (b == nblk - 1 ? CHUNK_END : 0u);
        compress(cv, m, (u64)c, 64u, fl);
        if (block_out) {
            u32* o = block_out + (r * nb + (long long)c * 16 + b) * 8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = cv[j];
        }
    }
    u32* o = cvs + t * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = cv[j];
}

// row r: the BLAKE3 root of its nch >= 2 chunk CVs (left-heavy parent fold, the key as the parents' CV, ROOT on top)
extern "C" __global__
void b3k_root(const u32* __restrict__ cvs, int nch, const u32* __restrict__ key, long long n, u32* __restrict__ root)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    u32 st[8][8];                  // a stack of pending subtree CVs (BLAKE3's incremental fold), nch <= 2^7
    int sp = 0;
    for (int c = 0; c < nch; ++c) {
        u32 cur[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) cur[j] = cvs[(r * nch + c) * 8 + j];
        // merge completed subtrees: after chunk c (0-based), c+1 chunks; merge while the total has trailing zeros,
        // but never the last chunk's merges (they happen in the final fold, with ROOT on the top one)
        if (c < nch - 1) {
            unsigned long long total = (unsigned long long)c + 1;
            #pragma unroll
            for (int j = 0; j < 8; ++j) st[sp][j] = cur[j];
            ++sp;
            while ((total & 1ull) == 0ull) {
                u32 m[16], cv[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) { m[j] = st[sp-2][j]; m[8+j] = st[sp-1][j]; cv[j] = key[j]; }
                compress(cv, m, 0ull, 64u, KEYED_HASH | PARENT);
                sp -= 2;
                #pragma unroll
                for (int j = 0; j < 8; ++j) st[sp][j] = cv[j];
                ++sp;
                total >>= 1;
            }
        } else {
            while (sp > 0) {
                u32 m[16], cv[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) { m[j] = st[sp-1][j]; m[8+j] = cur[j]; cv[j] = key[j]; }
                --sp;
                compress(cv, m, 0ull, 64u, KEYED_HASH | PARENT | (sp == 0 ? ROOT : 0u));
                #pragma unroll
                for (int j = 0; j < 8; ++j) cur[j] = cv[j];
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) root[r * 8 + j] = cur[j];
        }
    }
}
"""

_mods: dict[str, object] = {}


def _module(name: str):
    if name not in _mods:
        _mods[name] = cp.RawModule(code=_SRC if name == "sha" else _B3_SRC, options=("-std=c++17",))
    return _mods[name]


def _fn(mod: str, name: str):
    return _module(mod).get_function(name)


# ---------------------------------------------------------------------------------------------------------------------
# Host side of the framing: the constant head of a launch as (midstate, bytes hashed, tail)
# ---------------------------------------------------------------------------------------------------------------------

_K = (
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2)
_H0 = (0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19)
_M = 0xFFFFFFFF


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _M


def _compress(st: tuple, block: bytes) -> tuple:
    w = list(struct.unpack(">16I", block))
    for i in range(16, 64):
        s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & _M)
    a, b, c, d, e, f, g, h = st
    for i in range(64):
        t1 = (h + (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) + ((e & f) ^ (~e & g)) + _K[i] + w[i]) & _M
        t2 = ((_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & _M
        a, b, c, d, e, f, g, h = (t1 + t2) & _M, a, b, c, (d + t1) & _M, e, f, g
    return tuple((x + y) & _M for x, y in zip(st, (a, b, c, d, e, f, g, h)))


def midstate(head: bytes) -> tuple[np.ndarray, int, bytes]:
    """``head`` -> (the SHA-256 state after its whole 64-byte blocks (8,) uint32, the bytes they cover, the rest)."""
    st, k = _H0, 0
    while k + 64 <= len(head):
        st = _compress(st, head[k:k + 64])
        k += 64
    return np.asarray(st, dtype=np.uint32), k, head[k:]


def part(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def uint(v: int) -> bytes:
    return int(v).to_bytes(max(1, (int(v).bit_length() + 7) // 8), "big")


def head(tag: bytes, domain_id: bytes, *parts: bytes) -> bytes:
    """The constant head of a framed hash: frame, tag, then ``part`` of the domain id and of each of ``parts``."""
    return FRAME + len(tag).to_bytes(4, "big") + tag + part(domain_id) + b"".join(part(p) for p in parts)


class _Head:
    """A launch's head on the device: the midstate, the byte count it covers, the tail bytes."""

    _cache: dict = {}

    def __init__(self, data: bytes):
        mid, done, tail = midstate(data)
        self.mid = cp.asarray(mid)
        self.done = np.uint64(done)
        self.tail = cp.asarray(np.frombuffer(tail or b"\0", dtype=np.uint8))
        self.tail_len = np.int32(len(tail))

    @classmethod
    def of(cls, data: bytes) -> "_Head":
        h = cls._cache.get(data)
        if h is None:
            if len(cls._cache) > 4096:
                cls._cache.clear()
            h = cls._cache[data] = cls(data)
        return h


def _grid(n: int, block: int) -> tuple[int]:
    return (max(1, (n + block - 1) // block),)


# ---------------------------------------------------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------------------------------------------------

def depth_of(count: int) -> int:
    return max(0, (int(count) - 1).bit_length())


def leaves_into(out, domain_id: bytes, schema: str, values, *, vmode: int = VALUE_BYTES, vlen: int | None = None,
                start: int = 0, block: int = 256) -> None:
    """``out[i] = domain.leaf(start + i, start + i, schema, value_i)`` for the ``n = out.shape[0]`` values: ``values`` a
    device array of ``n`` rows of ``vlen`` bytes (``VALUE_BYTES``), or of ``n`` u16 / u32 words (``VALUE_U16`` /
    ``VALUE_U32``: the value is the word's 2 / 4 big-endian bytes)."""
    n = int(out.shape[0])
    if n == 0:
        return
    if vmode == VALUE_U16:
        vlen = 2
    elif vmode == VALUE_U32:
        vlen = 4
    h = _Head.of(head(LEAF, domain_id))
    suffix = cp.asarray(np.frombuffer(part(schema.encode("utf-8")) + int(vlen).to_bytes(8, "big"), dtype=np.uint8))
    _fn("sha", "fv3_leaf")(_grid(n, block), (block,), (h.mid, h.done, h.tail, h.tail_len, suffix, np.int32(suffix.size),
                                                          values, np.int32(vmode), np.int32(vlen), np.int64(start), np.int64(n), out))


def pads_into(out, domain_id: bytes, start: int, block: int = 256) -> None:
    n = int(out.shape[0])
    if n == 0:
        return
    h = _Head.of(head(PAD, domain_id))
    _fn("sha", "fv3_pad")(_grid(n, block), (block,), (h.mid, h.done, h.tail, h.tail_len, np.int64(start), np.int64(n), out))


def nodes_into(out, domain_id: bytes, level: int, children, block: int = 256) -> None:
    n = int(out.shape[0])
    h = _Head.of(head(NODE, domain_id, uint(level)))
    _fn("sha", "fv3_node")(_grid(n, block), (block,), (h.mid, h.done, h.tail, h.tail_len, children, np.int64(n), out))


class Tree:
    """A frame-v3 tree on the device: ``dev`` (2 width - 1, 32) uint8, level ``l`` at ``offsets[l]`` (leaves first)."""

    EAGER_BYTES = 1 << 22          # copy to the host at build time up to this size (row / output trees), else on first use

    def __init__(self, dev, depth: int):
        self.dev, self.depth = dev, depth
        width = 1 << depth
        self.offsets = [0]
        for lv in range(depth):
            self.offsets.append(self.offsets[-1] + (width >> lv))
        self._host = None
        if dev.nbytes <= self.EAGER_BYTES:
            self.host()

    @property
    def root(self) -> bytes:
        return (self._host[-1] if self._host is not None else cp.asnumpy(self.dev[-1])).tobytes()

    def level_dev(self, lv: int):
        return self.dev[self.offsets[lv]:self.offsets[lv] + ((1 << self.depth) >> lv)]

    def host(self) -> np.ndarray:
        if self._host is None:
            self._host = cp.asnumpy(self.dev)
        return self._host

    @property
    def levels(self) -> list[np.ndarray]:
        h = self.host()
        return [h[self.offsets[lv]:self.offsets[lv] + ((1 << self.depth) >> lv)] for lv in range(self.depth + 1)]


def build_tree(domain_id: bytes, count: int, schema: str, values, *, vmode: int = VALUE_BYTES, vlen: int | None = None,
               block: int = 256) -> Tree:
    """The frame-v3 tree of ``count`` leaves (ranks ``0..count``, positions = ranks) with pads to ``2^depth`` and the
    nodes level by level, all on the device (one launch per level)."""
    depth = depth_of(count)
    width = 1 << depth
    dev = cp.empty((2 * width - 1, 32), dtype=cp.uint8)
    leaves_into(dev[:count], domain_id, schema, values, vmode=vmode, vlen=vlen, block=block)
    pads_into(dev[count:width], domain_id, count, block=block)
    off = 0
    for lv in range(depth):
        n = width >> (lv + 1)
        nodes_into(dev[off + (width >> lv):off + (width >> lv) + n], domain_id, lv, dev[off:off + (width >> lv)], block=block)
        off += width >> lv
    return Tree(dev, depth)


# ---------------------------------------------------------------------------------------------------------------------
# Row digests (the values of row leaves)
# ---------------------------------------------------------------------------------------------------------------------

def blake3_keyed_rows(rows, key: bytes, *, block_out: bool = False, block: int = 128):
    """Keyed BLAKE3 of every row of ``rows`` (N, nbytes) uint8 on the device (nbytes a multiple of 64, at least 2 chunks):
    ``(cvs (N, n_chunks, 8) uint32 chunk chaining values, roots (N, 8) uint32 little-endian words = the 32-byte digest
    [, block_out (N, n_blocks, 8) the chaining value after every block])``."""
    rows = cp.ascontiguousarray(rows)
    if rows.dtype != cp.uint8:
        rows = rows.view(cp.uint8)
    n, nbytes = int(rows.shape[0]), int(rows.shape[1])
    if nbytes % 64 or nbytes <= 1024:
        raise ValueError("rows of whole blocks and more than one chunk")
    nch = -(-nbytes // 1024)
    kw = cp.asarray(np.frombuffer(key, dtype="<u4").astype(np.uint32))
    cvs = cp.empty((n, nch, 8), dtype=cp.uint32)
    bo = cp.empty((n, nbytes // 64, 8), dtype=cp.uint32) if block_out else None
    roots = cp.empty((n, 8), dtype=cp.uint32)
    if n:
        _fn("b3", "b3k_chunks")(_grid(n * nch, block), (block,), (rows, np.int64(nbytes), np.int32(nch), kw, np.int64(n), cvs,
                                                                 bo if bo is not None else np.uint64(0)))
        _fn("b3", "b3k_root")(_grid(n, block), (block,), (cvs, np.int32(nch), kw, np.int64(n), roots))
    return (cvs, roots, bo) if block_out else (cvs, roots)


def sha256_rows(rows, prefix: bytes = b"", block: int = 128):
    """``SHA-256(prefix || row)`` of every row of ``rows`` (N, nbytes) uint8 on the device -> (N, 32) uint8."""
    rows = cp.ascontiguousarray(rows)
    if rows.dtype != cp.uint8:
        rows = rows.view(cp.uint8)
    n, nbytes = int(rows.shape[0]), int(rows.shape[1])
    out = cp.empty((n, 32), dtype=cp.uint8)
    if n:
        h = _Head.of(prefix)
        _fn("sha", "sha256_rows")(_grid(n, block), (block,), (h.mid, h.done, h.tail, h.tail_len, rows, np.int64(nbytes),
                                                              np.int64(n), out))
    return out
