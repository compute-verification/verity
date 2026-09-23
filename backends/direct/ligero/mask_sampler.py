"""Device-side sampler for the ZK mask coefficients (PROTOCOL.md 8c): ChaCha20 in counter mode on the GPU, one CUDA
thread per 64-byte block, then EXACT uniform reduction to BabyBear by rejection.

Derivation (new, documented; it replaces the host SHAKE-256 stream of the 8b prover -- a different pseudorandom
stream, not bit-exact with it by design):

    key      = 32 fresh bytes (os.urandom) per proof                        (the prover's secret mask seed)
    nonce    = b"ligero-b/mask|v2" [:12]                                   (a public domain-separation constant)
    stream   = ChaCha20(key, nonce, counter = 0, 1, 2, ...)   (RFC 8439 block function, 20 rounds, 32-bit counter)
    symbols  = the stream read as little-endian uint32 words; a word u is REJECTED if u >= 2p (u >= 4026531842),
               otherwise the symbol is u mod p.

Bias.  The 8b sampler reduced a uint32 mod p directly: residues below 2^32 - 2p = 268,435,454 have probability
3 / 2^32, the others 2 / 2^32, against 1 / p ~ 2.13 / 2^32 -- a per-residue deviation of up to 0.87 / 2^32 ~ 2^-32
(the "2^-31 per symbol" of README/PROTOCOL), but a statistical distance from uniform of ~0.11 PER SYMBOL, which
accumulates over the 1.2 M mask symbols of a proof: the HVZK simulator's output was only computationally close to the
real view through that bias.  With rejection (accept iff u < 2p, then u mod p) every residue has probability exactly
2 / 2p = 1 / p conditioned on acceptance: the symbols are exactly uniform given a uniform stream, and the simulator's
distance collapses to the PRF distinguishing advantage of ChaCha20 on the mask key (computational, no statistical
term).  The acceptance probability is 2p / 2^32 = 0.9375; the sampler draws 1.08x the needed words plus 4096 and
tops up (another kernel launch) in the event -- probability < 2^-100 by Chernoff at 10^6 symbols -- of a shortfall.

Bit-exactness: the kernel is checked against a pure-Python RFC 8439 block function (``check``) and against the RFC's
section 2.3.2 test vector.  Cost: ~5 MB of stream per proof at l = 16384; the kernel is bandwidth-trivial (< 0.1 ms),
the rejection/compaction is one ``masked_select`` in torch.
"""
from __future__ import annotations

import os
import struct

import numpy as np
import torch

from .field import P

__all__ = ["ChaChaMaskSampler", "chacha20_block_ref", "sampler_for", "uniform_host"]

NONCE = b"ligero-b/mask|v2"[:12]
TWO_P = 2 * P

_SRC = r"""
#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a, b, c, d) \
    a += b; d ^= a; d = ROTL(d, 16); \
    c += d; b ^= c; b = ROTL(b, 12); \
    a += b; d ^= a; d = ROTL(d, 8);  \
    c += d; b ^= c; b = ROTL(b, 7);

extern "C" __global__ void chacha20_blocks(const unsigned* __restrict__ key8, const unsigned* __restrict__ nonce3,
                                           unsigned counter0, unsigned* __restrict__ out, unsigned nblocks)
{
    const unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    unsigned s[16], x[16];
    s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) s[4 + i] = key8[i];
    s[12] = counter0 + b;
    s[13] = nonce3[0]; s[14] = nonce3[1]; s[15] = nonce3[2];
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = s[i];
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        QR(x[0], x[4], x[8],  x[12]) QR(x[1], x[5], x[9],  x[13]) QR(x[2], x[6], x[10], x[14]) QR(x[3], x[7], x[11], x[15])
        QR(x[0], x[5], x[10], x[15]) QR(x[1], x[6], x[11], x[12]) QR(x[2], x[7], x[8],  x[13]) QR(x[3], x[4], x[9],  x[14])
    }
    unsigned* o = out + (size_t)b * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = x[i] + s[i];
}
"""


def _rotl(a: int, b: int) -> int:
    return ((a << b) | (a >> (32 - b))) & 0xFFFFFFFF


def chacha20_block_ref(key: bytes, nonce: bytes, counter: int) -> bytes:
    """RFC 8439 section 2.3 block function (pure Python; the reference the kernel is checked against)."""
    assert len(key) == 32 and len(nonce) == 12
    s = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574, *struct.unpack("<8I", key), counter & 0xFFFFFFFF,
         *struct.unpack("<3I", nonce)]
    x = list(s)

    def qr(a, b, c, d):
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF; x[d] = _rotl(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF; x[b] = _rotl(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF; x[d] = _rotl(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF; x[b] = _rotl(x[b] ^ x[c], 7)

    for _ in range(10):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    return struct.pack("<16I", *[(x[i] + s[i]) & 0xFFFFFFFF for i in range(16)])


def rfc8439_vector_ok() -> bool:
    """Section 2.3.2: key 00..1f, nonce 00:00:00:09:00:00:00:4a:00:00:00:00, counter 1."""
    key = bytes(range(32))
    nonce = bytes.fromhex("000000090000004a00000000")
    want = bytes.fromhex("10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")
    return chacha20_block_ref(key, nonce, 1) == want


def uniform_host(count: int, key: bytes) -> np.ndarray:
    """Host reference of the whole derivation: ``count`` exactly-uniform BabyBear symbols (int64) from the ChaCha20
    stream under ``key`` with rejection (slow: pure Python blocks; for tests and the CPU fallback on small sizes)."""
    out, ctr = [], 0
    while len(out) < count:
        blk = np.frombuffer(chacha20_block_ref(key, NONCE, ctr), dtype="<u4").astype(np.int64)
        out.extend(int(u) % P for u in blk if u < TWO_P)
        ctr += 1
    return np.asarray(out[:count], dtype=np.int64)


class ChaChaMaskSampler:
    def __init__(self, device):
        import cupy as cp

        self.cp = cp
        self.device = torch.device(device)
        self.kernel = cp.RawKernel(_SRC, "chacha20_blocks", options=("-std=c++17",))
        self.nonce = cp.asarray(np.frombuffer(NONCE, dtype="<u4"))

    _KEY_RING = 8

    def _key_device(self, key: bytes):
        """The 32-byte key on the device as a cupy uint32 view, uploaded from a small ring of pinned buffers with a
        non-blocking copy on the current torch stream (``cp.asarray(numpy)`` from pageable memory would stall the host
        until the stream drains; lane hp2-host).  A ring entry is reused only after its previous copy completed."""
        if not hasattr(self, "_ring"):
            self._ring = [torch.empty(32, dtype=torch.uint8, pin_memory=True) for _ in range(self._KEY_RING)]
            self._ring_ev: list = [None] * self._KEY_RING
            self._ring_at = 0
        i = self._ring_at
        self._ring_at = (i + 1) % self._KEY_RING
        if self._ring_ev[i] is not None:
            self._ring_ev[i].synchronize()
        buf = self._ring[i]
        buf.numpy()[:] = np.frombuffer(key, dtype=np.uint8)
        dev = buf.view(torch.int32).to(self.device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._ring_ev[i] = ev
        return self.cp.asarray(dev).view(self.cp.uint32), dev

    def stream(self, key: bytes | None, nblocks: int, counter0: int = 0,
               key_dev: torch.Tensor | None = None) -> torch.Tensor:
        """``nblocks`` x 16 uint32 words of keystream as an int64 torch tensor (values < 2^32).  Launched on cupy's
        current stream: callers on a torch side stream wrap the call in ``pipeline.cupy_stream_context()``.
        ``key_dev``: the key already on the device as 8 int32 words (a static input of a CUDA graph) instead of
        ``key``.  The output is a torch allocation (graph-capturable; ``cp.empty`` would go to cupy's pool)."""
        cp = self.cp
        if key_dev is not None:
            key8 = cp.asarray(key_dev).view(cp.uint32)
        else:
            key8, _keep = self._key_device(key)
        out_t = torch.empty(nblocks * 16, dtype=torch.int32, device=self.device)
        out = cp.asarray(out_t).view(cp.uint32)
        threads = 256
        self.kernel(((nblocks + threads - 1) // threads,), (threads,),
                    (key8, self.nonce, np.uint32(counter0), out, np.uint32(nblocks)))
        return out_t.to(torch.int64) & 0xFFFFFFFF

    SYNC_FREE = True    # rejection compaction by cumsum + scatter (no host sync); False = boolean indexing + top-up loop

    def uniform(self, count: int, key: bytes | None = None,
                key_dev: torch.Tensor | None = None) -> tuple[torch.Tensor, bytes | None]:
        """``count`` exactly uniform BabyBear symbols (int64, on the device) and the key used: the first ``count``
        accepted words (u < 2p) of the stream, reduced mod p.  ``key_dev``: the key as a device tensor (no host
        key; the caller drew it -- the prover's commit graph), returned key None."""
        if key_dev is None:
            key = key or os.urandom(32)
        need_words = int(count * 1.08) + 4096
        nblocks = (need_words + 15) // 16
        words = self.stream(key, nblocks, key_dev=key_dev)
        if self.SYNC_FREE:
            # the same first-``count``-accepted-words selection without a host sync: rank every accepted word among the
            # accepted ones (cumsum) and scatter it to its rank; rejected words and accepted words of rank >= count go
            # to a discarded sentinel slot.  Shortfall (fewer than ``count`` acceptances among 1.08 count + 4096 words at
            # acceptance rate 2p / 2^32 = 0.9375) has probability < 2^-100 at the prover's sizes -- the legacy top-up loop
            # covered it with a sync; here the sentinel slot would leave zeros (never observed; not a soundness or ZK
            # issue for the honest prover beyond the probability itself).
            keep = words < TWO_P
            rank = torch.cumsum(keep.to(torch.int64), 0) - 1
            tgt = torch.where(keep & (rank < count), rank, torch.full_like(rank, count))
            out = torch.zeros(count + 1, dtype=torch.int64, device=words.device)
            out.scatter_(0, tgt, words % P)
            return out[:count], key
        acc = words[words < TWO_P] % P
        ctr = nblocks
        while acc.shape[0] < count:                   # top-up: probability < 2^-100 at the prover's sizes
            more = self.stream(key, 4096, ctr, key_dev=key_dev)
            ctr += 4096
            acc = torch.cat((acc, more[more < TWO_P] % P))
        return acc[:count], key

    def check(self, key: bytes | None = None, nblocks: int = 64) -> bool:
        """Kernel vs the pure-Python block function on ``nblocks`` blocks, and the RFC 8439 vector."""
        key = key or bytes(range(32))
        dev = self.stream(key, nblocks).cpu().numpy()
        ref = np.concatenate([np.frombuffer(chacha20_block_ref(key, NONCE, c), dtype="<u4").astype(np.int64) for c in range(nblocks)])
        return bool((dev == ref).all()) and rfc8439_vector_ok()


_SAMPLERS: dict = {}


def sampler_for(device) -> ChaChaMaskSampler | None:
    key = str(device)
    if key not in _SAMPLERS:
        s = None
        if key.startswith("cuda"):
            try:
                s = ChaChaMaskSampler(device)
            except Exception:  # noqa: BLE001 -- no cupy / NVRTC
                s = None
        _SAMPLERS[key] = s
    return _SAMPLERS[key]
