"""Keyed fills and whole-buffer keyed digests for the volatile-memory secure-erasure calibration.

The secure-erasure audit asks a prover to hold, at one instant, a fresh-key fill `F_key` of all of its volatile memory
(HBM and host DRAM) and to answer a keyed digest over it by a deadline `T`. This module defines the fill stream, the
physical layouts it is written in, the digest over the written buffer, their CPU references, and the Triton kernels that
compute them on the device. `driver.py` times the kernels; it checks them bit-exactly against these references first.

The fill stream is a sequence of 32-bit words indexed by the logical word index `g` (block `b = g // WORDS`, in-block word
`r = g % WORDS`, `WORDS = 16384`, one 64 KiB block):

- `chachaR` (R = 8, 12, 20 rounds): `r = j * LANES + l` with `j < 16`, `l < LANES = 1024`, and the word is output word `j`
  of the ChaCha block function (RFC 8439 section 2.3, R rounds) on the input state `sigma, key[0..8], ctr_lo, ctr_hi,
  nonce[0], nonce[1]` with `ctr = b * LANES + l`: counter mode with a 64-bit block counter, so each lane of a device
  program computes one ChaCha block and every store is coalesced.
- `mix`: `fmix32(fmix32(g_lo ^ key[0]) ^ g_hi ^ key[1])`, the murmur3 finalizer twice; a keyed, non-cryptographic counter
  hash that runs at memory speed, the reference for what the memory system alone allows.

A layout maps the logical word `g = b * WORDS + r` of an `N`-block fill to a physical word of the buffer:

- `seq`: `g`.
- `blockstride`: `((b * A) mod N) * WORDS + r`, with `A` coprime to `N` (`stride_a`): whole 64 KiB blocks, scattered.
- `wordstride`: `r * N + b`, the transpose: consecutive words of a block land `N * 4` bytes apart, so no two stores of a
  warp share a memory transaction (the worst case for the memory system).

The digest is `hbm_audit/digest.py`'s keyed two-level polynomial hash over `GF(2^31 - 1)`, over all `N` physical blocks
in a digest order: `seq` (block `i` at position `i`) or `perm`, a keyed permutation of the blocks. The block hash points
`(g, z)` and the permutation's seed come from the digest key, a key of its own (`digest_key`).
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hbm_audit"))
import digest as D  # noqa: E402

BLOCK_BYTES = D.BLOCK_BYTES
WORDS = D.WORDS
LANES = 1024
CHACHA_WORDS = 16
assert LANES * CHACHA_WORDS == WORDS
MOD = D.MOD
M32 = 0xFFFFFFFF
SIGMA = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)  # "expand 32-byte k"

GENERATORS = {"mix": 0, "chacha8": 8, "chacha12": 12, "chacha20": 20}  # ChaCha rounds; 0 is the mix stream
LAYOUTS = {"seq": 0, "blockstride": 1, "wordstride": 2}
ORDERS = ("seq", "perm")


# --------------------------------------------------------------------------- keys


def _derive(seed: int, run: int, domain: str) -> bytes:
    return hashlib.sha256(f"verity/erase-calib/v0/{domain}/{seed}/{run}".encode()).digest()


def fill_key(seed: int, run: int) -> tuple[int, ...]:
    """Ten 32-bit words for run `run` of seed `seed`: the ChaCha key (eight words) and nonce (two). `mix` uses the first
    two."""
    b = _derive(seed, run, "fill") + _derive(seed, run, "fill-nonce")[:8]
    return tuple(int.from_bytes(b[4 * i : 4 * i + 4], "little") for i in range(10))


def digest_key(seed: int, run: int) -> tuple[int, int, int]:
    """`(g, z, perm_seed)`: the block-hash and fold points, both in `[2, p - 1]`, and the seed of the `perm` order."""
    b = _derive(seed, run, "digest")
    g = int.from_bytes(b[0:8], "little") % (MOD - 2) + 2
    z = int.from_bytes(b[8:16], "little") % (MOD - 2) + 2
    return g, z, int.from_bytes(b[16:24], "little") & ((1 << 63) - 1)


def stride_a(n: int) -> int:
    """The `blockstride` multiplier for `n` blocks: the first odd number at or above `0.618 n` coprime to `n`."""
    if n <= 2:
        return 1
    a = int(n * 0.6180339887) | 1
    while math.gcd(a, n) != 1:
        a += 2
    return a % n


# --------------------------------------------------------------------------- CPU references


def _rotl(x: np.ndarray, n: int) -> np.ndarray:
    return (x << np.uint32(n)) | (x >> np.uint32(32 - n))


def chacha_block(state: np.ndarray, rounds: int) -> np.ndarray:
    """The ChaCha block function on `state` (`[16, n] uint32`, one input state per column): `[16, n]` output words."""
    x = [state[i].astype(np.uint32).copy() for i in range(16)]

    def qr(a, b, c, d):
        x[a] += x[b]
        x[d] = _rotl(x[d] ^ x[a], 16)
        x[c] += x[d]
        x[b] = _rotl(x[b] ^ x[c], 12)
        x[a] += x[b]
        x[d] = _rotl(x[d] ^ x[a], 8)
        x[c] += x[d]
        x[b] = _rotl(x[b] ^ x[c], 7)

    for _ in range(rounds // 2):
        qr(0, 4, 8, 12), qr(1, 5, 9, 13), qr(2, 6, 10, 14), qr(3, 7, 11, 15)
        qr(0, 5, 10, 15), qr(1, 6, 11, 12), qr(2, 7, 8, 13), qr(3, 4, 9, 14)
    return np.stack([x[i] + state[i].astype(np.uint32) for i in range(16)])


def chacha_states(key: tuple[int, ...], ctr: np.ndarray) -> np.ndarray:
    """Input states `[16, n]` for 64-bit block counters `ctr` (`uint64`) under a ten-word `key`."""
    st = np.empty((16, ctr.size), dtype=np.uint32)
    st[0:4] = np.array(SIGMA, dtype=np.uint32)[:, None]
    st[4:12] = np.array(key[0:8], dtype=np.uint32)[:, None]
    st[12] = (ctr & np.uint64(M32)).astype(np.uint32)
    st[13] = (ctr >> np.uint64(32)).astype(np.uint32)
    st[14] = np.uint32(key[8])
    st[15] = np.uint32(key[9])
    return st


def fmix32(h: np.ndarray) -> np.ndarray:
    h = h ^ (h >> np.uint32(16))
    h = h * np.uint32(0x85EBCA6B)
    h = h ^ (h >> np.uint32(13))
    h = h * np.uint32(0xC2B2AE35)
    return h ^ (h >> np.uint32(16))


def logical_words(gen: str, key: tuple[int, ...], n_blocks: int, block_base: int = 0) -> np.ndarray:
    """The fill stream's words for logical blocks `block_base .. block_base + n_blocks`, in logical order (`uint32`)."""
    rounds = GENERATORS[gen]
    if rounds == 0:
        g = np.arange(block_base * WORDS, (block_base + n_blocks) * WORDS, dtype=np.uint64)
        lo = (g & np.uint64(M32)).astype(np.uint32)
        hi = (g >> np.uint64(32)).astype(np.uint32)
        return fmix32(fmix32(lo ^ np.uint32(key[0])) ^ hi ^ np.uint32(key[1]))
    ctr = np.arange(block_base * LANES, (block_base + n_blocks) * LANES, dtype=np.uint64)
    out = chacha_block(chacha_states(key, ctr), rounds)  # [16, n_blocks * LANES]
    return out.reshape(CHACHA_WORDS, n_blocks, LANES).transpose(1, 0, 2).reshape(-1)


def physical_index(layout: str, n_blocks: int) -> np.ndarray:
    """The physical word of each logical word `g` of an `n_blocks` fill (`int64 [n_blocks * WORDS]`)."""
    g = np.arange(n_blocks * WORDS, dtype=np.int64)
    b, r = g // WORDS, g % WORDS
    if layout == "seq":
        return g
    if layout == "blockstride":
        return ((b * stride_a(n_blocks)) % n_blocks) * WORDS + r
    if layout == "wordstride":
        return r * n_blocks + b
    raise ValueError(f"unknown layout {layout!r}")


def reference_fill(gen: str, key: tuple[int, ...], n_blocks: int, layout: str) -> np.ndarray:
    """The buffer the fill leaves: `uint32 [n_blocks, WORDS]`, physical blocks in address order."""
    out = np.empty(n_blocks * WORDS, dtype=np.uint32)
    out[physical_index(layout, n_blocks)] = logical_words(gen, key, n_blocks)
    return out.reshape(n_blocks, WORDS)


def powers_into(out, base: int):
    """`out[i] = base^i mod p` by doubling (log2(len) vector steps). `out` is any int64 array with slicing (numpy, or a
    torch device tensor, which is how the device digest builds its tables without a host loop)."""
    n = out.shape[0]
    out[0] = 1
    filled = 1
    while filled < n:
        m = min(filled, n - filled)
        out[filled : filled + m] = out[:m] * pow(base, filled, MOD) % MOD
        filled += m
    return out


def reference_block_hashes(blocks: np.ndarray, g: int) -> np.ndarray:
    """`P_g` of every row of `blocks` (`uint32 [n, WORDS]`): `sum_j w[j] g^(W-1-j) mod p`, as `digest.reference_block_hash`
    computes it word by word."""
    gpow = powers_into(np.empty(WORDS, dtype=np.int64), g)[::-1]
    w = blocks.astype(np.int64) % MOD
    return ((w * gpow[None, :]) % MOD).sum(axis=1) % MOD


def reference_digest(blocks: np.ndarray, idx: np.ndarray, g: int, z: int) -> int:
    """`sum_r P_g(blocks[idx[r]]) z^r mod p`."""
    h = reference_block_hashes(blocks, g)[idx]
    zp = powers_into(np.empty(len(idx), dtype=np.int64), z)
    return int(((h * zp) % MOD).sum() % MOD)


# --------------------------------------------------------------------------- device kernels

_FILL = None


def _build():
    """The fill kernel, imported lazily so this module imports without torch or Triton for the CPU tests."""
    global _FILL
    if _FILL is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fill_kernel

        _FILL = fill_kernel.fill_kernel
    return _FILL


def signed_words(words) -> list[int]:
    """32-bit words as the int32 values a device key tensor holds."""
    return [w - (1 << 32) if w >= (1 << 31) else w for w in words]


def device_fill(buf, key_dev, n_blocks: int, gen: str, layout: str, block_base: int = 0, num_warps: int = 8):
    """Write the `gen` fill of `n_blocks` blocks in `layout` into `buf` (int32, at least `n_blocks * WORDS` words).
    `key_dev` is the int32 [10+] key tensor on the device. Asynchronous on the current stream."""
    a = stride_a(n_blocks) if layout == "blockstride" else 1
    _build()[(n_blocks,)](buf, key_dev, n_blocks, a, block_base, ROUNDS=GENERATORS[gen], LAYOUT=LAYOUTS[layout],
                          L=LANES, NW=CHACHA_WORDS, num_warps=num_warps)


def device_digest(torch, blocks2d, idx, g: int, z: int, hbuf, zbuf, cbuf):
    """The digest of `blocks2d` (int32 [N, WORDS] view) in order `idx` (int64 [N] device), as a device scalar tensor.
    `hbuf`, `zbuf` (int64 [>= N]) and `cbuf` (int64 [INTERLEAVE]) are reused scratch; the key tables are built on the
    device by doubling, so no step loops over N on the host."""
    n = idx.numel()
    combine = powers_into(cbuf, g).flip(0)
    h = D.block_hashes_pre(blocks2d, idx, combine.contiguous(), pow(g, D.INTERLEAVE, MOD), hbuf[:n])
    zp = powers_into(zbuf[:n], z)
    return ((h % MOD) * zp % MOD).sum() % MOD
