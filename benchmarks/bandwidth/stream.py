"""The certified PCIe stream `stream/v1`: fresh randomness from the challenger, sent to the device at line rate during a
window and absorbed into the window's commitment, so the device's side link carries certified bytes.

~~~text
K_s      AES-128 key, 16 bytes: derive(source, "verity/bandwidth/stream/v1", {root, salt, nonce}).stream("aes-128", 16),
         the challenger's own randomness under the window's nonce; never sent to the device
S        AES-128(K_s, u128be 0) ‖ AES-128(K_s, u128be 1) ‖ …   (AES-128-CTR, counter block 0), n = 1024·C bytes, C > 64
K_a      SHA512(prefix("verity/bandwidth/stream/v1\0", "stream-blake3/v1", σ))[0:32], the device's key, from σ
D        BLAKE3-keyed(K_a, S), 32 bytes: chunk i of S (1,024 bytes) under counter i, BLAKE3's left-heavy tree of chunks
commit   the prover sends (root, D), and the verifier's clock stops on both
open(i)  chunk i's path in BLAKE3's tree: the sibling chaining values, leaf to top; the verifier regenerates chunk i from
         K_s (64 AES blocks at counter 64·i) and climbs to D, with ROOT on the last parent
samples  window key.uniform(C, ("stream", t)) for t < s, drawn only for an on-time commitment
~~~

The stream is uniform to the device: AES under a key it never sees. It exists only in the challenger's memory, and the
challenger sends it only after σ, so every byte the device knows of it crossed the link inside the window. A chunk the
device never received has an unknown chaining value, so the device's `D` is a guess or it opens that chunk with a path
that fails, and a sampled opening catches a device that skipped a fraction `f` of the chunks except with probability
`(1 − f)^s`. The verifier never hashes the stream: it checks `s` openings, each 64 AES blocks and about 30 BLAKE3
compressions.

The GPU's tree is bottom-up, an odd last node promoted, which is BLAKE3's left-heavy shape (`levels`): the absorb kernel
reduces each aligned group of `GROUP` chunks to one node in shared memory, and one final launch takes the groups' nodes
to the root.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.core.protocols.bandwidth import tags
from verity.ml.crypto import blake3
from verity.ml.randomness import Key, derive

STREAM_DOMAIN = "verity/bandwidth/stream/v1"
ABSORB_DOMAIN = b"verity/bandwidth/stream/v1\0"
ABSORB_NAME = "stream-blake3/v1"
CHUNK = blake3.CHUNK_LEN
GROUP = 64
"""Chunks the absorb kernel reduces to one node in shared memory (a CTA's share); `C > GROUP` keeps the root a parent
computed by the final launch."""

CV = tuple[int, ...]

# ---------------------------------------------------------------------------------------------------------------------
# AES-128 (FIPS 197), encryption only, plain Python: the challenger's PRG, which the host generator runs with AES-NI.


def _xtime(a: int) -> int:
    a <<= 1
    return (a ^ 0x11B) & 0xFF if a & 0x100 else a


def _sbox() -> tuple[int, ...]:
    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            x, y, p = a, b, 0
            while y:
                if y & 1:
                    p ^= x
                x, y = _xtime(x), y >> 1
            if p == 1:
                inv[a] = b
                break
    out = []
    for a in range(256):
        b = inv[a]
        s = b
        for k in range(1, 5):
            s ^= ((b << k) | (b >> (8 - k))) & 0xFF
        out.append(s ^ 0x63)
    return tuple(out)


SBOX = _sbox()


def expand_key(key: bytes) -> tuple[bytes, ...]:
    """The 11 round keys of AES-128."""
    if len(key) != 16:
        raise ValueError("an AES-128 key is 16 bytes")
    w = [list(key[4 * i:4 * i + 4]) for i in range(4)]
    rcon = 1
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = [SBOX[b] for b in t[1:] + t[:1]]
            t[0] ^= rcon
            rcon = _xtime(rcon)
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    return tuple(bytes(sum(w[4 * r:4 * r + 4], [])) for r in range(11))


def _mix(col: list[int]) -> list[int]:
    a0, a1, a2, a3 = col
    return [_xtime(a0) ^ _xtime(a1) ^ a1 ^ a2 ^ a3, a0 ^ _xtime(a1) ^ _xtime(a2) ^ a2 ^ a3,
            a0 ^ a1 ^ _xtime(a2) ^ _xtime(a3) ^ a3, _xtime(a0) ^ a0 ^ a1 ^ a2 ^ _xtime(a3)]


def encrypt_block(round_keys: Sequence[bytes], block: bytes) -> bytes:
    s = [b ^ k for b, k in zip(block, round_keys[0])]
    for r in range(1, 11):
        s = [SBOX[b] for b in s]
        s = [s[(i + 4 * (i % 4)) % 16] for i in range(16)]  # ShiftRows on the column-major state
        if r < 10:
            s = sum((_mix(s[4 * c:4 * c + 4]) for c in range(4)), [])
        s = [b ^ k for b, k in zip(s, round_keys[r])]
    return bytes(s)


def ctr(key: bytes, counter: int, blocks: int) -> bytes:
    """AES-128-CTR keystream: `blocks` blocks from the 128-bit big-endian counter block `counter`, which wraps."""
    rk = expand_key(key)
    return b"".join(encrypt_block(rk, ((counter + i) % (1 << 128)).to_bytes(16, "big")) for i in range(blocks))


# ---------------------------------------------------------------------------------------------------------------------
# The stream and its absorption.


def stream_key(source: bytes, root: bytes, salt: bytes, nonce: bytes) -> bytes:
    """`K_s`, the window's AES key: the challenger's secret `source` under the store and the window's nonce."""
    key: Key = derive(source, STREAM_DOMAIN, {"root": root, "salt": salt, "nonce": nonce})
    return key.stream("aes-128", 16)


def stream_bytes(key: bytes, n: int, offset: int = 0) -> bytes:
    """Bytes `[offset, offset + n)` of `S`; `offset` and `n` are multiples of 16."""
    if offset % 16 or n % 16:
        raise ValueError("the stream is read in whole AES blocks")
    return ctr(key, offset // 16, n // 16)


def absorb_key(seed: bytes) -> bytes:
    """`K_a`, the BLAKE3 key the device absorbs the stream under, from the window's seed σ."""
    return hashlib.sha512(tags.prefix(ABSORB_DOMAIN, ABSORB_NAME, seed)).digest()[:blake3.KEY_LEN]


def _key_words(key: bytes) -> CV:
    return tuple(int.from_bytes(key[4 * i:4 * i + 4], "little") for i in range(8))


def _block(left: CV, right: CV) -> bytes:
    return b"".join(w.to_bytes(4, "little") for w in (*left, *right))


def chunk_cv(key: bytes, i: int, chunk: bytes) -> CV:
    """Chunk `i`'s chaining value (counter `i`, not the root)."""
    if len(chunk) != CHUNK:
        raise ValueError(f"a stream chunk is {CHUNK} bytes")
    cv, flags = _key_words(key), blake3.KEYED_HASH
    for g in range(16):
        f = flags | (blake3.CHUNK_START if g == 0 else 0) | (blake3.CHUNK_END if g == 15 else 0)
        cv = tuple(blake3.compress(cv, chunk[64 * g:64 * g + 64], i, 64, f)[:8])
    return cv


def parent_cv(key: bytes, left: CV, right: CV, root: bool = False) -> CV:
    f = blake3.KEYED_HASH | blake3.PARENT | (blake3.ROOT if root else 0)
    return tuple(blake3.compress(_key_words(key), _block(left, right), 0, 64, f)[:8])


def cv_bytes(cv: CV) -> bytes:
    return b"".join(w.to_bytes(4, "little") for w in cv)


def sizes(chunks: int) -> tuple[int, ...]:
    """The bottom-up tree's level sizes over `chunks` leaves, leaves first."""
    if chunks <= GROUP:
        raise ValueError(f"a stream has more than {GROUP} chunks")
    out = [chunks]
    while out[-1] > 1:
        out.append((out[-1] + 1) // 2)
    return tuple(out)


def levels(key: bytes, cvs: Sequence[CV]) -> list[list[CV]]:
    """Every level of the tree over the chunks' chaining values, bottom-up with an odd last node promoted; the top is
    the root, computed under ROOT."""
    out = [list(cvs)]
    while len(out[-1]) > 1:
        cur = out[-1]
        top = len(cur) == 2
        out.append([parent_cv(key, cur[i], cur[i + 1], top) if i + 1 < len(cur) else cur[i]
                    for i in range(0, len(cur), 2)])
    return out


def absorb(key: bytes, stream: bytes) -> tuple[bytes, list[list[CV]]]:
    """`D` and the tree's levels, from the chunks up."""
    if len(stream) % CHUNK:
        raise ValueError(f"a stream is whole {CHUNK}-byte chunks")
    n = len(stream) // CHUNK
    sizes(n)
    lv = levels(key, [chunk_cv(key, i, stream[CHUNK * i:CHUNK * (i + 1)]) for i in range(n)])
    return cv_bytes(lv[-1][0]), lv


def group_nodes(lv: Sequence[Sequence[CV]]) -> list[CV]:
    """The level the absorb kernel writes, one node per aligned group of `GROUP` chunks."""
    return list(lv[GROUP.bit_length() - 1])


def path(lv: Sequence[Sequence[CV]], i: int) -> tuple[CV, ...]:
    """Chunk `i`'s opening: at each level its sibling, none where it is an odd last node (promoted)."""
    out = []
    for level in lv[:-1]:
        if i ^ 1 < len(level):
            out.append(level[i ^ 1])
        i >>= 1
    return tuple(out)


def verify(key: bytes, digest: bytes, chunks: int, i: int, chunk: bytes, opening: Sequence[CV]) -> bool:
    """Whether `opening` takes chunk `i` (the verifier's own bytes) to `digest` in a tree of `chunks` leaves."""
    n = sizes(chunks)
    if not 0 <= i < chunks:
        return False
    node, it = chunk_cv(key, i, chunk), iter(opening)
    for k, width in enumerate(n[:-1]):
        if i ^ 1 < width:
            sib = next(it, None)
            if sib is None:
                return False
            top = k == len(n) - 2
            node = parent_cv(key, node, sib, top) if i % 2 == 0 else parent_cv(key, sib, node, top)
        i >>= 1
    return next(it, None) is None and cv_bytes(node) == digest


def samples(window: Key, chunks: int, s: int) -> tuple[int, ...]:
    """The stream chunks a window opens, from the window's key, drawn only for an on-time commitment."""
    return tuple(window.uniform(chunks, ("stream", t)) for t in range(s))
