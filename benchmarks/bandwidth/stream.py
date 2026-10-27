"""The certified PCIe stream `stream/v1`: fresh randomness from the challenger, sent to the device at line rate during a
window and absorbed into the window's commitment, so the device's side link carries certified bytes.

~~~text
K_s      AES-128 key, 16 bytes: derive(source, "verity/bandwidth/stream/v1", {root, salt, nonce}).stream("aes-128", 16),
         the challenger's own randomness under the window's nonce; never sent to the device
S        AES-128(K_s, u128be 0) ‖ AES-128(K_s, u128be 1) ‖ …   (AES-128-CTR, counter block 0), n bytes, n = 65,536·G, G ≥ 2
M        S read group by group: chunk 64·q + c of M (C = 64·G chunks of 1,024 bytes) is S's AES blocks
         4096·q + 64·k + c for k = 0 … 63, in that order
K_a      SHA512(prefix("verity/bandwidth/stream/v1\0", "stream-blake3/v1", σ))[0:32], the device's key, from σ
D        BLAKE3-keyed(K_a, M), 32 bytes: chunk i under counter i, BLAKE3's left-heavy tree of chunks
commit   the prover sends (root, D), and the verifier's clock stops on both
open(i)  chunk i's path in BLAKE3's tree: the sibling chaining values, leaf to top; the verifier regenerates chunk i from
         K_s (its 64 AES blocks) and climbs to D, with ROOT on the last parent
samples  window key.uniform(C, ("stream", t)) for t < s, drawn only for an on-time commitment
~~~

`M` is `S` permuted within each 64 KiB group so that the device's 64 threads of a group, one chunk each, read `S`
contiguously: at the `j`-th load of compression `g` thread `c` reads AES block `4096·q + 64·(4g + j) + c`, so every warp
load is 512 contiguous bytes and every compression 4 KiB, from device memory or straight from the host's (zero copy). The
digest binds every byte of `S` all the same.

The stream is uniform to the device: AES under a key it never sees. It exists only in the challenger's memory, and the
challenger sends it only after σ, so every byte the device knows of it crossed the link inside the window. A chunk the
device never received has an unknown chaining value, so the device's `D` is a guess or it opens that chunk with a path
that fails, and a sampled opening catches a device that skipped a fraction `f` of the chunks except with probability
`(1 − f)^s`. The verifier never hashes the stream: it checks `s` openings, each 64 AES blocks and about 30 BLAKE3
compressions.

The GPU's tree is bottom-up, an odd last node promoted, which is BLAKE3's left-heavy shape (`levels`): the absorb kernel
reduces each group's `GROUP` chunks to one node in shared memory, and one final launch takes the groups' nodes to the
root.

`fold` is a far cheaper leaf, `stream-fold/v1`: each 1 KiB chunk of `S` to the XOR of its AES blocks, 16 bytes, which
the verifier checks whole against `S` regenerated, with no tree and no sampling (`bwstream --mode fold`, `foldlive`).
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
"""Chunks of a group: one CTA's share, reduced to one node in shared memory; at least two groups keep the root a
parent computed by the final launch."""
GROUP_BYTES = GROUP * CHUNK

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


def ctr(key: bytes, first: int, blocks: int) -> bytes:
    """AES-128-CTR keystream: `blocks` blocks from the 128-bit big-endian counter block `first`, which wraps."""
    rk = expand_key(key)
    return b"".join(encrypt_block(rk, ((first + i) % (1 << 128)).to_bytes(16, "big")) for i in range(blocks))


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


AES = 16
WORDS = CHUNK // AES
"""AES blocks a chunk: 64, its `k`-th from its group's `k`-th run of `GROUP` blocks."""


def counter(i: int, k: int) -> int:
    """The AES block of `S` (its counter) that is the `k`-th 16 bytes of chunk `i` of `M`."""
    q, c = divmod(i, GROUP)
    return q * (GROUP_BYTES // AES) + k * GROUP + c


def permuted(stream: bytes) -> bytes:
    """`M`, the absorbed message."""
    if len(stream) % GROUP_BYTES or len(stream) < 2 * GROUP_BYTES:
        raise ValueError(f"a stream is at least two whole {GROUP_BYTES}-byte groups")
    return b"".join(stream[AES * counter(i, k):AES * (counter(i, k) + 1)]
                    for i in range(len(stream) // CHUNK) for k in range(WORDS))


def chunk(key: bytes, i: int) -> bytes:
    """Chunk `i` of `M`, regenerated from `K_s` alone: what the verifier checks an opening against."""
    rk = expand_key(key)
    return b"".join(encrypt_block(rk, counter(i, k).to_bytes(16, "big")) for k in range(WORDS))


def absorb_key(seed: bytes) -> bytes:
    """`K_a`, the BLAKE3 key the device absorbs the stream under, from the window's seed σ."""
    return hashlib.sha512(tags.prefix(ABSORB_DOMAIN, ABSORB_NAME, seed)).digest()[:blake3.KEY_LEN]


def _key_words(key: bytes) -> CV:
    return tuple(int.from_bytes(key[4 * i:4 * i + 4], "little") for i in range(8))


def _pair(left: CV, right: CV) -> bytes:
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
    return tuple(blake3.compress(_key_words(key), _pair(left, right), 0, 64, f)[:8])


def cv_bytes(cv: CV) -> bytes:
    return b"".join(w.to_bytes(4, "little") for w in cv)


def sizes(chunks: int) -> tuple[int, ...]:
    """The bottom-up tree's level sizes over `chunks` leaves, leaves first."""
    if chunks % GROUP or chunks < 2 * GROUP:
        raise ValueError(f"a stream is at least two whole groups of {GROUP} chunks")
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
    """`D` over the stream `S`, and the tree's levels, from the chunks up."""
    m = permuted(stream)
    n = len(m) // CHUNK
    lv = levels(key, [chunk_cv(key, i, m[CHUNK * i:CHUNK * (i + 1)]) for i in range(n)])
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


FOLD_NAME = "stream-fold/v1"


def fold(stream: bytes) -> bytes:
    """`F`, the fold leaf `stream-fold/v1`: chunk `i` of `S` itself (bytes `[1024·i, 1024·(i + 1))`, not `M`'s) to the
    XOR of its 64 AES blocks, 16 bytes a chunk, in order.

    The verifier regenerates `S` from `K_s` and checks `F` whole, so the check is exact and costs the verifier one pass
    of AES. Byte `b` of chunk `i`'s fold is the XOR of the 64 stream bytes at offset `b` of its blocks, each uniform to
    a device that never fetched it, so a device that withheld any `X` bytes of `S` leaves at least `⌈X/64⌉` fold bytes
    uniform to it and passes with probability at most `2^(-8⌈X/64⌉) ≤ 2^(-X/8)` (336 bytes: `2^-42`); a skipped
    chunk, `2^-128`."""
    if len(stream) % CHUNK:
        raise ValueError(f"the fold reads whole {CHUNK}-byte chunks")
    out = bytearray()
    for i in range(0, len(stream), CHUNK):
        acc = 0
        for k in range(i, i + CHUNK, AES):
            acc ^= int.from_bytes(stream[k:k + AES], "little")
        out += acc.to_bytes(AES, "little")
    return bytes(out)


# ---------------------------------------------------------------------------------------------------------------------
# Pinned vectors (tests/stream_vectors.json), which `bwstream --mode gates` holds the host generators and the device to.

SOURCE = bytes(range(32))
ROOT = bytes(range(64, 128))
SALT = bytes(range(128, 144))
NONCE = bytes(range(160, 192))
SEED = bytes(range(192, 224))
CARRY = (1 << 32) - 13
"""A counter whose 64 blocks cross a 32-bit carry, which a generator adding to 32-bit lanes would get wrong."""


def vectors() -> dict:
    key = stream_key(SOURCE, ROOT, SALT, NONCE)
    k_a = absorb_key(SEED)
    out: dict = {"format": "verity-bandwidth/stream-vectors/v1", "source": SOURCE.hex(), "root": ROOT.hex(),
                 "salt": SALT.hex(), "nonce": NONCE.hex(), "key_s": key.hex(),
                 "keystream_low_counter": 0, "keystream_low": ctr(key, 0, 64).hex(),
                 "keystream_carry_counter": CARRY, "keystream_carry": ctr(key, CARRY, 64).hex(),
                 "seed": SEED.hex(), "key_a": k_a.hex()}
    for name, groups in (("two", 2), ("three", 3)):
        n = groups * GROUP_BYTES
        d, lv = absorb(k_a, stream_bytes(key, n))
        out[f"stream_{name}_bytes"] = n
        out[f"stream_{name}_digest"] = d.hex()
        out[f"stream_{name}_group_nodes"] = b"".join(cv_bytes(cv) for cv in group_nodes(lv)).hex()
        out[f"stream_{name}_fold"] = fold(stream_bytes(key, n)).hex()
    return out


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "tests" / "stream_vectors.json"
    target.write_text(json.dumps(vectors(), indent=1) + "\n")
