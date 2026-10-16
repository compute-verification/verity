"""The reference for `verity/kv-filler/fill-sk16/v0`, the filler's receipt (`fill.cu`): the signs, a block's sketch and
leaf, a unit's root, and a chunk's check word. The device must match it bit for bit (`bench.py`'s gate)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verity.ml.crypto.blake3 import compress  # noqa: E402

NAME = "verity/kv-filler/fill-sk16/v0"
BLK, UNIT_BLOCKS, ROWS = 4096, 32, 16
UNIT_BYTES = BLK * UNIT_BLOCKS
KEYED, START, END, PARENT = 16, 1, 2, 4


def signs(seed: bytes) -> bytes:
    """8 KiB: SHA-512(seed ‖ NAME/signs ‖ u32le i), i = 0..127; bit 4096 r + i (LSB first) is σ(r, i)'s."""
    tag = (NAME + "/signs").encode()
    return b"".join(hashlib.sha512(seed + tag + i.to_bytes(4, "little")).digest() for i in range(128))


def sign_matrix(sign_bytes: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8), bitorder="little").reshape(ROWS, BLK)
    return 1 - 2 * bits.astype(np.int64)


def key_words(seed: bytes) -> tuple[int, ...]:
    return tuple(int.from_bytes(seed[4 * i:4 * i + 4], "little") for i in range(8))


def sketches(blocks: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Each block's 16 int32 sums, blocks given as an (n, 4096) uint8 array read as int8."""
    return blocks.view(np.int8).astype(np.int64) @ sigma.T


def leaf(K: tuple[int, ...], s: np.ndarray, j: int) -> list[int]:
    return compress(K, s.astype("<i4").tobytes(), j, 64, KEYED | START | END)[:8]


def parent(K: tuple[int, ...], left: list[int], right: list[int], level: int, first: int) -> list[int]:
    msg = b"".join(w.to_bytes(4, "little") for w in left + right)
    return compress(K, msg, (level << 56) | first, 64, KEYED | PARENT)[:8]


def unit_root(unit: bytes, u: int, K: tuple[int, ...], sigma: np.ndarray) -> bytes:
    """Unit u's root: the tree over its 32 leaves, block j = 32 u + b."""
    s = sketches(np.frombuffer(unit, dtype=np.uint8).reshape(UNIT_BLOCKS, BLK), sigma)
    j0 = u * UNIT_BLOCKS
    nodes = [leaf(K, s[b], j0 + b) for b in range(UNIT_BLOCKS)]
    level = 1
    while len(nodes) > 1:
        half = 1 << (level - 1)
        nodes = [parent(K, nodes[2 * i], nodes[2 * i + 1], level, j0 + 2 * i * half) for i in range(len(nodes) // 2)]
        level += 1
    return b"".join(w.to_bytes(4, "little") for w in nodes[0])


def plan(region_bytes: list[int]) -> list[tuple[int, int]]:
    """Each unit of the plan over regions of these sizes: (region, byte offset)."""
    return [(r, k * UNIT_BYTES) for r, n in enumerate(region_bytes) for k in range(n // UNIT_BYTES)]


def chunk(regions: list[bytes], unit0: int, units: int, seed: bytes, sign_seed: bytes) -> tuple[list[bytes], int]:
    """Units [unit0, unit0 + units) of the plan: their roots, and the check word (XOR of each root's first LE word)."""
    K, sigma = key_words(seed), sign_matrix(signs(sign_seed))
    where = plan([len(r) for r in regions])
    roots = []
    for u in range(unit0, unit0 + units):
        r, off = where[u]
        roots.append(unit_root(regions[r][off:off + UNIT_BYTES], u, K, sigma))
    x = 0
    for root in roots:
        x ^= int.from_bytes(root[:4], "little")
    return roots, x


def chunk_seed(run_seed: bytes, c: int) -> bytes:
    """The top-up thread's chunk seed: the run's with its first 8 bytes XORed with the chunk's number."""
    s = bytearray(run_seed)
    for k in range(8):
        s[k] ^= (c >> (8 * k)) & 0xFF
    return bytes(s)


# --- verity/kv-filler/fill-sk16/v2: sketch-then-SHA-512, units by sequence number, check words by epoch -------------
# Unit s of a run is plan unit s mod P in pass p = s div P. Pass p's signs are its own (`signs_v2`). Block j's (its plan
# index) word is the first 4 bytes, big-endian, of SHA-512(TAG_BLK ‖ its 16 sums as i32be ‖ u64be j ‖ u64be p), one
# compression; a unit's word is the XOR of its 32 blocks', and an epoch's (2^shift consecutive s) the XOR of its units'.

NAME_V2 = "verity/kv-filler/fill-sk16/v2"
TAG_BLK, TAG_SGN = b"fill-sk16/v2/blk", b"fill-sk16/v2/sgn"


def signs_v2(run_seed: bytes, p: int) -> bytes:
    """Pass p's 8 KiB: SHA-512(TAG_SGN ‖ run seed ‖ u64be p ‖ u64be i), i = 0..127; bit 4096 r + i (LSB first) is σ(r, i)'s."""
    head = TAG_SGN + run_seed + p.to_bytes(8, "big")
    return b"".join(hashlib.sha512(head + i.to_bytes(8, "big")).digest() for i in range(128))


def block_word(s: np.ndarray, j: int, p: int) -> int:
    msg = TAG_BLK + s.astype(">i4").tobytes() + j.to_bytes(8, "big") + p.to_bytes(8, "big")
    return int.from_bytes(hashlib.sha512(msg).digest()[:4], "big")


def unit_word(unit: bytes, u: int, p: int, sigma: np.ndarray) -> int:
    s = sketches(np.frombuffer(unit, dtype=np.uint8).reshape(UNIT_BLOCKS, BLK), sigma)
    x = 0
    for b in range(UNIT_BLOCKS):
        x ^= block_word(s[b], u * UNIT_BLOCKS + b, p)
    return x


def epoch(regions: list[bytes], s0: int, n: int, run_seed: bytes) -> int:
    """The check word of units s0 .. s0 + n − 1."""
    where = plan([len(r) for r in regions])
    P = len(where)
    x, sigma_of = 0, {}
    for s in range(s0, s0 + n):
        p, u = divmod(s, P)
        if p not in sigma_of:
            sigma_of = {p: sign_matrix(signs_v2(run_seed, p))}
        r, off = where[u]
        x ^= unit_word(regions[r][off:off + UNIT_BYTES], u, p, sigma_of[p])
    return x
