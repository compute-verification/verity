"""The fused tag's blocks over the plain fragment layout, and a twin of `tf.cu`'s lane stream.  Standard library only.

The store is the word codec's plain fragment layout (`benchmarks/word_codec`): row block `nb` (16 rows of W, N x K e4m3
bytes) holds its K / 32 tiles in order, a tile 32 lanes x 16 bytes, lane `l = 4 g + tq` holding its m16n8k32 A fragment
`a0 a1 a2 a3`: rows g and g + 8, columns 4 tq .. 4 tq + 3 and 16 + 4 tq .. 16 + 4 tq + 3 of the tile.  A block is one
lane's 16 bytes over `span` consecutive tiles of one row block:

~~~text
unit u   = nb * (K / 32 / span) + s       tiles s * span .. s * span + span - 1 of row block nb
block j  = jbase + 32 u + lane            E_j = lane's 16 bytes of each tile of the unit, in tile order
y_j      = BLAKE3-keyed(K, u64be j || E_j)  ro-blake3/v1, verity/core/protocols/bandwidth/tags.py
~~~

so the blocks are a fixed permutation of the store's bytes, and the warp that multiplies a unit's tiles hashes them from
its registers.  `lane_tag` is the kernel's order of work on one block: a compression per four tiles, the last 8 bytes
of each quad carried into the next, then the tail chunk and the parents.

Chunk blocks (`fused2_kernel`) make a lane's 16 bytes of 64 tiles, D, one BLAKE3 chunk, and `lanes` consecutive lanes
one block of `lanes` KiB, each D rotated left by 8 bytes so that chunk i of `u64be j || E_j` is whole in lane i:

~~~text
unit u   = s * (N / 16) + nb               tiles 64 s .. 64 s + 63 of row block nb
block j  = jbase + (32 / lanes) u + lane / lanes
E_j      = D_0[8:] D_0[:8] ... D_(L-1)[8:] D_(L-1)[:8]    chunk i = P_i || D_i[8:], P_0 = u64be j, P_i = D_(i-1)[:8]
~~~

`chunk_tag` is the kernel's order of work: the chunks, the 8-byte tail D_(L-1)[:8], the parents and the root.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.ml.crypto import blake3

LANES = 32
LANE_BYTES = 16
TILE_COLS = 32
SPANS = (32, 64, 128)

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
LAYERS = 32  # Llama-3.1-8B


def plain_fragments(w: bytes, n: int, k: int) -> bytes:
    """W (n x k bytes, row-major) in the plain fragment layout."""
    if len(w) != n * k or n % 16 or k % TILE_COLS:
        raise ValueError("W is n x k bytes, n a multiple of 16 and k of 32")
    out = bytearray()
    for nb in range(n // 16):
        for t in range(k // TILE_COLS):
            for lane in range(LANES):
                g, tq = divmod(lane, 4)
                for hc in range(2):
                    for h in range(2):
                        row, col = nb * 16 + 8 * h + g, t * TILE_COLS + 16 * hc + 4 * tq
                        out += w[row * k + col:row * k + col + 4]
    return bytes(out)


def spans(k: int, span: int) -> int:
    if span not in SPANS or k % (TILE_COLS * span):
        raise ValueError(f"a span is one of {SPANS} tiles and divides K / 32")
    return k // TILE_COLS // span


def block_count(n: int, k: int, span: int) -> int:
    return n // 16 * spans(k, span) * LANES


def blocks(store: bytes, n: int, k: int, span: int) -> list[bytes]:
    """Every block of the store, in index order (block j at position j - jbase)."""
    tiles, ns = k // TILE_COLS, spans(k, span)
    out = []
    for nb in range(n // 16):
        for s in range(ns):
            for lane in range(LANES):
                at = [((nb * tiles + t) * LANES + lane) * LANE_BYTES for t in range(s * span, (s + 1) * span)]
                out.append(b"".join(store[a:a + LANE_BYTES] for a in at))
    return out


def _words(b: bytes) -> tuple[int, ...]:
    return tuple(int.from_bytes(b[4 * i:4 * i + 4], "little") for i in range(len(b) // 4))


def _bytes(ws: Sequence[int]) -> bytes:
    return b"".join(w.to_bytes(4, "little") for w in ws)


def lane_tag(key: bytes, j: int, block: bytes) -> bytes:
    """`tf.cu`'s `LaneTag` on one block; equal to `blake3.keyed(key, u64be j || block)`."""
    span = len(block) // LANE_BYTES
    if span not in SPANS or len(block) % LANE_BYTES:
        raise ValueError(f"a block is 16 bytes times one of {SPANS}")
    kw = _words(key)
    keyed = blake3.KEYED_HASH
    cv, carry, left = kw, j.to_bytes(8, "big"), None
    for qd in range(span // 4):
        quad = block[64 * qd:64 * qd + 64]
        m, carry = carry + quad[:56], quad[56:]
        ci = qd & 15
        fl = keyed | (blake3.CHUNK_START if ci == 0 else 0) | (blake3.CHUNK_END if span > 32 and ci == 15 else 0)
        cv = tuple(blake3.compress(cv, m, qd >> 4, 64, fl)[:8])
        if span == 128 and qd == 15:
            left, cv = cv, kw
    tail = carry + bytes(56)
    if span == 32:
        return _bytes(blake3.compress(cv, tail, 0, 8, keyed | blake3.CHUNK_END | blake3.ROOT)[:8])
    last = 1 if span == 64 else 2
    t = blake3.compress(kw, tail, last, 8, keyed | blake3.CHUNK_START | blake3.CHUNK_END)[:8]
    if span == 128:
        cv = blake3.compress(kw, _bytes(left) + _bytes(cv), 0, 64, keyed | blake3.PARENT)[:8]
    return _bytes(blake3.compress(kw, _bytes(cv) + _bytes(t), 0, 64, keyed | blake3.PARENT | blake3.ROOT)[:8])


CHUNK_TILES = 64  # a lane's 16 bytes of 64 tiles: one BLAKE3 chunk
GROUPS = (1, 2, 4, 8)  # lanes (chunks) a chunk block takes


def chunk_spans(k: int) -> int:
    if k % (TILE_COLS * CHUNK_TILES):
        raise ValueError(f"K is a multiple of {TILE_COLS * CHUNK_TILES}")
    return k // (TILE_COLS * CHUNK_TILES)


def chunk_block_count(n: int, k: int, lanes: int) -> int:
    if lanes not in GROUPS:
        raise ValueError(f"a chunk block is one of {GROUPS} lanes")
    return n // 16 * chunk_spans(k) * LANES // lanes


def lane_chunks(store: bytes, n: int, k: int) -> list[bytes]:
    """Every lane's 1,024 bytes D in `tf.cu`'s unit order: unit u = s * (n / 16) + nb, then lane."""
    tiles = k // TILE_COLS
    out = []
    for s in range(chunk_spans(k)):
        for nb in range(n // 16):
            for lane in range(LANES):
                at = [((nb * tiles + t) * LANES + lane) * LANE_BYTES
                      for t in range(s * CHUNK_TILES, (s + 1) * CHUNK_TILES)]
                out.append(b"".join(store[a:a + LANE_BYTES] for a in at))
    return out


def chunk_blocks(store: bytes, n: int, k: int, lanes: int) -> list[bytes]:
    """Every chunk block, in index order: `lanes` consecutive lanes' D, each rotated left by 8 bytes."""
    if lanes not in GROUPS:
        raise ValueError(f"a chunk block is one of {GROUPS} lanes")
    ds = lane_chunks(store, n, k)
    return [b"".join(d[8:] + d[:8] for d in ds[i:i + lanes]) for i in range(0, len(ds), lanes)]


def chunk_tag(key: bytes, j: int, ds: Sequence[bytes]) -> bytes:
    """`tf.cu`'s `ChunkTag` on one block from its lanes' D; equal to `blake3.keyed(key, u64be j || E_j)`."""
    if len(ds) not in GROUPS or any(len(d) != 16 * CHUNK_TILES for d in ds):
        raise ValueError(f"a chunk block is one of {GROUPS} lanes' 1,024 bytes")
    kw = _words(key)
    keyed = blake3.KEYED_HASH
    cvs = []
    for i, d in enumerate(ds):
        msg = (j.to_bytes(8, "big") if i == 0 else ds[i - 1][:8]) + d[8:]
        cv = kw
        for c in range(16):
            fl = keyed | (blake3.CHUNK_START if c == 0 else 0) | (blake3.CHUNK_END if c == 15 else 0)
            cv = tuple(blake3.compress(cv, msg[64 * c:64 * c + 64], i, 64, fl)[:8])
        cvs.append(cv)
    tail = blake3.compress(kw, ds[-1][:8] + bytes(56), len(ds), 8,
                           keyed | blake3.CHUNK_START | blake3.CHUNK_END)[:8]
    while len(cvs) > 1:
        cvs = [tuple(blake3.compress(kw, _bytes(cvs[i]) + _bytes(cvs[i + 1]), 0, 64, keyed | blake3.PARENT)[:8])
               for i in range(0, len(cvs), 2)]
    return _bytes(blake3.compress(kw, _bytes(cvs[0]) + _bytes(tail), 0, 64, keyed | blake3.PARENT | blake3.ROOT)[:8])


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha512(b"\x01" + left + right).digest()


def tree_root(leaves: Sequence[bytes]) -> bytes:
    """The RFC 6962 SHA-512 tree's root, bottom-up with an odd last node promoted (as `tf.cu` builds it)."""
    level = list(leaves)
    if not level:
        raise ValueError("a tree has at least one leaf")
    while len(level) > 1:
        nxt = [node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def store_blocks(span: int) -> int:
    """Blocks of every linear of Llama-3.1-8B's 32 layers at this span (the store one step's tags cover)."""
    return LAYERS * sum(block_count(n, k, span) for n, k in SHAPES.values())


def store_chunk_blocks(lanes: int) -> int:
    """`store_blocks` for chunk blocks of `lanes` lanes."""
    return LAYERS * sum(chunk_block_count(n, k, lanes) for n, k in SHAPES.values())
