"""`ro-blake3/v1` over a wide store as `wt.cu`'s tag warps compute it: E's blocks of `TAG_BLOCK` = 4,088 bytes, so
`u64be j ‖ E_j` is four whole BLAKE3 chunks, the last block short; 16 tags a leaf; the RFC 6962 SHA-512 tree over the
leaves (`pous.verifier`'s shape: split at the largest power of two below the count).

The tag, key, prefixes and leaves are `verity/core/protocols/bandwidth/tags.py`'s, imported from the source tree (it
is stdlib only), so the pod's Python needs no `verity` install. `tags_fast` recomputes every tag with the PyPI
`blake3` when that is installed; `tags.py`'s own BLAKE3 (pure Python, ~3 ms a block) checks a sample.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TAG_BLOCK = 4088
PER_LEAF = 16
NAME = "ro-blake3/v1"
REPO = Path(__file__).resolve().parents[2]


def tags_module():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from verity.core.protocols.bandwidth import tags

    return tags


def block_count(nbytes: int) -> int:
    return -(-nbytes // TAG_BLOCK)


def block(store: bytes, jl: int) -> bytes:
    return store[TAG_BLOCK * jl:TAG_BLOCK * (jl + 1)]


def key(seed: bytes) -> bytes:
    """K, which as 8 little-endian words is the BLAKE3 key the device's constant memory holds."""
    return tags_module().Blake3Tag().key(seed)


def leaf_prefix(seed: bytes) -> bytes:
    T = tags_module()
    return T.prefix(T.LEAF_DOMAIN, NAME, seed)


def tag(seed: bytes, j: int, blk: bytes) -> bytes:
    """`tags.py`'s y_j."""
    return tags_module().Blake3Tag().keyed(seed)(j, blk)


def tags_fast(store: bytes, j0: int, seed: bytes) -> list[bytes] | None:
    """Every block's tag (block jl at index j0 + jl) with the PyPI `blake3`, or None without it."""
    try:
        import blake3 as fast
    except ImportError:
        return None
    k = key(seed)
    return [fast.blake3((j0 + jl).to_bytes(8, "big") + block(store, jl), key=k).digest()
            for jl in range(block_count(len(store)))]


def leaves(seed: bytes, ys: list[bytes]) -> list[bytes]:
    T = tags_module()
    return T.leaves(T.Blake3Tag(), seed, ys)


def node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha512(b"\x01" + left + right).digest()


def root(nodes: list[bytes]) -> bytes:
    """RFC 6962's tree hash over the leaves."""
    if len(nodes) == 1:
        return nodes[0]
    mid = 1 << ((len(nodes) - 1).bit_length() - 1)
    return node(root(nodes[:mid]), root(nodes[mid:]))
