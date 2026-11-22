"""The public size classes of a proof block's bits out, the public grid counts are rounded up to, and the name a Definition
takes in advice rows (`id_digest`), as `verity/partition/v2` fixed them.
"""
from __future__ import annotations

import hashlib
from typing import Any

#: an advice row's bytes
ROW_BYTES = 128
GRID_BITS = 4
#: the widest public size class of a proof block's bits out (the claims' `Claims.Sampling.sizeClassMax`), provisional: at 16,
#: `SIZE_CLASSES` is the literal 16-bit limit alone
SIZE_CLASS_MAX = 262_144
#: the public size classes, the powers of two from 16 to `SIZE_CLASS_MAX`, ascending (the claims' `sizeClasses`)
SIZE_CLASSES = tuple(16 << i for i in range((SIZE_CLASS_MAX // 16).bit_length()))


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def grid(x: int, bits: int = GRID_BITS) -> int:
    """The least y >= x with at most `bits` significant bits: x itself below 2^bits, else x rounded up to a multiple of
    2^(bit_length(x) - bits).  y < x * (1 + 2^(1 - bits)), so fewer than one in 2^(bits - 1) of y's units are fillers."""
    if not (_is_int(x) and x >= 0 and _is_int(bits) and bits >= 1):
        raise ValueError(f"grid of a count >= 0 at bits >= 1, not {x!r}, {bits!r}")
    e = max(0, x.bit_length() - bits)
    return -(-x >> e) << e


def id_digest(fn: Any) -> str:
    """The SHA-512 a Definition is named by in advice: of its specialization id, `fn.id`."""
    return hashlib.sha512(b"verity/partition/v2/definition\0" + fn.id.encode()).hexdigest()


def size_class(bits: int) -> int | None:
    """A block's size class: the least of `SIZE_CLASSES` that holds its ``bits`` out (`restrict.front`), None past the widest."""
    return next((w for w in SIZE_CLASSES if bits <= w), None)


__all__ = ["GRID_BITS", "ROW_BYTES", "SIZE_CLASSES", "SIZE_CLASS_MAX", "grid", "id_digest", "size_class"]
