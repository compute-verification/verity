"""Bit layout: Lean's `Bits L = Fin L → Bool` as a Python int whose bit `i` is Lean's bit `i`.

Every string of blocks is laid out by `codeEquiv B ℓ`, whose `finProdFinEquiv` puts bit `j` of block `i` at
`i * ℓ + j`.  So component `v` of a label is bits `v * wc .. v * wc + wc - 1`, label `u` of `W` is bits
`u * m .. u * m + m - 1`, and bytes are little-endian with bit `i` at bit `i % 8` of byte `i // 8`.
"""

from __future__ import annotations

from collections.abc import Sequence

from .params import SALT_BITS, Params


def mask(bits: int) -> int:
    return (1 << bits) - 1


def split(x: int, count: int, width: int) -> list[int]:
    """`(codeEquiv count width).symm`: `count` blocks of `width` bits."""
    m = mask(width)
    return [(x >> (i * width)) & m for i in range(count)]


def join(parts: Sequence[int], width: int) -> int:
    """`codeEquiv count width`: the concatenation of the blocks, block 0 lowest."""
    x = 0
    for i, part in enumerate(parts):
        if not 0 <= part < 1 << width:
            raise ValueError(f"block {i} does not fit in {width} bits")
        x |= part << (i * width)
    return x


def component(label: int, v: int, wc: int) -> int:
    return (label >> (v * wc)) & mask(wc)


def transpose(labels: Sequence[int], n: int, wc: int) -> list[int]:
    """Lean `transpose C v u = C u v`: label `v` of the result collects component `v` of every input label."""
    if len(labels) != n:
        raise ValueError(f"transpose needs {n} labels, got {len(labels)}")
    columns = [split(label, n, wc) for label in labels]
    return [join([columns[u][v] for u in range(n)], wc) for v in range(n)]


def from_bytes(data: bytes, bits: int) -> int:
    """The `bits`-bit string of exactly `ceil(bits / 8)` bytes; the padding bits must be zero."""
    if type(data) is not bytes or len(data) != (bits + 7) // 8:
        raise ValueError(f"expected {(bits + 7) // 8} bytes for {bits} bits")
    x = int.from_bytes(data, "little")
    if x >> bits:
        raise ValueError("nonzero padding bits")
    return x


def to_bytes(x: int, bits: int) -> bytes:
    if not 0 <= x < 1 << bits:
        raise ValueError(f"value does not fit in {bits} bits")
    return x.to_bytes((bits + 7) // 8, "little")


def salt_value(salt: bytes) -> int:
    return from_bytes(salt, SALT_BITS)


def tag(salt: int, segment: int) -> int:
    """`tag_s = salt || u64(s)`: Lean `tagOf salt` for segment 0 (the salt, then zeros)."""
    if not 0 <= salt < 1 << SALT_BITS:
        raise ValueError(f"the salt is {SALT_BITS} bits")
    if not 0 <= segment < 1 << 64:
        raise ValueError("the segment index is 64 bits")
    return salt | (segment << SALT_BITS)


def w_labels(params: Params, w: int) -> list[list[int]]:
    """`W` as each segment's `n` lower inputs (Lean `Wlab`, segment-major)."""
    labels = split(w, params.blocks, params.m)
    return [labels[s * params.n:(s + 1) * params.n] for s in range(params.segments)]


def w_value(params: Params, segments: Sequence[Sequence[int]]) -> int:
    """Lean `Wbits`: the inverse of `w_labels`."""
    return join([label for segment in segments for label in segment], params.m)
