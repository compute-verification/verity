"""Tensors in `W`: a model's tensors laid out and zero-padded to whole segments, and per-tensor decode.

`TensorLayout.of` packs tensors back to back in the order given, or each from a segment boundary with `align=True`;
`TensorLayout.of_ranges` takes byte ranges already fixed, such as a safetensors data section.  A tensor decodes from the
segments it overlaps and nothing else, since segments decode independently, with any codec.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .codec import Codec, Geometry
from .primitives import Primitives


@dataclass(frozen=True, slots=True)
class TensorRange:
    """Tensor `name`'s bytes in `W`: `size` of them from byte `offset`."""

    name: str
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class TensorLayout:
    """`params` has as many segments as `W` needs; `tensors` are their byte ranges in `W`, in order."""

    params: Geometry
    tensors: tuple[TensorRange, ...]

    @classmethod
    def of(cls, base: Geometry, sizes: Iterable[tuple[str, int]], align: bool = False) -> TensorLayout:
        seg = base.segment_bytes
        placements, at, names = [], 0, set()
        for name, size in sizes:
            if type(name) is not str or name in names:
                raise ValueError(f"tensor names must be distinct strings: {name!r}")
            if type(size) is not int or size < 0:
                raise ValueError(f"tensor {name!r} needs a nonnegative size")
            if align and size and at % seg:
                at += seg - at % seg
            placements.append(TensorRange(name, at, size))
            names.add(name)
            at += size
        return cls(base.with_segments(max(1, -(-at // seg))), tuple(placements))

    @classmethod
    def of_ranges(cls, base: Geometry, ranges: Iterable[tuple[str, int, int]]) -> TensorLayout:
        """The layout with each tensor at `(name, offset, size)` as given: distinct names, disjoint ranges.  `W` ends at
        the last range, padded to whole segments; bytes between ranges are zero in `pack`."""
        placements = sorted((TensorRange(name, offset, size) for name, offset, size in ranges), key=lambda p: p.offset)
        names, end = set(), 0
        for p in placements:
            if type(p.name) is not str or p.name in names:
                raise ValueError(f"tensor names must be distinct strings: {p.name!r}")
            if type(p.offset) is not int or type(p.size) is not int or p.offset < 0 or p.size < 0:
                raise ValueError(f"tensor {p.name!r} needs a nonnegative offset and size")
            if p.size and p.offset < end:
                raise ValueError(f"tensor {p.name!r} overlaps the one before it")
            names.add(p.name)
            end = max(end, p.offset + p.size)
        return cls(base.with_segments(max(1, -(-end // base.segment_bytes))), tuple(placements))

    def placement(self, name: str) -> TensorRange:
        for p in self.tensors:
            if p.name == name:
                return p
        raise KeyError(f"no tensor {name!r} in the layout")

    def segments(self, name: str) -> range:
        """The segments tensor `name` overlaps."""
        p, seg = self.placement(name), self.params.segment_bytes
        return range(p.offset // seg, -(-(p.offset + p.size) // seg)) if p.size else range(0)

    def blocks(self, name: str) -> range:
        """The labels `decode_tensor` reads for tensor `name`."""
        segs, n = self.segments(name), self.params.blocks_per_segment
        return range(segs.start * n, segs.stop * n)

    def pack(self, data: Mapping[str, bytes]) -> bytes:
        """`W`: every tensor's bytes at its place, zeros elsewhere."""
        if set(data) != {p.name for p in self.tensors}:
            raise ValueError("pack needs exactly the layout's tensors")
        w = bytearray(self.params.w_bytes)
        for p in self.tensors:
            if len(data[p.name]) != p.size:
                raise ValueError(f"tensor {p.name!r} has {len(data[p.name])} bytes, the layout says {p.size}")
            w[p.offset:p.offset + p.size] = data[p.name]
        return bytes(w)

    def unpack(self, w: bytes, name: str) -> bytes:
        p = self.placement(name)
        return w[p.offset:p.offset + p.size]

    def manifest(self) -> dict:
        return {"params": self.params.manifest(), "tensors": [[p.name, p.offset, p.size] for p in self.tensors]}


def decode_tensor(layout: TensorLayout, prims: Primitives, salt: bytes, name: str, block: Callable[[int], bytes],
                  codec: Codec | None = None) -> bytes:
    """Tensor `name` from the labels of the segments it overlaps; `block(i)` returns label `i` of `C`.  The codec
    defaults to the layout's parameters' own."""
    params, p = layout.params, layout.placement(name)
    codec = codec or params.codec
    segs, n = layout.segments(name), params.blocks_per_segment
    data = b"".join(codec.decode_segment(params, prims, salt, s, [block(s * n + v) for v in range(n)]) for s in segs)
    start = p.offset - segs.start * params.segment_bytes if p.size else 0
    return data[start:start + p.size]
