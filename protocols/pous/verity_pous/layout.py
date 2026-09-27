"""Tensors in `W`: a model's tensors laid out in order and zero-padded to whole segments, and per-tensor decode.

Tensors are packed back to back, or each from a segment boundary with `align=True`.  A tensor decodes from the
segments it overlaps and nothing else, since segments decode independently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .params import Params
from .primitives import Primitives
from .scheme import decode_segment


@dataclass(frozen=True, slots=True)
class Placement:
    name: str
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class Layout:
    """`params` has as many segments as `W` needs; `tensors` are their byte ranges in `W`, in order."""

    params: Params
    tensors: tuple[Placement, ...]

    @classmethod
    def of(cls, base: Params, sizes: Iterable[tuple[str, int]], align: bool = False) -> Layout:
        seg = base.segment_bytes
        placements, at, names = [], 0, set()
        for name, size in sizes:
            if type(name) is not str or name in names:
                raise ValueError(f"tensor names must be distinct strings: {name!r}")
            if type(size) is not int or size < 0:
                raise ValueError(f"tensor {name!r} needs a nonnegative size")
            if align and size and at % seg:
                at += seg - at % seg
            placements.append(Placement(name, at, size))
            names.add(name)
            at += size
        return cls(base.with_segments(max(1, -(-at // seg))), tuple(placements))

    def placement(self, name: str) -> Placement:
        for p in self.tensors:
            if p.name == name:
                return p
        raise KeyError(f"no tensor {name!r} in the layout")

    def segments(self, name: str) -> range:
        """The segments tensor `name` overlaps."""
        p, seg = self.placement(name), self.params.segment_bytes
        return range(p.offset // seg, -(-(p.offset + p.size) // seg)) if p.size else range(0)

    def blocks(self, name: str) -> range:
        """The blocks `decode_tensor` reads for tensor `name`."""
        segs = self.segments(name)
        return range(segs.start * self.params.n, segs.stop * self.params.n)

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


def decode_tensor(layout: Layout, prims: Primitives, salt: bytes, name: str, block: Callable[[int], bytes]) -> bytes:
    """Tensor `name` from the blocks of the segments it overlaps; `block(i)` returns block `i` of `C`."""
    params, p = layout.params, layout.placement(name)
    segs = layout.segments(name)
    data = b"".join(decode_segment(params, prims, salt, s, [block(s * params.n + v) for v in range(params.n)])
                    for s in segs)
    start = p.offset - segs.start * params.segment_bytes if p.size else 0
    return data[start:start + p.size]
