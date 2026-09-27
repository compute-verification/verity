"""`W` for POUS: the data section of the checkpoint's one `.safetensors` file, as the reference `verity_pous.layout.Layout`
of its tensors back to back (so `W` is the data section, zero-padded to whole segments).

The verifier derives `W` from the checkpoint of record alone, and so does the server; the header gives every tensor's dtype and
shape, the reference layout its byte range and segments.  Stdlib and `verity_pous` only.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from verity_pous.layout import Layout
from verity_pous.params import OPERATING_POINT, Params


@dataclass(frozen=True)
class Checkpoint:
    path: str
    data_offset: int
    data_bytes: int
    dtypes: dict[str, str]
    shapes: dict[str, tuple[int, ...]]
    layout: Layout

    @property
    def params(self) -> Params:
        return self.layout.params

    def segment_range(self, names: list[str]) -> tuple[int, int]:
        """The segments `[first, last + 1)` that hold every byte of `names`."""
        spans = [self.layout.segments(n) for n in names]
        return min(r.start for r in spans), max(r.stop for r in spans)

    def read_w(self) -> bytes:
        """`W`: the data section, then zeros to a whole number of segments (`Layout.pack` of the tensors)."""
        with open(self.path, "rb") as f:
            f.seek(self.data_offset)
            data = f.read(self.data_bytes)
        if len(data) != self.data_bytes:
            raise ValueError(f"{self.path}: short data section")
        return data + bytes(self.params.w_bytes - self.data_bytes)


def read(path: str | Path, base: Params = OPERATING_POINT) -> Checkpoint:
    """The checkpoint of one `.safetensors` file; refuses a data section with gaps, overlaps or trailing bytes."""
    path = str(path)
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        f.seek(0, 2)
        size = f.tell()
    header.pop("__metadata__", None)
    entries = sorted((int(t["data_offsets"][0]), int(t["data_offsets"][1]), name) for name, t in header.items())
    layout = Layout.of(base, [(name, b - a) for a, b, name in entries])
    for (a, b, name), p in zip(entries, layout.tensors):
        if (p.offset, p.offset + p.size) != (a, b):
            raise ValueError(f"{path}: data section is not contiguous at {name}")
    data_bytes = entries[-1][1] if entries else 0
    if 8 + hlen + data_bytes != size:
        raise ValueError(f"{path}: {size - 8 - hlen - data_bytes} trailing bytes after the last tensor")
    return Checkpoint(path, 8 + hlen, data_bytes, {n: t["dtype"] for n, t in header.items()},
                      {n: tuple(t["shape"]) for n, t in header.items()}, layout)
