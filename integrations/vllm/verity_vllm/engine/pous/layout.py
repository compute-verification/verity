"""`W` for POUS: the data sections of the checkpoint's `.safetensors` files, in file order, as the reference
`verity_pous.layout.Layout` of their tensors at their header ranges (`Layout.of_ranges`), so `W` is those sections back to back,
zero-padded to whole segments.

The verifier derives `W` from the checkpoint of record alone (its files' sha256 are pinned in `manifests/checkpoints.json`, which
is the commitment that precedes the salt), and so does the server; the headers give every tensor's dtype and shape, the reference
layout its byte range and segments.  Stdlib and `verity_pous` only.
"""
from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from verity_pous.layout import Layout
from verity_pous.params import OPERATING_POINT, Params


@dataclass(frozen=True)
class Section:
    path: str
    data_offset: int
    data_bytes: int


@dataclass(frozen=True)
class Checkpoint:
    sections: tuple[Section, ...]
    dtypes: dict[str, str]
    shapes: dict[str, tuple[int, ...]]
    layout: Layout

    @property
    def params(self) -> Params:
        return self.layout.params

    @property
    def data_bytes(self) -> int:
        return sum(s.data_bytes for s in self.sections)

    def segment_range(self, names: list[str]) -> tuple[int, int]:
        """The segments `[first, last + 1)` that hold every byte of `names`."""
        spans = [self.layout.segments(n) for n in names]
        return min(r.start for r in spans), max(r.stop for r in spans)

    def read_w(self) -> bytes:
        """`W`: the data sections in file order, then zeros to a whole number of segments."""
        parts = []
        for s in self.sections:
            with open(s.path, "rb") as f:
                f.seek(s.data_offset)
                data = f.read(s.data_bytes)
            if len(data) != s.data_bytes:
                raise ValueError(f"{s.path}: short data section")
            parts.append(data)
        return b"".join(parts) + bytes(self.params.w_bytes - self.data_bytes)


def _section(path: str) -> tuple[Section, list[tuple[int, int, str]], dict]:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        f.seek(0, 2)
        size = f.tell()
    header.pop("__metadata__", None)
    entries = sorted((int(t["data_offsets"][0]), int(t["data_offsets"][1]), name) for name, t in header.items())
    at = 0
    for a, b, name in entries:
        if a != at:
            raise ValueError(f"{path}: data section is not contiguous at {name}")
        at = b
    if 8 + hlen + at != size:
        raise ValueError(f"{path}: {size - 8 - hlen - at} trailing bytes after the last tensor")
    return Section(path, 8 + hlen, at), entries, header


def read(paths: str | Path | Sequence[str | Path], base: Params = OPERATING_POINT) -> Checkpoint:
    """The checkpoint of one or more `.safetensors` files, in the order given; refuses a data section with gaps, overlaps or
    trailing bytes, and a tensor name in two files."""
    files = [str(paths)] if isinstance(paths, (str, Path)) else [str(p) for p in paths]
    sections, ranges, dtypes, shapes = [], [], {}, {}
    start = 0
    for path in files:
        section, entries, header = _section(path)
        for a, b, name in entries:
            if name in dtypes:
                raise ValueError(f"tensor {name} is in two checkpoint files")
            ranges.append((name, start + a, b - a))
            dtypes[name], shapes[name] = header[name]["dtype"], tuple(header[name]["shape"])
        sections.append(section)
        start += section.data_bytes
    return Checkpoint(tuple(sections), dtypes, shapes, Layout.of_ranges(base, ranges))
