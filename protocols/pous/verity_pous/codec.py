"""The codec seam: what setup, layout, the verifier and the audit need from a scheme.

A codec encodes and decodes one segment of `W` at a time, on bytes, so the whole of `W` never becomes one integer.
`scheme.P3` is the two-layer scheme; a dense single-layer scheme (POUS `docs/scheme-choice-e2e.md`) is a second codec
with the same two functions.  Everything else reads the geometry (`Geometry`) and the codec, never P3's structure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class Geometry(Protocol):
    """What a codec's parameters say about sizes: `segments` segments of `blocks_per_segment` blocks of `m` bits, and
    `segment_bytes` bytes of `W` per segment (rate 1).  `params.Params` is P3's."""

    segments: int

    @property
    def blocks_per_segment(self) -> int: ...

    @property
    def m(self) -> int: ...

    @property
    def block_bytes(self) -> int: ...

    @property
    def blocks(self) -> int: ...

    @property
    def segment_bytes(self) -> int: ...

    @property
    def w_bytes(self) -> int: ...

    def with_segments(self, segments: int) -> Any: ...

    def manifest(self) -> dict: ...


type EncodeSegment = Callable[[Any, Any, bytes, int, bytes], list[bytes]]
type DecodeSegment = Callable[[Any, Any, bytes, int, Sequence[bytes]], bytes]


@dataclass(frozen=True, slots=True)
class Codec:
    """`encode_segment(params, prims, salt, s, w_seg) -> blocks` and `decode_segment(params, prims, salt, s, blocks)
    -> w_seg`, for segment `s`."""

    name: str
    encode_segment: EncodeSegment
    decode_segment: DecodeSegment


def split(params: Geometry, w: bytes | Iterable[bytes]) -> Iterator[bytes]:
    """`W` as its segments: slices of `w` when it is bytes, else the given segments, each checked for size."""
    size = params.segment_bytes
    if isinstance(w, (bytes, bytearray, memoryview)):
        if len(w) != params.w_bytes:
            raise ValueError(f"W has {len(w)} bytes, the parameters say {params.w_bytes}")
        view = memoryview(w)
        for s in range(params.segments):
            yield bytes(view[s * size:(s + 1) * size])
        return
    count = 0
    for seg in w:
        if len(seg) != size:
            raise ValueError(f"segment {count} has {len(seg)} bytes, not {size}")
        count += 1
        yield bytes(seg)
    if count != params.segments:
        raise ValueError(f"W has {count} segments, the parameters say {params.segments}")


def encode_segments(codec: Codec, params: Geometry, prims: Any, w: bytes | Iterable[bytes],
                    salt: bytes) -> Iterator[list[bytes]]:
    """Each segment's blocks, in order."""
    for s, seg in enumerate(split(params, w)):
        yield codec.encode_segment(params, prims, salt, s, seg)
