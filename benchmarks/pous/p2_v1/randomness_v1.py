"""P2 v1.1's frozen derivation: `verity.ml.randomness` v1 (SHA-256, as of verity `f5df3bbc`), kept here so the kit's
§7 vectors and every pin of them hold after core moved to SHA-512; memory-accounting's P2 v4 retires it.

The parts the kit reads, with v1's bytes and behaviour (stdlib only):

* ``frame(x)`` = tag || len(body) as 8 bytes big-endian || body, with tag ``b`` for bytes, ``s`` for UTF-8 str, ``i``
  for a nonnegative int as minimal big-endian bytes (0 is one zero byte).  An index or a tuple of parts frames each
  part and concatenates.
* ``derive`` = SHA-256(``verity/randomness/derive/v1\\0`` || frame(source) || frame(domain) || frame(ctx)), where ctx
  concatenates frame(key) || frame(value) over the context sorted by key.
* ``stream(index)`` block c = SHA-256(``verity/randomness/stream/v1\\0`` || key || frame(frames(index)) || c as 8 bytes
  big-endian), for c = 0, 1, ...; the stream is the blocks' concatenation.
* ``uniform(n, index)`` reads w-byte big-endian words from ``stream(("u", *index))``, w = ceil((bitlen(n) + 64) / 8),
  and returns the first word below the largest multiple of n that fits, reduced mod n.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

DOMAIN = re.compile(r"verity/[a-z0-9-]+/[a-z0-9-]+/v[0-9]+")
_DERIVE = b"verity/randomness/derive/v1\0"
_STREAM = b"verity/randomness/stream/v1\0"

type Part = bytes | str | int
type Index = Part | tuple[Part, ...]


def frame(part: Part) -> bytes:
    if isinstance(part, bytes):
        tag, body = b"b", part
    elif isinstance(part, str):
        tag, body = b"s", part.encode("utf-8")
    elif isinstance(part, int) and not isinstance(part, bool) and part >= 0:
        tag, body = b"i", part.to_bytes(max(1, (part.bit_length() + 7) // 8), "big")
    else:
        raise TypeError(f"parts are bytes, str or nonnegative int, got {type(part).__name__}")
    return tag + len(body).to_bytes(8, "big") + body


def _parts(index: Index) -> tuple[Part, ...]:
    return index if isinstance(index, tuple) else (index,)


def _frames(index: Index) -> bytes:
    return frame(b"".join(frame(p) for p in _parts(index)))


@dataclass(frozen=True, slots=True)
class Key:
    """A derived 32-byte key.  Its value is secret when its source is, so it is kept out of `repr`."""

    value: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, bytes) or len(self.value) != 32:
            raise ValueError("a Key is 32 bytes")

    def blocks(self, index: Index) -> Iterator[bytes]:
        head = _STREAM + self.value + _frames(index)
        c = 0
        while True:
            yield hashlib.sha256(head + c.to_bytes(8, "big")).digest()
            c += 1

    def stream(self, index: Index, n: int) -> bytes:
        """The first `n` bytes of the counter-mode stream at `index` (a prefix of every longer read)."""
        out = bytearray()
        for block in self.blocks(index):
            if len(out) >= n:
                break
            out += block
        return bytes(out[:n])

    def uniform(self, n: int, index: Index = 0) -> int:
        """An exactly uniform integer in [0, n)."""
        if n <= 0:
            raise ValueError("uniform(n) needs n > 0")
        width = (n.bit_length() + 64 + 7) // 8
        limit = (1 << 8 * width) - (1 << 8 * width) % n
        buf = b""
        for block in self.blocks(("u", *_parts(index))):
            buf += block
            while len(buf) >= width:
                x, buf = int.from_bytes(buf[:width], "big"), buf[width:]
                if x < limit:
                    return x % n
        raise AssertionError("unreachable: the stream is unbounded")


def derive(source: bytes, domain: str, context: Mapping[str, Part]) -> Key:
    if not isinstance(source, bytes) or not source:
        raise TypeError("source must be nonempty bytes: a beacon round or a key holder's secret")
    if not DOMAIN.fullmatch(domain):
        raise ValueError(f"domain {domain!r} is not verity/{{protocol}}/{{purpose}}/v{{n}}")
    if not all(isinstance(key, str) for key in context):
        raise TypeError("context keys are str")
    ctx = b"".join(frame(key) + frame(context[key]) for key in sorted(context))
    return Key(hashlib.sha256(_DERIVE + frame(source) + frame(domain) + frame(ctx)).digest())
