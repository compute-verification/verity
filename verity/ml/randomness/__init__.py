"""verity.ml.randomness: the one challenge derivation every protocol draws through (decision 42).

    derive(source, domain, context) -> Key

`source` is a key holder's secret (an auditor key, a device's challenger key, a seed derived earlier).  It is never a prover's own commitment: Fiat-Shamir over a prover's root lets the prover re-commit until the
sample misses its incorrect units, so outer challenges must not use it (Fiat-Shamir stays inside proofs).  `domain`
is ``verity/{protocol}/{purpose}/v{n}``.  `context` names the bindings the challenge must follow (registered roots, a
Program digest, a request header, a device), so a challenge cannot be computed before they exist.

A `Key` drives exact integer samplers over counter-mode streams.  Every draw is addressed by an `index`, so the draws
for different items are independent streams, and any shard of a population computes its own draws without the rest
(`Key.shard` hands a worker a sub-key).  Stdlib only.

Byte formats (pinned by ``verity/ml/randomness/tests/vectors.json``):

* ``frame(x)`` = tag || len(body) as 8 bytes big-endian || body, with tag ``b`` for bytes, ``s`` for UTF-8 str, ``i``
  for a nonnegative int as minimal big-endian bytes (0 is one zero byte).  An index or a tuple of parts frames each
  part and concatenates.
* ``derive`` = the first 32 bytes of SHA-512(``verity/randomness/derive/v2\\0`` || frame(source) || frame(domain) ||
  frame(ctx)), where ctx concatenates frame(key) || frame(value) over the context sorted by key.
* ``stream(index)`` block c = SHA-512(``verity/randomness/stream/v2\\0`` || key || frame(frames(index)) || c as 8 bytes
  big-endian), all 64 bytes, for c = 0, 1, ...; the stream is the blocks' concatenation.
* ``shard(index)`` = the first 32 bytes of SHA-512(``verity/randomness/shard/v2\\0`` || key || frame(frames(index))).
* A `Key` is 32 bytes: SHA-512 truncated to 256 bits is SHA-512 (one hash, SP 800-107 §5.1), and a 256-bit key keeps
  128-bit security against a quantum search.
* ``uniform(n, index)`` reads w-byte big-endian words from ``stream(("u", *index))``, w = ceil((bitlen(n) + 64) / 8),
  and returns the first word below the largest multiple of n that fits, reduced mod n (exact, no modulo bias).
* ``bernoulli(p, index)`` = uniform(denominator) on ``("b", *index)`` < numerator; p = 0 and p = 1 draw nothing.
* ``subset(k, n, index)``: a sparse partial Fisher-Yates whose i-th swap draws uniform(n - i) on ``("s", *index, i)``,
  returned sorted.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

DOMAIN = re.compile(r"verity/[a-z0-9-]+/[a-z0-9-]+/v[0-9]+")
_DERIVE = b"verity/randomness/derive/v2\0"
_STREAM = b"verity/randomness/stream/v2\0"
_SHARD = b"verity/randomness/shard/v2\0"

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


def uniform_from_words(n: int, words: Iterator[int], width_bits: int) -> int:
    """The exact reduction: the first word below the largest multiple of `n` under 2^width_bits, mod `n`."""
    if n <= 0:
        raise ValueError("uniform(n) needs n > 0")
    space = 1 << width_bits
    if n > space:
        raise ValueError("uniform(n) needs n <= 2^width_bits")
    limit = space - space % n
    for x in words:
        if x < limit:
            return x % n
    raise AssertionError("word source exhausted")


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
            yield hashlib.sha512(head + c.to_bytes(8, "big")).digest()
            c += 1

    def stream(self, index: Index, n: int) -> bytes:
        """The first `n` bytes of the counter-mode stream at `index` (a prefix of every longer read)."""
        out = bytearray()
        for block in self.blocks(index):
            if len(out) >= n:
                break
            out += block
        return bytes(out[:n])

    def shard(self, index: Index) -> Key:
        """An independent sub-key for one shard of a population."""
        return Key(hashlib.sha512(_SHARD + self.value + _frames(index)).digest()[:32])

    def _uniform(self, n: int, index: tuple[Part, ...]) -> int:
        if n <= 0:
            raise ValueError("uniform(n) needs n > 0")
        width = (n.bit_length() + 64 + 7) // 8

        def words() -> Iterator[int]:
            buf = b""
            for block in self.blocks(index):
                buf += block
                while len(buf) >= width:
                    yield int.from_bytes(buf[:width], "big")
                    buf = buf[width:]

        return uniform_from_words(n, words(), 8 * width)

    def uniform(self, n: int, index: Index = 0) -> int:
        """An exactly uniform integer in [0, n)."""
        return self._uniform(n, ("u", *_parts(index)))

    def bernoulli(self, p: Fraction, index: Index = 0) -> bool:
        """True with probability exactly `p`."""
        p = Fraction(p)
        if not 0 <= p <= 1:
            raise ValueError("bernoulli(p) needs 0 <= p <= 1")
        if p in (0, 1):
            return p == 1
        return self._uniform(p.denominator, ("b", *_parts(index))) < p.numerator

    def subset(self, k: int, n: int, index: Index = 0) -> tuple[int, ...]:
        """A uniformly random k-subset of range(n), sorted.  O(k) time and memory, whatever `n` is."""
        if not 0 <= k <= n:
            raise ValueError("subset needs 0 <= k <= n")
        parts = _parts(index)
        swapped: dict[int, int] = {}
        for i in range(k):
            j = i + self._uniform(n - i, ("s", *parts, i))
            swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
        return tuple(sorted(swapped.get(i, i) for i in range(k)))


def derive(source: bytes, domain: str, context: Mapping[str, Part]) -> Key:
    if not isinstance(source, bytes) or not source:
        raise TypeError("source must be nonempty bytes: a key holder's secret")
    if not DOMAIN.fullmatch(domain):
        raise ValueError(f"domain {domain!r} is not verity/{{protocol}}/{{purpose}}/v{{n}}")
    if not all(isinstance(key, str) for key in context):
        raise TypeError("context keys are str")
    ctx = b"".join(frame(key) + frame(context[key]) for key in sorted(context))
    return Key(hashlib.sha512(_DERIVE + frame(source) + frame(domain) + frame(ctx)).digest()[:32])


__all__ = ["DOMAIN", "Index", "Key", "Part", "derive", "frame", "uniform_from_words"]
