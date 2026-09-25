"""verity.randomness: the one challenge derivation every protocol draws through.

    derive(source, domain, context) -> Key

`source` is a secret or public value (a beacon round, a device's challenger key, a seed derived earlier), `domain` is
`verity/{protocol}/{purpose}/v{n}`, and `context` is a mapping of the values the challenge must follow (registered
roots, a request header, the device, the scope).  A challenge cannot be computed before the roots in its context
exist, and a beacon source is the first round published after they were registered, so ordering is a property of
the derivation rather than of a separate check.  Fiat-Shamir over a prover's own root is deliberately not a source:
a prover can re-commit until such a sample misses its incorrect units.

`Key` drives exact samplers through counter-mode streams, so every draw is reproducible by anyone holding the source.
Stdlib only.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

_DOMAIN = re.compile(r"verity/[a-z0-9-]+/[a-z0-9-]+/v[0-9]+")
_TAG = b"verity/randomness/derive/v1\0"

type Part = bytes | str | int


def _encode(part: Part) -> bytes:
    if isinstance(part, bytes):
        tag, body = b"b", part
    elif isinstance(part, str):
        tag, body = b"s", part.encode("utf-8")
    elif isinstance(part, int) and not isinstance(part, bool) and part >= 0:
        tag, body = b"i", part.to_bytes(max(1, (part.bit_length() + 7) // 8), "big")
    else:
        raise TypeError(f"context parts are bytes, str or nonnegative int, got {type(part).__name__}")
    return tag + len(body).to_bytes(8, "big") + body


def _frame(*parts: Part) -> bytes:
    return b"".join(_encode(p) for p in parts)


@dataclass(frozen=True, slots=True)
class Key:
    """A derived 32-byte key.  Every sampler takes an `index` so draws for different items are independent streams."""

    value: bytes

    def stream(self, index: Part | tuple[Part, ...], n: int) -> bytes:
        parts = index if isinstance(index, tuple) else (index,)
        return hashlib.shake_256(_frame(self.value, *parts)).digest(n)

    def below(self, n: int, index: Part | tuple[Part, ...] = 0) -> int:
        """Uniform integer in [0, n), by rejection from 128-bit words (exact, no modulo bias)."""
        if n <= 0:
            raise ValueError("below(n) needs n > 0")
        parts = index if isinstance(index, tuple) else (index,)
        limit = (1 << 128) - (1 << 128) % n
        attempt = 0
        while True:
            x = int.from_bytes(self.stream((*parts, attempt), 16), "big")
            if x < limit:
                return x % n
            attempt += 1

    def bernoulli(self, p: Fraction, index: Part | tuple[Part, ...] = 0) -> bool:
        return self.below(p.denominator, index) < p.numerator

    def subset(self, k: int, n: int, index: Part | tuple[Part, ...] = 0) -> tuple[int, ...]:
        """A uniformly random k-subset of range(n), sorted (partial Fisher-Yates)."""
        if not 0 <= k <= n:
            raise ValueError("subset needs 0 <= k <= n")
        parts = index if isinstance(index, tuple) else (index,)
        pool = list(range(n))
        for i in range(k):
            j = i + self.below(n - i, (*parts, i))
            pool[i], pool[j] = pool[j], pool[i]
        return tuple(sorted(pool[:k]))


def derive(source: bytes, domain: str, context: Mapping[str, Part]) -> Key:
    if not isinstance(source, bytes) or not source:
        raise TypeError("source must be nonempty bytes")
    if not _DOMAIN.fullmatch(domain):
        raise ValueError(f"domain {domain!r} is not verity/{{protocol}}/{{purpose}}/v{{n}}")
    ctx = _frame(*(x for key in sorted(context) for x in (key, context[key])))
    return Key(hashlib.sha256(_TAG + _frame(source, domain, ctx)).digest())


__all__ = ["Key", "derive"]
