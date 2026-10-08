"""The registered schemes behind `protocol.Scheme`, looked up by name.

Each scheme is one module that implements only what differs between schemes (encoding and decoding one segment) and
states its certificate at its parameters:

- `band` (`band-chain/d12/v1`, the default): the secure-first scheme.
- `dense` (`dense-chain/v1`): selectable.
- `p3` (`p3/v1`): research.
- `p2` (`p2-16448/v4`, and `p2-16448/v2` with the ChaCha8 key expander the spec does not allow): **experimental**,
  never the default (an abstract-M1 result under the pinned `P2SlackFamilyUncond`, whose instantiation assumption is not
  accepted).

Their conformance vectors are pinned by name under `tests/fixtures/schemes/` (`tests/pous_scheme_vectors.py` writes
them).
"""

from __future__ import annotations

from .. import band as band_codec
from .. import dense as dense_codec
from .. import scheme as p3_codec
from ..codec import Geometry
from ..params import OPERATING_POINT
from ..protocol import Scheme
from .band import Band
from .dense import Dense
from .p2 import CHACHA8, P2, P2ChaCha, P2Params
from .p2 import POINT as P2_POINT
from .p3 import P3

REGISTRY: dict[str, tuple[type[Scheme], Geometry]] = {
    band_codec.BAND12.name: (Band, band_codec.SECURE_FIRST),
    dense_codec.DENSE.name: (Dense, dense_codec.POINT_64K),
    p3_codec.P3.name: (P3, OPERATING_POINT),
    P2Params(*P2_POINT).codec.name: (P2, P2Params(*P2_POINT)),
    P2Params(*P2_POINT, expander=CHACHA8).codec.name: (P2ChaCha, P2Params(*P2_POINT, expander=CHACHA8)),
}
"""Each registered name, with its class and its registered point (one segment)."""

DEFAULT = band_codec.BAND12.name
"""The secure-first scheme."""

EXPERIMENTAL = frozenset(name for name, (cls, _) in REGISTRY.items() if cls.experimental)
"""The registered names that are experimental: never the default."""


def get(name: str = DEFAULT, segments: int = 1, params: Geometry | None = None) -> Scheme:
    """The registered scheme `name`, at its registered point for `segments` segments of `W`, or at `params` (which
    must give the same name: a band of another in-degree is another scheme)."""
    try:
        cls, point = REGISTRY[name]
    except KeyError:
        raise KeyError(f"no POUS scheme {name!r}; registered: {', '.join(sorted(REGISTRY))}") from None
    scheme = cls(point.with_segments(segments) if params is None else params)
    if scheme.name != name:
        raise ValueError(f"these parameters make {scheme.name}, not {name}")
    return scheme


__all__ = ["DEFAULT", "EXPERIMENTAL", "REGISTRY", "Band", "Dense", "P2", "P2ChaCha", "P2Params", "P3", "get"]
