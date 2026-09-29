"""The named PoUW schemes.  `get(name)` returns a configured `verity_pouw.protocol.PoUWScheme`; each name has a section
in ``PROTOCOL.md`` and pinned vectors under ``tests/vectors``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..protocol import PoUWScheme


def _ncp(perm: str) -> Callable[..., PoUWScheme]:
    def make(**config: Any) -> PoUWScheme:
        from .ncp import NCP
        return NCP(perm, **config)
    return make


def _pearl(**config: Any) -> PoUWScheme:
    from .pearl_kw import PearlKW
    return PearlKW(**config)


#: name -> factory of the configured scheme (keyword arguments configure it, as a commitment scheme's ports do)
SCHEMES: dict[str, Callable[..., PoUWScheme]] = {
    "ncp-v1": _ncp("word"),
    "ncp-v1-shift24": _ncp("shift24"),
    "pearl-fp8-v4": _pearl,
}


def get(name: str, **config: Any) -> PoUWScheme:
    """The scheme registered under `name`; `KeyError` names the registered ones."""
    try:
        make = SCHEMES[name]
    except KeyError:
        raise KeyError(f"no PoUW scheme {name!r}; registered: {', '.join(SCHEMES)}") from None
    return make(**config)


__all__ = ["SCHEMES", "get"]
