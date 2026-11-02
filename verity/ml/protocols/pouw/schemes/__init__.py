"""The named PoUW schemes.  `get(name)` returns a configured `verity.core.protocols.pouw.protocol.PoUWScheme`; each name has a section
in ``PROTOCOL.md`` and pinned vectors under ``tests/vectors``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verity.core.protocols.pouw.protocol import PoUWScheme


def _ncp(perm: str) -> Callable[..., PoUWScheme]:
    def make(**config: Any) -> PoUWScheme:
        from .ncp import NCP
        return NCP(perm, **config)
    return make


def _pearl(**config: Any) -> PoUWScheme:
    from .pearl_kw import PearlKW
    return PearlKW(**config)


def _pearl_c(forming: str, device: str = "h100", hashing: str = "h0") -> Callable[..., PoUWScheme]:
    def make(**config: Any) -> PoUWScheme:
        from .pearl_c import PearlC
        from .pearl_c_device import DEVICES
        return PearlC(forming=forming, device=DEVICES[device], hashing=hashing, **config)
    return make


def _pearl_c4(fmt: str) -> Callable[..., PoUWScheme]:
    def make(**config: Any) -> PoUWScheme:
        from .pearl_c4 import PearlC4
        return PearlC4(fmt, **config)
    return make


#: name -> factory of the configured scheme (keyword arguments configure it, as a commitment scheme's ports do)
SCHEMES: dict[str, Callable[..., PoUWScheme]] = {
    "ncp-v1": _ncp("word"),
    "ncp-v1-shift24": _ncp("shift24"),
    "pearl-fp8-v4": _pearl,
    "pearl-c-h100-v0": _pearl_c("v0"),
    "pearl-c-h100-v1": _pearl_c("v1"),
    "pearl-c-h100-v1-h3": _pearl_c("v1", hashing="h3"),
    "pearl-c-sm120-v1": _pearl_c("v1", "sm120"),
    "pearl-c-sm120-v1-h3": _pearl_c("v1", "sm120", hashing="h3"),
    "pearl-c-nvfp4-v0": _pearl_c4("nvfp4"),
}


def get(name: str, **config: Any) -> PoUWScheme:
    """The scheme registered under `name`; `KeyError` names the registered ones."""
    try:
        make = SCHEMES[name]
    except KeyError:
        raise KeyError(f"no PoUW scheme {name!r}; registered: {', '.join(SCHEMES)}") from None
    return make(**config)


__all__ = ["SCHEMES", "get"]
