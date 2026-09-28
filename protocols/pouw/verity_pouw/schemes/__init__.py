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


def _fp8_is(forming: str) -> Callable[..., PoUWScheme]:
    def make(**config: Any) -> PoUWScheme:
        from .fp8_is import DEFAULT_PRICES, FORMINGS, FP8IS
        return FP8IS(FORMINGS[forming](config.get("prices") or DEFAULT_PRICES), **config)
    return make


#: name -> factory of the configured scheme (keyword arguments configure it, as a commitment scheme's ports do)
SCHEMES: dict[str, Callable[..., PoUWScheme]] = {
    "ncp-v1": _ncp("word"),
    "ncp-v1-shift24": _ncp("shift24"),
    "pearl-fp8-v4": _pearl,
    "fp8-is-h100-d3s-v0": _fp8_is("d3s-v0"),
    "fp8-is-h100-d3s-v0-f16": _fp8_is("d3s-v0-f16"),
    "fp8-is-h100-d3s-v0-y16": _fp8_is("d3s-v0-y16"),
    "fp8-is-h100-d3s-v0-y16-f16": _fp8_is("d3s-v0-y16-f16"),
    "fp8-is-h100-d3m-v0": _fp8_is("d3m-v0"),
    "fp8-is-h100-d3m-v0-f16": _fp8_is("d3m-v0-f16"),
    "fp8-is-h100-d3m-v0-y16": _fp8_is("d3m-v0-y16"),
    "fp8-is-h100-d3m-v0-y16-f16": _fp8_is("d3m-v0-y16-f16"),
    "fp8-is-h100-d3-v0": _fp8_is("d3-v0"),
    "fp8-is-h100-d2-draft": _fp8_is("d2-draft"),
    "fp8-is-h100-fpncp-v0": _fp8_is("fpncp-v0"),
}


def get(name: str, **config: Any) -> PoUWScheme:
    """The scheme registered under `name`; `KeyError` names the registered ones."""
    try:
        make = SCHEMES[name]
    except KeyError:
        raise KeyError(f"no PoUW scheme {name!r}; registered: {', '.join(SCHEMES)}") from None
    return make(**config)


__all__ = ["SCHEMES", "get"]
