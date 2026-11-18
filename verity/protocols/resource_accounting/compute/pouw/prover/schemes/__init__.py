"""The named PoUW schemes.  `get(name)` returns a configured `verity.protocols.resource_accounting.compute.pouw.verifier.protocol.PoUWScheme`; each name's module
docstring states its steps, and its pinned vectors are under ``verifier/tests/vectors``.  A name in `EXPERIMENTAL` is
named here and brought by an experimental package, which `register`s its factory when imported: PoUW imports nothing of
``verity/protocols/experimental/``, so until then `get` raises `NotImplementedError` naming the seam."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import PoUWScheme


def _ncp(perm: str) -> Callable[..., PoUWScheme]:
    def make(**config: Any) -> PoUWScheme:
        from .ncp import NCP
        return NCP(perm, **config)
    return make


def _pearl(**config: Any) -> PoUWScheme:
    from .pearl_kw import PearlKW
    return PearlKW(**config)


def _pearl_c(device: str) -> Callable[..., PoUWScheme]:
    def make(**config: Any) -> PoUWScheme:
        from .pearl_c import PearlC
        from .pearl_c_device import DEVICES
        return PearlC(device=DEVICES[device], hashing="h3", **config)
    return make


#: name -> the seam of the experimental package that registers it (Pearl-C4: ``verity/protocols/experimental/pearl_c4/``)
EXPERIMENTAL: dict[str, str] = {"pearl-c-nvfp4-v0": "compute-accounting/pearl-c4-experimental"}


def _experimental(name: str) -> Callable[..., PoUWScheme]:
    def make(**config: Any) -> PoUWScheme:
        raise NotImplementedError(EXPERIMENTAL[name])
    return make


#: name -> factory of the configured scheme (keyword arguments configure it, as a commitment scheme's ports do)
SCHEMES: dict[str, Callable[..., PoUWScheme]] = {
    "ncp-v1": _ncp("word"),
    "ncp-v1-shift24": _ncp("shift24"),
    "pearl-fp8-v4": _pearl,
    "pearl-c-h100-v1-h3": _pearl_c("h100"),
    "pearl-c-sm120-v1-h3": _pearl_c("sm120"),
    "pearl-c-nvfp4-v0": _experimental("pearl-c-nvfp4-v0"),
}


def get(name: str, **config: Any) -> PoUWScheme:
    """The scheme registered under `name`; `KeyError` names the registered ones."""
    try:
        make = SCHEMES[name]
    except KeyError:
        raise KeyError(f"no PoUW scheme {name!r}; registered: {', '.join(SCHEMES)}") from None
    return make(**config)


def register(name: str, make: Callable[..., PoUWScheme]) -> None:
    """Install ``make`` as the factory of ``name``, a name `EXPERIMENTAL` holds: its package calls this when imported."""
    if name not in EXPERIMENTAL:
        raise KeyError(f"{name!r} is not registered by an experimental package; those are: {', '.join(EXPERIMENTAL)}")
    SCHEMES[name] = make


__all__ = ["EXPERIMENTAL", "SCHEMES", "get", "register"]
