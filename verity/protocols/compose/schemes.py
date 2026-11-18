"""Every resource scheme, by name: the one table from a scheme's name to the module that declares it, which a reader
(an application's table, the adapter, the benchmarks, the composition's `credited`) reads instead of importing each
scheme's package by name. The prover and the verifier read the same table, so no verdict is given under a name the
verifier never registered.

A new scheme adds its name here, one line; `get` imports its module on first use, which registers it
(`verity.protocols.resource_accounting.scheme.register`), so a reader loads only the schemes an application lists;
`load` imports every module and returns the registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType

from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.scheme import Scheme

_COMPUTE = "verity.protocols.resource_accounting.compute"
_MEMORY = "verity.protocols.resource_accounting.memory"

#: each scheme's name, the name an application lists, to the module that registers it when imported
DECLARATIONS: Mapping[str, str] = MappingProxyType({
    "bandwidth": f"{_MEMORY}.poub.scheme",
    "dither": "verity.protocols.experimental.dither.scheme",
    "network-attester": "verity.protocols.resource_accounting.network.scheme",
    "pearl-plus": f"{_COMPUTE}.pouw.scheme",
    "pous": f"{_MEMORY}.pous.scheme",
    "wipe": f"{_MEMORY}.pous.scheme",
})


def get(name: str) -> Scheme:
    """The scheme named ``name``, its module imported on first use; `KeyError` names the declared ones."""
    if name not in S.SCHEMES:
        if name not in DECLARATIONS:
            raise KeyError(f"no scheme {name!r}; declared: {', '.join(sorted(DECLARATIONS))}")
        import_module(DECLARATIONS[name])
        if name not in S.SCHEMES:
            raise KeyError(f"{DECLARATIONS[name]} registers no scheme {name!r}; it registers what this table must name")
    return S.SCHEMES[name]


def load() -> Mapping[str, Scheme]:
    """Every declared scheme, by name, once each declaring module is imported."""
    for module in sorted(set(DECLARATIONS.values())):
        import_module(module)
    return MappingProxyType(dict(S.SCHEMES))


__all__ = ["DECLARATIONS", "get", "load"]
