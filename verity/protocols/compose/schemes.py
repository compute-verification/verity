"""Every resource scheme, registered: the one list of the modules that declare one, which a reader of every scheme (an
application's table, the adapter, the benchmarks, the composition's `credited`) loads instead of importing each scheme's
package by name. The prover and the verifier load the same list, so no verdict is given under a name the verifier never
registered.

A new declaration adds its module here, one line; `load` imports each, which registers it
(`verity.protocols.resource_accounting.scheme.register`), and returns the registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType

from verity.protocols.resource_accounting.scheme import SCHEMES, Scheme

#: the modules that declare a resource scheme, each registering its schemes when imported
DECLARATIONS = (
    "verity.protocols.experimental.dither.scheme",
    "verity.protocols.resource_accounting.compute.pouw.scheme",
    "verity.protocols.resource_accounting.memory.poub.scheme",
    "verity.protocols.resource_accounting.network.scheme",
)


def load() -> Mapping[str, Scheme]:
    """Every declared scheme, by name, once each declaration module is imported."""
    for module in DECLARATIONS:
        import_module(module)
    return MappingProxyType(dict(SCHEMES))


__all__ = ["DECLARATIONS", "load"]
