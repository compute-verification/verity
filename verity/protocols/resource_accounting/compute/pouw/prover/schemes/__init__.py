"""The PoUW kernel schemes' builders. The schemes are named once, in
`verity.protocols.resource_accounting.compute.pouw.verifier.kernel_schemes` (`KERNEL_SCHEMES`); `get(name)` builds the
configured `verity.protocols.resource_accounting.compute.pouw.verifier.protocol.PoUWScheme` an entry names, and
`of(scheme, card, pinned)` the one a resource scheme runs on a card (its declared ``kernels``). Each family's module docstring states its steps, and
its pinned vectors are under ``verifier/tests/vectors``. A family an experimental package builds (an entry's ``seam``)
is `register`ed by that package when imported: until then building it raises `NotImplementedError` naming the seam."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verity.protocols.resource_accounting.compute.pouw.verifier.kernel_schemes import KERNEL_SCHEMES, KernelScheme, named
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import PoUWScheme


def _ncp(**config: Any) -> PoUWScheme:
    from .ncp import NCP
    return NCP(**config)


def _pearl_kw(**config: Any) -> PoUWScheme:
    from .pearl_kw import PearlKW
    return PearlKW(**config)


def _pearl_c(*, device: str, **config: Any) -> PoUWScheme:
    from .pearl_c import PearlC
    from .pearl_c_device import DEVICES
    return PearlC(device=DEVICES[device], **config)


#: family -> builder of a configured scheme from its entry's parameters (further keyword arguments configure it, as a
#: commitment scheme's ports do)
BUILDERS: dict[str, Callable[..., PoUWScheme]] = {"ncp": _ncp, "pearl-kw": _pearl_kw, "pearl-c": _pearl_c}


def build(entry: KernelScheme, **config: Any) -> PoUWScheme:
    """The scheme ``entry`` names, configured by ``config`` over its parameters."""
    if entry.family not in BUILDERS:
        if entry.seam:
            raise NotImplementedError(f"{entry.name}: built by {entry.seam}, which registers {entry.family!r} when "
                                      "imported")
        raise KeyError(f"{entry.name} is a {entry.family} scheme, which no PoUWScheme builds")
    s = BUILDERS[entry.family](**entry.parameters, **config)
    if s.name != entry.name:
        raise ValueError(f"the scheme built for {entry.name!r} names itself {s.name!r}")
    return s


def get(name: str, **config: Any) -> PoUWScheme:
    """The scheme named ``name``; `KeyError` names the named ones."""
    return build(named(name), **config)


def of(scheme: str, card: str, pinned: bool, **config: Any) -> PoUWScheme:
    """The scheme the resource scheme ``scheme`` runs on the card whose short name is ``card``, with registration
    pinning the weights B or not (its ``kernels``); `LookupError` where it declares none or refuses the card."""
    from verity.protocols.compose import schemes
    from verity.protocols.resource_accounting.scheme import declared

    return get(declared(schemes.get(scheme), "kernels")(card, pinned=pinned), **config)


def register(family: str, make: Callable[..., PoUWScheme]) -> None:
    """Install ``make`` as the builder of ``family``, one an entry's ``seam`` names: its package calls this when
    imported."""
    seams = {k.family: k.seam for k in KERNEL_SCHEMES if k.seam}
    if family not in seams:
        raise KeyError(f"{family!r} is not built by an experimental package; those are: {', '.join(seams)}")
    BUILDERS[family] = make


__all__ = ["BUILDERS", "build", "get", "of", "register"]
