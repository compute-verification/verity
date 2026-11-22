"""The PoUW kernel schemes' builders. The schemes are named once, in
`verity.protocols.accounting.compute.pouw.verifier.kernel_schemes` (`KERNEL_SCHEMES`); `get(name)` builds the
configured `verity.protocols.accounting.compute.pouw.verifier.protocol.PoUWScheme` an entry names, and
`of(scheme, card, pinned=, arch=)` the one a resource scheme runs on a card (its declared ``kernels``, which
`runs_on` builds). Each family's module docstring states its steps, and
its pinned vectors are under ``verifier/tests/vectors``. A family an experimental package builds (an entry's ``seam``)
is `register`ed by that package when imported: until then building it raises `NotImplementedError` naming the seam."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verity.protocols.accounting.compute.pouw.verifier.kernel_schemes import KERNEL_SCHEMES, KernelScheme, named
from verity.protocols.accounting.compute.pouw.verifier.protocol import PoUWScheme
from verity.protocols.accounting.scheme import Kernels, Scheme, declared


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


def of(scheme: Scheme, card: str, *, pinned: bool, arch: str, **config: Any) -> PoUWScheme:
    """The scheme the resource scheme ``scheme`` runs on the card whose short name is ``card`` and whose card file's
    ``arch`` is ``arch``, with registration pinning the weights B or not (its ``kernels``); `LookupError` where it
    declares none or refuses the card."""
    return get(declared(scheme, "kernels")(card, pinned=pinned, arch=arch), **config)


def _runs(k: KernelScheme) -> str:
    return f"{k.name} on {' or '.join(k.cards)}{'' if k.pinned is None else ' pinned' if k.pinned else ' chosen'}"


def runs_on(*names: str) -> Kernels:
    """A resource scheme's ``kernels`` when it runs the entries ``names``: given a card's short name, ``pinned=`` and
    ``arch=`` (its card file's ``arch``, which its caller reads), the one entry's name (the name the build is found by
    everywhere) built for that architecture and run on that card with that pinning.
    `LookupError` names the architectures there are builds for when the card's has none, and the cards and pinnings the
    builds run on when the card or its pinning isn't one."""
    rows = tuple(named(n) for n in names)
    if any(not k.cards for k in rows):
        raise ValueError(f"{', '.join(k.name for k in rows if not k.cards)}: a resource scheme runs only an entry naming "
                         "its build's architecture and cards")

    def kernels(card: str, *, pinned: bool, arch: str) -> str:
        built = [k for k in rows if k.arch == arch]
        if not built:
            raise LookupError(f"{card}'s architecture {arch} has no build here: "
                              f"{', '.join(f'{k.name} is built for {k.arch}' for k in rows)}")
        found = [k for k in built if card in k.cards and k.pinned in (pinned, None)]
        if len(found) != 1:
            raise LookupError(f"{len(found)} builds for {arch} run on {card} with B {'pinned' if pinned else 'chosen'}: "
                              f"they run {', '.join(_runs(k) for k in built)}")
        return found[0].name

    return kernels


def register(family: str, make: Callable[..., PoUWScheme]) -> None:
    """Install ``make`` as the builder of ``family``, one an entry's ``seam`` names: its package calls this when
    imported."""
    seams = {k.family: k.seam for k in KERNEL_SCHEMES if k.seam}
    if family not in seams:
        raise KeyError(f"{family!r} is not built by an experimental package; those are: {', '.join(seams)}")
    BUILDERS[family] = make


__all__ = ["BUILDERS", "build", "get", "of", "register", "runs_on"]
