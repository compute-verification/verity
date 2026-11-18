"""PoUW's kernel schemes, each stated once: every `protocol.PoUWScheme` a name stands for, and every circuit scheme.

A kernel scheme's ``name`` is the string a proof's identifier carries (`identifier`, where ``identified``) and a record
stores. Its ``family`` and ``parameters`` say what builds it (`verity.protocols.resource_accounting.compute.pouw.prover
.schemes.get`, which has a builder for each family but ``ncp-v2``, whose Programs are `prover.circuit.traced_as`'s).

One that a resource scheme (`verity.protocols.resource_accounting.scheme`) runs names the architecture its build is
for once (``arch``, a card file's ``arch``), the cards the scheme runs that build on by their card files' ``short``
(``calibration/hardware/cards``), and whether registration pins the weights B (``pinned``; None where it takes either):
a build is made for an architecture and allowed on a card, so a new card of a built architecture is one more name in
``cards``. A resource scheme declares the entries it runs as its ``kernels`` (`runs_on`), which reads a card's
architecture from its file.

``seam`` names the experimental package that builds the family, which registers its builder when imported
(`prover.schemes.register`): PoUW imports nothing of ``verity/protocols/experimental/``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class KernelScheme:
    name: str
    family: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    arch: str = ""
    cards: tuple[str, ...] = ()
    pinned: bool | None = None
    identified: bool = True
    seam: str = ""

    def __post_init__(self) -> None:
        if bool(self.cards) != bool(self.arch):
            raise ValueError(f"{self.name}: a build's architecture and the cards it runs on are named together")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


_PEARL_C4 = "verity/protocols/experimental/pearl_c4/"
_DITHER = "verity/protocols/experimental/dither/"

KERNEL_SCHEMES: tuple[KernelScheme, ...] = (
    KernelScheme("ncp-v1", "ncp", {"perm": "word"}),
    KernelScheme("ncp-v1-shift24", "ncp", {"perm": "shift24"}),
    KernelScheme("pearl-fp8-v4", "pearl-kw"),
    KernelScheme("pearl-c-sm120-v1-h3", "pearl-c", {"device": "sm120", "hashing": "h3"}),
    KernelScheme("pearl-c-sm120-unpromoted-v1-h3", "pearl-c", {"device": "sm120-unpromoted", "hashing": "h3"},
                 identified=False),
    KernelScheme("pearl-c-h100-v1-h3", "pearl-c", {"device": "h100", "hashing": "h3"},
                 arch="sm_90a", cards=("h100", "h200")),
    KernelScheme("pearl-c-nvfp4-v0", "pearl-c4", {"fmt": "nvfp4", "hashing": "h0"}, seam=_PEARL_C4),
    KernelScheme("ncp-v2", "ncp-v2", {"perm": "word"}, identified=False),
    KernelScheme("ncp-v2-shift24", "ncp-v2", {"perm": "shift24"}, identified=False),
    KernelScheme("dither", "dither", identified=False, seam=_DITHER),
)

BY_NAME: Mapping[str, KernelScheme] = MappingProxyType({k.name: k for k in KERNEL_SCHEMES})
if len(BY_NAME) != len(KERNEL_SCHEMES):
    raise ValueError("a kernel scheme is named twice")
#: the names a proof's identifier v1 takes, in the table's order
IDENTIFIED: tuple[str, ...] = tuple(k.name for k in KERNEL_SCHEMES if k.identified)


def named(name: str) -> KernelScheme:
    """The kernel scheme named ``name``; `KeyError` names them all."""
    try:
        return BY_NAME[name]
    except (KeyError, TypeError):
        raise KeyError(f"no PoUW scheme {name!r}; registered: {', '.join(BY_NAME)}") from None


def of_family(family: str) -> tuple[KernelScheme, ...]:
    return tuple(k for k in KERNEL_SCHEMES if k.family == family)


def names(family: str) -> tuple[str, ...]:
    """The names of ``family``'s kernel schemes, in the table's order."""
    return tuple(k.name for k in of_family(family))


CARD_FILES = Path(__file__).resolve().parents[6] / "calibration/hardware/cards"


def _runs(k: KernelScheme) -> str:
    return f"{k.name} on {' or '.join(k.cards)}{'' if k.pinned is None else ' pinned' if k.pinned else ' chosen'}"


def runs_on(*names: str) -> Callable[..., str]:
    """A resource scheme's ``kernels`` when it runs the entries ``names``: given a card's short name and ``pinned=``,
    the one entry's name (the name the build is found by everywhere) built for that card's architecture (its card file's
    ``arch``) and run on that card with that pinning. `LookupError` names the architectures there are builds for when
    the card's has none, and the cards and pinnings the builds run on when the card or its pinning isn't one."""
    rows = tuple(named(n) for n in names)
    if any(not k.cards for k in rows):
        raise ValueError(f"{', '.join(k.name for k in rows if not k.cards)}: a resource scheme runs only an entry naming "
                         "its build's architecture and cards")

    def kernels(card: str, *, pinned: bool) -> str:
        from verity.device import gpu_card_called

        arch = gpu_card_called(card, directory=CARD_FILES).get("arch")
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


__all__ = ["BY_NAME", "CARD_FILES", "IDENTIFIED", "KERNEL_SCHEMES", "KernelScheme", "named", "names", "of_family", "runs_on"]
