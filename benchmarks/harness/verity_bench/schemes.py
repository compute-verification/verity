"""A benchmark's useful-work scheme, given by its registered name (`verity.protocols.resource_accounting.scheme`), and
the PoUW kernel scheme it runs on a card, given by the card's short name (`kit.card_short`).

- `useful_work(name)`: the registered compute scheme ``name``; ValueError naming the registered ones otherwise.
- `served(scheme)`: ``scheme`` where an adapter entry attaches it to a served run; `NotImplementedError` otherwise.
- `kernel_scheme(name, card, pinned)`: the PoUW kernel scheme ``name`` runs on ``card``.
- `arch(card)`: the build target of the card record whose ``short`` is ``card``.
"""

from __future__ import annotations

from verity.protocols.resource_accounting import scheme as registry

__all__ = ["arch", "kernel_scheme", "served", "useful_work"]


def useful_work(name: str) -> registry.Scheme:
    """The compute scheme registered under ``name``; ValueError naming the registered compute schemes otherwise."""
    # the registry holds a scheme only once the package declaring it is imported
    from verity.protocols.experimental.dither import scheme as _dither  # noqa: F401
    from verity.protocols.resource_accounting.compute.pouw import scheme as _pouw  # noqa: F401

    compute = sorted(n for n, s in registry.SCHEMES.items() if s.resource == "compute")
    if name not in compute:
        raise ValueError(f"--protocol {name!r} is no registered compute scheme; registered: {', '.join(compute)}")
    return registry.get(name)


def served(scheme: registry.Scheme) -> registry.Scheme:
    """``scheme``, where an adapter entry attaches it to a served run (`Scheme.option`); `NotImplementedError` naming
    what it lacks otherwise."""
    if not scheme.option:
        raise NotImplementedError(
            f"{scheme.name} is not served: no adapter entry attaches it to a served run, so no kernel computes its "
            f"workload's products while serving and no verifier ({scheme.protocol}) answers for a served run of it")
    return scheme


def arch(card: str) -> str:
    """The ``arch`` of the GPU card record whose ``short`` is ``card``; ValueError naming the shorts otherwise."""
    from verity.device import load_card, skus

    from verity_bench.kit import cards_directory

    directory = cards_directory()
    cards = {c.get("short"): c for c in (load_card(s, directory=directory) for s in skus("gpu", directory=directory))}
    if card not in cards:
        raise ValueError(f"no card record under {directory} has short {card!r}; the shorts are "
                         f"{', '.join(sorted(s for s in cards if s))}")
    return cards[card]["arch"]


def kernel_scheme(name: str, card: str, pinned: bool) -> str:
    """The PoUW kernel scheme the served useful-work scheme ``name`` runs on the card ``card`` (its short name), with
    registration pinning the weights or not (`pouw.verifier.kernel_schemes.resolve`, whose `KeyError` names the table)."""
    from verity.protocols.resource_accounting.compute.pouw.verifier.kernel_schemes import resolve

    return resolve(served(useful_work(name)).name, card, pinned).name
