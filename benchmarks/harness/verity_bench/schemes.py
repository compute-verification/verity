"""A benchmark's useful-work scheme, given by its registered name (`verity.protocols.resource_accounting.scheme`), and
the PoUW kernel scheme it runs on a card, given by the card's short name (`kit.card_short`).

- `useful_work(name)`: the registered compute scheme ``name``; ValueError naming the registered ones otherwise.
- `attached(name, weights, given)`: ``name`` attached to a benchmark's exchange with no engine, its run opened.
- `served(scheme)`: ``scheme`` where an adapter entry attaches it to a served run; `NotImplementedError` otherwise.
- `kernel_scheme(name, card, pinned)`: the PoUW kernel scheme ``name`` runs on ``card``.
- `arch(short)`: the build target of the GPU card whose ``short`` is ``short`` (`verity.device.gpu_card_called`).
"""

from __future__ import annotations

from verity.protocols.resource_accounting import scheme as registry

__all__ = ["arch", "attached", "kernel_scheme", "served", "useful_work"]


def useful_work(name: str) -> registry.Scheme:
    """The compute scheme declared under ``name`` (`compose.schemes.get`); ValueError naming the declared compute schemes
    otherwise, every declaration loaded only to name them (`compose.schemes.load`)."""
    from verity.protocols.compose import schemes

    try:
        if (found := schemes.get(name)).resource == "compute":
            return found
    except KeyError:
        pass
    compute = sorted(n for n, s in schemes.load().items() if s.resource == "compute")
    raise ValueError(f"--protocol {name!r} is no registered compute scheme; registered: {', '.join(compute)}")


def attached(name: str, weights: dict, given: dict) -> object:
    """What scheme ``name``'s ``attach`` returned on an exchange with no engine: ``weights`` registered by key, phase 0 open."""
    from verity import hooks as H
    from verity.protocols.compose import schemes as declarations
    from verity_vllm.messages import Exchange

    exchange = Exchange()
    on = exchange.attach(declarations.get(name), given)
    exchange.emit(H.RegisteredValues(tuple((str(k), w) for k, w in weights.items()), operands={str(k): k for k in weights}))
    exchange.emit(H.PhaseOpened(0, 0))
    return on


def served(scheme: registry.Scheme) -> registry.Scheme:
    """``scheme``, where an adapter entry attaches it to a served run (`Scheme.option`); `NotImplementedError` naming
    what it lacks otherwise."""
    if not scheme.option:
        raise NotImplementedError(
            f"{scheme.name} is not served: no adapter entry attaches it to a served run, so no kernel computes its "
            f"workload's products while serving and no verifier ({scheme.protocol}) answers for a served run of it")
    return scheme


def arch(short: str) -> str:
    """The ``arch`` of the GPU card whose ``short`` is ``short`` (`verity.device.gpu_card_called`)."""
    from verity.device import gpu_card_called

    from verity_bench.kit import cards_directory

    return gpu_card_called(short, directory=cards_directory())["arch"]


def kernel_scheme(name: str, card: str, pinned: bool) -> str:
    """The PoUW kernel scheme the served useful-work scheme ``name`` runs on the card ``card`` (its short name), with
    registration pinning the weights or not (the scheme's declared ``kernels``, `kernel_schemes.runs_on`, which refuses a
    card its build doesn't run on by name)."""
    return registry.declared(served(useful_work(name)), "kernels")(card, pinned=pinned)
