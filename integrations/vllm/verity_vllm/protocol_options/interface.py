"""What every protocol option implements, and what the selector hands it (`verity_vllm.protocol_options`).

An adapter is one module of this package (`sampled_proofs`, `pous`, `pouw`) that defines:

* ``NAME``: its protocol, one of `config.PROTOCOLS`;
* ``schemes() -> tuple[str, ...]``: the scheme names its protocol package registers;
* ``refusals(scheme, ctx) -> tuple[str, ...]``: why this run cannot enable it (empty: it can), decided from the `Context`
  alone, before the engine is built;
* ``install(scheme, target) -> Installed``: install it on the built model;
* optionally ``EXECUTES: tuple[str, ...]``: the sites where the option replaces the computation rather than wrapping
  it (PoUW: ``LinearBase.quant_method.apply``).  A site has at most one executor, innermost; the others wrap it from
  outside (`protocol_options.SITES`).  What sampled proofs traces an executed site as is not the adapter's to say: it
  is the protocol package's circuit (`protocol_options.CIRCUITS`, `protocol_options.traced_as`), and an executor whose
  scheme has none is refused beside sampled proofs, since it would be traced as the stock computation;
* optionally ``keeps_weight_copy(scheme, ctx) -> bool``: True when the installed option keeps a weight-derived copy
  (PoUW's prepared int7 or FP10 operands) resident across forwards.  With POUS enabled, the selector refuses such an
  option (`protocol_options.refusals`): its GEMMs would read the copy, not what POUS stores.  Absent means False;
* optionally ``commit_weights(scheme, target) -> Mapping``: its commitment to the plaintext weights of record, made in
  the composition's first phase, before any option changes the model or reads a beacon (`install`).  ``install`` then
  finds every option's in ``target.weight_commitments``.

An adapter holds no protocol logic: the scheme, its commitments, the salt and the audit are its protocol package's
(`verity.core.protocols.pouw`, `verity.core.protocols.pous`, `verity.core.protocols.sampled_proofs`), which never import this one.  It finds its hook sites by type,
never by module name, and makes every patch and service through ``target.hooks(entry)``: an `engine.hooks.Hooks` tagged
with its entry, so the composition uninstalls it and a committer recognises it (`engine.hooks.live(tag)`).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from verity_vllm.engine.hooks import Hooks

#: Where a composition runs: ``serve`` is `verity_vllm.LLM`; ``capture`` is a row's Match capture (its executors only, so the
#: observed run executes what the Commit serves); ``commit`` is the Commit stage of a row.
STAGES = ("serve", "capture", "commit")


@dataclass(frozen=True)
class Context:
    """What an adapter's refusals read.  ``enabled`` is the whole set in its one spelling (`config.parse_protocols`), so
    an adapter sees what it is composed with; ``settings`` holds each protocol's own keys of the protocols config
    (``{name: {...}}``), which are never hashed."""

    stage: str
    enabled: tuple[str, ...]
    world: int = 1
    enforce_eager: bool = True
    settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")

    def scheme(self, name: str) -> str | None:
        """The scheme ``name`` is enabled with, None when it is off."""
        return next((e.partition(":")[2] for e in self.enabled if e.partition(":")[0] == name), None)

    def settings_of(self, name: str) -> Mapping[str, Any]:
        return self.settings.get(name) or {}


@dataclass(frozen=True)
class Target:
    """What an adapter installs onto: the served model (on this rank) and the run's `Context`.  ``made`` keeps every
    `Hooks` handed out, so an install that raises halfway leaves nothing behind; ``weight_commitments`` holds every
    option's commitment to the plaintext weights, by entry, made before any install."""

    model: Any
    context: Context
    rank: int = 0
    made: list[Hooks] = field(default_factory=list)
    weight_commitments: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def hooks(self, entry: str) -> Hooks:
        """A fresh `Hooks` tagged with ``entry`` (``pous:band-chain/d12/v1``), for every patch and service the option makes."""
        h = Hooks(tag=entry)
        self.made.append(h)
        return h


@runtime_checkable
class Installed(Protocol):
    """An installed option.  `commitment` and `info` stay readable after `close`: the Commit writes its record once every
    option is off."""

    name: str
    hooks: Hooks

    def commitment(self) -> dict[str, Any]:
        """What the option commits to, as JSON: the composition record's entry (`Composition.record`)."""

    def info(self) -> dict[str, Any]:
        """Certificate, counters and endpoints, recorded beside the run and never hashed."""

    def weights(self) -> Mapping[str, Any] | None:
        """The plaintext of every parameter or buffer the option released from the model, by its qualname in
        ``model.named_parameters()`` / ``named_buffers()`` (tied names map to one tensor object); None when it released
        none.  Readers of live weights read through it while the option is installed (`protocol_options.weights_view`)."""

    def close(self) -> None:
        """Uninstall the option's hooks, then release its device state.  Idempotent."""


@dataclass(eq=False)
class Option:
    """An `Installed` an adapter can return as it is: ``commits`` and ``details`` are its commitment and info,
    ``decoded`` returns the plaintext of what it released (`Installed.weights`), and ``release`` runs once after its
    hooks are off."""

    name: str
    hooks: Hooks
    commits: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    decoded: Callable[[], Mapping[str, Any]] | None = None
    release: Callable[[], Any] | None = None
    closed: bool = False

    def commitment(self) -> dict[str, Any]:
        return dict(self.commits)

    def info(self) -> dict[str, Any]:
        return {"entries": len(self.hooks.entries), **self.details}

    def weights(self) -> Mapping[str, Any] | None:
        return None if self.decoded is None or self.closed else self.decoded()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.hooks.uninstall()
        finally:
            if self.release is not None:
                self.release()
