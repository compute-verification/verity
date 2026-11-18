"""PoUS on a served run, through the run's exchange (`verity.hooks`): the body of the ``pous`` scheme's ``requests`` and
``attach`` (`memory/pous/scheme.py`). It imports nothing of a serving system: a handle is a tensor it reads through the
weight store (`weight_store`), and the adapter's hooks emit the events and answer the requests.

`Served.install` files, in this order of the run:

* ``declared``: PoUS's settings, `Declared.settings["pous"]`, which `scheme.settings` checked (``challenges``,
  `check_settings`); what the launcher passes in ``ctx`` is checked when it is attached (`launcher_refusals`): the
  auditor's coin source (``salt_source``), ``libp2v1.so``'s directory (``p2_lib``) and ``experimental``, none of them an
  application's setting, since a coin fixed in an application would be no live coin and a path is the machine's;
* ``memory``: at `RegisteredValues`, ``W`` is the values a block reads as its weight operand (``operands``), in the order
  of their blocks' keys; the commitment to it
  (`weight_store.commit`) before any salt exists, the salt from the auditor's coin flipped after it
  (`weight_store.salt_from`, `weight_store.salt`), ``C*`` encoded on the weights' device (`weight_store.encode`), and a
  `ReleaseValue` for each weight of ``W``, whose ``restore`` decodes it from ``C*``;
* ``blocks``: a `ComputeBlockWith` whose kernel decodes a due block's registered weight (`BlockDue.weights["weight"]`)
  from ``C*`` into a fresh buffer and hands it inward, so the blocks inside it read what PoUS stores. It is filed after
  PoUW's (an executor's), so it runs outside it.

It is the `verity.hooks.Commits` of its run: ``commitment`` (``W``'s, the salt, ``C*``'s root), ``record``, ``summary``,
and ``release``, which frees ``C*`` when the workload stops. The vLLM adapter's memory hook hands it the weights and
the blocks: after ``load_model`` it attaches the runner's linear sites once a scheme computes blocks, emits
`RegisteredValues` with each site's weight as an operand, then releases what was asked (`engine/kv_blocks.py`).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from verity import hooks as H

NAME = "pous"
EAGER = H.RequireProperty("eager", True, "compiled execution: PoUS's decode at each use is not graph-capturable, so pous "
                          "runs eager")


def requests(ctx: Any) -> tuple[H.Request, ...]:
    return (EAGER,)


def source_refusal(source: Any) -> str | None:
    """Why ``salt_source`` can't give the auditor's coin, or None."""
    if source is None or source == "":
        return ("no salt_source: the salt is setup.sample_salt(source, commitment), the source the auditor's coin drawn after "
                "the commitment (its hex, a callable of the commitment, or {'command': argv})")
    if callable(source):
        return None
    if isinstance(source, str):
        try:
            return None if bytes.fromhex(source) else "salt_source is empty"
        except ValueError:
            return f"salt_source {source[:16]!r}... is not hex"
    if isinstance(source, Mapping) and set(source) == {"command"} and isinstance(source["command"], (list, tuple)) \
            and source["command"]:
        return None
    return f"salt_source is hex, a callable of the commitment or {{'command': argv}}, not {type(source).__name__}"


def check_settings(values: Mapping[str, Any]) -> str | None:
    """`Scheme.check_settings`: what PoUS refuses of its settings' values beyond their types."""
    k = values.get("challenges")
    return None if type(k) is int and k >= 1 else f"challenges is a positive integer, not {k!r}"


#: what a launcher passes PoUS in ``ctx``, beside ``settings``
LAUNCHER = ("salt_source", "p2_lib", "experimental")


def launcher_refusals(scheme: str, ctx: Mapping[str, Any]) -> tuple[str, ...]:
    """Why PoUS can't run ``scheme`` with what its launcher passes in ``ctx`` (`LAUNCHER`)."""
    from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws

    out = []
    if (registered := ws.registered(scheme)) is None:
        out.append(f"pous registers no scheme {scheme!r}")
    elif (reason := registered.experimental) and not ctx.get("experimental"):
        out.append(f"{scheme} is experimental ({reason}); a launcher's experimental = true runs it anyway")
    if why := source_refusal(ctx.get("salt_source")):
        out.append(why)
    return tuple(out)


@dataclass(eq=False)
class Served:
    """PoUS on one served run, held under ``scheme`` (``p2-16448/v4``) with what its launcher passed (``ctx``), its
    settings from the declared build."""

    scheme: str
    ctx: Mapping[str, Any]
    settings: dict[str, Any] | None = None
    store: Any = None
    exchange: Any = None
    shapes: dict[str, tuple[tuple[int, ...], Any]] = field(default_factory=dict)
    commits: dict[str, Any] = field(default_factory=dict)
    answered: int = 0
    _decode: Any = None

    @property
    def entry(self) -> str:
        return f"{NAME}:{self.scheme}"

    def install(self, exchange: Any) -> None:
        if why := launcher_refusals(self.scheme, self.ctx):
            raise ValueError(f"{self.entry}: {'; '.join(why)}")
        self.exchange = exchange
        exchange.subscribe(NAME, "declared", self.declared)
        exchange.subscribe(NAME, "memory", self.registered)
        exchange.request(NAME, H.ComputeBlockWith(self.kernel))

    def declared(self, event: H.Declared) -> None:
        if NAME not in event.settings:
            raise ValueError(f"{self.entry}: the build declared no settings under {NAME!r}")
        self.settings = dict(event.settings[NAME])

    def registered(self, event: Any) -> None:
        if not isinstance(event, H.RegisteredValues):
            return
        from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws

        if self.settings is None:
            raise RuntimeError(f"{self.entry}: the weights were registered before the run declared its settings")
        if self.store is not None:
            raise RuntimeError(f"{self.entry}: the weights were registered twice, and C* is encoded once")
        tensors = dict(sorted(((n, h) for n, h in event.values if n in event.operands), key=lambda v: event.operands[v[0]]))
        if not tensors:
            raise ValueError(f"{self.entry}: no registered value is a block's weight operand, so W is empty")
        lib = self.ctx.get("p2_lib") or None
        layout, commitment = ws.commit(self.scheme, tensors)
        source, how = ws.salt_from(self.entry, self.ctx["salt_source"], commitment, layout.params)
        self.store = ws.encode(layout, tensors, commitment, ws.salt(source, commitment), how, lib)
        self._decode = ws.decoder(self.store, lib)
        self.shapes = {n: (tuple(t.shape), t.dtype) for n, t in tensors.items()}
        q = self.store.params
        self.commits = {"scheme": self.scheme, "commitment": commitment.hex(), "salt": self.store.salt.hex(),
                        "storage_root": self.store.root.hex(), "blocks": q.blocks, "segments": q.segments,
                        "layout": ws.layout_digest(layout)}
        for name in tensors:
            self.exchange.request(NAME, H.ReleaseValue(name, (lambda n: lambda: self.decoded(n))(name)))

    def decoded(self, name: str) -> Any:
        """Registered weight ``name`` decoded from ``C*`` into a fresh buffer on ``C*``'s device."""
        from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws

        shape, dtype = self.shapes[name]
        out = self.store.rows.new_empty(shape, dtype=dtype)
        self._decode(self.store.layout.placement(name), ws.flat(out))
        return out

    def kernel(self, due: H.BlockDue, original: Any) -> Any:
        name = due.weights.get("weight")
        if name not in self.shapes:
            w = due.inputs.get("weight")
            if w is not None and w.numel() == 0:
                raise RuntimeError(f"{self.entry}: block {due.block}'s weight {name} is released, and PoUS holds no "
                                   f"weight by that name ({len(self.shapes)} it holds)")
            return original(None)
        return original({"weight": self.decoded(name)})

    def commitment(self) -> Mapping[str, Any]:
        """What PoUS commits to: the scheme, ``W``'s commitment, the salt, ``C*``'s root and layout."""
        return dict(self.commits)

    def record(self) -> Mapping[str, Any]:
        return {"scheme": self.scheme, **(self.settings or {})}

    def summary(self) -> Mapping[str, Any]:
        return {"answered": self.answered, "weights": len(self.shapes)}

    def release(self) -> None:
        if self.store is not None:
            self.store.free()

    def answer(self, index: int) -> bytes:
        """The prover's reply to challenge ``index``: block ``index`` of ``C*``."""
        from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws

        self.answered += 1
        return ws.answer(self.store, index)


def attach(exchange: Any, ctx: Mapping[str, Any], scheme: str) -> Served:
    """PoUS under ``scheme`` on ``exchange``, with what its launcher passed in ``ctx`` (`LAUNCHER`)."""
    s = Served(scheme, ctx)
    s.install(exchange)
    return s


__all__ = ["EAGER", "LAUNCHER", "NAME", "Served", "attach", "check_settings", "launcher_refusals",
           "requests", "source_refusal"]
