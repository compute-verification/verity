"""PoUS on a served run, through the run's exchange (`verity.hooks`): the body of the ``pous`` scheme's ``requests`` and
``attach`` (`memory/pous/scheme.py`). It imports nothing of a serving system: a handle is a tensor it reads through the
weight store (`weight_store`), and the adapter's hooks emit the events and answer the requests.

`Served.install` files, in this order of the run:

* ``declared``: PoUS's settings, `Declared.settings["pous"]` over the declaration's defaults, each checked (`refusals`);
* ``memory``: at `RegisteredValues`, ``W`` is the values marked linear, in the order registered; the commitment to it
  (`weight_store.commit`) before any salt exists, the salt from the auditor's coin flipped after it
  (`weight_store.salt_from`, `weight_store.salt`), ``C*`` encoded on the weights' device (`weight_store.encode`), and a
  `ReleaseValue` for each weight of ``W``, whose ``restore`` decodes it from ``C*``;
* ``blocks``: a `ComputeBlockWith` whose kernel decodes a due block's registered weight (`BlockDue.weights["weight"]`)
  from ``C*`` into a fresh buffer and hands it inward, so the blocks inside it read what PoUS stores. It is filed after
  PoUW's (an executor's), so it runs outside it.

`attach` raises until a served engine hands its linear layers' calls to the exchange (`WAITS_ON`): without that, the
release would empty weights no kernel decodes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from verity import hooks as H

NAME = "pous"
GOAL = "memory-accounting/memory-protocols-declared"
WAITS_ON = ("a served engine's linear layers called through the exchange: a build that, after vLLM's load_model, calls "
            "verity_vllm.protocols.linear.vllm_site.discover(kv_blocks.model_of(runner)) and vllm_site.attach(sites, "
            "hooks), so each call is a BlockDue whose weights['weight'] is the registered value's name")
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


def refusals(scheme: str, own: Mapping[str, Any], defaults: Mapping[str, Any]) -> tuple[str, ...]:
    """Why PoUS can't run ``scheme`` with its settings ``own`` (`Declared.settings["pous"]`), whose keys are
    ``defaults``' (the declaration's settings)."""
    from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws

    out = []
    if stray := sorted(set(own) - set(defaults)):
        out.append(f"unknown pous settings {stray} (known: {', '.join(sorted(defaults))})")
    k = own.get("challenges", defaults.get("challenges"))
    if type(k) is not int or k < 1:
        out.append(f"challenges is a positive integer, not {k!r}")
    if (registered := ws.registered(scheme)) is None:
        out.append(f"pous registers no scheme {scheme!r}")
    elif (reason := registered.experimental) and not own.get("experimental"):
        out.append(f"{scheme} is experimental ({reason}); settings pous.experimental = true runs it anyway")
    if why := source_refusal(own.get("salt_source")):
        out.append(why)
    return tuple(out)


@dataclass(eq=False)
class Served:
    """PoUS on one served run, held under ``scheme`` (``p2-16448/v4``) with the declaration's ``defaults``."""

    scheme: str
    defaults: Mapping[str, Any]
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
        self.exchange = exchange
        exchange.subscribe(NAME, "declared", self.declared)
        exchange.subscribe(NAME, "memory", self.registered)
        exchange.request(NAME, H.ComputeBlockWith(self.kernel))

    def declared(self, event: H.Declared) -> None:
        own = dict(event.settings.get(NAME, {}))
        if why := refusals(self.scheme, own, self.defaults):
            raise ValueError(f"{self.entry}: {'; '.join(why)}")
        self.settings = {**self.defaults, **own}

    def registered(self, event: Any) -> None:
        if not isinstance(event, H.RegisteredValues):
            return
        from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws

        if self.settings is None:
            raise RuntimeError(f"{self.entry}: the weights were registered before the run declared its settings")
        if self.store is not None:
            raise RuntimeError(f"{self.entry}: the weights were registered twice, and C* is encoded once")
        tensors = {n: h for n, h in event.values if n in event.linear}
        if not tensors:
            raise ValueError(f"{self.entry}: no registered value is a linear weight, so W is empty")
        lib = self.settings.get("p2_lib") or None
        layout, commitment = ws.commit(self.scheme, tensors)
        source, how = ws.salt_from(self.entry, self.settings["salt_source"], commitment, layout.params)
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
            return original(None)
        return original({"weight": self.decoded(name)})

    def answer(self, index: int) -> bytes:
        """The prover's reply to challenge ``index``: block ``index`` of ``C*``."""
        from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws

        self.answered += 1
        return ws.answer(self.store, index)


def attach(exchange: Any, ctx: Any) -> None:
    raise NotImplementedError(f"pous attaches through the exchange once there is {WAITS_ON} (goal {GOAL}); until then "
                              "the vLLM adapter's pous.install holds it, and Served(scheme, POUS.settings).install(exchange) "
                              "is this body")


__all__ = ["EAGER", "GOAL", "NAME", "Served", "WAITS_ON", "attach", "refusals", "requests", "source_refusal"]
