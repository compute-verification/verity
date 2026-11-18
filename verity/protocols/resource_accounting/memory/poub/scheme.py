"""PoUB's two declarations as resource schemes (`verity.protocols.resource_accounting.scheme`), both of read bandwidth,
registered when this module is imported (`verity.protocols.compose.schemes.DECLARATIONS`).

They are two protocols, each with its own verifier and its own store, and no setting turns one into the other:

- `BANDWIDTH` (``bandwidth``, the name onsite's applications list): PoUB-P2, ``verity/remote-bandwidth/p2/v0``, over
  PoUS's registered C*. Its prover reads PoUS's store and the receipts PoUS's decode computed: it attaches through no
  hook of the served run and has no setting.
- `POUB` (``poub``, the adapter's option and the benchmarks' session): the remote session ``verity/remote-bandwidth/v0``
  over the node's own store (`prover.node_store`), which `prover.binder.Binder` keeps current from the served run's
  ``memory`` and ``steps`` events, subscribed under `POUB`'s name. ``refresh`` is its one setting. The Commit's
  session over that store is its process `SESSION_PROCESS` (`prover.commit_session`), which reads what the build
  declared.

No verifier loads this module: each verifier states its own id, component name and deadline, and PoUB's tests hold them
equal to the declarations. It reads each one's parties and channels from its verifier's manifest (`verifier.p2_parties`,
`verifier.session_parties`).

Neither declares a block: PoUB's opening unit (`verifier.commitments.gates.block_check`, ``BlockOpeningCheck_v2``) is
proved in the read session's own window (`prover.window`), not as a template of the served circuit's first level, and
P2's openings go to its verifier in the clear. Each says its own term (`Scheme.delta_scheme`) as a `Term`: the sketch
bound's error `δSketch g b Q P N εrank` (`DELTA_SCHEME`) at its receipts' layout (g, b), a function of the prover's hash
budget ``Q``, the session's passes ``P``, the store's blocks ``N`` and the sign blocks' rank tail ``rank_tail`` (the
Lean's εrank, which is not the audit's error ε).
The two sides are the route form's: `ResourceAccounting.Remote.Run.Paid` is `Verity.Scheme`'s charge exactly
(`ResourceAccounting.Remote.Run.scheme`, security/Definitions/Poub/Remote/Scheme.lean:56); the deployed forms are what
its `GetsBack` gives. The term and the sides are not yet one statement: `Run.scheme`'s own term is `Paid`'s δ
(`δscheme := δ`, linked by `PaidBoundsBreaks`, stated and unproved), which the deployed route gives as
`RemoteRunRoutes`' hash term (Guarantees/RateAuditDeployed.lean:164), while `δSketch` is the error of `SketchBound`
(Assumptions/RemoteHeld.lean:108), a bound on the trace's correct receipts, and no statement takes it to `Paid`.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from importlib import import_module
from types import MappingProxyType
from typing import Any

from dataclasses import replace

from verity import hooks as H
from verity.commitments import domains
from verity.protocol import Channel
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.poub.verifier import p2_parties as P2
from verity.protocols.resource_accounting.memory.poub.verifier import session_parties as RS

#: P2's allowance from a window's boundary message to its roots' deadline, which the session's parameters carry
#: (`verifier.p2_parties.SESSION`); the remote session's is a parameter of each session (`verifier.audit.Params`)
DEADLINE_NS = 10**9
#: the process `POUB` registers (`Scheme.processes`): the Commit's session, which an adapter looks up by this name
SESSION_PROCESS = "poub-session"
#: what the parts of `POUB` on a served run that are declared and not built wait on
GOAL = "memory-accounting/memory-protocols-declared"
#: what `refresh` waits on: a session with the setting on accepted by the Lean verifier
REFRESH_GOAL = "memory-accounting/refresh-session-accepted"
#: the sketch bound's error at a fixed receipt layout, `δSketch g b Q P N εrank`
#: (security/Definitions/Poub/Guarantees/RemoteSketch.lean): g the guess term's bits, b the bits of the sign keys'
#: birthday term
DELTA_SKETCH = "ResourceAccounting.Guarantees.Remote.δSketch"


def delta_sketch(g: int, b: int) -> S.Term:
    """`DELTA_SKETCH` at the layout (``g``, ``b``) as a scheme's own term: its parameters are the prover's SHA-512
    compressions over the session ``Q``, its passes ``P``, the store's blocks ``N`` and the signs' rank tail
    ``rank_tail`` (the Lean's εrank, a rational here), and its value
    ``Q/2^g + (Q + P)·(rank_tail + N/2^512) + (Q + P)²/2^b``."""
    def value(Q: int, P: int, N: int, rank_tail: Fraction) -> Fraction:
        return Fraction(Q, 2**g) + (Q + P) * (Fraction(rank_tail) + Fraction(N, 2**512)) + Fraction((Q + P)**2, 2**b)
    return S.Term(f"{DELTA_SKETCH} {g} {b}", ("Q", "P", "N", "rank_tail"), value)


#: each scheme's own term: `DELTA_SKETCH` at its receipts' layout, the conclusion of `SketchBoundP2` (rank 12, 256-bit
#: sign keys) and of `SketchBoundDeployed` (rank 27, 512-bit sign keys)
DELTA_SCHEME = {"bandwidth": delta_sketch(96, 257), "poub": delta_sketch(216, 513)}
#: the ways to a correct receipt without a read from node memory, as the session's ledger names them
#: (`verifier.audit.GammaTerms`)
CHARGES = ("chip", "counting", "rebuild", "host", "link", "carrier")
_SKETCH = ("ResourceAccounting.Assumptions.SketchHashRandom",
           "ResourceAccounting.Assumptions.SketchRank",
           "ResourceAccounting.Remote.Sketch.Assumptions.KeyBudget",
           "ResourceAccounting.Remote.Sketch.Assumptions.Present",
           "ResourceAccounting.Remote.Sketch.Assumptions.Receipted")
_OUTSIDE = "the opening unit (BlockOpeningCheck_v2) is proved outside the workload's circuit: "


def _named(role: str) -> str:
    return P2.CLOSER_PARTY if role == P2.VERIFIER else role


def _bandwidth_attach(exchange: Any, ctx: Any) -> Any:
    """``bandwidth`` on a served run (`Scheme.attach`): P2's prover over the store PoUS holds on ``exchange``
    (`prover.p2_stage.bandwidth.Attached`), PoUS attached before it (``depends``); refused by name where it is not."""
    from verity.protocols.resource_accounting.memory.pous.prover import served

    storage = served.of(exchange)
    if storage is None:
        raise LookupError("bandwidth reads the store pous holds on the same exchange, and no pous is attached on it: "
                          "attach pous before bandwidth (its depends)")
    return import_module("verity.protocols.resource_accounting.memory.poub.prover.p2_stage.bandwidth").Attached(
        storage, ctx)


def _refresh(values: Mapping[str, Any]) -> str | None:
    return f"refresh waits on {REFRESH_GOAL}: a session with it on, accepted by the Lean verifier" \
        if values["refresh"] else None


def _bandwidth_reading(row: Mapping[str, Any]) -> S.Reading:
    q = row["quantity"]
    return S.Reading(int(q["q_bytes"]), Fraction(str(q["eps"])), "quantity.eps")


#: why neither read audit has a check of a record: its Lean verifier answers the session as it runs, one event a line
_LIVE = ("its Lean verifier ({}) checks the session live, an event a line, and no entry checks a finished record "
         "against the verifier's registration")


#: the route's named premise, unproved: the incorrect receipts are counted (`Remote.Run.Counted`)
_COUNTED = "ResourceAccounting.Remote.Run.Counted"

_REQUIRES = {
    "log-order": "the verifier party's own log of the session",
    "log-times": "the verifier party's clock on the auditor's host",
    "coins-at-rate": "a challenger in the verifier's process on its host's entropy; a pass's seed is derived from its "
                     "window's challenge and the previous pass's root",
}

BANDWIDTH = S.register(S.Scheme(
    name="bandwidth",
    protocol=domains.POUB_P2,
    resource="read",
    blocks=S.Lacks(_OUTSIDE + "no block of the workload's circuit carries its cost"),
    assumptions=(*_SKETCH, "ResourceAccounting.Assumptions.WorkHardnessP2",
                 "ResourceAccounting.Remote.Held.Assumptions.SquaringCeilingP2"),
    log=(
        S.Order("registration", "challenge"),
        S.Order("roots", "boundary", direction="after", within_ns=DEADLINE_NS,
                assumption="ResourceAccounting.Remote.Held.Assumptions.SquaringCeilingP2"),
        S.Order("roots", "openings"),
    ),
    charges=tuple(S.Charge(name, f"no P2 record carries it: verifier.audit.GammaTerms.{name} at P2's units",
                           bounded=False) for name in CHARGES),
    delta_scheme=DELTA_SCHEME["bandwidth"],
    requires={
        **_REQUIRES,
        "one-labelling": "its own trees: PoUS's vk root over the store and a receipt root a pass",
        "one-design": _OUTSIDE + "its own sample of receipts a window and its own pooled bound, the openings sent to "
                                 "the verifier in the clear",
    },
    over="the credited bits: the committed receipts' words, committed · b bits (`Remote.Run.committed`)",
    under="what the run paid for words: its tier reads, the state held at the pass boundaries and the bits rebuilt "
          "(`Remote.Run.paid`), in bits",
    attach=_bandwidth_attach,
    parties=(*(replace(r, name=_named(r.name)) for r in P2.MANIFEST.parties), S.VERIFIER),
    channels=(*(replace(c, sender=_named(c.sender), receiver=_named(c.receiver)) for c in P2.MANIFEST.channels),
              Channel(P2.CLOSER_PARTY, S.VERIFIER.name, ("BandwidthAnswer",), after=("Openings",))),
    channel_schemas=P2.MANIFEST.messages,
    colocated=(),
    needs=frozenset({"receipts"}),
    depends=("pous",),
    record="bandwidth",
    symbol="Q_cert",
    published=("bytes", Fraction(1, 8)),
    check=S.Lacks(_LIVE.format("p2_lean, as p2_parties.drive runs it")),
    reader=_bandwidth_reading,
    premises=(_COUNTED,),
    **dict.fromkeys(("tables", "output_format", "product_family", "y_committed", "salt_open_root", "kernels"),
                    S.Lacks("not a kernel scheme")),
))

POUB = S.register(S.Scheme(
    name="poub",
    protocol=domains.POUB_PROTOCOL,
    resource="read",
    blocks=S.Lacks(_OUTSIDE + "no block of the workload's circuit carries its cost"),
    assumptions=(*_SKETCH, "ResourceAccounting.Assumptions.WorkHardness",
                 "ResourceAccounting.Remote.Held.Assumptions.SquaringCeiling"),
    log=(S.Order("registration", "challenge"), S.Order("roots", "openings")),
    charges=tuple(S.Charge(name, f"gamma.{name}") for name in CHARGES),
    delta_scheme=DELTA_SCHEME["poub"],
    settings={"refresh": False},
    check_settings=_refresh,
    hooks=("declared", "memory", "steps"),
    requests=lambda ctx: requests(ctx),
    attach=lambda exchange, ctx: attach(exchange, ctx),
    processes={SESSION_PROCESS: lambda *a, **kw: session(*a, **kw)},
    requires={
        **_REQUIRES,
        "one-labelling": "its own trees: the node store's registration root and a receipt root a pass",
        "one-design": _OUTSIDE + "in the read session's own window (prover.window), whose circuit is not built "
                                 "(memory-accounting/read-session-in-zero-knowledge); today its own sample of "
                                 "receipts a window and its own pooled bound",
    },
    over="the credited bits: the committed receipts' words, committed · b bits (`Remote.Run.committed`)",
    under="what the run paid for words: its tier reads, the state held at the pass boundaries and the bits rebuilt "
          "(`Remote.Run.paid`), in bits",
    parties=RS.MANIFEST.parties,
    channels=RS.MANIFEST.channels,
    channel_schemas=RS.MANIFEST.messages,
    colocated=(),
    needs=frozenset(),
    depends=(),
    record=S.Lacks("no combined record has a row for the remote session: its verdict is its own session's "
                   "(benchmarks/poub_session)"),
    symbol=S.Lacks("the remote session publishes no certified quantity in a combined record"),
    published=S.Lacks("the remote session publishes no certified quantity in a combined record"),
    check=S.Lacks(_LIVE.format("session.lean_verifier, as session.verifier runs it")),
    reader=S.Lacks("no combined record has a row for the remote session"),
    premises=("ResourceAccounting.Assumptions.WorkHardness",
              "ResourceAccounting.Remote.Held.Assumptions.SquaringCeiling", _COUNTED),
    **dict.fromkeys(("tables", "output_format", "product_family", "y_committed", "salt_open_root", "kernels"),
                    S.Lacks("not a kernel scheme")),
))

#: a freed range is reused as late as the workload allows, so one freed inside a window is not written again in it
REUSE_OLDEST_FREED = H.ReuseOldestFreed()
#: what a session in lockstep with the run asks of it (one engine step a pass, `prover.node_store.serving.Lockstep`),
#: in the properties a served run declares (`verity.hooks.Declared`): one GPU's store, a step's outputs read before the
#: next step is scheduled, and no range brought back holding the values another input wrote
SESSION: tuple[H.Request, ...] = (
    H.RequireProperty("world", 1, "a store is one GPU's: a TP Commit's session is not wired yet"),
    H.RequireProperty("async_steps", False, "a pass follows each step's schedule and its output in turn"),
    H.RequireProperty("prefix_cache", False, "a prefix hit brings a page back with its KV"),
)


def requests(ctx: Mapping[str, Any]) -> tuple[H.Request, ...]:
    """`POUB`'s requests of a served run (`Scheme.requests`), by what ``ctx`` says of it: `REUSE_OLDEST_FREED` unless
    ``fifo`` is False; `SESSION` with ``session`` (a session in lockstep with the run); and with ``ranges`` that the
    workload's memory hold exactly that many (a session whose store's size is its registration's)."""
    return (*((REUSE_OLDEST_FREED,) if ctx.get("fifo", True) else ()),
            *(SESSION if ctx.get("session") else ()),
            *(() if ctx.get("ranges") is None else (H.FixRanges(ctx["ranges"]),)))


#: `POUB`'s prover, by what the declaration hands out, each as ``module:name``. The prover is a distribution of its
#: own (it runs on a GPU through torch, which `verity` never imports; ``tests/import_order.toml`` lists this file as
#: its loader), so the declaration names it and loads it on first use, as `verity.protocols.compose.schemes` loads a
#: declaration by its scheme's name.
PROVER: Mapping[str, str] = MappingProxyType({
    "binder": "verity.protocols.resource_accounting.memory.poub.prover.binder:Binder",
    SESSION_PROCESS: "verity.protocols.resource_accounting.memory.poub.prover.commit_session:CommitSession",
})


def prover(name: str) -> Any:
    """What `PROVER` names ``name``, its module imported on first use."""
    module, attribute = PROVER[name].split(":")
    return getattr(import_module(module), attribute)


def attach(exchange: Any, ctx: Mapping[str, Any]) -> Any:
    """`POUB` on a served run (`Scheme.attach`): the binder over ``ctx["store"]`` (a `NodeMemory`), built with
    ``ctx["binder"]``'s keywords, installed on ``exchange``, which files its requests; returns the binder."""
    return prover("binder")(ctx["store"], **ctx.get("binder", {})).install(exchange)


def session(verifier: str, **kw: Any) -> Any:
    """The process `SESSION_PROCESS`: the Commit's session at ``verifier`` (``HOST:PORT``), `prover.commit_session`'s
    `CommitSession` with its keywords. An adapter hands it ``exchange``, a call that gives the exchange of the build
    about to be made, and drives what it gets back."""
    return prover(SESSION_PROCESS)(verifier, **kw)


__all__ = ["BANDWIDTH", "CHARGES", "DEADLINE_NS", "DELTA_SCHEME", "DELTA_SKETCH", "GOAL", "POUB", "PROVER",
           "REFRESH_GOAL", "REUSE_OLDEST_FREED", "SESSION", "SESSION_PROCESS", "attach", "delta_sketch", "prover",
           "requests", "session"]
