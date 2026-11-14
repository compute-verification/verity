"""A protocol's manifest, and the placement check.

The manifest is a typed object in each protocol's package. Its `Role`s and `Channel`s are also what a protocol declares
to a deployment configuration (`config.Component`), which is where it says what it needs of the workload:

- its parties (`Role`): each one's principal (`developer`, `auditor` or `anyone`), its cardinality (`per` server, link,
  site, or `one`), and what its machine must provide (`needs`: `clock` and `coins` are sources the runtime attaches, the
  rest capabilities such as `gpu` or `signing-key`);
- its channels (`Channel`): who may send which messages to whom, under what latency bound, whether they carry coins,
  and the messages sent before them (`after`);
- its messages: the wire schemas (`verity.channel.Schema`), each with its kind.

`ProtocolDef` is the protocol as a run takes it: its manifest with a factory for each of its parties (`party.Party`).

`place` checks a binding of the parties to a site's machines: each party's principal owns its machine (`principal`), the
machine provides what the party needs (`needs`), a coin-carrying channel passes no developer-owned machine (`coins-hop`),
and each channel's link meets its latency bound (`latency`). It refuses anything else by name, or, in a `simulated`
placement, lets those four stand as the premises the run assumes and lists them (`PlaceResult.unmet`). A binding that
names an unknown party or machine, leaves a party unbound, binds a `per = "one"` party twice or puts a channel between
machines with no link is refused in either mode.

Monitoring periods: ticks, spans of ticks and the monitoring period, the Python mirror of the code's
`CoreProtocol.Period` (`verity/protocols/interface/CoreProtocol/Period.lean`), which is the code's copy of
`security/Definitions/Core/Period.lean`. The difftest port `core-periods-difftest` holds this section to that Lean
(`verity/tests/tests/test_periods_difftest.py`).

- `TickSpan(start, len)`: the ticks ``[start, start + len)`` on the one clock (`OneClock`), ending at `stop`. `ticks` is
  their range, and `within` says one span lies inside another.
- `Cut(span, boundaries)`: a span cut at its boundaries, from its start to its stop, each no later than the next. Its
  `parts` are the spans between consecutive boundaries.
- `MonitoringPeriod(span, execution, sampling)`: a span, its cut into its execution blocks' spans, and for each execution
  block its span's cut into its sampling blocks' spans. `MonitoringPeriod.of` builds one from boundaries.
- The reporting grid every monitoring-period report states: `MONITORING_PERIODS`, the monitoring periods a result is
  presented for (week, month, year), each with its length in seconds, and `DELTAS`, the δs its bounds are stated at
  (1/100 and 10⁻⁶). PoUS's wipe table, PoUB's session table and the block record each pool their figures over every
  period at every δ.
- `block_count(seconds, block_s, machines)`: the blocks a monitoring period of `seconds` holds over `machines` machines,
  each audited in whole `block_s`-second blocks: the one count every period table scales a block's figures by.

The JSON is the Lean's: a span is ``{"len": n, "start": n}`` and a monitoring period ``{"execution": [τ, …], "sampling":
[[τ, …], …], "span": SPAN}``. `from_json` accepts exactly what `to_json` writes. Anything that isn't a span, a cut or a
monitoring period raises `PeriodError`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from verity.channel import NAME, MessageTypes, Refused, Schema

if TYPE_CHECKING:
    from .party import Party

PRINCIPALS = ("developer", "auditor", "anyone")
OWNERS = ("developer", "auditor")
PER = ("server", "link", "site", "one")
SOURCES = ("clock", "coins")
"""Needs the runtime meets with a source party on the same machine, on a channel of that name."""


@dataclass(frozen=True)
class Role:
    """A party of the manifest."""

    name: str
    principal: str
    per: str = "one"
    needs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not NAME.fullmatch(self.name) or self.name in SOURCES:
            raise ValueError(f"a party's name is a name ({NAME.pattern}) other than {SOURCES}, not {self.name!r}")
        if self.principal not in PRINCIPALS or self.per not in PER:
            raise ValueError(f"{self.name}: a principal is one of {PRINCIPALS} and per one of {PER}")


@dataclass(frozen=True)
class Channel:
    """Messages `sender` may send `receiver`, in a link whose latency is at most `latency_ms`, after the messages
    `after`."""

    sender: str
    receiver: str
    messages: tuple[str, ...]
    latency_ms: int | None = None
    carries_coins: bool = False
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Manifest:
    protocol: str
    version: int
    params: str
    parties: tuple[Role, ...]
    channels: tuple[Channel, ...]
    messages: tuple[Schema, ...]

    def __post_init__(self) -> None:
        names = [r.name for r in self.parties]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.protocol} names a party twice: {names}")
        types = {s.type for s in self.messages}
        pairs = set()
        for c in self.channels:
            if c.sender not in names or c.receiver not in names or c.sender == c.receiver:
                raise ValueError(f"{self.protocol}: a channel joins two of its parties, not {c.sender} to {c.receiver}")
            if (c.sender, c.receiver) in pairs:
                raise ValueError(f"{self.protocol}: one channel from {c.sender} to {c.receiver}")
            pairs.add((c.sender, c.receiver))
            if not set(c.messages) | set(c.after) <= types:
                raise ValueError(f"{self.protocol}: {sorted(set(c.messages + c.after) - types)} on {c.sender} -> "
                                 f"{c.receiver} are not its messages")
        MessageTypes(self.messages)

    @property
    def registry(self) -> MessageTypes:
        return MessageTypes(self.messages)

    def role(self, name: str) -> Role:
        r = next((r for r in self.parties if r.name == name), None)
        if r is None:
            raise Refused("unknown-party", f"{self.protocol} has no party {name!r}: its parties are "
                                           f"{[r.name for r in self.parties]}")
        return r

    def channel(self, sender: str, receiver: str) -> Channel | None:
        return next((c for c in self.channels if (c.sender, c.receiver) == (sender, receiver)), None)

    def sends(self, sender: str, receiver: str) -> MessageTypes:
        """The messages `sender` may send `receiver`: what the runtime lets it send, and what `receiver` accepts."""
        c = self.channel(sender, receiver)
        return self.registry.only(c.messages if c else ())

    def peers(self, party: str) -> tuple[str, ...]:
        return tuple(sorted({c.receiver for c in self.channels if c.sender == party}
                            | {c.sender for c in self.channels if c.receiver == party}))


@dataclass(frozen=True)
class ProtocolDef:
    """A protocol as a run takes it: its manifest, and a factory for each of its parties."""

    manifest: Manifest
    parties: Mapping[str, Callable[[], Party]]


# --- placement ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Host:
    """A machine a party can be placed on: its owner and what it provides."""

    name: str
    owner: str
    provides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.owner not in OWNERS:
            raise ValueError(f"{self.name}: a machine's owner is one of {OWNERS}, not {self.owner!r}")


@dataclass(frozen=True)
class Route:
    """A link between two machines, through the machines `via`, at `latency_ms`."""

    ends: tuple[str, str]
    latency_ms: int
    via: tuple[str, ...] = ()


@dataclass(frozen=True)
class Topology:
    """A site's machines and the routes between them, which `place` binds parties to."""

    machines: tuple[Host, ...]
    links: tuple[Route, ...] = ()

    def machine(self, name: str) -> Host:
        m = next((m for m in self.machines if m.name == name), None)
        if m is None:
            raise Refused("unknown-machine", f"the site has no machine {name!r}")
        return m

    def link(self, a: str, b: str) -> Route | None:
        if a == b:
            return Route((a, b), 0)
        return next((k for k in self.links if set(k.ends) == {a, b}), None)


@dataclass(frozen=True)
class PlaceResult:
    """The check's result: `placed`, or `simulated` with the premises it leaves unmet as (name, detail)."""

    mode: str
    bindings: Mapping[str, str]
    unmet: tuple[tuple[str, str], ...] = field(default=())

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode, "bindings": dict(sorted(self.bindings.items())),
                "unmet": [{"refused": n, "detail": d} for n, d in self.unmet]}


def _role_of(instance: str) -> tuple[str, str]:
    role, _, scope = instance.partition("@")
    return role, scope


def place(manifest: Manifest, site: Topology, bindings: Mapping[str, str], *, simulated: bool = False) -> PlaceResult:
    """Check that `bindings` (party instance `ROLE` or `ROLE@SCOPE` -> machine) places `manifest`'s parties on `site`."""
    unmet: list[tuple[str, str]] = []
    instances: dict[str, list[tuple[str, str]]] = {r.name: [] for r in manifest.parties}
    for inst, machine in sorted(bindings.items()):
        role, scope = _role_of(inst)
        manifest.role(role)
        site.machine(machine)
        instances[role].append((scope, machine))
    for r in manifest.parties:
        got = instances[r.name]
        if not got:
            raise Refused("unbound", f"{manifest.protocol}'s party {r.name} is bound to no machine")
        if r.per == "one" and len(got) != 1:
            raise Refused("cardinality", f"{r.name} is one party, bound {len(got)} times")
        for scope, machine in got:
            m = site.machine(machine)
            if r.principal != "anyone" and m.owner != r.principal:
                unmet.append(("principal", f"{r.name} is the {r.principal}'s, on {machine}, which the {m.owner} owns"))
            missing = sorted(set(r.needs) - set(m.provides))
            if missing:
                unmet.append(("needs", f"{r.name} needs {missing}, which {machine} doesn't provide"))
    for c in manifest.channels:
        for s_scope, s_machine in instances[c.sender]:
            for r_scope, r_machine in instances[c.receiver]:
                if s_scope and r_scope and s_scope != r_scope:
                    continue
                link = site.link(s_machine, r_machine)
                if link is None:
                    raise Refused("no-link", f"{c.sender} on {s_machine} and {c.receiver} on {r_machine} have no link")
                if c.latency_ms is not None and link.latency_ms > c.latency_ms:
                    slow = (f"{c.sender} -> {c.receiver} is bounded by {c.latency_ms} ms, and its link takes "
                            f"{link.latency_ms} ms")
                    unmet.append(("latency", slow))
                hops = [v for v in link.via if site.machine(v).owner == "developer"]
                if c.carries_coins and hops:
                    unmet.append(("coins-hop", f"{c.sender} -> {c.receiver} carries coins through the developer's {hops}"))
    if unmet and not simulated:
        name, detail = unmet[0]
        raise Refused(name, detail)
    return PlaceResult("simulated" if simulated else "placed", dict(bindings), tuple(unmet))


def site_of(machines: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any]] = ()) -> Topology:
    """A site from its description as data: `{"name", "owner", "provides"}` a machine, `{"ends", "latency_ms", "via"}` a
    link."""
    return Topology(tuple(Host(m["name"], m["owner"], tuple(m.get("provides", ()))) for m in machines),
                tuple(Route(tuple(k["ends"]), k["latency_ms"], tuple(k.get("via", ()))) for k in links))


# --- monitoring periods -------------------------------------------------------------------------------------------------

#: The monitoring periods a result is presented for, each with its length in seconds.
MONITORING_PERIODS: tuple[tuple[str, int], ...] = (("week", 7 * 86400), ("month", 30 * 86400), ("year", 365 * 86400))
#: The δs a period's bounds are stated at.
DELTAS: tuple[Fraction, ...] = (Fraction(1, 100), Fraction(1, 10 ** 6))


class PeriodError(ValueError):
    """A span, cut or monitoring period that isn't one, or JSON that encodes none."""


def block_count(seconds: int, block_s: int, machines: int = 1) -> int:
    """The blocks a monitoring period of `seconds` holds over `machines` machines, each machine's span cut into whole
    `block_s`-second blocks: ``machines · ⌊seconds / block_s⌋``, the execution blocks of each machine's `MonitoringPeriod`
    times the machines (a remainder shorter than a block is no block)."""
    _nat(seconds, "seconds")
    _nat(machines, "machines")
    if _nat(block_s, "block_s") == 0:
        raise PeriodError("block_s: a block lasts at least a second")
    return machines * (seconds // block_s)


def _nat(v: Any, what: str) -> int:
    if type(v) is not int or v < 0:
        raise PeriodError(f"{what}: not a natural number")
    return v


def _object(j: Any, keys: tuple[str, ...], what: str) -> dict[str, Any]:
    if not isinstance(j, dict):
        raise PeriodError(f"{what}: not an object")
    if sorted(j) != list(keys):
        raise PeriodError(f"{what}: the keys must be {list(keys)}")
    return j


def _ticks(j: Any, what: str) -> tuple[int, ...]:
    if not isinstance(j, list):
        raise PeriodError(f"{what}: not a list")
    return tuple(_nat(t, what) for t in j)


@dataclass(frozen=True)
class TickSpan:
    """A span of ticks: the ticks ``[start, start + len)``."""

    start: int
    len: int

    def __post_init__(self) -> None:
        _nat(self.start, "start")
        _nat(self.len, "len")

    @property
    def stop(self) -> int:
        """The tick after the span's last."""
        return self.start + self.len

    @property
    def ticks(self) -> range:
        return range(self.start, self.stop)

    def within(self, outer: TickSpan) -> bool:
        """Whether this span starts no earlier than `outer` and stops no later."""
        return outer.start <= self.start and self.stop <= outer.stop

    def to_json(self) -> dict[str, int]:
        return {"len": self.len, "start": self.start}

    @classmethod
    def from_json(cls, j: Any) -> TickSpan:
        o = _object(j, ("len", "start"), "span")
        return cls(_nat(o["start"], "span.start"), _nat(o["len"], "span.len"))


@dataclass(frozen=True)
class Cut:
    """The span `span` cut at `boundaries`: the first its start, the last its stop, each no later than the next."""

    span: TickSpan
    boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        bs = tuple(_nat(b, "boundary") for b in self.boundaries)
        object.__setattr__(self, "boundaries", bs)
        if not bs or bs[0] != self.span.start or bs[-1] != self.span.stop or any(a > b for a, b in zip(bs, bs[1:])):
            raise PeriodError(f"the boundaries {list(bs)} don't cut {self.span}")

    @property
    def parts(self) -> tuple[TickSpan, ...]:
        """The spans between consecutive boundaries."""
        return tuple(TickSpan(a, b - a) for a, b in zip(self.boundaries, self.boundaries[1:]))


@dataclass(frozen=True)
class MonitoringPeriod:
    """A monitoring period: its span, the cut of its span into its execution blocks' spans, and for each execution block
    the cut of its span into its sampling blocks' spans."""

    span: TickSpan
    execution: Cut
    sampling: tuple[Cut, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling", tuple(self.sampling))
        if self.execution.span != self.span:
            raise PeriodError(f"the execution blocks cut {self.execution.span}, not the span {self.span}")
        if len(self.sampling) != len(self.execution.parts):
            raise PeriodError(f"{len(self.sampling)} sampling cuts for {len(self.execution.parts)} execution blocks")
        for j, (block, cut) in enumerate(zip(self.execution.parts, self.sampling)):
            if cut.span != block:
                raise PeriodError(f"execution block {j}'s sampling cut is of {cut.span}, not of its span {block}")

    @classmethod
    def of(cls, span: TickSpan, execution: Sequence[int], sampling: Sequence[Sequence[int]]) -> MonitoringPeriod:
        """The monitoring period of `span` whose execution blocks' boundaries are `execution`, and whose execution block
        `j`'s sampling blocks' boundaries are `sampling[j]`."""
        cut = Cut(span, tuple(execution))
        if len(sampling) != len(cut.parts):
            raise PeriodError(f"{len(sampling)} sampling cuts for {len(cut.parts)} execution blocks")
        return cls(span, cut, tuple(Cut(block, tuple(bs)) for block, bs in zip(cut.parts, sampling)))

    @property
    def execution_blocks(self) -> int:
        return len(self.sampling)

    def execution_span(self, j: int) -> TickSpan:
        """Execution block `j`'s span."""
        return self.execution.parts[j]

    def sampling_span(self, j: int, t: int) -> TickSpan:
        """Sampling block `t`'s span in execution block `j`."""
        return self.sampling[j].parts[t]

    def to_json(self) -> dict[str, Any]:
        return {"execution": list(self.execution.boundaries), "sampling": [list(c.boundaries) for c in self.sampling],
                "span": self.span.to_json()}

    @classmethod
    def from_json(cls, j: Any) -> MonitoringPeriod:
        o = _object(j, ("execution", "sampling", "span"), "monitoring period")
        span = TickSpan.from_json(o["span"])
        execution = _ticks(o["execution"], "monitoring period.execution")
        if not isinstance(o["sampling"], list):
            raise PeriodError("monitoring period.sampling: not a list")
        return cls.of(span, execution, [_ticks(bs, "monitoring period.sampling") for bs in o["sampling"]])


__all__ = ["DELTAS", "MONITORING_PERIODS", "Channel", "Cut", "Manifest", "MonitoringPeriod", "PeriodError", "PlaceResult",
           "ProtocolDef", "Role", "TickSpan", "Topology", "block_count", "place", "site_of"]
