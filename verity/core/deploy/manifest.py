"""A protocol's manifest, and the placement check.

The manifest is a typed object in each protocol's package (`render` writes it as TOML, which is only a rendering). Its
`Role`s and `Channel`s are also what a protocol declares to a deployment configuration (`config.Component`):

- its parties (`Role`): each one's principal (`developer`, `auditor` or `anyone`), its cardinality (`per` server, link,
  site, or `one`), and what its machine must provide (`needs`: `clock` and `coins` are sources the runtime attaches, the
  rest capabilities such as `gpu` or `signing-key`);
- its channels (`Channel`): who may send which messages to whom, under what latency bound, whether they carry coins,
  and the messages sent before them (`after`);
- its messages: the wire schemas (`verity.core.wire.Schema`), each with its kind;
- what it needs of the workload (`needs`: `windows`, `store`, ...), which a workload's `provides` must cover
  (`workload`).

`place` checks a binding of the parties to a site's machines: each party's principal owns its machine (`principal`), the
machine provides what the party needs (`needs`), a coin-carrying channel passes no developer-owned machine (`coins-hop`),
and each channel's link meets its latency bound (`latency`). It refuses anything else by name, or, in a `simulated`
placement, lets those four stand as the premises the run assumes and lists them (`Placement.unmet`). A binding that
names an unknown party or machine, leaves a party unbound, binds a `per = "one"` party twice or puts a channel between
machines with no link is refused in either mode.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from verity.core.wire import NAME, Refused, Registry, Schema

PRINCIPALS = ("developer", "auditor", "anyone")
OWNERS = ("developer", "auditor")
PER = ("server", "link", "site", "one")
SOURCES = ("clock", "coins")
"""Needs the runtime meets with a source party on the same machine, on a channel of that name."""
STRUCTURAL = ("unknown-party", "unknown-machine", "unbound", "cardinality", "no-link")
PREMISES = ("principal", "needs", "coins-hop", "latency")
"""What a `simulated` placement may leave unmet: the premises a run on a real site would rest on."""


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
    needs: tuple[str, ...] = ()

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
        Registry(self.messages)

    @property
    def registry(self) -> Registry:
        return Registry(self.messages)

    def role(self, name: str) -> Role:
        r = next((r for r in self.parties if r.name == name), None)
        if r is None:
            raise Refused("unknown-party", f"{self.protocol} has no party {name!r}: its parties are "
                                           f"{[r.name for r in self.parties]}")
        return r

    def channel(self, sender: str, receiver: str) -> Channel | None:
        return next((c for c in self.channels if (c.sender, c.receiver) == (sender, receiver)), None)

    def sends(self, sender: str, receiver: str) -> Registry:
        """The messages `sender` may send `receiver`: what the runtime lets it send, and what `receiver` accepts."""
        c = self.channel(sender, receiver)
        return self.registry.only(c.messages if c else ())

    def peers(self, party: str) -> tuple[str, ...]:
        return tuple(sorted({c.receiver for c in self.channels if c.sender == party}
                            | {c.sender for c in self.channels if c.receiver == party}))


def workload(manifest: Manifest, provides: Iterable[str]) -> None:
    """Refuse a workload that doesn't provide what the protocol needs of it (`workload-needs`)."""
    missing = sorted(set(manifest.needs) - set(provides))
    if missing:
        raise Refused("workload-needs", f"{manifest.protocol} needs {missing} of the workload, which doesn't provide them")


def _toml(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return "[" + ", ".join(_toml(x) for x in v) + "]"


def render(m: Manifest) -> str:
    """The manifest as TOML, for reading: the typed object is the manifest."""
    out = [f"protocol = {_toml(m.protocol)}", f"version = {m.version}", f"params = {_toml(m.params)}"]
    if m.needs:
        out.append(f"needs = {_toml(m.needs)}")
    for r in m.parties:
        out += ["", f"[party.{r.name}]", f"principal = {_toml(r.principal)}", f"per = {_toml(r.per)}",
                f"needs = {_toml(r.needs)}"]
    for c in m.channels:
        out += ["", "[[channel]]", f"from = {_toml(c.sender)}", f"to = {_toml(c.receiver)}",
                f"messages = {_toml(c.messages)}"]
        if c.latency_ms is not None:
            out.append(f"latency_ms = {c.latency_ms}")
        if c.carries_coins:
            out.append("carries_coins = true")
        if c.after:
            out.append(f"after = {_toml(c.after)}")
    for s in m.messages:
        out += ["", f"[message.{s.type}]", f"version = {s.version}", f"kind = {_toml(s.kind)}"]
    return "\n".join(out) + "\n"


# --- placement ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Machine:
    name: str
    owner: str
    provides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.owner not in OWNERS:
            raise ValueError(f"{self.name}: a machine's owner is one of {OWNERS}, not {self.owner!r}")


@dataclass(frozen=True)
class Link:
    """A link between two machines, through the machines `via`, at `latency_ms`."""

    ends: tuple[str, str]
    latency_ms: int
    via: tuple[str, ...] = ()


@dataclass(frozen=True)
class Site:
    machines: tuple[Machine, ...]
    links: tuple[Link, ...] = ()

    def machine(self, name: str) -> Machine:
        m = next((m for m in self.machines if m.name == name), None)
        if m is None:
            raise Refused("unknown-machine", f"the site has no machine {name!r}")
        return m

    def link(self, a: str, b: str) -> Link | None:
        if a == b:
            return Link((a, b), 0)
        return next((k for k in self.links if set(k.ends) == {a, b}), None)


@dataclass(frozen=True)
class Placement:
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


def place(manifest: Manifest, site: Site, bindings: Mapping[str, str], *, simulated: bool = False) -> Placement:
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
                    unmet.append(("latency", f"{c.sender} -> {c.receiver} is bounded by {c.latency_ms} ms, and its link "
                                             f"takes {link.latency_ms} ms"))
                hops = [v for v in link.via if site.machine(v).owner == "developer"]
                if c.carries_coins and hops:
                    unmet.append(("coins-hop", f"{c.sender} -> {c.receiver} carries coins through the developer's {hops}"))
    if unmet and not simulated:
        name, detail = unmet[0]
        raise Refused(name, detail)
    return Placement("simulated" if simulated else "placed", dict(bindings), tuple(unmet))


def site_of(machines: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any]] = ()) -> Site:
    """A site from its description as data: `{"name", "owner", "provides"}` a machine, `{"ends", "latency_ms", "via"}` a
    link."""
    return Site(tuple(Machine(m["name"], m["owner"], tuple(m.get("provides", ()))) for m in machines),
                tuple(Link(tuple(k["ends"]), k["latency_ms"], tuple(k.get("via", ()))) for k in links))


__all__ = ["OWNERS", "PER", "PREMISES", "PRINCIPALS", "SOURCES", "STRUCTURAL", "Channel", "Link", "Machine", "Manifest",
           "Placement", "Role", "Site", "place", "render", "site_of", "workload"]
