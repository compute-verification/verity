"""The registration log in Python: `Core.Registration` (`interface/CoreProtocol/Registration.lean`), the type every
subprotocol and both composites read, entry for entry and refusal for refusal. The composition suite's difftest holds
this module to the Lean (`composition/tests/test_registration_difftest.py`).

A deployment's registration is one append-only log of entries, each naming the head of the log before it (`previous`):
the digest of the log's last entry, or its composite's genesis when the log is empty, so the head names the whole log.
An entry is one of two formats:

* `verity/registration/hardware/v1`: the cards, locations and links it adds to the site (`Site`);
* `verity/registration/information/v1`: its items, each registered once in the log (`item_key`): θ's hiding roots, the
  builder B (its digest and its public cap on one declaration's advice, in bits), each subprotocol's public Program and
  parameters by digest, and the proof system. Registration registers B, not the circuit it builds.

~~~text
entry digest   SHA-512(format ‖ 0 ‖ canon(entry))                          the entry's format and previous among its keys
card digest    SHA-512("verity/registration/card/v1" ‖ 0 ‖ canon(card))
genesis        SHA-512("verity/registration/genesis/v1" ‖ 0 ‖ canon({"composite": NAME}))
the log        {"composite": "onsite" | "remote", "entries": [ENTRY, ...]}   committed; its digest is its head
~~~

`check` holds each entry to the log before it and raises `Refused` at the first entry that fails, by the first check it
fails (`REFUSALS`, in order). A component reads its view of an accepted log (`View`): the site, θ, the builder, the
proof system and its own Program and parameters (`View.program`). Decoding takes exactly what the encoders write and
raises `Malformed` otherwise.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from verity.core.wire import canonical

HARDWARE = "verity/registration/hardware/v1"
INFORMATION = "verity/registration/information/v1"
CARD = "verity/registration/card/v1"
GENESIS = "verity/registration/genesis/v1"
PROTOCOL = "verity/registration/v1"
OUTSIDE = "outside"
"""The rest of the world: an end every site has, whose id no location may take."""
COMPOSITES = ("onsite", "remote")
KINDS = ("site", "host", "device", "memory", "certifier", "challenger", "outside")
TIERS = ("hbm", "dram", "disk")
OWNERS = ("prover", "auditor")
PARENTS: dict[str, tuple[str | None, ...]] = {
    "site": (None,), "host": ("site",), "device": ("host",), "memory": ("device", "host"), "certifier": ("site",),
    "challenger": ("site",), "outside": ()}
"""The kinds of parent a location of each kind may have, None being the root."""
LINKS = (("host", "host"), ("host", "outside"), ("certifier", "host"), ("certifier", "outside"), ("challenger", "host"))
"""The pairs of kinds a link may join, in either order."""
AUDITOR_KINDS = ("certifier", "challenger")
REFUSALS = ("previous", "duplicate-id", "unknown-ref", "kind", "card-digest", "capacity", "owner", "uncertified",
            "challenger-reach", "duplicate-information")


class Malformed(ValueError):
    """Bytes or JSON that aren't what the encoders write."""


class Refused(Exception):
    """The check refuses entry `entry` by `name`, one of `REFUSALS`."""

    def __init__(self, entry: int, name: str) -> None:
        super().__init__(f"entry {entry}: {name}")
        self.entry, self.name = entry, name


def tagged(tag: str, obj: Any) -> str:
    """`SHA-512(tag ‖ 0 ‖ canon(obj))` as hex."""
    return hashlib.sha512(tag.encode() + b"\0" + canonical(obj)).hexdigest()


@dataclass(frozen=True)
class Card:
    """An SKU's data-sheet card: tensor-core multiply-adds a tick, bits from device memory to the chip a tick, bits of
    on-chip state and bytes of device memory."""

    sku: str
    tc: int
    bw: int
    chip: int
    mem: int

    def to_json(self) -> dict[str, Any]:
        return {"bw": self.bw, "chip": self.chip, "mem": self.mem, "sku": self.sku, "tc": self.tc}

    def digest(self) -> str:
        return tagged(CARD, self.to_json())


@dataclass(frozen=True)
class Loc:
    """A location and its owner: a device names its card's digest, a memory leaf its tier and bytes, the auditor's
    devices their keys."""

    id: str
    kind: str
    parent: str | None = None
    owner: str = "prover"
    card: str = ""
    tier: str | None = None
    bytes: int = 0
    key: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "card": self.card, "id": self.id, "key": self.key, "kind": self.kind,
                "owner": self.owner, "parent": self.parent, "tier": self.tier}


@dataclass(frozen=True)
class Link:
    """Two ends, a capacity `beta` in bits a tick each way and a latency in ticks."""

    id: str
    ends: tuple[str, str]
    beta: int
    latency: int

    def to_json(self) -> dict[str, Any]:
        return {"beta": self.beta, "ends": list(self.ends), "id": self.id, "latency": self.latency}


@dataclass(frozen=True)
class Hardware:
    cards: tuple[Card, ...] = ()
    locs: tuple[Loc, ...] = ()
    links: tuple[Link, ...] = ()


@dataclass(frozen=True)
class Information:
    """The items an information entry registers, each canonical JSON with its `kind`: `theta` (`weights`, `state`),
    `builder` (`digest`, `cap`), `program` (`protocol`, `program`, `params`) or `proof-system` (`id`, `digest`)."""

    items: tuple[Mapping[str, Any], ...] = ()


def item_key(item: Mapping[str, Any]) -> str:
    """What an item registers, which the log registers once."""
    return "program " + item["protocol"] if item["kind"] == "program" else item["kind"]


@dataclass(frozen=True)
class Entry:
    previous: str
    body: Hardware | Information

    @property
    def format(self) -> str:
        return HARDWARE if isinstance(self.body, Hardware) else INFORMATION

    def to_json(self) -> dict[str, Any]:
        b = self.body
        if isinstance(b, Hardware):
            return {"cards": [k.to_json() for k in b.cards], "format": HARDWARE,
                    "links": [k.to_json() for k in b.links], "locs": [x.to_json() for x in b.locs],
                    "previous": self.previous}
        return {"format": INFORMATION, "items": [dict(i) for i in b.items], "previous": self.previous}

    def digest(self) -> str:
        return tagged(self.format, self.to_json())


def genesis(composite: str) -> str:
    """The head of a composite's empty log."""
    return tagged(GENESIS, {"composite": composite})


def head(composite: str, log: Sequence[Entry]) -> str:
    """The log's head, which names it."""
    return log[-1].digest() if log else genesis(composite)


def chain(composite: str, bodies: Sequence[Hardware | Information]) -> list[Entry]:
    """The log of `bodies` in order, each entry following the head of the log before it."""
    log: list[Entry] = []
    for b in bodies:
        log.append(Entry(head(composite, log), b))
    return log


def committed(composite: str, log: Sequence[Entry]) -> bytes:
    """The log as committed: the canonical JSON the protocol's commitment is."""
    return canonical({"composite": composite, "entries": [e.to_json() for e in log]})


# --- decoding: exactly what the encoders write ------------------------------------------------------------------------

def _exactly(j: Any, keys: Sequence[str], what: str) -> None:
    if not isinstance(j, dict) or sorted(j) != list(keys):
        raise Malformed(f"{what}: the keys must be [{', '.join(keys)}]")


def _str(j: Any, what: str) -> str:
    if not isinstance(j, str):
        raise Malformed(f"{what}: not a string")
    return j


def _nat(j: Any, what: str) -> int:
    if not isinstance(j, int) or isinstance(j, bool) or j < 0:
        raise Malformed(f"{what}: not a natural number")
    return j


def _list(j: Any, what: str) -> list:
    if not isinstance(j, list):
        raise Malformed(f"{what}: not a list")
    return j


def _one_of(j: Any, names: Sequence[str], what: str) -> str:
    if _str(j, what) not in names:
        raise Malformed(f"{what}: {j} is not one of {list(names)}")
    return j


def _hex64(j: Any, what: str) -> str:
    """`Flock.unhex` of 64 bytes: lowercase hex of a SHA-512 digest."""
    s = _str(j, what)
    if len(s) % 2 or not set(s) <= set("0123456789abcdef"):
        raise Malformed(f"{what}: not lowercase hex")
    if len(s) != 128:
        raise Malformed(f"{what}: not a SHA-512 digest (64 bytes)")
    return s


def _optional(f: Callable[[Any], Any], j: Any) -> Any:
    return None if j is None else f(j)


def card_of(j: Any) -> Card:
    _exactly(j, ["bw", "chip", "mem", "sku", "tc"], "a card")
    return Card(_str(j["sku"], "sku"), _nat(j["tc"], "tc"), _nat(j["bw"], "bw"), _nat(j["chip"], "chip"),
                _nat(j["mem"], "mem"))


def loc_of(j: Any) -> Loc:
    _exactly(j, ["bytes", "card", "id", "key", "kind", "owner", "parent", "tier"], "a location")
    return Loc(_str(j["id"], "id"), _one_of(j["kind"], KINDS, "kind"),
               _optional(lambda p: _str(p, "parent"), j["parent"]), _one_of(j["owner"], OWNERS, "owner"),
               _str(j["card"], "card"), _optional(lambda t: _one_of(t, TIERS, "tier"), j["tier"]),
               _nat(j["bytes"], "bytes"), _str(j["key"], "key"))


def link_of(j: Any) -> Link:
    _exactly(j, ["beta", "ends", "id", "latency"], "a link")
    ends = _list(j["ends"], "ends")
    if len(ends) != 2:
        raise Malformed("a link: ends: not two ends")
    return Link(_str(j["id"], "id"), (_str(ends[0], "an end"), _str(ends[1], "an end")), _nat(j["beta"], "beta"),
                _nat(j["latency"], "latency"))


ITEM_KEYS = {"theta": ["kind", "state", "weights"], "builder": ["cap", "digest", "kind"],
             "program": ["kind", "params", "program", "protocol"], "proof-system": ["digest", "id", "kind"]}


def item_of(j: Any) -> dict[str, Any]:
    if not isinstance(j, dict) or "kind" not in j:
        raise Malformed("an item: kind missing")
    kind = _str(j["kind"], "an item's kind")
    if kind not in ITEM_KEYS:
        raise Malformed(f"an item: kind {kind} is not one of {list(ITEM_KEYS)}")
    _exactly(j, ITEM_KEYS[kind], f"a {kind} item")
    if kind == "theta":
        return {"kind": kind, "state": _hex64(j["state"], "state"), "weights": _hex64(j["weights"], "weights")}
    if kind == "builder":
        return {"cap": _nat(j["cap"], "cap"), "digest": _hex64(j["digest"], "digest"), "kind": kind}
    if kind == "program":
        return {"kind": kind, "params": _hex64(j["params"], "params"), "program": _hex64(j["program"], "program"),
                "protocol": _str(j["protocol"], "protocol")}
    return {"digest": _hex64(j["digest"], "digest"), "id": _str(j["id"], "id"), "kind": kind}


def entry_of(j: Any) -> Entry:
    if not isinstance(j, dict) or "format" not in j:
        raise Malformed("an entry: format missing")
    fmt = _str(j["format"], "an entry's format")
    if fmt == HARDWARE:
        _exactly(j, ["cards", "format", "links", "locs", "previous"], "a hardware entry")
        previous = _hex64(j["previous"], "previous")
        return Entry(previous, Hardware(tuple(card_of(k) for k in _list(j["cards"], "cards")),
                                        tuple(loc_of(x) for x in _list(j["locs"], "locs")),
                                        tuple(link_of(k) for k in _list(j["links"], "links"))))
    if fmt == INFORMATION:
        _exactly(j, ["format", "items", "previous"], "an information entry")
        previous = _hex64(j["previous"], "previous")
        return Entry(previous, Information(tuple(item_of(i) for i in _list(j["items"], "items"))))
    raise Malformed(f"an entry: format {fmt} is not one of [{HARDWARE}, {INFORMATION}]")


def log_of(j: Any) -> tuple[str, list[Entry]]:
    """The committed log's composite and entries."""
    _exactly(j, ["composite", "entries"], "the log")
    return _one_of(j["composite"], COMPOSITES, "the log's composite"), [entry_of(e) for e in _list(j["entries"], "entries")]


# --- the site's reading -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    """The cards, locations and links the log's hardware entries add, in order."""

    cards: tuple[Card, ...] = ()
    locs: tuple[Loc, ...] = ()
    links: tuple[Link, ...] = ()

    def add(self, h: Hardware) -> Site:
        return Site(self.cards + h.cards, self.locs + h.locs, self.links + h.links)

    def loc(self, i: str) -> Loc | None:
        return next((x for x in self.locs if x.id == i), None)

    def kind(self, i: str) -> str | None:
        if i == OUTSIDE:
            return "outside"
        x = self.loc(i)
        return None if x is None else x.kind

    def known(self, i: str) -> bool:
        return i == OUTSIDE or any(x.id == i for x in self.locs)

    def ids(self) -> list[str]:
        return [x.id for x in self.locs] + [k.id for k in self.links] + [OUTSIDE]

    def card(self, x: Loc) -> Card | None:
        return next((k for k in self.cards if k.digest() == x.card), None)

    def held(self, i: str) -> int:
        return sum(x.bytes for x in self.locs if x.kind == "memory" and x.parent == i)

    def joins(self, k: Link, a: str, b: str) -> bool:
        return (self.kind(k.ends[0]), self.kind(k.ends[1])) in ((a, b), (b, a))

    def boundary(self, k: Link) -> bool:
        return self.joins(k, "host", "outside") or self.joins(k, "certifier", "host")

    def certified(self, k: Link) -> bool:
        return "certifier" in (self.kind(k.ends[0]), self.kind(k.ends[1]))

    def far(self, i: str) -> list[str]:
        return [e for k in self.links if i in k.ends for e in k.ends if e != i]

    def host(self, x: Loc) -> str | None:
        if x.parent is None:
            return None
        if self.kind(x.parent) == "host":
            return x.parent
        p = self.loc(x.parent)
        return None if p is None else p.parent

    def reached(self) -> list[str]:
        return [e for k in self.links if "challenger" in (self.kind(k.ends[0]), self.kind(k.ends[1])) for e in k.ends]

    def to_json(self) -> dict[str, Any]:
        return {"cards": [k.to_json() for k in self.cards], "links": [k.to_json() for k in self.links],
                "locs": [x.to_json() for x in self.locs]}


# --- the check --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Auditor:
    """The auditor's data sheet for each SKU, and the key of each of its own devices."""

    sheets: tuple[Card, ...] = ()
    keys: tuple[tuple[str, str], ...] = ()

    def sheet(self, sku: str) -> Card | None:
        return next((k for k in self.sheets if k.sku == sku), None)

    def key(self, i: str) -> str | None:
        return next((k for d, k in self.keys if d == i), None)


def _nodup(xs: Sequence[str]) -> bool:
    return len(set(xs)) == len(xs)


def hardware_refusal(auditor: Auditor, composite: str, site: Site, h: Hardware) -> str | None:
    """The first check a hardware entry adding `h` to `site` fails, or None."""
    t = site.add(h)
    digests = [k.digest() for k in t.cards]
    if not (_nodup(t.ids()) and _nodup(digests)):
        return "duplicate-id"
    if not (all(x.parent is None or t.known(x.parent) for x in h.locs)
            and all(t.known(k.ends[0]) and t.known(k.ends[1]) for k in h.links)):
        return "unknown-ref"
    if not (all((None if x.parent is None else t.kind(x.parent)) in PARENTS[x.kind] and (x.kind == "memory") == (x.tier is not None)
                for x in h.locs)
            and all(k.ends[0] != k.ends[1] and any(t.joins(k, a, b) for a, b in LINKS) and k.beta > 0 for k in h.links)
            and all(x.kind != "certifier" or [t.kind(e) for e in t.far(x.id)] in (["host", "outside"], ["outside", "host"])
                    for x in t.locs)):
        return "kind"
    if not (all(auditor.sheet(k.sku) == k for k in h.cards)
            and all(x.kind != "device" or x.card in digests for x in h.locs)):
        return "card-digest"
    if not (all(x.kind != "memory" or x.bytes > 0 for x in h.locs)
            and all(x.kind != "device" or (k := t.card(x)) is None or t.held(x.id) <= k.mem for x in t.locs)):
        return "capacity"
    if not all((x.owner == "auditor" and auditor.key(x.id) == x.key) if x.kind in AUDITOR_KINDS else x.owner == "prover"
               for x in h.locs):
        return "owner"
    if composite == "onsite" and not all(not t.boundary(k) or t.certified(k) for k in t.links):
        return "uncertified"
    if composite == "onsite":
        reached = t.reached()
        if not all(x.kind != "memory" or (g := t.host(x)) is None or g in reached for x in t.locs):
            return "challenger-reach"
    return None


@dataclass(frozen=True)
class State:
    """The log so far, folded: its site, its information items in order, its head and its length."""

    site: Site
    items: tuple[Mapping[str, Any], ...]
    head: str
    entries: int

    def refusal(self, auditor: Auditor, composite: str, e: Entry) -> str | None:
        if e.previous != self.head:
            return "previous"
        if isinstance(e.body, Hardware):
            return hardware_refusal(auditor, composite, self.site, e.body)
        return None if _nodup([item_key(i) for i in self.items + e.body.items]) else "duplicate-information"

    def add(self, e: Entry) -> State:
        if isinstance(e.body, Hardware):
            return replace(self, site=self.site.add(e.body), head=e.digest(), entries=self.entries + 1)
        return replace(self, items=self.items + e.body.items, head=e.digest(), entries=self.entries + 1)

    def view(self) -> View:
        first = lambda kind: next((dict(i) for i in self.items if i["kind"] == kind), None)  # noqa: E731
        return View(self.site, first("theta"), first("builder"), first("proof-system"),
                    tuple(dict(i) for i in self.items if i["kind"] == "program"), self.head)


def check(auditor: Auditor, composite: str, log: Sequence[Entry]) -> State:
    """The log folded entry by entry, each held to the log before it; `Refused` at the first entry that fails."""
    s = State(Site(), (), genesis(composite), 0)
    for e in log:
        if r := s.refusal(auditor, composite, e):
            raise Refused(s.entries, r)
        s = s.add(e)
    return s


@dataclass(frozen=True)
class View:
    """What a component reads of an accepted log."""

    site: Site
    theta: dict[str, Any] | None
    builder: dict[str, Any] | None
    proof_system: dict[str, Any] | None
    programs: tuple[dict[str, Any], ...] = field(default=())
    head: str = ""

    def program(self, protocol: str) -> dict[str, Any] | None:
        """The subprotocol `protocol`'s registered Program and parameters."""
        return next((p for p in self.programs if p["protocol"] == protocol), None)

    def to_json(self) -> dict[str, Any]:
        return {"builder": self.builder, "programs": list(self.programs), "proof_system": self.proof_system,
                "site": self.site.to_json(), "theta": self.theta}


# --- the core protocol ------------------------------------------------------------------------------------------------

def params_of(j: Any) -> tuple[str, Auditor]:
    """The auditor's parameters: `{"composite", "keys": [{"id", "key"}], "sheets": [CARD]}`."""
    _exactly(j, ["composite", "keys", "sheets"], "the params")
    sheets = tuple(card_of(k) for k in _list(j["sheets"], "sheets"))
    keys = []
    for k in _list(j["keys"], "keys"):
        _exactly(k, ["id", "key"], "a key")
        keys.append((_str(k["id"], "id"), _str(k["key"], "key")))
    if not _nodup([k.sku for k in sheets]):
        raise Malformed("the params: a data sheet's SKU is named twice")
    if not _nodup([d for d, _ in keys]):
        raise Malformed("the params: a device's key is named twice")
    return _one_of(j["composite"], COMPOSITES, "the params' composite"), Auditor(sheets, tuple(keys))


def params(composite: str, auditor: Auditor) -> dict[str, Any]:
    return {"composite": composite, "keys": [{"id": d, "key": k} for d, k in auditor.keys],
            "sheets": [k.to_json() for k in auditor.sheets]}


def check_log(p: tuple[str, Auditor], c: tuple[str, list[Entry]]) -> tuple:
    """`Core.Registration.checkLog`: `("accept", public, profile)` or `("reject", reason)`."""
    (composite, auditor), (theirs, log) = p, c
    if theirs != composite:
        return "reject", f"composite: the log is {theirs}'s, not {composite}'s"
    try:
        s = check(auditor, composite, log)
    except Refused as r:
        return "reject", str(r)
    return "accept", s.view().to_json(), {"entries": s.entries}
