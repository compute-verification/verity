"""The site log in Python: the registration log the Site is folded from, its site entries and the digests they name
things by, and the check each entry passes. It mirrors the spec's site log (`Site.Registration`, `Site.ofLog`, the
digests and `Registration.Refusal`) and its check (`Registration.Entry.refusal`, `Registration.check`), check for
check and in the same order, over the Site's types in `verity.device`.

The log (`Registration`) is append-only: each site entry adds sheets, locations and links and names the head of the log
before it (`Entry.previous`), the digest of its last entry or, for an empty log, its composite's genesis. `of_log`
is the site a log registers, and `check_registration` holds one entry to the log before it, refusing by the first check
it fails (`CHECKS`, the spec's `Site.Entry` checks by name and in its order, each with its refusal; `REFUSALS`, the
spec's `Registration.Refusal` in kebab case); `check` holds each entry so, then the whole log by `inventory` at its
end (`incomplete`):

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicate-id` | no id is registered twice, `outside`'s included, and no sheet twice |
| `unknown-ref` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows, so an auditor's device is at the site and holds only the auditor's devices, and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity or, where it says why no certifier splits it, at none; each attester splits exactly one uplink |
| `sheet-digest` | each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered sheet |
| `figures` | each new sheet's figures are within the auditor's tolerance of its estimate for the SKU (`DeviceFigures.accepted`), and its memory figures are the auditor's; each new memory leaf directly under a host states the auditor's figures for it (its host, a rate above 0, its bytes), and no other new location a rate; each new site states the site's unit of time, above 0, and no other new location one; each new host states the auditor's figures for it (its on-chip state and each tier's shared rate, none 0), and no other new location either |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory |
| `owner` | each new attester, challenger or auditor's device is the auditor's own, with its key and its key's scheme, one of `SCHEMES`; every other new location is the prover's and states no scheme; each new auditor's device states its time base, the auditor's for it, and the auditor's figure for its clock's drift, below 1 where there is one, and no other new location states a time base or a drift |
| `inventory` | the log, at its end: under each host, every memory leaf the auditor has figures for, at least its count of devices of each SKU it lists, and each way out it lists, as a link under that id with an end at the host at its rate |
| `uncertified` | onsite: every boundary link is certified or says why no certifier splits it |
| `challenger-reach` | onsite: the host of every memory leaf is at an end of a memory attester's link |

`Registration.node_entry` is a node's entry, read from its node file and cards (`verity.device`'s `load_node`,
`load_card`, `ways_out` and `leaves`). Digests are SHA-512 over canonical JSON (`verity.channel.canonical`) under the
registration log's tags (`SITE`, `GENESIS`).
"""


from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from verity.device import (AUDITOR_KINDS, COMPOSITES, LINKS, OUTSIDE, PARENTS, SCHEMES, Auditor, Link, Loc, Malformed,
                           Sheet, Site, _exactly, _list, _lookup, _Node, _one_of, _str, tagged)

SITE = "verity/registration/site/v2"
"""A site entry's format: the tag its digest is taken under."""
GENESIS = "verity/registration/genesis/v1"
REFUSALS = ("previous", "duplicate-id", "unknown-ref", "kind", "sheet-digest", "figures", "capacity", "owner",
            "inventory", "uncertified", "challenger-reach")
"""The refusals, in the spec's order: `check_registration` makes each check but `inventory` in this order, and
`check` makes `inventory` once, at the log's end."""
TIME_UNIT = Fraction(1, 10 ** 9)
"""The unit of time `Registration.node_entry` gives a site it adds, in seconds: a nanosecond."""
NO_LINE_RATE = "no line rate stated"
"""Why no attester splits a node's interface that reads no line rate: `Registration.node_entry`'s `unmediated`."""
STORAGE_PATH = "storage path"
"""Why no attester splits a node's storage path: `Registration.node_entry`'s `unmediated`."""


class Refused(Exception):
    """`check_registration` refuses the entry at index `entry` of its log by `name`, one of `REFUSALS`."""

    def __init__(self, name: str, detail: str, entry: int) -> None:
        super().__init__(f"entry {entry}: {name}: {detail}")
        self.name, self.detail, self.entry = name, detail, entry


def genesis(composite: str) -> str:
    """The head of a composite's empty log."""
    return tagged(GENESIS, {"composite": composite})


@dataclass(frozen=True)
class Entry:
    """A site entry of the registration log (`Registration.Entry.site`): the sheets, locations and links it adds, and
    the head of the log it follows (`previous`). It is checked as a whole against the log before it."""

    previous: str
    sheets: tuple[Sheet, ...] = ()
    locs: tuple[Loc, ...] = ()
    links: tuple[Link, ...] = ()

    def __post_init__(self) -> None:
        _str(self.previous, "an entry's previous")
        for f in ("sheets", "locs", "links"):
            object.__setattr__(self, f, tuple(getattr(self, f)))

    def to_json(self) -> dict[str, Any]:
        return {"format": SITE, "links": [k.to_json() for k in self.links], "locs": [x.to_json() for x in self.locs],
                "previous": self.previous, "sheets": [k.to_json() for k in self.sheets]}

    @classmethod
    def of_json(cls, j: Any) -> Entry:
        j = _exactly(j, ("format", "links", "locs", "previous", "sheets"), "an entry")
        if j["format"] != SITE:
            raise Malformed(f"an entry: format {j['format']!r} is not {SITE}")
        return cls(j["previous"], tuple(Sheet.of_json(k) for k in _list(j["sheets"], "sheets")),
                   tuple(Loc.of_json(x) for x in _list(j["locs"], "locs")),
                   tuple(Link.of_json(k) for k in _list(j["links"], "links")))

    def digest(self) -> str:
        return tagged(SITE, self.to_json())


@dataclass(frozen=True)
class Registration:
    """The registration log of a composite's site: its entries, oldest first. It is append-only: an entry is checked
    against the log before it (`check_registration`), and nothing after it changes it."""

    composite: str
    entries: tuple[Entry, ...] = ()

    def __post_init__(self) -> None:
        _one_of(self.composite, COMPOSITES, "the log's composite")
        object.__setattr__(self, "entries", tuple(self.entries))

    @property
    def head(self) -> str:
        """The digest of the last entry, or the composite's genesis for an empty log: it names the whole log."""
        return self.entries[-1].digest() if self.entries else genesis(self.composite)

    def prefix(self, n: int) -> Registration:
        return Registration(self.composite, self.entries[:n])

    def append(self, e: Entry) -> Registration:
        """The log with `e` appended, unchecked."""
        return Registration(self.composite, self.entries + (e,))

    def entry(self, sheets: Iterable[Sheet] = (), locs: Iterable[Loc] = (), links: Iterable[Link] = ()) -> Entry:
        """An entry adding `sheets`, `locs` and `links`, following this log's head."""
        return Entry(self.head, tuple(sheets), tuple(locs), tuple(links))

    def node_entry(self, node: Mapping[str, Any], card: Mapping[str, Any], host: str, site: str = "site",
                   attester: tuple[str, str, str] | None = None, *, host_card: Mapping[str, Any],
                   ways: Iterable[tuple[str, int | None, bool]], leaves: Any,
                   time_unit: Fraction = TIME_UNIT, attester_host: Loc | None = None) -> Entry:
        """**A node's entry**, following this log's head: what a node file's table (`load_node(kind)`),
        the tables of the GPU card and the host card it names (`load_card(sku)` each), the file's ways
        out (`ways`: `ways_out(node, host)`, the one reader of their rates) and its leaves
        (`leaves`: `leaves(node, host)`, the one reader of theirs) register as host `host` at site
        `site`. It adds the sheet `Sheet.of_file` gives the card, and the site's location at the unit of time
        `time_unit` in seconds, each only where the log has none; the host, stating its figures (`node_figures`): its
        on-chip state in bits and the rate each tier's leaves share; the node's `device.count` devices `host/gpuI`, each
        naming the sheet, with one `hbm` leaf `host/gpuI/hbm` holding the sheet's device memory, whole; for each
        `memory.NAME` of the file a leaf `host/NAME` under the host, at the tier and bytes it states and its rate in
        bits a second, in the file's order; and for each way out a link under the way's id from the host to the
        outside, at the way's bits a second, and latency 0. The file names no host, so the caller gives its id, and
        every id under it starts with that. The auditor's figures for the node are `node_figures` of the same tables,
        so `check` takes the entry against an auditor that holds them.

        A way out is an interface of the file's (`host/NAME`) or its storage path (`host/storage`: what the disks
        that are not the server's own carry off the node, at the rate that bounds the node's disks together). Its link
        is a boundary link that no attester splits, and it says what is so of a node none of whose ways out the
        auditor's hardware sits on. Two kinds say why no attester splits them (`Link.unmediated`, its text from the
        way's kind): an interface that reads no line rate, whose link states no capacity (`NO_LINE_RATE`), and the
        storage path, which no attester can sit on (`STORAGE_PATH`). A site with either is not mediated, and `onsite`
        takes their reasons. An interface with a rate says none, so `onsite` refuses it by `uncertified` until a
        network attester splits it. `ways` is held to the file: an entry for each `interface.NAME`, in the file's order, with
        a rate exactly where its table states one, then the storage path's, with its rate, exactly where the file has a
        `storage` table. `leaves` is held to it too: each `memory.NAME` by id, tier and bytes, in the file's order. The
        rates' values are the caller's, read by `ways_out` and `leaves`.

        **An entry always registers a link out.** A file that states no interface is refused: its entry would not say
        how the node reaches the outside.

        `attester` is the id, the key and the key's scheme of a network attester of the auditor's that stands in
        front of the node without splitting its interface (a process on another machine, say). It adds the network attester
        and the two links of a certified way out, host to attester (`host/ID`) and attester to outside (`ID/outside`), each at the interfaces' rates summed, beside the ways' own links, which
        stay. The storage path's rate is not in that sum and no attester is linked to the path: a network attester's links
        say nothing of it. A network attester on a node any of whose interfaces has no rate is refused, by its id, since its
        links' rate would be below what it splits (a missing rate is never read as 0, nor left out of a sum). A network attester splits one host's way out, so each node names its own, and the key and scheme are the
        ones the auditor holds for the id or `owner` refuses the entry. None of the auditor's other devices is added.

        `attester_host` is the auditor's device the network attester runs in (`auditor_host`'s location, which states its
        time base and its clock's drift): the entry adds it where the log has none, at `site`, and the network attester in
        it, whose clock is then that host's (`Site.clock`, `Site.drift`). With none the network attester is under the site,
        where it reads no clock and times nothing.

        A file that states no `dram` leaf or no `disk` leaf is refused: its entry would pass `check` for `remote` with
        a host that holds less memory than the node has, and whatever reads the site would take the node for smaller
        than it is."""
        n = _Node.read(node, host_card, host, ways, leaves)
        if attester_host is not None:
            if attester is None:
                raise Malformed("`attester_host` is the auditor's device a network certifier runs in, and no attester "
                                "is given; register the node without one, or name the network attester")
            if not isinstance(attester_host, Loc) or (attester_host.kind, attester_host.parent) != (
                    "auditor-host", site):
                raise Malformed(f"`attester_host` is an auditor's device at site {site} (`auditor_host`'s location), "
                                f"not {attester_host!r}")
        rates = [bw for _, bw, split in n.ways if split and bw is not None]
        unrated = [i for i, bw, split in n.ways if split and bw is None]
        if attester is not None and unrated:
            name = attester[0] if isinstance(attester, (tuple, list)) and attester else attester
            raise Malformed(f"network attester {name!r}: the interfaces {unrated} the node file states have no rate, "
                            "so its links' rate, the interfaces' summed, would be below what it splits: an interface "
                            "that reads no line rate takes no attester. Register the node without one")
        sheet = Sheet.of_file(card)
        if sheet.sku != n.sku:
            raise Malformed(f"the node's devices are {n.sku!r}, and the card given is {sheet.sku!r}'s")
        before, digest = of_log(self), sheet.digest()
        locs = [] if before.known(site) else [Loc(site, "site", time_unit=time_unit)]
        locs.append(Loc(host, "host", site, chip=n.chip, shared=n.shared))
        for gpu in (f"{host}/gpu{i}" for i in range(n.count)):
            locs += [Loc(gpu, "device", host, sheet=digest),
                     Loc(f"{gpu}/hbm", "memory", gpu, tier="hbm", bytes=sheet.mem)]
        locs += [Loc(i, "memory", host, tier=tier, bytes=size, bw=bw) for i, tier, size, bw in n.leaves]
        links = [Link(i, (host, OUTSIDE), bw, 0, STORAGE_PATH if not split else None if bw else NO_LINE_RATE)
                 for i, bw, split in n.ways]
        if attester is not None:
            if not (isinstance(attester, (tuple, list)) and len(attester) == 3):
                raise Malformed(f"`attester` is a network certifier's id, its key and the key's scheme, not "
                                f"{attester!r}")
            device, key, scheme = attester
            beta = sum(rates)
            if attester_host is not None and not before.known(attester_host.id):
                locs.append(attester_host)
            locs.append(Loc(device, "certifier", site if attester_host is None else attester_host.id, "auditor",
                            key=key, scheme=scheme))
            links += [Link(f"{host}/{device}", (host, device), beta, 0),
                      Link(f"{device}/{OUTSIDE}", (device, OUTSIDE), beta, 0)]
        return self.entry(() if any(k.digest() == digest for k in before.sheets) else (sheet,), locs, links)

    def to_json(self) -> dict[str, Any]:
        return {"composite": self.composite, "entries": [e.to_json() for e in self.entries]}

    @classmethod
    def of_json(cls, j: Any) -> Registration:
        j = _exactly(j, ("composite", "entries"), "the log")
        return cls(j["composite"], tuple(Entry.of_json(e) for e in _list(j["entries"], "entries")))



def add(s: Site, e: Entry) -> Site:
    """The site `e` extends `s` to, unchecked (`Site.add`)."""
    return Site(s.sheets + e.sheets, s.locs + e.locs, s.links + e.links)


def of_log(log: Registration | Sequence[Entry]) -> Site:
    """**The site of a log** (`Site.ofLog`): its entries' sheets, locations and links, in the log's order. The site of a
    prefix of the log is the site as registered when that prefix was the log."""
    s = Site()
    for e in log.entries if isinstance(log, Registration) else log:
        s = add(s, e)
    return s


def _dups(xs: Sequence[str]) -> list[str]:
    return sorted({x for x in xs if xs.count(x) > 1})


def _fresh(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """No id is registered twice, the outside's among them, and no sheet twice."""
    t = add(s, e)
    digests = [k.digest() for k in t.sheets]
    if dup := _dups(t.ids()) + [k.sku for k in t.sheets if digests.count(k.digest()) > 1]:
        return f"{dup} registered twice"
    return None


def _refs(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new location's parent and each new link's ends name something registered."""
    t = add(s, e)
    for x in e.locs:
        if x.parent is not None and not t.known(x.parent):
            return f"{x.id}'s parent {x.parent!r} isn't registered"
    for k in e.links:
        if bad := [n for n in k.ends if not t.known(n)]:
            return f"link {k.id} names {bad}, which aren't registered"
    return None


def _kinds(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Parents and tiers by kind, so an auditor's device is at the site and holds only the auditor's devices; links
    between kinds a link may join, each at a positive capacity or, where it says no certifier splits it, at none, and
    each attester splitting exactly one uplink."""
    t = add(s, e)
    for x in e.locs:
        if ((None if x.parent is None else t.kind(x.parent)) not in PARENTS[x.kind]
                or (x.kind == "memory") != (x.tier is not None)):
            return f"{x.kind} {x.id} can't sit under {x.parent!r}, or has a tier exactly when it isn't memory"
    for k in e.links:
        if (k.ends[0] == k.ends[1] or not any(t.joins(k, p, q) for p, q in LINKS) or k.beta == 0
                or (k.beta is None and k.unmediated is None)):
            return (f"link {k.id} can't join {list(k.ends)} at capacity {k.beta}: a link states no capacity only where "
                    "it says why no certifier splits it (unmediated)")
    for x in t.locs:
        if x.kind == "certifier" and (far := [t.kind(n) for n in t.far(x.id)]) not in (["host", "outside"],
                                                                                       ["outside", "host"]):
            return f"attester {x.id} doesn't split one uplink: it links to {far}"
    return None


def _sheets(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered sheet."""
    for k in e.sheets:
        if a.sheet(k.sku) is None:
            return f"sheet {k.sku} ({k.digest()[:16]}...) names an SKU the auditor has no data sheet for"
    digests = [k.digest() for k in add(s, e).sheets]
    for x in e.locs:
        if x.kind == "device" and x.sheet not in digests:
            return f"{x.id}'s sheet {x.sheet[:16]}... isn't registered"
    return None


def _figures(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new sheet is accepted against the auditor's sheet for its SKU: its figures against the estimate, and its
    memory figures exactly the auditor's."""
    for k in e.sheets:
        if (b := a.sheet(k.sku)) is not None and not (k.figures.accepted(a.estimate(b))
                                                       and (k.bw, k.chip, k.mem) == (b.bw, b.chip, b.mem)):
            return (f"sheet {k.sku}'s figures aren't within the auditor's tolerance of its estimate, or its memory "
                    "figures aren't the auditor's")
    return None


def _rates(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new memory leaf directly under a host states the auditor's figures for it: the host it is under, a rate
    that is not 0, and its bytes; every other new location states no rate."""
    t = add(s, e)
    for x in e.locs:
        if x.kind == "memory" and x.parent is not None and t.kind(x.parent) == "host":
            f = _lookup(a.memory, x.id)
            if f is None or (x.parent, x.bw, x.bytes) != (f.host, f.bw, f.bytes) or f.bw == 0:
                return (f"memory {x.id} under host {x.parent} states {x.bytes} bytes at {x.bw} bits a second, and the "
                        f"auditor's figures for it are {f}; they must be the same, at a rate above 0")
        elif x.bw is not None:
            return f"{x.kind} {x.id} states a rate, which only a memory leaf directly under a host does"
    return None


def _timed(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new location of kind `site` states the site's unit of time, which is not 0 and is the one the site states;
    every other new location states none."""
    t = add(s, e)
    for x in e.locs:
        if x.kind == "site":
            if x.time_unit is None or x.time_unit == 0 or t.time_unit != x.time_unit:
                return (f"site {x.id} states the unit of time {x.time_unit}, and the site's is {t.time_unit}; a site "
                        "states the site's, above 0")
        elif x.time_unit is not None:
            return f"{x.kind} {x.id} states a unit of time, which only a site does"
    return None


def _chips(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new host states the auditor's figures for it, its on-chip state in bits and each tier's shared rate, none
    of them 0; every other new location states neither."""
    for x in e.locs:
        if x.kind == "host":
            h = _lookup(a.hosts, x.id)
            if h is None or (x.chip, x.shared) != (h.chip, h.shared) or any(bw == 0 for _, bw in h.shared):
                return (f"host {x.id} states chip {x.chip} and shared rates {list(x.shared)}, and the auditor's "
                        f"figures for it are {h}; they must be the same, with no rate 0")
        elif x.chip is not None or x.shared:
            return f"{x.kind} {x.id} states on-chip state or shared rates, which only a host does"
    return None


def _capacity(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory."""
    t = add(s, e)
    for x in e.locs:
        if x.kind == "memory" and x.bytes == 0:
            return f"memory {x.id} holds nothing"
    for x in t.locs:
        if x.kind == "device" and (k := t.sheet_of(x)) is not None and t.held(x.id) > k.mem:
            return f"{x.id}'s memory leaves hold {t.held(x.id)} bytes, over its sheet's {k.mem}"
    return None


def _owners(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new attester, challenger or auditor's device is the auditor's own, owned by it, with its key and its key's
    scheme, one of `SCHEMES`; every other new location is the prover's and states no scheme; and each new auditor's
    device states its time base, the auditor's for it, held as its key is, and the auditor's figure for its clock, below
    1 where there is one, and no other new location states a time base or a clock."""
    for x in e.locs:
        if x.kind in AUDITOR_KINDS:
            if x.owner != "auditor" or a.key(x.id) != x.key or a.scheme(x.id) != x.scheme:
                return f"{x.kind} {x.id} isn't the auditor's own, with its key and scheme"
            if x.scheme not in SCHEMES:
                return f"{x.kind} {x.id}'s key is of scheme {x.scheme!r}, not one of {list(SCHEMES)}"
        elif x.owner != "prover" or x.scheme != "":
            return f"{x.kind} {x.id} is a prover's location, registered as the {x.owner}'s, or states a scheme"
        if x.kind == "auditor-host":
            if x.timebase is None or x.timebase != _lookup(a.timebases, x.id):
                return (f"auditor-host {x.id} states the time base {x.timebase!r}, and the auditor's for it is "
                        f"{_lookup(a.timebases, x.id)!r}; it must state that one")
            if x.drift != _lookup(a.drift, x.id) or (x.drift is not None and x.drift >= 1):
                return (f"auditor-host {x.id} states the drift {x.drift}, and the auditor's figure is "
                        f"{_lookup(a.drift, x.id)}; it must be that, below 1")
        elif x.timebase is not None or x.drift is not None:
            return (f"{x.kind} {x.id} states a time base or a drift, which only an auditor's device does: a device of "
                    "the auditor's reads the clock of the host it is in")
    return None


def _covered(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Onsite: every boundary link is certified, or says why no certifier splits it."""
    t = add(s, e)
    if c == "onsite" and (bare := [k.id for k in t.links if t.is_boundary(k) and not t.certified(k)
                                   and k.unmediated is None]):
        return f"boundary links {bare} have no attester splitting them and say no reason none does"
    return None


def _reached(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Onsite: the host of every memory leaf is at an end of a memory attester's link."""
    t = add(s, e)
    if c == "onsite":
        reached = t.reached()
        if far := sorted({h for x in t.locs if x.kind == "memory" and (h := t.host_of(x)) is not None} - set(reached)):
            return f"hosts {far} hold the prover's memory, and no memory attester links to them"
    return None


CHECKS: dict[str, tuple[str, Callable[[str, Auditor, Site, Entry], str | None]]] = {
    "Fresh": ("duplicate-id", _fresh), "Refs": ("unknown-ref", _refs), "Kinds": ("kind", _kinds),
    "Sheets": ("sheet-digest", _sheets), "Figures": ("figures", _figures), "Rates": ("figures", _rates),
    "Timed": ("figures", _timed), "Chips": ("figures", _chips), "Capacity": ("capacity", _capacity),
    "Owners": ("owner", _owners), "Covered": ("uncertified", _covered), "Reached": ("challenger-reach", _reached)}
"""A site entry's checks (`Site.Entry`'s), by their names in the spec, in the order `Site.Entry.refusal` makes them,
each with the refusal it gives and the check: given the composite, the auditor, the site of the log before the entry
and the entry, what it fails, or None."""


def _first(composite: str, s: Site, head: str, e: Entry, auditor: Auditor) -> tuple[str, str] | None:
    """The first check `e` fails against the site `s` and head `head` of the log before it, with what failed."""
    if e.previous != head:
        return "previous", f"the entry follows {e.previous[:16]}..., not the log's head {head[:16]}..."
    for name, check in CHECKS.values():
        if (detail := check(composite, auditor, s, e)) is not None:
            return name, detail
    return None


def incomplete(s: Site, auditor: Auditor) -> str | None:
    """What the site lacks of the auditor's figures (`Registration.Complete`), or None: under each host, every memory
    leaf the auditor has figures for, at least the count of devices of each SKU it lists, and every way out it lists,
    a link under the way's id with an end at the host at the way's rate."""
    for h in (x for x in s.locs if x.kind == "host"):
        for i, f in auditor.memory:
            if f.host == h.id and not any(x.id == i and x.kind == "memory" and x.parent == h.id for x in s.locs):
                return f"host {h.id} registers no memory leaf {i}, which the auditor's figures place under it"
        if (g := _lookup(auditor.hosts, h.id)) is None:
            continue
        for sku, n in g.cards:
            have = sum(1 for d in s.locs if d.kind == "device" and d.parent == h.id
                       and (k := s.sheet_of(d)) is not None and k.sku == sku)
            if n > have:
                return f"host {h.id} registers {have} devices of {sku}, and the auditor's figures say {n}"
        for w, beta in g.ways:
            if not any(k.id == w and h.id in k.ends and k.beta == beta for k in s.links):
                return f"host {h.id} registers no link {w} at {beta} bits a second, a way out the auditor's figures list"
    return None


def entry_refusal(composite: str, site: Site, head: str, e: Entry, auditor: Auditor) -> str | None:
    """**The refusal** (`Registration.Entry.refusal`): the first check `e` fails against the site `site` and head `head`
    of the log before it, or None."""
    r = _first(composite, site, head, e, auditor)
    return None if r is None else r[0]


def refusal(log: Registration, e: Entry, auditor: Auditor) -> str | None:
    """`entry_refusal` against the log before `e`."""
    return entry_refusal(log.composite, of_log(log), log.head, e, auditor)


def check_registration(log: Registration, e: Entry, auditor: Auditor) -> Registration:
    """One entry held to the log before it: the log with `e` appended, or `Refused` by the first check it fails."""
    if r := _first(log.composite, of_log(log), log.head, e, auditor):
        raise Refused(*r, len(log.entries))
    return log.append(e)


def check(log: Registration, auditor: Auditor) -> Site:
    """**A valid log** (`Registration.check`): each entry held to the log before it, and the site the log registers;
    `Refused` at the first entry that fails, or, where every entry passes and the site lacks hardware the auditor's
    figures list (`Complete`), `Refused` by `inventory` at the log's length. The log is held to the auditor's
    inventory at its end and not entry by entry, so a later entry may add hardware under a registered host."""
    s, head = Site(), genesis(log.composite)
    for i, e in enumerate(log.entries):
        if r := _first(log.composite, s, head, e, auditor):
            raise Refused(*r, i)
        s, head = add(s, e), e.digest()
    if lack := incomplete(s, auditor):
        raise Refused("inventory", lack, len(log.entries))
    return s
