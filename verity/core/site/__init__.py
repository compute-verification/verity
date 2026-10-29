"""The Site in code: the registered hardware, folded from the registration log, the check each entry passes, and the
views the protocols read. It mirrors the spec's `Verity.Site` (`Security/Definitions/Core/Site.lean`): its sheets, locations,
links and log, `Site.ofLog`, and `Registration.Entry.refusal`, check for check and in the same order.

The Site is a tree of locations, each with its owner: the site at the root, its hosts, a host's devices, memory leaves
by tier and bytes under a device or a host, and at the site the auditor's devices, its network certifiers and memory
challengers (`PARENTS`). Links join two ends, each a location or `OUTSIDE`, with a capacity `beta` in bits a tick each
way and a latency in ticks (`LINKS`). A network certifier is the leaf that splits an uplink: one link from a host to it,
one from it to the outside. Hosts carry no compute: a device names its sheet by digest. A sheet (`Sheet`), a card, is
the figures the prover states for an SKU: its compute figures (`DeviceFigures`: the SKU, the tensor clock and the peak
rate of each operand class) and its memory figures (bits from device memory to the chip a tick, bits of on-chip state and
bytes of device memory). The auditor's data sheets, public and looked up by SKU, are sheets too, whose figures are its
estimate of the SKU's (`Auditor.estimate`); they come from the card files (`verity/hardware/cards/<sku>.toml`):
`Sheet.of_file` reads one as `verity.hardware.load(sku)` returns it.

The log (`Registration`) is append-only: each site entry adds sheets, locations and links and names the head of the log
before it (`Entry.previous`), the digest of its last entry or, for an empty log, its composite's genesis. `Site.of_log`
is the site a log registers, and `check_registration` holds one entry to the log before it, refusing by the first check
it fails (`REFUSALS`, the spec's `Registration.Refusal` in kebab case):

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicate-id` | no id is registered twice, `outside`'s included, and no sheet twice |
| `unknown-ref` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity; each certifier splits exactly one uplink |
| `sheet-digest` | each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered sheet |
| `figures` | each new sheet's figures are within the auditor's tolerance of its estimate for the SKU (`DeviceFigures.accepted`), and its memory figures are the auditor's |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory |
| `owner` | each new certifier or challenger is one of the auditor's own devices with its key, and every other new location is the prover's |
| `uncertified` | onsite: every boundary link is certified |
| `challenger-reach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |

The views a protocol reads (`Site.sheet`, `device`, `node`, `tiers`, `boundary`, `units`, `registered`) are functions of
the site alone, but `device`, which reads the auditor's sheet too. Digests are SHA-512 over canonical JSON
(`verity.core.wire.canonical`) under the registration log's tags; a rational is the string `n` or `n/d` in lowest terms.
Stdlib and the wire only.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from verity.core.wire import canonical

SITE = "verity/registration/site/v1"
"""A site entry's format: the tag its digest is taken under."""
SHEET = "verity/registration/sheet/v1"
GENESIS = "verity/registration/genesis/v1"
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
OPERAND_CLASSES = ("uniform-s8", "uniform-s2", "uniform-u2", "uniform-u4", "zero")
"""The operand classes of a tensor-core MAC (`OperandClass`, in kebab case): int8 operands drawn uniformly over
[-128, 127], [-2, 1], [0, 3] and [0, 15], and the one value 0."""
REFUSALS = ("previous", "duplicate-id", "unknown-ref", "kind", "sheet-digest", "figures", "capacity", "owner",
            "uncertified", "challenger-reach")
"""The checks `check_registration` makes, in the order it makes them."""
TICK = Fraction(1, 10 ** 9)
"""A tick, in seconds: a nanosecond."""


class Malformed(ValueError):
    """A value no `Site` type can hold: an unknown kind, tier, owner, composite or operand class, a negative number, or
    JSON that isn't what the encoders write."""


class Refused(Exception):
    """`check_registration` refuses the entry at index `entry` of its log by `name`, one of `REFUSALS`."""

    def __init__(self, name: str, detail: str, entry: int) -> None:
        super().__init__(f"entry {entry}: {name}: {detail}")
        self.name, self.detail, self.entry = name, detail, entry


def tagged(tag: str, obj: Any) -> str:
    """`SHA-512(tag ‖ 0 ‖ canonical(obj))` as hex."""
    return hashlib.sha512(tag.encode() + b"\0" + canonical(obj)).hexdigest()


def genesis(composite: str) -> str:
    """The head of a composite's empty log."""
    return tagged(GENESIS, {"composite": composite})


def _str(v: Any, what: str) -> str:
    if not isinstance(v, str):
        raise Malformed(f"{what}: {v!r} is not a string")
    return v


def _nat(v: Any, what: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise Malformed(f"{what}: {v!r} is not a natural number")
    return v


def _one_of(v: Any, names: Sequence[str], what: str) -> str:
    if _str(v, what) not in names:
        raise Malformed(f"{what}: {v!r} is not one of {list(names)}")
    return v


def _exactly(j: Any, keys: Sequence[str], what: str) -> Mapping[str, Any]:
    if not isinstance(j, Mapping) or sorted(j) != sorted(keys):
        raise Malformed(f"{what}: the keys must be {sorted(keys)}")
    return j


def _list(j: Any, what: str) -> list[Any]:
    if not isinstance(j, list):
        raise Malformed(f"{what}: not a list")
    return j


_RATIONAL = re.compile(r"(0|[1-9][0-9]*)(/[1-9][0-9]*)?")


def _rat(v: Any, what: str) -> Fraction:
    if isinstance(v, bool) or not isinstance(v, (int, Fraction)) or v < 0:
        raise Malformed(f"{what}: {v!r} is not a nonnegative rational")
    return Fraction(v)


def _rat_json(v: Any, what: str) -> Fraction:
    """A nonnegative rational as the JSON writes it: the string `n` or `n/d` in lowest terms, Lean's `toString` of a
    `Rat`, so one rational has one encoding."""
    if not isinstance(v, str) or not _RATIONAL.fullmatch(v) or str(q := Fraction(v)) != v:
        raise Malformed(f"{what}: {v!r} is not a rational's string n or n/d in lowest terms")
    return q


def within_tolerance(tau: Fraction, e: Fraction, x: Fraction) -> bool:
    """`x` is within relative tolerance `tau` of `e`, in both directions (`WithinTolerance`): `(1 - tau)·e ≤ x ≤
    (1 + tau)·e`, with `1 - tau` truncated at 0, so a tolerance of 1 or more sets no lower bound."""
    return max(Fraction(0), 1 - tau) * e <= x <= (1 + tau) * e


@dataclass(frozen=True)
class DeviceFigures:
    """A device's compute figures, which its sheet carries (`DeviceFigures`): its SKU, its tensor clock in Hz and the
    peak rate of tensor-core MACs a second of each operand class (`OPERAND_CLASSES`), each a nonnegative rational.
    `peak_rate` is given as a mapping from every class to its rate, or as one rate every class shares, and is held in
    `OPERAND_CLASSES`' order."""

    sku: str
    clock: Fraction
    peak_rate: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        _str(self.sku, "a figure's sku")
        object.__setattr__(self, "clock", _rat(self.clock, f"{self.sku}'s clock"))
        r: Any = self.peak_rate
        if isinstance(r, Mapping):
            r = tuple(_exactly(r, OPERAND_CLASSES, f"{self.sku}'s peak rates")[c] for c in OPERAND_CLASSES)
        elif not isinstance(r, (tuple, list)):
            r = (r,) * len(OPERAND_CLASSES)
        if len(r) != len(OPERAND_CLASSES):
            raise Malformed(f"{self.sku}'s peak rates: one for each of {list(OPERAND_CLASSES)}")
        object.__setattr__(self, "peak_rate", tuple(_rat(x, f"{self.sku}'s peak rate") for x in r))

    def rate(self, c: str) -> Fraction:
        """The peak rate of operand class `c`."""
        return self.peak_rate[OPERAND_CLASSES.index(c)]

    def accepted(self, est: DeviceEstimate) -> bool:
        """**Registration's check of the figures** against the auditor's estimate `est` (`DeviceFigures.Accepted`): the
        SKU is the estimate's, the clock and every peak rate are within the tolerance of the estimate's in both
        directions, and every peak rate is positive."""
        e, tau = est.figures, est.tolerance
        return (self.sku == e.sku and within_tolerance(tau, e.clock, self.clock)
                and all(within_tolerance(tau, a, d) for a, d in zip(e.peak_rate, self.peak_rate, strict=True))
                and all(d > 0 for d in self.peak_rate))

    def bounding(self, est: DeviceEstimate) -> DeviceFigures:
        """**The figures the bound reads** (`DeviceFigures.bounding`): the larger clock and the larger peak rate of each
        class, of these and the estimate's. The SKU is these figures'."""
        e = est.figures
        return DeviceFigures(self.sku, max(self.clock, e.clock), tuple(map(max, self.peak_rate, e.peak_rate)))

    def to_json(self) -> dict[str, Any]:
        return {"clock": str(self.clock), "peak_rate": {c: str(r) for c, r in zip(OPERAND_CLASSES, self.peak_rate)},
                "sku": self.sku}

    @classmethod
    def of_json(cls, j: Any) -> DeviceFigures:
        j = _exactly(j, ("clock", "peak_rate", "sku"), "a sheet's figures")
        rates = _exactly(j["peak_rate"], OPERAND_CLASSES, "the peak rates")
        return cls(_str(j["sku"], "sku"), _rat_json(j["clock"], "clock"),
                   tuple(_rat_json(rates[c], f"{c}'s peak rate") for c in OPERAND_CLASSES))


@dataclass(frozen=True)
class DeviceEstimate:
    """The auditor's estimate of one SKU's figures, with the relative tolerance registration allows around them
    (`DeviceEstimate`)."""

    figures: DeviceFigures
    tolerance: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "tolerance", _rat(self.tolerance, "a tolerance"))


@dataclass(frozen=True)
class Sheet:
    """A sheet, a card: the figures the prover states for an SKU. Its compute figures (`figures`), and its memory
    figures in the device model's terms at a tick: bits from device memory to the chip (`bw`), bits of on-chip state
    (`chip`) and bytes of device memory (`mem`). A device names its sheet by digest. The auditor's data sheet for an SKU
    is a sheet too, whose figures are its estimate of the SKU's."""

    figures: DeviceFigures
    bw: int
    chip: int
    mem: int

    def __post_init__(self) -> None:
        if not isinstance(self.figures, DeviceFigures):
            raise Malformed(f"a sheet's figures: {self.figures!r} are not DeviceFigures")
        for f in ("bw", "chip", "mem"):
            _nat(getattr(self, f), f"sheet {self.sku}'s {f}")

    @property
    def sku(self) -> str:
        """The sheet's SKU, its figures'."""
        return self.figures.sku

    @classmethod
    def of_file(cls, card: Mapping[str, Any], tick: Fraction = TICK) -> Sheet:
        """The sheet a card file gives (`verity.hardware.load(sku)`'s table). Its figures: the SKU, the SM clock in Hz,
        and as every operand class's peak rate the card's s8 MACs a SM-clock (`rate.tensor`; every class is int8) over
        its SMs at that clock. Its memory figures at ticks of `tick` seconds, as `Accel.Card.toDevice` reads the card:
        the bandwidth a ceiling rounded up, and the stores' sizes exact."""
        def v(entry: Mapping[str, Any]) -> Fraction:
            return Fraction(str(entry["value"]))

        sms, store = v(card["sm"]["count"]), card["store"]
        chip = (32 * sms * (v(store["reg"]) + v(store["reg_extra"])) + 8 * (sms * v(store["smem"]) + v(store["l2"])))
        mem = v(store["dram"])
        if chip.denominator != 1 or mem.denominator != 1:
            raise Malformed(f"card {card['sku']}: its stores' bits and device memory's bytes are whole")
        clock = v(card["clock"]["sm"]) * 10 ** 6
        figures = DeviceFigures(_str(card["sku"], "sku"), clock, sms * v(card["rate"]["tensor"]) * clock)
        return cls(figures, math.ceil(8 * v(card["memory"]["bandwidth"]) * tick), int(chip), int(mem))

    def to_json(self) -> dict[str, Any]:
        return {"bw": self.bw, "chip": self.chip, "figures": self.figures.to_json(), "mem": self.mem}

    @classmethod
    def of_json(cls, j: Any) -> Sheet:
        j = _exactly(j, ("bw", "chip", "figures", "mem"), "a sheet")
        return cls(DeviceFigures.of_json(j["figures"]), _nat(j["bw"], "bw"), _nat(j["chip"], "chip"),
                   _nat(j["mem"], "mem"))

    def digest(self) -> str:
        return tagged(SHEET, self.to_json())


@dataclass(frozen=True)
class Loc:
    """A location and its owner. A device names its sheet's digest, a memory leaf its tier and bytes, the auditor's
    devices their keys; a field the location's kind doesn't use is ignored."""

    id: str
    kind: str
    parent: str | None = None
    owner: str = "prover"
    sheet: str = ""
    tier: str | None = None
    bytes: int = 0
    key: str = ""

    def __post_init__(self) -> None:
        _str(self.id, "a location's id")
        _one_of(self.kind, KINDS, f"{self.id}'s kind")
        if self.parent is not None:
            _str(self.parent, f"{self.id}'s parent")
        _one_of(self.owner, OWNERS, f"{self.id}'s owner")
        _str(self.sheet, f"{self.id}'s sheet")
        if self.tier is not None:
            _one_of(self.tier, TIERS, f"{self.id}'s tier")
        _nat(self.bytes, f"{self.id}'s bytes")
        _str(self.key, f"{self.id}'s key")

    def to_json(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "id": self.id, "key": self.key, "kind": self.kind, "owner": self.owner,
                "parent": self.parent, "sheet": self.sheet, "tier": self.tier}

    @classmethod
    def of_json(cls, j: Any) -> Loc:
        j = _exactly(j, ("bytes", "id", "key", "kind", "owner", "parent", "sheet", "tier"), "a location")
        return cls(j["id"], j["kind"], j["parent"], j["owner"], j["sheet"], j["tier"], j["bytes"], j["key"])


@dataclass(frozen=True)
class Link:
    """A link between two ends, with its capacity `beta` in bits a tick each way and its latency in ticks."""

    id: str
    ends: tuple[str, str]
    beta: int
    latency: int

    def __post_init__(self) -> None:
        _str(self.id, "a link's id")
        if not isinstance(self.ends, tuple) or len(self.ends) != 2:
            raise Malformed(f"link {self.id}: its ends are a pair")
        for e in self.ends:
            _str(e, f"link {self.id}'s end")
        _nat(self.beta, f"link {self.id}'s beta")
        _nat(self.latency, f"link {self.id}'s latency")

    def to_json(self) -> dict[str, Any]:
        return {"beta": self.beta, "ends": list(self.ends), "id": self.id, "latency": self.latency}

    @classmethod
    def of_json(cls, j: Any) -> Link:
        j = _exactly(j, ("beta", "ends", "id", "latency"), "a link")
        ends = _list(j["ends"], "a link's ends")
        if len(ends) != 2:
            raise Malformed("a link's ends: not two ends")
        return cls(j["id"], (ends[0], ends[1]), j["beta"], j["latency"])


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

    def to_json(self) -> dict[str, Any]:
        return {"composite": self.composite, "entries": [e.to_json() for e in self.entries]}

    @classmethod
    def of_json(cls, j: Any) -> Registration:
        j = _exactly(j, ("composite", "entries"), "the log")
        return cls(j["composite"], tuple(Entry.of_json(e) for e in _list(j["entries"], "entries")))


@dataclass(frozen=True)
class Auditor:
    """What the auditor brings to the check: its data sheet for each SKU, public, whose figures are its estimate of the
    SKU's; the relative tolerance registration allows around an estimate; and the key of each of its own devices."""

    sheets: tuple[Sheet, ...] = ()
    tolerance: Fraction = Fraction(0)
    keys: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sheets", tuple(self.sheets))
        object.__setattr__(self, "tolerance", _rat(self.tolerance, "the auditor's tolerance"))
        object.__setattr__(self, "keys", tuple((_str(d, "a device"), _str(k, "a key")) for d, k in self.keys))

    @classmethod
    def of_files(cls, cards: Iterable[Mapping[str, Any]], tolerance: Fraction = Fraction(0),
                 keys: Mapping[str, str] | None = None, tick: Fraction = TICK) -> Auditor:
        """The auditor whose data sheets come from these card files (`verity.hardware.load(sku)` each), at this
        tolerance, and its devices' keys."""
        return cls(tuple(Sheet.of_file(c, tick) for c in cards), tolerance, tuple((keys or {}).items()))

    def sheet(self, sku: str) -> Sheet | None:
        """The data sheet for `sku`: the first under it."""
        return next((k for k in self.sheets if k.sku == sku), None)

    def estimate(self, a: Sheet) -> DeviceEstimate:
        """**The auditor's estimate** its sheet `a` states: `a`'s figures, at the auditor's tolerance."""
        return DeviceEstimate(a.figures, self.tolerance)

    def key(self, i: str) -> str | None:
        """The key of the auditor's device `i`: the first under it."""
        return next((k for d, k in self.keys if d == i), None)

    def to_json(self) -> dict[str, Any]:
        return {"keys": [{"id": d, "key": k} for d, k in self.keys], "sheets": [k.to_json() for k in self.sheets],
                "tolerance": str(self.tolerance)}

    @classmethod
    def of_json(cls, j: Any) -> Auditor:
        j = _exactly(j, ("keys", "sheets", "tolerance"), "the auditor")
        keys = [_exactly(k, ("id", "key"), "a key") for k in _list(j["keys"], "keys")]
        return cls(tuple(Sheet.of_json(k) for k in _list(j["sheets"], "sheets")),
                   _rat_json(j["tolerance"], "the auditor's tolerance"), tuple((k["id"], k["key"]) for k in keys))


@dataclass(frozen=True)
class Site:
    """**The registered hardware**: its sheets, its tree of locations and its links, as a log registers them
    (`of_log`). A part of it, a host and what lies inside (`node`), is a `Site` too."""

    sheets: tuple[Sheet, ...] = ()
    locs: tuple[Loc, ...] = ()
    links: tuple[Link, ...] = ()

    def add(self, e: Entry) -> Site:
        """The site `e` extends this one to, unchecked."""
        return Site(self.sheets + e.sheets, self.locs + e.locs, self.links + e.links)

    @classmethod
    def of_log(cls, log: Registration | Sequence[Entry]) -> Site:
        """**The site of a log**: its entries' sheets, locations and links, in the log's order. The site of a prefix of
        the log is the site as registered when that prefix was the log."""
        s = cls()
        for e in log.entries if isinstance(log, Registration) else log:
            s = s.add(e)
        return s

    # --- lookups, as the spec states them -----------------------------------------------------------------------------

    def loc(self, i: str) -> Loc | None:
        """The location an id names: the first registered under it."""
        return next((x for x in self.locs if x.id == i), None)

    def kind(self, i: str) -> str | None:
        """The kind of what an id names: the outside's, or its location's; None when nothing is registered under it."""
        if i == OUTSIDE:
            return "outside"
        x = self.loc(i)
        return None if x is None else x.kind

    def known(self, i: str) -> bool:
        """An id names something: the outside, or a registered location."""
        return i == OUTSIDE or any(x.id == i for x in self.locs)

    def ids(self) -> list[str]:
        """Every registered id: the locations', the links' and the outside's."""
        return [x.id for x in self.locs] + [k.id for k in self.links] + [OUTSIDE]

    def sheet_of(self, x: Loc) -> Sheet | None:
        """A device's sheet: the first registered sheet under the digest it names."""
        return next((k for k in self.sheets if k.digest() == x.sheet), None)

    def held(self, i: str) -> int:
        """The bytes of the memory leaves directly under `i`."""
        return sum(x.bytes for x in self.locs if x.kind == "memory" and x.parent == i)

    def joins(self, k: Link, a: str, b: str) -> bool:
        """Link `k` joins kinds `a` and `b`, in either order."""
        return (self.kind(k.ends[0]), self.kind(k.ends[1])) in ((a, b), (b, a))

    def is_boundary(self, k: Link) -> bool:
        """A boundary link of the prover's: a host's link to the outside, or to the certifier that splits its uplink."""
        return self.joins(k, "host", "outside") or self.joins(k, "certifier", "host")

    def certified(self, k: Link) -> bool:
        """A certifier is at one of the link's ends."""
        return "certifier" in (self.kind(k.ends[0]), self.kind(k.ends[1]))

    def far(self, i: str) -> list[str]:
        """The far ends of `i`'s links: for each link with an end at `i`, its other end."""
        return [e for k in self.links if i in k.ends for e in k.ends if e != i]

    def host_of(self, x: Loc) -> str | None:
        """The host a memory leaf or a device is in: its parent when that is a host, else its parent's parent."""
        if x.parent is None:
            return None
        if self.kind(x.parent) == "host":
            return x.parent
        p = self.loc(x.parent)
        return None if p is None else p.parent

    def reached(self) -> list[str]:
        """What the memory challengers reach: the ends of their links."""
        return [e for k in self.links if "challenger" in (self.kind(k.ends[0]), self.kind(k.ends[1])) for e in k.ends]

    # --- the views a protocol reads -----------------------------------------------------------------------------------

    def registered(self, i: str) -> bool:
        """A location or a link is registered under `i`."""
        return any(x.id == i for x in self.locs) or any(k.id == i for k in self.links)

    @property
    def units(self) -> tuple[str, ...]:
        """The devices, in the log's order: the locations that compute (hosts carry no compute)."""
        return tuple(x.id for x in self.locs if x.kind == "device")

    def sheet(self, d: str) -> Sheet | None:
        """Device `d`'s data sheet, or None when `d` is no device or names no registered sheet."""
        x = self.loc(d)
        return self.sheet_of(x) if x is not None and x.kind == "device" else None

    def device(self, d: str, auditor: Auditor, tick: Fraction = TICK, cuda: int = 0, oracle: int = 0,
               cap: Mapping[str, int] | None = None) -> dict[str, Any] | None:
        """**Device `d`'s model** (`Accounting.Device`, `Site.device`), read off its sheet `k` and the auditor's sheet
        `a` for `k`'s SKU: `k`'s on-chip state (`chip`) and memory bandwidth (`bw`), `k`'s device memory in bits (`mem`),
        and as its tensor-core MACs a tick (`rate["tc"]`) the highest peak rate of the figures the bound reads
        (`DeviceFigures.bounding`: class by class, the larger of `k`'s and `a`'s estimate's) over a tick of `tick`
        seconds, rounded up. No sheet states a device's CUDA-core ops or oracle queries a tick, or its side channels'
        capacities (`cap`), so they are parameters. None when `d` names no registered sheet or the auditor has none for
        its SKU."""
        k = self.sheet(d)
        a = None if k is None else auditor.sheet(k.sku)
        if k is None or a is None:
            return None
        f = k.figures.bounding(auditor.estimate(a))
        return {"bw": k.bw, "cap": dict(cap or {}), "chip": k.chip, "mem": 8 * k.mem,
                "rate": {"cuda": cuda, "oracle": oracle, "tc": math.ceil(tick * max(f.peak_rate))}}

    def host(self, i: str) -> str | None:
        """The host a location is in: itself when it is a host."""
        x = self.loc(i)
        if x is None:
            return None
        return x.id if x.kind == "host" else self.host_of(x)

    def _within(self, x: Loc, i: str) -> bool:
        """`x` is `i` or lies under it: `i` is on its chain of parents, followed at most once per location."""
        for _ in range(len(self.locs) + 1):
            if x.id == i:
                return True
            if x.parent is None or (x := self.loc(x.parent)) is None:
                return False
        return False

    def part(self, i: str) -> Site:
        """The part of the site at location `i`: `i` and every location under it, the sheets its devices name, and the
        links with an end in it."""
        locs = tuple(x for x in self.locs if self._within(x, i))
        ids, digests = {x.id for x in locs}, {x.sheet for x in locs if x.kind == "device"}
        return Site(tuple(k for k in self.sheets if k.digest() in digests), locs,
                    tuple(k for k in self.links if k.ends[0] in ids or k.ends[1] in ids))

    def node(self, h: str) -> Site | None:
        """Host `h` as registered, its devices, their memory and its own, and its links; None when `h` is no host."""
        return self.part(h) if self.kind(h) == "host" else None

    def tiers(self, h: str) -> dict[str, int]:
        """The bytes of memory at location `h` (a host's, its devices' among them), by tier."""
        p = self.part(h)
        return {t: sum(x.bytes for x in p.locs if x.kind == "memory" and x.tier == t) for t in TIERS}

    def boundary(self, h: str | None = None) -> list[Link]:
        """The boundary links (`is_boundary`) with an end at host `h`, or every boundary link."""
        return [k for k in self.links if self.is_boundary(k) and (h is None or h in k.ends)]

    def to_json(self) -> dict[str, Any]:
        return {"links": [k.to_json() for k in self.links], "locs": [x.to_json() for x in self.locs],
                "sheets": [k.to_json() for k in self.sheets]}


def _dups(xs: Sequence[str]) -> list[str]:
    return sorted({x for x in xs if xs.count(x) > 1})


def _first(composite: str, s: Site, head: str, e: Entry, auditor: Auditor) -> tuple[str, str] | None:
    """The first check `e` fails against the site `s` and head `head` of the log before it, with what failed."""
    if e.previous != head:
        return "previous", f"the entry follows {e.previous[:16]}..., not the log's head {head[:16]}..."
    t = s.add(e)
    digests = [k.digest() for k in t.sheets]
    if dup := _dups(t.ids()) + [k.sku for k in t.sheets if digests.count(k.digest()) > 1]:
        return "duplicate-id", f"{dup} registered twice"
    for x in e.locs:
        if x.parent is not None and not t.known(x.parent):
            return "unknown-ref", f"{x.id}'s parent {x.parent!r} isn't registered"
    for k in e.links:
        if bad := [n for n in k.ends if not t.known(n)]:
            return "unknown-ref", f"link {k.id} names {bad}, which aren't registered"
    for x in e.locs:
        if ((None if x.parent is None else t.kind(x.parent)) not in PARENTS[x.kind]
                or (x.kind == "memory") != (x.tier is not None)):
            return "kind", f"{x.kind} {x.id} can't sit under {x.parent!r}, or has a tier exactly when it isn't memory"
    for k in e.links:
        if k.ends[0] == k.ends[1] or not any(t.joins(k, a, b) for a, b in LINKS) or k.beta == 0:
            return "kind", f"link {k.id} can't join {list(k.ends)} at capacity {k.beta}"
    for x in t.locs:
        if x.kind == "certifier" and (far := [t.kind(n) for n in t.far(x.id)]) not in (["host", "outside"],
                                                                                       ["outside", "host"]):
            return "kind", f"certifier {x.id} doesn't split one uplink: it links to {far}"
    for k in e.sheets:
        if auditor.sheet(k.sku) is None:
            return "sheet-digest", f"sheet {k.sku} ({k.digest()[:16]}...) names an SKU the auditor has no data sheet for"
    for x in e.locs:
        if x.kind == "device" and x.sheet not in digests:
            return "sheet-digest", f"{x.id}'s sheet {x.sheet[:16]}... isn't registered"
    for k in e.sheets:
        if (a := auditor.sheet(k.sku)) is not None and not (k.figures.accepted(auditor.estimate(a))
                                                             and (k.bw, k.chip, k.mem) == (a.bw, a.chip, a.mem)):
            return "figures", (f"sheet {k.sku}'s figures aren't within the auditor's tolerance of its estimate, or its "
                               "memory figures aren't the auditor's")
    for x in e.locs:
        if x.kind == "memory" and x.bytes == 0:
            return "capacity", f"memory {x.id} holds nothing"
    for x in t.locs:
        if x.kind == "device" and (k := t.sheet_of(x)) is not None and t.held(x.id) > k.mem:
            return "capacity", f"{x.id}'s memory leaves hold {t.held(x.id)} bytes, over its sheet's {k.mem}"
    for x in e.locs:
        if x.kind in AUDITOR_KINDS and (x.owner != "auditor" or auditor.key(x.id) != x.key):
            return "owner", f"{x.kind} {x.id} isn't one of the auditor's own devices"
        if x.kind not in AUDITOR_KINDS and x.owner != "prover":
            return "owner", f"{x.kind} {x.id} is a prover's location, registered as the {x.owner}'s"
    if composite == "onsite" and (bare := [k.id for k in t.links if t.is_boundary(k) and not t.certified(k)]):
        return "uncertified", f"boundary links {bare} have no certifier splitting them"
    if composite == "onsite":
        reached = t.reached()
        if far := sorted({h for x in t.locs if x.kind == "memory" and (h := t.host_of(x)) is not None} - set(reached)):
            return "challenger-reach", f"hosts {far} hold the prover's memory, and no memory challenger links to them"
    return None


def entry_refusal(composite: str, site: Site, head: str, e: Entry, auditor: Auditor) -> str | None:
    """**The refusal** (`Registration.Entry.refusal`): the first check `e` fails against the site `site` and head `head`
    of the log before it, or None."""
    r = _first(composite, site, head, e, auditor)
    return None if r is None else r[0]


def refusal(log: Registration, e: Entry, auditor: Auditor) -> str | None:
    """`entry_refusal` against the log before `e`."""
    return entry_refusal(log.composite, Site.of_log(log), log.head, e, auditor)


def check_registration(log: Registration, e: Entry, auditor: Auditor) -> Registration:
    """One entry held to the log before it: the log with `e` appended, or `Refused` by the first check it fails."""
    if r := _first(log.composite, Site.of_log(log), log.head, e, auditor):
        raise Refused(*r, len(log.entries))
    return log.append(e)


def check(log: Registration, auditor: Auditor) -> Site:
    """**A valid log**: each entry held to the log before it, and the site the log registers; `Refused` at the first
    entry that fails."""
    s, head = Site(), genesis(log.composite)
    for i, e in enumerate(log.entries):
        if r := _first(log.composite, s, head, e, auditor):
            raise Refused(*r, i)
        s, head = s.add(e), e.digest()
    return s
