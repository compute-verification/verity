"""The Site in code: the registered hardware, folded from the registration log, the check each entry passes, and the
views the protocols read. It mirrors the spec's `Verity.Site` (`security/Definitions/Core/Site.lean`): its sheets, locations,
links and log, `Site.ofLog`, and `Registration.Entry.refusal`, check for check and in the same order.

The Site is a tree of locations, each with its owner: the site at the root, stating the site's unit of time
(`Site.time_unit`), its hosts, a host's devices, memory leaves by tier and bytes under a device or a host, and at the
site the auditor's devices, its network certifiers and memory challengers (`PARENTS`), each with its key, the key's
scheme (`SCHEMES`) and how far its clock may drift. A host states its on-chip state and the rate its leaves of a tier
share, and a leaf under a host the rate it is read at. Links join two ends, each a location or `OUTSIDE`, with a
capacity `beta` in bits a second each way and a latency in the site's units of time (`LINKS`). A boundary link no
certifier splits may say why none does (`unmediated`), and a link states no capacity only where it does; a site is
mediated when no link says so (`Site.mediated`). A network certifier is the leaf that splits an uplink: one link from
a host to it, one from it to the outside. Hosts carry no compute: a device names its sheet by digest. A sheet
(`Sheet`), a card, is the figures the prover states for an SKU: its compute figures (`DeviceFigures`: the SKU, the
tensor clock and the peak rate of each operand class) and its memory figures (bits read from or written to device
memory a second, bits of on-chip state and bytes of device memory). The auditor's data sheets, public and looked up
by SKU, are sheets too, whose figures are its estimate of the SKU's (`Auditor.estimate`); they come from the card files
(`verity/hardware/cards/<sku>.toml`): `Sheet.of_file` reads one as `verity.hardware.cards.load(sku)` returns it. The
auditor's figures for each host and each leaf under one (`HostFigures`, `MemoryFigures`) come from the node files:
`node_figures` reads a node file's table (`verity.hardware.nodes.load(kind)`), its host card's, its leaves
(`verity.hardware.nodes.leaves`) and its ways out (`verity.hardware.nodes.ways_out`). `Registration.node_entry` reads
the same and its GPU card's, states those figures on the node's locations, and registers each way out, an interface or
the storage path, as a boundary link that no certifier splits. `clock_drift` reads an auditor's host's time base card
(`kind = "timebase"`): by its `drift.max` mark, the host's premise (`ClockPremise`) or the figure `Auditor.drift`
takes, never both; `device_drift` is the drift an auditor's device in that host states, which `node_entry`'s
certifier and `Auditor.of_files`' `clocks` take, and a premise gives the device's `Account.Premised`
(`ClockPremise.premised`).

The log (`Registration`) is append-only: each site entry adds sheets, locations and links and names the head of the log
before it (`Entry.previous`), the digest of its last entry or, for an empty log, its composite's genesis. `Site.of_log`
is the site a log registers, and `check_registration` holds one entry to the log before it, refusing by the first check
it fails (`CHECKS`, the spec's `Site.Entry` checks by name and in its order, each with its refusal; `REFUSALS`, the
spec's `Registration.Refusal` in kebab case); `check` holds each entry so, then the whole log by `inventory` at its
end (`incomplete`):

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicate-id` | no id is registered twice, `outside`'s included, and no sheet twice |
| `unknown-ref` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity or, where it says why no certifier splits it, at none; each certifier splits exactly one uplink |
| `sheet-digest` | each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered sheet |
| `figures` | each new sheet's figures are within the auditor's tolerance of its estimate for the SKU (`DeviceFigures.accepted`), and its memory figures are the auditor's; each new memory leaf directly under a host states the auditor's figures for it (its host, a rate above 0, its bytes), and no other new location a rate; each new site states the site's unit of time, above 0, and no other new location one; each new host states the auditor's figures for it (its on-chip state and each tier's shared rate, none 0), and no other new location either |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory |
| `owner` | each new certifier or challenger is one of the auditor's own devices, with its key, its key's scheme, one of `SCHEMES`, and the auditor's figure for its clock's drift, below 1 where there is one; every other new location is the prover's and states no scheme and no drift |
| `inventory` | the log, at its end: under each host, every memory leaf the auditor has figures for, at least its count of devices of each SKU it lists, and each way out it lists, as a link under that id with an end at the host at its rate |
| `uncertified` | onsite: every boundary link is certified or says why no certifier splits it |
| `challenger-reach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |

The views a protocol reads (`Site.sheet`, `device`, `node`, `tiers`, `held`, `read_rate`, `over`, `time_unit`,
`mediated`, `boundary`, `units`, `registered`) are functions of the site alone, but `device`, which reads the
auditor's sheet too. Digests are SHA-512 over canonical JSON
(`verity.configuration.wire.canonical`) under the registration log's tags; a rational is the string `n` or `n/d` in lowest terms.
Stdlib and the wire only.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from verity.configuration.wire import canonical

SITE = "verity/registration/site/v2"
"""A site entry's format: the tag its digest is taken under."""
SHEET = "verity/registration/sheet/v2"
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
SCHEMES = ("lms-sha512/n64-w1-h20/v0", "hss-lms-sha512/n64-w1-h20/n64-w1-h20/v0")
"""The signature schemes a registered key may be of: one LMS key that signs, and a root LMS key that certifies the
keys that sign."""
OPERAND_CLASSES = ("uniform-s8", "uniform-s2", "uniform-u2", "uniform-u4", "zero")
"""The operand classes of a tensor-core MAC (`OperandClass`, in kebab case): int8 operands drawn uniformly over
[-128, 127], [-2, 1], [0, 3] and [0, 15], and the one value 0."""
REFUSALS = ("previous", "duplicate-id", "unknown-ref", "kind", "sheet-digest", "figures", "capacity", "owner",
            "inventory", "uncertified", "challenger-reach")
"""The refusals, in the spec's order: `check_registration` makes each check but `inventory` in this order, and
`check` makes `inventory` once, at the log's end."""
TIME_UNIT = Fraction(1, 10 ** 9)
"""The unit of time `Registration.node_entry` gives a site it adds, in seconds: a nanosecond."""
NO_LINE_RATE = "no line rate stated"
"""Why no certifier splits a node's interface that reads no line rate: `Registration.node_entry`'s `unmediated`."""
STORAGE_PATH = "storage path"
"""Why no certifier splits a node's storage path: `Registration.node_entry`'s `unmediated`."""


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


_RATIONAL = re.compile(r"(0|[1-9][0-9]*)(?:/([1-9][0-9]*))?")
_SHORT = 512
"""A count of decimal digits Python's `int` and `str` convert under every setting of the interpreter's limit on integer
strings. Past that limit they refuse a number Lean's `Nat` reads, so a longer one is converted in pieces this short."""


def _nat_of(s: str) -> int:
    """The natural number the ASCII decimal digits `s` write, at any length."""
    if len(s) <= _SHORT:
        return int(s)
    k = len(s) // 2
    return _nat_of(s[:-k]) * 10 ** k + _nat_of(s[-k:])


def _nat_str(n: int) -> str:
    """A natural number's decimal digits, at any length."""
    if n < 10 ** _SHORT:
        return str(n)
    k = n.bit_length() * 3 // 20  # about half of n's digits, a bit being log10(2) = 0.301 of one
    high, low = divmod(n, 10 ** k)
    return _nat_str(high) + _nat_str(low).zfill(k)


def _rat(v: Any, what: str) -> Fraction:
    if isinstance(v, bool) or not isinstance(v, (int, Fraction)) or v < 0:
        raise Malformed(f"{what}: {v!r} is not a nonnegative rational")
    return Fraction(v)


def _rat_json(v: Any, what: str) -> Fraction:
    """A nonnegative rational as the JSON writes it: the string `n` or `n/d` in lowest terms, Lean's `toString` of a
    `Rat`, so one rational has one encoding. It reads the strings `Core.Ratio.ofString` reads, of any length: `n` and
    `d` in ASCII digits with no leading zero, `d` at least 2 and coprime to `n`."""
    m = _RATIONAL.fullmatch(v) if isinstance(v, str) else None
    if m is not None:
        n, d = _nat_of(m[1]), _nat_of(m[2] or "1")
        if m[2] is None or (d > 1 and math.gcd(n, d) == 1):
            return Fraction(n, d)
    raise Malformed(f"{what}: {v!r} is not a rational's string n or n/d in lowest terms")


def _rat_str(q: Fraction) -> str:
    """A nonnegative rational's one encoding, which `_rat_json` reads back."""
    n = _nat_str(q.numerator)
    return n if q.denominator == 1 else f"{n}/{_nat_str(q.denominator)}"


def _whole(entry: Any, what: str) -> int:
    """The whole number a card or node file's entry states, read from its value's decimal text."""
    try:
        n = Fraction(str(entry["value"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise Malformed(f"{what}: {entry!r} is not an entry with a number as its value") from None
    if n < 0 or n.denominator != 1:
        raise Malformed(f"{what}: {n} is not a whole number")
    return int(n)


def _way(way: Any, host: str) -> tuple[str, int | None, bool]:
    """A way out of host `host`'s node as `verity.hardware.nodes.ways_out` gives it: its id under the host, what it
    carries in whole bits a second, or `None` where its file states no rate, and whether a certifier may split it."""
    try:
        name, bw, split = way
    except (TypeError, ValueError):
        raise Malformed(f"a way out is its id, its bits a second and whether a certifier may split it, not {way!r}") from None
    if (not isinstance(name, str) or not name.startswith(f"{host}/") or not isinstance(split, bool)
            or isinstance(bw, bool) or not (bw is None or (isinstance(bw, int) and bw > 0))):
        raise Malformed(f"a way out of {host}: {way!r}; its id is {host}/NAME, its rate a whole number of bits a "
                        "second above 0, or None where its file states none, and its split a bool")
    return name, bw, split


def _leaf(leaf: Any, host: str) -> tuple[str, str, int, int]:
    """A memory leaf of host `host`'s as `verity.hardware.nodes.leaves` gives it: its id under the host, its tier, its
    bytes and the most it is read back at, in whole bits a second."""
    try:
        name, tier, size, bw = leaf
    except (TypeError, ValueError):
        raise Malformed(f"a leaf is its id, its tier, its bytes and its bits a second, not {leaf!r}") from None
    if (not isinstance(name, str) or not name.startswith(f"{host}/") or tier not in TIERS or isinstance(size, bool)
            or not isinstance(size, int) or isinstance(bw, bool) or not isinstance(bw, int) or bw <= 0):
        raise Malformed(f"a leaf of {host}: {leaf!r}; its id is {host}/NAME, its tier one of {list(TIERS)}, its bytes "
                        "a whole number, and its rate a whole number of bits a second above 0")
    return name, tier, size, bw


@dataclass(frozen=True)
class _Node:
    """What a node's files register as one host: its devices' SKU and count, its leaves (each id, tier, bytes and
    bits a second), the rate each tier's leaves share, its ways out (each id, bits a second or None, and whether a
    certifier may split it) and its host's on-chip state in bits."""

    sku: str
    count: int
    leaves: list[tuple[str, str, int, int]]
    shared: tuple[tuple[str, int], ...]
    ways: list[tuple[str, int | None, bool]]
    chip: int

    @classmethod
    def read(cls, node: Mapping[str, Any], host_card: Mapping[str, Any], host: str, ways: Any, leaves: Any) -> _Node:
        """The node file's table, its host card's, and its ways out and leaves as `verity.hardware.nodes` reads them,
        held to the file: the ways each by name and in order, with a rate exactly where the file states one, and the
        leaves each by name, tier and bytes, in the file's order. Refused where the file states no `dram` or no `disk`
        leaf, or no interface."""
        try:
            sku, count = node["device"]["sku"], _whole(node["device"]["count"], "a node's device.count")
            stated_leaves = [(f"{host}/{name}", leaf["tier"], _whole(leaf["bytes"], f"a node's memory.{name}.bytes"))
                             for name, leaf in node.get("memory", {}).items()]
            stated = [(f"{host}/{name}", True, "rate" in port.keys())
                      for name, port in node.get("interface", {}).items()]
            stated += [(f"{host}/storage", False, True)] if "storage" in node else []
            host_sku = node["host"]["sku"]
        except (AttributeError, KeyError, TypeError) as e:
            raise Malformed(f"a node file's table: {e!r}; it has device.sku, device.count, host.sku, for each "
                            "memory.NAME a tier and bytes, and for each interface.NAME a table") from None
        try:
            out = [_way(way, host) for way in ways]
        except TypeError:
            raise Malformed(f"`ways` is {ways!r}; give the node file's ways out, "
                            "`verity.hardware.nodes.ways_out(node, host)`") from None
        given = [(name, split, bw is not None) for name, bw, split in out]
        if given != stated:
            raise Malformed(f"the ways out given are {given}, and the node file states {stated} under host {host!r} "
                            "(each interface, then its storage path, with whether a certifier may split it and "
                            "whether it has a rate); give `verity.hardware.nodes.ways_out(node, host)`")
        try:
            each, together = leaves
            each, together = list(each), dict(together)
        except (TypeError, ValueError):
            raise Malformed(f"`leaves` is {leaves!r}; give the node file's leaves and the rate each tier's share, "
                            "`verity.hardware.nodes.leaves(node, host)`") from None
        each = [_leaf(leaf, host) for leaf in each]
        if [leaf[:3] for leaf in each] != stated_leaves:
            raise Malformed(f"the leaves given are {[leaf[:3] for leaf in each]}, and the node file states "
                            f"{stated_leaves} under host {host!r} (each id, tier and bytes); give "
                            "`verity.hardware.nodes.leaves(node, host)`")
        shared = tuple((t, together[t]) for t in TIERS if t in together)
        if (len(shared) != len(together) or
                not all(isinstance(bw, int) and not isinstance(bw, bool) and bw > 0 for _, bw in shared)):
            raise Malformed(f"the tiers' shared rates given are {together!r}; each is a tier's, a whole number of bits "
                            "a second above 0; give `verity.hardware.nodes.leaves(node, host)`")
        unread = [tier for tier in ("dram", "disk") if all(t != tier for _, t, _ in stated_leaves)]
        if unread:
            raise Malformed(f"the node file states no {' and no '.join(unread)} leaf of its host's, so its entry would "
                            "understate the node's memory; read them on a node of its kind (`lsmem -b --summary=only; "
                            "lsblk -b -d -n -o NAME,SIZE,TYPE` under `research run --on NODE`) and state each as a "
                            "memory.NAME table: the memory the host is given, and every block device")
        if not any(split for _, _, split in out):
            raise Malformed("the node file states no interface, so its entry would not say how the node reaches the "
                            "outside; read the node's interfaces on a node of its kind (`ip route; ip -6 route; "
                            "ethtool IF` under `research run --on NODE`) and state each one the routes name as an "
                            "interface.NAME table: its rate, or where it reads none its driver and what was read")
        try:
            card_sku, chip = host_card["sku"], 8 * _whole(host_card["store"]["chip"], "a host card's store.chip")
        except (KeyError, TypeError):
            raise Malformed("the host card given has no sku or no store.chip; give the card the node file names "
                            f"by host.sku, `verity.hardware.cards.load({host_sku!r})`") from None
        if card_sku != host_sku:
            raise Malformed(f"the node's host is {host_sku!r}, and the host card given is {card_sku!r}'s")
        return cls(sku, count, each, shared, out, chip)

    def figures(self, host: str) -> tuple[tuple[tuple[str, MemoryFigures], ...], HostFigures]:
        """The auditor's figures for the node's leaves and for its host."""
        return (tuple((i, MemoryFigures(host, bw, size)) for i, _, size, bw in self.leaves),
                HostFigures(self.chip, ((self.sku, self.count),), self.shared,
                            tuple((i, bw) for i, bw, _ in self.ways)))


def node_figures(node: Mapping[str, Any], host_card: Mapping[str, Any], host: str, *, ways: Any,
                 leaves: Any) -> tuple[tuple[tuple[str, MemoryFigures], ...], HostFigures]:
    """**The auditor's figures for a node registered as host `host`** (`Auditor.memory` and one of `Auditor.hosts`),
    from a node file's table (`verity.hardware.nodes.load(kind)`), its host card's table (`verity.hardware.cards.load`
    of its `host.sku`), its ways out (`verity.hardware.nodes.ways_out(node, host)`) and its leaves
    (`verity.hardware.nodes.leaves(node, host)`). Each leaf's figures: the host, its bits a second and its bytes. The
    host's: its on-chip state in bits, eight times the host card's `store.chip`; its cards, the file's `device.sku` at
    its `device.count`; for each tier, the bits a second its leaves share; and its ways out, each by its link's id at
    its bits a second, or None where its file states none. `Registration.node_entry` states the same figures on the
    node's locations, which `check` holds to these."""
    return _Node.read(node, host_card, host, ways, leaves).figures(host)


@dataclass(frozen=True)
class ClockPremise:
    """**A host's clock drift taken on a premise**: how far the clock of the auditor's device in host `host` may run
    from real time, as a share of it, where no data sheet bounds it, with the premise that says so (`name`, as a row
    that rests on it prints it), the time base card's mark (`premise`) and its source. It is the Lean's
    `Account.Premised`, and never the auditor's figure for a device (`Auditor.drift`): a row that reads it names
    `name` among its premises, and none of record does."""

    host: str
    name: str
    drift: Fraction
    mark: str
    source: str

    def premised(self, device: str) -> tuple[str, Fraction, str]:
        """The drift of the auditor's device `device`, which runs in the host, taken on this premise: the Lean's
        `Account.Premised`, its device, drift and premise's name, in that order."""
        return _str(device, "the device's id"), self.drift, self.name


def device_drift(clock: ClockPremise | Fraction | None) -> Fraction | None:
    """**The drift an auditor's device states** (`Loc.drift`, and its entry in `Auditor.drift`, which `owner` holds
    it to), from the clock of the host it runs in as `clock_drift` gives it: the figure, or None for a premise or no
    clock. A number taken on a premise is never the device's figure."""
    if clock is None or isinstance(clock, ClockPremise):
        return None
    if isinstance(clock, bool) or not isinstance(clock, (int, Fraction)):
        raise Malformed(f"a host's clock is `clock_drift`'s, a premise or a figure, not {clock!r}")
    return _rat(clock, "a host's clock drift")


def clock_drift(host: str, card: Mapping[str, Any]) -> ClockPremise | Fraction:
    """**The drift of the clock in host `host`**, as its time base's card (`verity.hardware.cards.load(sku)`, a card
    of kind `timebase`) states it in `drift.max`: by the entry's mark, a premise or the auditor's figure, never both.
    Marked `premise`, it is the host's premise, `ClockRateWithin(<host>, <drift.max>)` with the number as the card
    writes it in scientific notation; marked `ceiling`, it is the figure `Auditor.drift` takes for the auditor's
    devices in the host, with no premise. A `measurement` is only a lower bound on the ceiling and a `placeholder`
    no bound, so either is refused, as are a card of another kind and a drift that is not a share below 1 in `s/s`."""
    _str(host, "the host's name")
    if not host:
        raise Malformed("the host's name is empty; name the auditor's host whose clock the card times")
    sku = card.get("sku") if isinstance(card, Mapping) else None
    if not isinstance(card, Mapping) or card.get("kind") != "timebase":
        raise Malformed(f"card {sku!r} is no time base (its kind is "
                        f"{card.get('kind') if isinstance(card, Mapping) else None!r}); give the card of the clock "
                        f"host {host}'s device reads, `verity.hardware.cards.load(sku)` of a card of kind \"timebase\"")
    try:
        entry = card["drift"]["max"]
        value, unit, mark, source = entry["value"], entry["unit"], entry["mark"], entry["source"]
        drift = Fraction(str(value))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise Malformed(f"time base {sku!r} states no drift.max entry {{value, unit, mark, source}} with a number as "
                        "its value") from None
    if unit != "s/s" or not 0 <= drift < 1:
        raise Malformed(f"time base {sku!r}'s drift.max is {value} {unit}; state it in s/s, a share of real time "
                        "below 1")
    if mark == "ceiling":
        return drift
    if mark != "premise":
        raise Malformed(f"time base {sku!r}'s drift.max is marked {mark!r}, which bounds no clock; mark it \"ceiling\" "
                        "where a page bounds it, or \"premise\" with the premise its source states")
    number = format(Decimal(str(value)).normalize(), "e")
    return ClockPremise(host, f"ClockRateWithin({host}, {number})", drift, mark, _str(source, f"{sku}'s source"))


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
        return {"clock": _rat_str(self.clock),
                "peak_rate": {c: _rat_str(r) for c, r in zip(OPERAND_CLASSES, self.peak_rate)}, "sku": self.sku}

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
    figures in the device model's terms: bits read from or written to device memory a second (`bw`), bits of on-chip
    state (`chip`) and bytes of device memory (`mem`). A device names its sheet by digest. The auditor's data sheet for an SKU
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
    def of_file(cls, card: Mapping[str, Any]) -> Sheet:
        """The sheet a card file gives (`verity.hardware.cards.load(sku)`'s table). Its figures: the SKU, the SM clock in Hz,
        and as every operand class's peak rate the card's s8 MACs a SM-clock (`rate.tensor`; every class is int8) over
        its SMs at that clock. Its memory figures as `Accel.Card.toDevice` reads the card: the bandwidth in bits a
        second, a ceiling rounded up, and the stores' sizes exact."""
        def v(entry: Mapping[str, Any]) -> Fraction:
            return Fraction(str(entry["value"]))

        sms, store = v(card["sm"]["count"]), card["store"]
        chip = (32 * sms * (v(store["reg"]) + v(store["reg_extra"])) + 8 * (sms * v(store["smem"]) + v(store["l2"])))
        mem = v(store["dram"])
        if chip.denominator != 1 or mem.denominator != 1:
            raise Malformed(f"card {card['sku']}: its stores' bits and device memory's bytes are whole")
        clock = v(card["clock"]["sm"]) * 10 ** 6
        figures = DeviceFigures(_str(card["sku"], "sku"), clock, sms * v(card["rate"]["tensor"]) * clock)
        return cls(figures, math.ceil(8 * v(card["memory"]["bandwidth"])), int(chip), int(mem))

    def to_json(self) -> dict[str, Any]:
        return {"bw": self.bw, "chip": self.chip, "figures": self.figures.to_json(), "mem": self.mem}

    @classmethod
    def of_json(cls, j: Any) -> Sheet:
        j = _exactly(j, ("bw", "chip", "figures", "mem"), "a sheet")
        return cls(DeviceFigures.of_json(j["figures"]), _nat(j["bw"], "bw"), _nat(j["chip"], "chip"),
                   _nat(j["mem"], "mem"))

    def digest(self) -> str:
        return tagged(SHEET, self.to_json())


def _pairs(v: Any, what: str) -> list[tuple[Any, Any]]:
    """`v` as a list of pairs, or `Malformed`."""
    if isinstance(v, (str, Mapping)) or not isinstance(v, Iterable):
        raise Malformed(f"{what}: {v!r} is not a list of pairs")
    v = list(v)
    if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in v):
        raise Malformed(f"{what}: {v!r} is not a list of pairs")
    return [(a, b) for a, b in v]


def _shared(v: Any, what: str) -> tuple[tuple[str, int], ...]:
    """A list of a tier and a rate in bits a second, each: the rates a host's leaves of a tier share."""
    return tuple((_one_of(t, TIERS, f"{what}'s tier"), _nat(bw, f"{what}'s rate")) for t, bw in _pairs(v, what))


def _shared_json(shared: Sequence[tuple[str, int]]) -> list[dict[str, Any]]:
    return [{"bw": bw, "tier": t} for t, bw in shared]


def _shared_of(j: Any, what: str) -> tuple[tuple[str, int], ...]:
    return tuple((s["tier"], s["bw"]) for s in (_exactly(s, ("bw", "tier"), what) for s in _list(j, what)))


def _opt(v: Any, read: Any, what: str) -> Any:
    return None if v is None else read(v, what)


@dataclass(frozen=True)
class Loc:
    """A location and its owner. A device names its sheet's digest, a memory leaf its tier and bytes, the auditor's
    devices their keys; a field the location's kind doesn't use is ignored.

    `bw` is the bits a second read from or written to a memory leaf directly under a host, and None on every other
    location: a memory leaf under a device reads at its device's sheet's `bw` (`Site.read_rate`). A disk leaf's `bw`
    bounds the bytes that come off the disk, not a read the host returns from memory, so a read rate over a host's
    disk leaves counts the host's DRAM tier's rate beside them, once. `chip` is a host's on-chip state in bits, as a
    sheet's is, and None on every other location. `shared` is, for a tier, the bits a second read from or written to a
    host's leaves of that tier together, where one ceiling holds them all; empty on every other location. `key` is the
    public key one of the auditor's devices is registered under, and `scheme` the id of the signature scheme that key
    is of (`SCHEMES`), which says how many levels of keys stand under it; both empty on every other location. `drift`
    is how far the clock of one of the auditor's devices may run from real time, as a share of it; None on every other
    location, and a device that states none times nothing. `time_unit` is the site's unit of time in seconds, on a
    location of kind `site` and None on every other (`Site.time_unit`)."""

    id: str
    kind: str
    parent: str | None = None
    owner: str = "prover"
    sheet: str = ""
    tier: str | None = None
    bytes: int = 0
    key: str = ""
    scheme: str = ""
    bw: int | None = None
    chip: int | None = None
    shared: tuple[tuple[str, int], ...] = ()
    drift: Fraction | None = None
    time_unit: Fraction | None = None

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
        _str(self.scheme, f"{self.id}'s scheme")
        _opt(self.bw, _nat, f"{self.id}'s bw")
        _opt(self.chip, _nat, f"{self.id}'s chip")
        object.__setattr__(self, "shared", _shared(self.shared, f"{self.id}'s shared"))
        object.__setattr__(self, "drift", _opt(self.drift, _rat, f"{self.id}'s drift"))
        object.__setattr__(self, "time_unit", _opt(self.time_unit, _rat, f"{self.id}'s time unit"))

    def to_json(self) -> dict[str, Any]:
        return {"bw": self.bw, "bytes": self.bytes, "chip": self.chip,
                "drift": None if self.drift is None else _rat_str(self.drift), "id": self.id, "key": self.key,
                "kind": self.kind, "owner": self.owner, "parent": self.parent, "scheme": self.scheme,
                "shared": _shared_json(self.shared), "sheet": self.sheet, "tier": self.tier,
                "time_unit": None if self.time_unit is None else _rat_str(self.time_unit)}

    @classmethod
    def of_json(cls, j: Any) -> Loc:
        j = _exactly(j, ("bw", "bytes", "chip", "drift", "id", "key", "kind", "owner", "parent", "scheme", "shared",
                         "sheet", "tier", "time_unit"), "a location")
        return cls(j["id"], j["kind"], j["parent"], j["owner"], j["sheet"], j["tier"], j["bytes"], j["key"],
                   j["scheme"], j["bw"], j["chip"], _shared_of(j["shared"], "a shared rate"),
                   _opt(j["drift"], _rat_json, "a location's drift"),
                   _opt(j["time_unit"], _rat_json, "a location's time unit"))


@dataclass(frozen=True)
class MemoryFigures:
    """The auditor's figures for one memory leaf directly under a host: the host it is under, the bits a second read
    from or written to it, and the bytes it holds."""

    host: str
    bw: int
    bytes: int

    def __post_init__(self) -> None:
        _str(self.host, "a memory leaf's host")
        _nat(self.bw, "a memory leaf's bw")
        _nat(self.bytes, "a memory leaf's bytes")


@dataclass(frozen=True)
class HostFigures:
    """The auditor's figures for one host: its on-chip state in bits, the cards it carries, each SKU with its count,
    for a tier the bits a second its leaves of that tier read or write together, and the ways traffic leaves it, each
    by its link's id with the bits a second it carries each way, or None where it states none."""

    chip: int
    cards: tuple[tuple[str, int], ...] = ()
    shared: tuple[tuple[str, int], ...] = ()
    ways: tuple[tuple[str, int | None], ...] = ()

    def __post_init__(self) -> None:
        _nat(self.chip, "a host's chip")
        object.__setattr__(self, "cards", tuple((_str(s, "a card's sku"), _nat(n, "a card's count"))
                                                for s, n in _pairs(self.cards, "a host's cards")))
        object.__setattr__(self, "shared", _shared(self.shared, "a host's shared"))
        object.__setattr__(self, "ways", tuple((_str(i, "a way's id"), _opt(b, _nat, "a way's beta"))
                                               for i, b in _pairs(self.ways, "a host's ways")))


@dataclass(frozen=True)
class Link:
    """A link between two ends, with its capacity `beta` in bits a second each way and its latency in the site's units
    of time. `unmediated` is, on a boundary link no certifier splits, the reason none does; None on every other link. A
    link states no capacity (`beta` None) only where it carries `unmediated`: an interface that reads no line rate has
    none, and no amount is read from it."""

    id: str
    ends: tuple[str, str]
    beta: int | None
    latency: int
    unmediated: str | None = None

    def __post_init__(self) -> None:
        _str(self.id, "a link's id")
        if not isinstance(self.ends, tuple) or len(self.ends) != 2:
            raise Malformed(f"link {self.id}: its ends are a pair")
        for e in self.ends:
            _str(e, f"link {self.id}'s end")
        if self.beta is not None:
            _nat(self.beta, f"link {self.id}'s beta")
        _nat(self.latency, f"link {self.id}'s latency")
        if self.unmediated is not None:
            _str(self.unmediated, f"link {self.id}'s unmediated")

    def to_json(self) -> dict[str, Any]:
        return {"beta": self.beta, "ends": list(self.ends), "id": self.id, "latency": self.latency,
                "unmediated": self.unmediated}

    @classmethod
    def of_json(cls, j: Any) -> Link:
        j = _exactly(j, ("beta", "ends", "id", "latency", "unmediated"), "a link")
        ends = _list(j["ends"], "a link's ends")
        if len(ends) != 2:
            raise Malformed("a link's ends: not two ends")
        return cls(j["id"], (ends[0], ends[1]), j["beta"], j["latency"], j["unmediated"])


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
                   certifier: tuple[str, str, str] | None = None, *, host_card: Mapping[str, Any],
                   ways: Iterable[tuple[str, int | None, bool]], leaves: Any,
                   time_unit: Fraction = TIME_UNIT, clock: ClockPremise | Fraction | None = None) -> Entry:
        """**A node's entry**, following this log's head: what a node file's table (`verity.hardware.nodes.load(kind)`),
        the tables of the GPU card and the host card it names (`verity.hardware.cards.load(sku)` each), the file's ways
        out (`ways`: `verity.hardware.nodes.ways_out(node, host)`, the one reader of their rates) and its leaves
        (`leaves`: `verity.hardware.nodes.leaves(node, host)`, the one reader of theirs) register as host `host` at site
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
        is a boundary link that no certifier splits, and it says what is so of a node none of whose ways out the
        auditor's hardware sits on. Two kinds say why no certifier splits them (`Link.unmediated`, its text from the
        way's kind): an interface that reads no line rate, whose link states no capacity (`NO_LINE_RATE`), and the
        storage path, which no certifier can sit on (`STORAGE_PATH`). A site with either is not mediated, and `onsite`
        takes their reasons. An interface with a rate says none, so `onsite` refuses it by `uncertified` until a
        certifier splits it. `ways` is held to the file: an entry for each `interface.NAME`, in the file's order, with
        a rate exactly where its table states one, then the storage path's, with its rate, exactly where the file has a
        `storage` table. `leaves` is held to it too: each `memory.NAME` by id, tier and bytes, in the file's order. The
        rates' values are the caller's, read by `ways_out` and `leaves`.

        **An entry always registers a link out.** A file that states no interface is refused: its entry would not say
        how the node reaches the outside.

        `certifier` is the id, the key and the key's scheme of a network certifier of the auditor's that stands in
        front of the node without splitting its interface (a process on another machine, say). It adds the certifier
        under the site, stating the drift of the clock of the host it runs in (`clock`, `clock_drift`'s: the figure,
        or no drift for a premise or no clock given; `device_drift`), and the two links of a certified way out, host to certifier (`host/ID`) and
        certifier to outside (`ID/outside`), each at the interfaces' rates summed, beside the ways' own links, which
        stay. The storage path's rate is not in that sum and no certifier is linked to the path: a certifier's links
        say nothing of it. A certifier on a node none of whose interfaces has a rate is refused, since its links would
        have none. A certifier splits one host's way out, so each node names its own, and the key and scheme are the
        ones the auditor holds for the id or `owner` refuses the entry. None of the auditor's other devices is added.

        A file that states no `dram` leaf or no `disk` leaf is refused: its entry would pass `check` for `remote` with
        a host that holds less memory than the node has, and whatever reads the site would take the node for smaller
        than it is."""
        n = _Node.read(node, host_card, host, ways, leaves)
        if clock is not None and certifier is None:
            raise Malformed("`clock` is the clock of the host a network certifier runs in, and no certifier is given; "
                            "register the node without one, or name the certifier")
        rates = [bw for _, bw, split in n.ways if split and bw is not None]
        if certifier is not None and not rates:
            raise Malformed(f"no interface the node file states has a rate ({[i for i, _, split in n.ways if split]})"
                            ", so a network certifier's links would have none: an interface that reads no line rate "
                            "takes no certifier. Register the node without one")
        sheet = Sheet.of_file(card)
        if sheet.sku != n.sku:
            raise Malformed(f"the node's devices are {n.sku!r}, and the card given is {sheet.sku!r}'s")
        before, digest = Site.of_log(self), sheet.digest()
        locs = [] if before.known(site) else [Loc(site, "site", time_unit=time_unit)]
        locs.append(Loc(host, "host", site, chip=n.chip, shared=n.shared))
        for gpu in (f"{host}/gpu{i}" for i in range(n.count)):
            locs += [Loc(gpu, "device", host, sheet=digest),
                     Loc(f"{gpu}/hbm", "memory", gpu, tier="hbm", bytes=sheet.mem)]
        locs += [Loc(i, "memory", host, tier=tier, bytes=size, bw=bw) for i, tier, size, bw in n.leaves]
        links = [Link(i, (host, OUTSIDE), bw, 0, STORAGE_PATH if not split else None if bw else NO_LINE_RATE)
                 for i, bw, split in n.ways]
        if certifier is not None:
            if not (isinstance(certifier, (tuple, list)) and len(certifier) == 3):
                raise Malformed(f"`certifier` is a network certifier's id, its key and the key's scheme, not "
                                f"{certifier!r}")
            device, key, scheme = certifier
            beta = sum(rates)
            locs.append(Loc(device, "certifier", site, "auditor", key=key, scheme=scheme, drift=device_drift(clock)))
            links += [Link(f"{host}/{device}", (host, device), beta, 0),
                      Link(f"{device}/{OUTSIDE}", (device, OUTSIDE), beta, 0)]
        return self.entry(() if any(k.digest() == digest for k in before.sheets) else (sheet,), locs, links)

    def to_json(self) -> dict[str, Any]:
        return {"composite": self.composite, "entries": [e.to_json() for e in self.entries]}

    @classmethod
    def of_json(cls, j: Any) -> Registration:
        j = _exactly(j, ("composite", "entries"), "the log")
        return cls(j["composite"], tuple(Entry.of_json(e) for e in _list(j["entries"], "entries")))


def _lookup(pairs: Iterable[tuple[str, Any]], i: str) -> Any:
    """The value under `i` in a list of pairs: the first (Lean's `List.lookup`), or None."""
    return next((v for k, v in pairs if k == i), None)


@dataclass(frozen=True)
class Auditor:
    """What the auditor brings to the check: its data sheet for each SKU, public, whose figures are its estimate of the
    SKU's; the relative tolerance registration allows around an estimate; the key of each of its own devices and the
    scheme it is of, by the device's id (`keys`, each `(id, key, scheme)`); its figures for each memory leaf directly
    under a host, by the leaf's id (`memory`); its figures for each host, by the host's id (`hosts`); and how far each
    of its own devices' clocks may run from real time, by the device's id (`drift`). The first under an SKU or an id
    is the one the check reads. That these figures are the hardware is a public input: no check holds it and no
    premise says it, and a result is over the registered hardware."""

    sheets: tuple[Sheet, ...] = ()
    tolerance: Fraction = Fraction(0)
    keys: tuple[tuple[str, str, str], ...] = ()
    memory: tuple[tuple[str, MemoryFigures], ...] = ()
    hosts: tuple[tuple[str, HostFigures], ...] = ()
    drift: tuple[tuple[str, Fraction], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sheets", tuple(self.sheets))
        object.__setattr__(self, "tolerance", _rat(self.tolerance, "the auditor's tolerance"))
        keys = list(self.keys)
        if not all(isinstance(k, (tuple, list)) and len(k) == 3 for k in keys):
            raise Malformed(f"the auditor's keys: {keys!r}; each is a device's id, its key and the key's scheme")
        object.__setattr__(self, "keys", tuple((_str(d, "a device"), _str(k, "a key"), _str(c, "a scheme"))
                                               for d, k, c in keys))
        for f, cls in (("memory", MemoryFigures), ("hosts", HostFigures)):
            pairs = tuple((_str(i, f"an id of the auditor's {f}"), v) for i, v in _pairs(getattr(self, f), f))
            if not all(isinstance(v, cls) for _, v in pairs):
                raise Malformed(f"the auditor's {f}: each is an id and its {cls.__name__}")
            object.__setattr__(self, f, pairs)
        object.__setattr__(self, "drift", tuple((_str(i, "a device"), _rat(q, f"{i}'s drift"))
                                                for i, q in _pairs(self.drift, "the auditor's drift")))

    @classmethod
    def of_files(cls, cards: Iterable[Mapping[str, Any]], tolerance: Fraction = Fraction(0),
                 keys: Mapping[str, tuple[str, str]] | None = None, *,
                 memory: Iterable[tuple[str, MemoryFigures]] = (), hosts: Iterable[tuple[str, HostFigures]] = (),
                 drift: Iterable[tuple[str, Fraction]] = (),
                 clocks: Mapping[str, ClockPremise | Fraction] | None = None) -> Auditor:
        """The auditor whose data sheets come from these card files (`verity.hardware.cards.load(sku)` each), at this
        tolerance, with its devices' keys, each a key and its scheme by the device's id, and these figures
        (`Registration.node_figures` gives a node's). `clocks` gives, by the id of each of its devices, the clock of
        the host it runs in, `clock_drift`'s: a figure is the device's drift, after `drift`'s; a premise gives none
        (`device_drift`), and is the row's to name (`ClockPremise.premised`)."""
        clocked = [(d, q) for d, c in (clocks or {}).items() if (q := device_drift(c)) is not None]
        return cls(tuple(Sheet.of_file(c) for c in cards), tolerance,
                   tuple((d, k, c) for d, (k, c) in (keys or {}).items()), tuple(memory), tuple(hosts),
                   (*drift, *clocked))

    def sheet(self, sku: str) -> Sheet | None:
        """The data sheet for `sku`: the first under it."""
        return next((k for k in self.sheets if k.sku == sku), None)

    def estimate(self, a: Sheet) -> DeviceEstimate:
        """**The auditor's estimate** its sheet `a` states: `a`'s figures, at the auditor's tolerance."""
        return DeviceEstimate(a.figures, self.tolerance)

    def key(self, i: str) -> str | None:
        """The key of the auditor's device `i`: the first under it."""
        return next((k for d, k, _ in self.keys if d == i), None)

    def scheme(self, i: str) -> str | None:
        """The scheme of the key of the auditor's device `i`: the first key's under it."""
        return next((c for d, _, c in self.keys if d == i), None)

    def to_json(self) -> dict[str, Any]:
        return {"drift": [{"drift": _rat_str(q), "id": i} for i, q in self.drift],
                "hosts": [{"cards": [{"count": n, "sku": sku} for sku, n in h.cards], "chip": h.chip, "id": i,
                           "shared": _shared_json(h.shared), "ways": [{"beta": b, "id": w} for w, b in h.ways]}
                          for i, h in self.hosts],
                "keys": [{"id": d, "key": k, "scheme": c} for d, k, c in self.keys],
                "memory": [{"bw": m.bw, "bytes": m.bytes, "host": m.host, "id": i} for i, m in self.memory],
                "sheets": [k.to_json() for k in self.sheets], "tolerance": _rat_str(self.tolerance)}

    @classmethod
    def of_json(cls, j: Any) -> Auditor:
        j = _exactly(j, ("drift", "hosts", "keys", "memory", "sheets", "tolerance"), "the auditor")
        drift = [_exactly(d, ("drift", "id"), "a device's drift") for d in _list(j["drift"], "drift")]
        keys = [_exactly(k, ("id", "key", "scheme"), "a key") for k in _list(j["keys"], "keys")]
        memory = [_exactly(m, ("bw", "bytes", "host", "id"), "a memory leaf's figures")
                  for m in _list(j["memory"], "memory")]
        hosts = [_exactly(h, ("cards", "chip", "id", "shared", "ways"), "a host's figures")
                 for h in _list(j["hosts"], "hosts")]

        def host(h: Mapping[str, Any]) -> HostFigures:
            cards = [_exactly(k, ("count", "sku"), "a host's card") for k in _list(h["cards"], "a host's cards")]
            ways = [_exactly(w, ("beta", "id"), "a host's way out") for w in _list(h["ways"], "a host's ways")]
            return HostFigures(h["chip"], tuple((k["sku"], k["count"]) for k in cards),
                               _shared_of(h["shared"], "a shared rate"), tuple((w["id"], w["beta"]) for w in ways))

        return cls(tuple(Sheet.of_json(k) for k in _list(j["sheets"], "sheets")),
                   _rat_json(j["tolerance"], "the auditor's tolerance"),
                   tuple((k["id"], k["key"], k["scheme"]) for k in keys),
                   tuple((m["id"], MemoryFigures(m["host"], m["bw"], m["bytes"])) for m in memory),
                   tuple((h["id"], host(h)) for h in hosts),
                   tuple((d["id"], _rat_json(d["drift"], "a device's drift")) for d in drift))


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

    @property
    def time_unit(self) -> Fraction | None:
        """The site's unit of time in seconds: the one its first location of kind `site` states. Its views round a
        rate at it, and its links' latency is counted in it."""
        x = next((x for x in self.locs if x.kind == "site"), None)
        return None if x is None else x.time_unit

    @property
    def mediated(self) -> bool:
        """No link of the site says that no certifier splits it."""
        return all(k.unmediated is None for k in self.links)

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

    def device(self, d: str, auditor: Auditor, cuda: int = 0, oracle: int = 0,
               cap: Mapping[str, int] | None = None) -> dict[str, Any] | None:
        """**Device `d`'s model** (`Accounting.Device`, `Site.device`), at ticks of the site's unit of time
        (`time_unit`), read off its sheet `k` and the auditor's sheet `a` for `k`'s SKU: `k`'s on-chip state (`chip`),
        `k`'s device memory in bits (`mem`), and a tick's bits from device memory to the chip (`bw`) and tensor-core
        MACs (`rate["tc"]`), each a rate a second over one tick, rounded up: `k`'s `bw`, and the highest peak rate of
        the figures the bound reads (`DeviceFigures.bounding`: class by class, the larger of `k`'s and `a`'s
        estimate's). No sheet states a device's CUDA-core ops or oracle queries a tick, or its side channels'
        capacities (`cap`), so they are parameters. None when the site states no unit of time, `d` names no registered
        sheet or the auditor has none for its SKU."""
        tick, k = self.time_unit, self.sheet(d)
        a = None if k is None else auditor.sheet(k.sku)
        if tick is None or k is None or a is None:
            return None
        f = k.figures.bounding(auditor.estimate(a))
        return {"bw": math.ceil(tick * k.bw), "cap": dict(cap or {}), "chip": k.chip, "mem": 8 * k.mem,
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

    def read_rate(self, xs: Iterable[str]) -> int | None:
        """**The bits a second readable from the memory leaves `xs`** (`Site.readRate`): a host's leaves of one tier
        at the sum of their own registered rates, or at the host's shared rate for the tier where that is less; and
        each device with a leaf among them at its sheet's `bw`, once. A disk leaf's rate bounds the bytes that come off
        the disk, and a host can return a disk leaf's bytes from memory, so a host with a disk leaf among `xs` counts
        its whole DRAM tier's rate, whichever of its DRAM leaves are among them. None when an id names nothing, or a
        location states no rate and is under no device with a registered sheet."""
        ys = [self.loc(x) for x in xs]
        if any(y is None for y in ys):
            return None
        devices = [y.parent for y in ys if y.bw is None]
        if any(d is None for d in devices):
            return None
        sheets = [self.sheet(d) for d in dict.fromkeys(devices)]
        if any(k is None for k in sheets):
            return None
        rated = [y for y in ys if y.bw is not None]

        def tier(zs: Iterable[Loc], g: tuple[str | None, str | None]) -> int:
            total = sum(y.bw for y in zs if (y.parent, y.tier) == g and y.bw is not None)
            h = None if g[0] is None else self.loc(g[0])
            ceiling = None if h is None or g[1] is None else _lookup(h.shared, g[1])
            return total if ceiling is None else min(total, ceiling)

        disks = list(dict.fromkeys(y.parent for y in rated if y.tier == "disk"))
        groups = dict.fromkeys([(y.parent, y.tier) for y in rated] + [(h, "dram") for h in disks])
        return (sum(tier(self.locs, g) if g[1] == "dram" and g[0] in disks else tier(rated, g) for g in groups)
                + sum(k.bw for k in sheets))

    def over(self, rate: int, ticks: int) -> int | None:
        """What a rate of `rate` bits a second carries over `ticks` of the site's units of time, rounded up once
        (`Site.over`); None when the site states no unit of time."""
        u = self.time_unit
        return None if u is None else math.ceil(u * rate * ticks)

    def boundary(self, h: str | None = None) -> list[Link]:
        """The boundary links (`is_boundary`) with an end at host `h`, or every boundary link."""
        return [k for k in self.links if self.is_boundary(k) and (h is None or h in k.ends)]

    def to_json(self) -> dict[str, Any]:
        return {"links": [k.to_json() for k in self.links], "locs": [x.to_json() for x in self.locs],
                "sheets": [k.to_json() for k in self.sheets]}

    @classmethod
    def of_json(cls, j: Any) -> Site:
        j = _exactly(j, ("links", "locs", "sheets"), "a site")
        return cls(tuple(Sheet.of_json(k) for k in _list(j["sheets"], "sheets")),
                   tuple(Loc.of_json(x) for x in _list(j["locs"], "locs")),
                   tuple(Link.of_json(k) for k in _list(j["links"], "links")))


def _dups(xs: Sequence[str]) -> list[str]:
    return sorted({x for x in xs if xs.count(x) > 1})


def _fresh(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """No id is registered twice, the outside's among them, and no sheet twice."""
    t = s.add(e)
    digests = [k.digest() for k in t.sheets]
    if dup := _dups(t.ids()) + [k.sku for k in t.sheets if digests.count(k.digest()) > 1]:
        return f"{dup} registered twice"
    return None


def _refs(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new location's parent and each new link's ends name something registered."""
    t = s.add(e)
    for x in e.locs:
        if x.parent is not None and not t.known(x.parent):
            return f"{x.id}'s parent {x.parent!r} isn't registered"
    for k in e.links:
        if bad := [n for n in k.ends if not t.known(n)]:
            return f"link {k.id} names {bad}, which aren't registered"
    return None


def _kinds(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Parents and tiers by kind, links between kinds a link may join, each at a positive capacity or, where it says
    no certifier splits it, at none, and each certifier splitting exactly one uplink."""
    t = s.add(e)
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
            return f"certifier {x.id} doesn't split one uplink: it links to {far}"
    return None


def _sheets(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered sheet."""
    for k in e.sheets:
        if a.sheet(k.sku) is None:
            return f"sheet {k.sku} ({k.digest()[:16]}...) names an SKU the auditor has no data sheet for"
    digests = [k.digest() for k in s.add(e).sheets]
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
    t = s.add(e)
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
    t = s.add(e)
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
    t = s.add(e)
    for x in e.locs:
        if x.kind == "memory" and x.bytes == 0:
            return f"memory {x.id} holds nothing"
    for x in t.locs:
        if x.kind == "device" and (k := t.sheet_of(x)) is not None and t.held(x.id) > k.mem:
            return f"{x.id}'s memory leaves hold {t.held(x.id)} bytes, over its sheet's {k.mem}"
    return None


def _owners(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Each new certifier or challenger is one of the auditor's own devices, owned by it, with its key, its key's
    scheme, one of `SCHEMES`, and the auditor's figure for its clock, below 1 where there is one; every other new
    location is the prover's and states no scheme and no clock."""
    for x in e.locs:
        if x.kind in AUDITOR_KINDS:
            if x.owner != "auditor" or a.key(x.id) != x.key or a.scheme(x.id) != x.scheme:
                return f"{x.kind} {x.id} isn't one of the auditor's own devices, with its key and scheme"
            if x.scheme not in SCHEMES:
                return f"{x.kind} {x.id}'s key is of scheme {x.scheme!r}, not one of {list(SCHEMES)}"
            if x.drift != _lookup(a.drift, x.id) or (x.drift is not None and x.drift >= 1):
                return (f"{x.kind} {x.id} states the drift {x.drift}, and the auditor's figure is "
                        f"{_lookup(a.drift, x.id)}; it must be that, below 1")
        elif x.owner != "prover" or x.scheme != "" or x.drift is not None:
            return (f"{x.kind} {x.id} is a prover's location, registered as the {x.owner}'s, or states a scheme or a "
                    "drift")
    return None


def _covered(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Onsite: every boundary link is certified, or says why no certifier splits it."""
    t = s.add(e)
    if c == "onsite" and (bare := [k.id for k in t.links if t.is_boundary(k) and not t.certified(k)
                                   and k.unmediated is None]):
        return f"boundary links {bare} have no certifier splitting them and say no reason none does"
    return None


def _reached(c: str, a: Auditor, s: Site, e: Entry) -> str | None:
    """Onsite: the host of every memory leaf is at an end of a memory challenger's link."""
    t = s.add(e)
    if c == "onsite":
        reached = t.reached()
        if far := sorted({h for x in t.locs if x.kind == "memory" and (h := t.host_of(x)) is not None} - set(reached)):
            return f"hosts {far} hold the prover's memory, and no memory challenger links to them"
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
    return entry_refusal(log.composite, Site.of_log(log), log.head, e, auditor)


def check_registration(log: Registration, e: Entry, auditor: Auditor) -> Registration:
    """One entry held to the log before it: the log with `e` appended, or `Refused` by the first check it fails."""
    if r := _first(log.composite, Site.of_log(log), log.head, e, auditor):
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
        s, head = s.add(e), e.digest()
    if lack := incomplete(s, auditor):
        raise Refused("inventory", lack, len(log.entries))
    return s
