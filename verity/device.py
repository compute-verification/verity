"""The Site in code: the registered hardware, folded from the registration log, the check each entry passes, and the
views the protocols read; and the readers of the card files and the node files, the data sheets it is registered from.
It mirrors the spec's `Verity.Site` (`security/Definitions/Core/Site.lean`): its sheets, locations,
links and log, `Site.ofLog`, and `Registration.Entry.refusal`, check for check and in the same order.

The Site is a tree of locations, each with its owner: the site at the root, its hosts, a host's devices, memory leaves
by tier and bytes under a device or a host, and at the site the auditor's devices, its network certifiers and memory
challengers (`PARENTS`). Links join two ends, each a location or `OUTSIDE`, with a capacity `beta` in bits a tick each
way and a latency in ticks (`LINKS`). A network certifier is the leaf that splits an uplink: one link from a host to it,
one from it to the outside. Hosts carry no compute: a device names its sheet by digest. A sheet (`Sheet`), a card, is
the figures the prover states for an SKU: its compute figures (`DeviceFigures`: the SKU, the tensor clock and the peak
rate of each operand class) and its memory figures (bits from device memory to the chip a tick, bits of on-chip state and
bytes of device memory). The auditor's data sheets, public and looked up by SKU, are sheets too, whose figures are its
estimate of the SKU's (`Auditor.estimate`); they come from the card files (below): `Sheet.of_file` reads one as
`load_card(sku)` returns it. A node's entry comes from its files too: `Registration.node_entry` reads a node file's
table (`load_node(kind)`), its GPU card's and the file's ways out (`ways_out`), and registers each way out that has a
rate, an interface or the storage path, as a boundary link that no certifier splits.

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
(`verity.channel.canonical`) under the registration log's tags; a rational is the string `n` or `n/d` in lowest terms.

**The card files**: one TOML file per SKU (`<sku>.toml` in `CARDS`), the one source of each card's numbers. A card is a
GPU's (`kind = "gpu"`), a host's (`kind = "host"`: a server's CPUs and DRAM, apart from the GPUs it carries), a
server's (`kind = "server"`: a server model's network ports, apart from its CPUs, DRAM and GPUs) or a time base's
(`kind = "timebase"`: the clock an auditor's device reads, apart from the host it runs in; its one section is `drift`:
`max`, the most the clock's rate may differ from the SI second's (`s/s`), and `read`, the largest reading of it on
record, a `measurement`). Every number in a card file is an entry `{value, unit, mark, source}`. `mark` is `ceiling`
(no program on that SKU exceeds it, whatever firmware it runs: what the model's premises need), `measurement` (a
measured value, so only a lower bound on the ceiling), `premise` (no page and no run bounds the number; it is assumed
under the premise its source states, of whom it names. A row that uses it prints the premise for its device's host, and
no figure of record uses it) or `placeholder` (none of these: open). `source` names a data sheet, a document or an
`art:` id (a run's record), and a node by its kind, never its host name. The Lean card tables
(`security/Definitions/Core/Accel`) are held to the GPU cards that name one (`lean`), number by number
(`verity/hardware/tests/test_lean.py`). `rate` is the one calculator of a GPU's peak rates, Lean's `card.ρ`: a reader
that needs the tensor peak at some clock calls it rather than multiplying the card's numbers itself. `readings` is the
tensor peak under each reading of its clock the card states (`READINGS`): a reader that quotes a figure under every
reading iterates it rather than naming a clock itself.

~~~python
from verity.device import load_card, number, rate, readings, skus
skus("host")                       # ['xeon-6776p-2s', 'xeon-platinum-8468-2s']
drift = load_card("nebius-vm-monotonic-raw")["drift"]["max"]  # {'value': 0.0005, 'unit': 's/s', 'mark': 'premise', ...}
number(drift)                      # Fraction(1, 2000): the auditor's figure for its device's drift, under that premise
card = load_card("h200-sxm-141gb")
card["clock"]["sm"]["value"]       # 1980 (MHz)
card["memory"]["bandwidth"]        # {'value': 4916736000000.0, 'unit': 'B/s', 'mark': 'ceiling', 'source': ...}
load_card("dgx-h200")["link"]["rate"]   # {'value': 5011000000000.0, 'unit': 'bit/s', 'mark': 'ceiling', 'source': ...}
rate(card)                         # Fraction(989429760000000): s8 MACs a second, 132 SMs x 4096 x clock.tensor 1830 MHz
rate(card, mhz=1980)               # Fraction(1070530560000000): the same at clock.sm's 1980 MHz, the second reading
readings(card)                     # {'tensor': Fraction(989429760000000), 'sm': Fraction(1070530560000000)}: both
rate(card, fmt="f16", mhz=1700)    # the f16 MACs a second at a 1700 MHz clock
~~~

**The node files**: one TOML file per kind of node (`<kind>.toml` in `NODES`), holding what a node's registration
states that no card does. A node is a host and the GPUs it carries. Its file names the GPU card and how many of it the
node carries (`device.sku`, `device.count`), the host card (`host.sku`), the host's own memory, one table a leaf
(`memory.NAME`: its `tier`, `dram` or `disk`, its `bytes` and its `rate`, the most its bytes are read back at, in
`B/s`, a ceiling), and the node's ways out: the interfaces that carry traffic off the node, one table each
(`interface.NAME`: its `rate`, the line rate each way in `bit/s`, or, where the interface reads none, its `driver` and
the `source` of that read, and no rate), and the storage path, once (`storage`, below). A card's own memory is its
card's `store.dram`, whole, and no clock or peak of a card is here: those are the cards'. Every number is an entry
`{value, unit, mark, source}` under the cards' rules, so what is not a number is not written as one, and a leaf or an
interface with no read on record has no table. A kind is the card and its count, never a host's name, and a file names
no machine, no address and no provider: whoever registers a node gives its host's id (`Registration.node_entry`, which
gives no entry of a file with no `dram` or no `disk` leaf, none of a file with no interface, and none of a file with no
way out that has a rate).

**A leaf's rate** is the tightest bound on record that holds whatever the node runs. The DRAM leaf's is the host
card's `memory.bandwidth_max`, the server's channels. A disk's is its own interface's where the disk states one (a
drive of the server's own: its link's lanes at their rate, less the coding), and else the host card's
`disk.read_max`, the lanes into the server's sockets, which every byte of a disk enters by. That figure of the host
card bounds the host's disks together too, so `leaves` gives it beside each leaf's own, once a tier: a reader that
adds the leaves' rates of a tier takes the smaller of the sum and the tier's. Neither a limit a provider sets on a
disk nor a measured rate is a ceiling, so neither is a leaf's rate.

**The ways out** are what a byte leaves the node by, and `ways_out` is their one reader. An interface is one, and a
network certifier may split it. **The storage path** is the other: a disk that is not the server's own (a network
disk, which can be attached to another machine or copied through whoever provides it) takes what is written to it
off the node past every interface, so no certifier can split it. A file states it once, as `storage`: `leaves`,
the names of the disk leaves that ride it, and `source`, why they leave. It states no rate of its own: its rate is
the one that bounds the node's disks together, `leaves`'s for `disk`. A server whose drives are all its own has no
`storage` table and no such way out.

~~~python
from verity.device import Auditor, Registration, check, leaves, load_card, load_node, ways_out
node = load_node("rtx-pro-6000-server-x8")
gpu = load_card(node["device"]["sku"])
log = Registration("remote")
entry = log.node_entry(node, gpu, host="node0", ways=ways_out(node, "node0"))
site = check(log.append(entry), Auditor.of_files([gpu]))
site.units                          # ('node0/gpu0', ..., 'node0/gpu7')
site.tiers("node0")                 # {'hbm': 824633720832, 'dram': 1872605741056, 'disk': 6266358333440}
[(k.id, k.beta) for k in site.boundary("node0")]   # [('node0/eth0', 400), ('node0/storage', 5585)], bits a nanosecond
each, tier = leaves(node, "node0")
[(leaf.id, leaf.bw) for leaf in each]   # [('node0/dram', 9011200000000), ('node0/vda', 5584738461544), ...], bit/s
tier                                # {'dram': 9011200000000, 'disk': 5584738461544}, each tier's leaves together
ways_out(node, "node0")            # (Way('node0/eth0', 400000000000, True), Way('node0/storage', 5584738461544, False))
~~~

`CARDS` and `NODES`, the directories the sheets are in, are `hardware/cards/` and `hardware/nodes/` beside the `verity`
package's directory, the first of this file's resolved parents so named. The readers of card files (`skus`,
`load_card`) and of node files (`node_kinds`, `load_node`) take `directory`, and every reader of the node files that
reads a card file (`load_node`, `leaves`, `ways_out`) takes `cards_directory`, the directories the files are in, so
that a reader whose copy of them is elsewhere names where it is; `None` is `CARDS` for a card file and `NODES` for a
node file, read when the reader is called.

Stdlib, the wire (`verity.channel`) and `verity.errors` only.
"""


from __future__ import annotations

import hashlib
import math
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

from verity.channel import canonical
from verity.errors import InvalidArtifact

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
    """A way out of host `host`'s node as `ways_out` gives it: its id under the host, what it
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
        """The sheet a card file gives (`load_card(sku)`'s table). Its figures: the SKU, the clock of the card's default
        reading (`clock.default`, one of `READINGS`) in Hz, and as every operand class's peak rate the tensor's peak at
        that reading, `readings(card)[card["clock"]["default"]]` (every class is int8, `rate.tensor`'s unit). Its memory
        figures at ticks of `tick` seconds, as `Accel.Card.toDevice` reads the card: the bandwidth a ceiling rounded up,
        and the stores' sizes exact."""
        def v(entry: Mapping[str, Any]) -> Fraction:
            return Fraction(str(entry["value"]))

        sms, store = v(card["sm"]["count"]), card["store"]
        chip = (32 * sms * (v(store["reg"]) + v(store["reg_extra"])) + 8 * (sms * v(store["smem"]) + v(store["l2"])))
        mem = v(store["dram"])
        if chip.denominator != 1 or mem.denominator != 1:
            raise Malformed(f"card {card['sku']}: its stores' bits and device memory's bytes are whole")
        default = card["clock"]["default"]
        clock = v(card["clock"][default]) * 10 ** 6
        figures = DeviceFigures(_str(card["sku"], "sku"), clock, readings(card)[default])
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

    def node_entry(self, node: Mapping[str, Any], card: Mapping[str, Any], host: str, site: str = "site",
                   tick: Fraction = TICK, certifier: tuple[str, str] | None = None, *,
                   ways: Iterable[tuple[str, int | None, bool]]) -> Entry:
        """**A node's entry**, following this log's head: what a node file's table (`load_node(kind)`),
        the table of the GPU card it names (`load_card(sku)`) and the file's ways out
        (`ways`: `ways_out(node, host)`, the one reader of their rates) register as host `host`
        at site `site`. It adds the sheet `Sheet.of_file` gives the card and the site's location, each only where the
        log has none; the host; the node's `device.count` devices `host/gpuI`, each naming the sheet, with one `hbm`
        leaf `host/gpuI/hbm` holding the sheet's device memory, whole; for each `memory.NAME` of the file a leaf
        `host/NAME` under the host, at the tier and bytes it states, in the file's order; and for each way out that
        has a rate a link under the way's id from the host to the outside, at the bits that rate moves in a tick of
        `tick` seconds, rounded up, and latency 0. The file names no host, so the caller gives its id, and every id
        under it starts with that.

        A way out is an interface of the file's (`host/NAME`) or its storage path (`host/storage`: what the disks
        that are not the server's own carry off the node, at the rate that bounds the node's disks together). Its link
        is a boundary link that no certifier splits: it says what is so of a node none of whose ways out the
        auditor's hardware sits on, and `onsite` refuses it by `uncertified`. An interface with no rate registers no
        link. `ways` is held to the file: an entry for each `interface.NAME`, in the file's order, with a rate exactly
        where its table states one, then the storage path's, with its rate, exactly where the file has a `storage`
        table. The rates' values are the caller's, read by `ways_out`.

        **An entry always registers a link out.** A file that states no interface is refused: its entry would not say
        how the node reaches the outside. So is a file none of whose ways out has a rate, whose entry would hold no
        link out.

        `certifier` is the id and the key of a network certifier of the auditor's that stands in front of the node
        without splitting its interface (a process on another machine, say). It adds the certifier under the site and
        the two links of a certified way out, host to certifier (`host/ID`) and certifier to outside (`ID/outside`),
        each at the interfaces' rates summed, beside the ways' own links, which stay. The storage path's rate is not
        in that sum and no certifier is linked to the path: a certifier's links say nothing of it. A certifier on a
        node none of whose interfaces has a rate is refused, since its links would have none. A certifier splits one
        host's way out, so each node names its own, and the key is the one the auditor holds for the id or `owner`
        refuses the entry. None of the auditor's other devices is added.

        A file that states no `dram` leaf or no `disk` leaf is refused: its entry would pass `check` for `remote` with
        a host that holds less memory than the node has, and whatever reads the site would take the node for smaller
        than it is."""
        try:
            sku, count = node["device"]["sku"], _whole(node["device"]["count"], "a node's device.count")
            leaves = [(name, leaf["tier"], _whole(leaf["bytes"], f"a node's memory.{name}.bytes"))
                      for name, leaf in node.get("memory", {}).items()]
            stated = [(f"{host}/{name}", True, "rate" in port.keys())
                      for name, port in node.get("interface", {}).items()]
            stated += [(f"{host}/storage", False, True)] if "storage" in node else []
        except (AttributeError, KeyError, TypeError) as e:
            raise Malformed(f"a node file's table: {e!r}; it has device.sku, device.count, for each memory.NAME a "
                            "tier and bytes, and for each interface.NAME a table") from None
        try:
            out = [_way(way, host) for way in ways]
        except TypeError:
            raise Malformed(f"`ways` is {ways!r}; give the node file's ways out, "
                            "`ways_out(node, host)`") from None
        given = [(name, split, bw is not None) for name, bw, split in out]
        if given != stated:
            raise Malformed(f"the ways out given are {given}, and the node file states {stated} under host {host!r} "
                            "(each interface, then its storage path, with whether a certifier may split it and "
                            "whether it has a rate); give `ways_out(node, host)`")
        unread = [tier for tier in ("dram", "disk") if all(t != tier for _, t, _ in leaves)]
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
        if all(bw is None for _, bw, _ in out):
            raise Malformed(f"no way out the node file states has a rate ({[name for name, _, _ in out]}), so its "
                            "entry would hold no link out and nothing would bound what leaves the node; state the "
                            "line rate of an interface that reads one, or the storage path its disks ride")
        rates = [bw for _, bw, split in out if split and bw is not None]
        if certifier is not None and not rates:
            raise Malformed(f"no interface the node file states has a rate ({[name for name, _, split in out if split]})"
                            ", so a network certifier's links would have none: an interface that reads no line rate "
                            "registers no link and takes no certifier. Register the node without one")
        sheet = Sheet.of_file(card, tick)
        if sheet.sku != sku:
            raise Malformed(f"the node's devices are {sku!r}, and the card given is {sheet.sku!r}'s")
        before, digest = Site.of_log(self), sheet.digest()
        locs = [] if before.known(site) else [Loc(site, "site")]
        locs.append(Loc(host, "host", site))
        for gpu in (f"{host}/gpu{i}" for i in range(count)):
            locs += [Loc(gpu, "device", host, sheet=digest),
                     Loc(f"{gpu}/hbm", "memory", gpu, tier="hbm", bytes=sheet.mem)]
        locs += [Loc(f"{host}/{name}", "memory", host, tier=tier, bytes=n) for name, tier, n in leaves]
        links = [Link(name, (host, OUTSIDE), math.ceil(bw * tick), 0) for name, bw, _ in out if bw is not None]
        if certifier is not None:
            try:
                device, key = certifier
            except (TypeError, ValueError):
                raise Malformed(f"`certifier` is a network certifier's id and its key, not {certifier!r}") from None
            beta = math.ceil(sum(rates) * tick)
            locs.append(Loc(device, "certifier", site, "auditor", key=key))
            links += [Link(f"{host}/{device}", (host, device), beta, 0),
                      Link(f"{device}/{OUTSIDE}", (device, OUTSIDE), beta, 0)]
        return self.entry(() if any(k.digest() == digest for k in before.sheets) else (sheet,), locs, links)

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
        """The auditor whose data sheets come from these card files (`load_card(sku)` each), at this
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
                "tolerance": _rat_str(self.tolerance)}

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

    @classmethod
    def of_json(cls, j: Any) -> Site:
        j = _exactly(j, ("links", "locs", "sheets"), "a site")
        return cls(tuple(Sheet.of_json(k) for k in _list(j["sheets"], "sheets")),
                   tuple(Loc.of_json(x) for x in _list(j["locs"], "locs")),
                   tuple(Link.of_json(k) for k in _list(j["links"], "links")))


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


# --- the card files and the node files ---------------------------------------------------------------------------------

_PACKAGE = next(p for p in Path(__file__).resolve().parents if p.name == "verity")
"""The `verity` package's directory, the first of this file's resolved parents so named: the sheets are in the
top-level `hardware/` beside it, whatever this file's depth in the package, and wherever a link to the package that
imported it is (a `research run` job imports `verity` through one)."""
CARDS = _PACKAGE.parent / "hardware" / "cards"
CARD_KINDS = ("gpu", "host", "server", "timebase")
MARKS = ("ceiling", "measurement", "premise", "placeholder")
ENTRY = frozenset({"value", "unit", "mark", "source"})
WORD = 4  # bytes in a 32-bit word, the unit of every store and of device memory's rate


def skus(kind: str | None = None, *, directory: Path | None = None) -> list[str]:
    """The SKUs with a card file in ``directory`` (``CARDS`` when ``None``), or those of one ``kind``."""
    every = sorted(p.stem for p in (CARDS if directory is None else directory).glob("*.toml"))
    return every if kind is None else [sku for sku in every if load_card(sku, directory=directory)["kind"] == kind]


def load_card(sku: str, *, directory: Path | None = None) -> dict:
    """The card file of ``sku`` in ``directory`` (``CARDS`` when ``None``) as parsed TOML, refused unless every number
    in it carries its unit, mark and source, and a GPU's unless its ``clock.default`` is one of ``READINGS``."""
    directory = CARDS if directory is None else directory
    path = directory / f"{sku}.toml"
    if not path.is_file():
        raise KeyError(f"no card {sku!r}: the cards are {skus(directory=directory)} ({directory}/<sku>.toml)")
    card = tomllib.loads(path.read_text(encoding="utf-8"))
    if card.get("sku") != sku:
        raise InvalidArtifact(f"{path.name}: sku = {card.get('sku')!r}; set it to the file's name, {sku!r}")
    if card.get("kind") not in CARD_KINDS:
        raise InvalidArtifact(f"{path.name}: kind = {card.get('kind')!r}; set it to one of {CARD_KINDS}")
    check_card(card, path.name)
    if card["kind"] == "gpu" and card.get("clock", {}).get("default") not in READINGS:
        raise InvalidArtifact(f"{path.name}: clock.default = {card.get('clock', {}).get('default')!r}; set it to the "
                              f"reading its figures are stated at, one of {READINGS}")
    return card


def number(entry: dict) -> Fraction:
    """A card entry's value as an exact rational, read from its decimal text (``1.597568e12`` is that number exactly)."""
    return Fraction(str(entry["value"]))


def rate(card: dict, resource: str = "tensor", *, fmt: str | None = None, mhz: float | None = None) -> Fraction:
    """Lean's ``card.ρ resource``: the units a second of ``resource`` on the whole GPU, ``sm.count × rate.<resource> ×``
    its clock in Hz (for the tensor ``clock.tensor`` where the card has one and ``clock.sm`` where it doesn't, for the
    rest ``clock.sm``, unless ``mhz`` names the clock), and for device memory (``dram``) ``memory.bandwidth`` in 32-bit
    words. With ``fmt``, the tensor rate in MACs of that operand format (tensor units over ``weight.<fmt>``).
    ``verity/hardware/tests/test_lean.py`` holds it to every Lean table's arms."""
    if card["kind"] != "gpu":
        raise ValueError(f"{card['sku']} is a {card['kind']} card; rate is a GPU's")
    if resource == "dram":
        if fmt is not None or mhz is not None:
            raise ValueError("device memory's rate is memory.bandwidth: no SM clock or operand format scales it")
        return number(card["memory"]["bandwidth"]) / WORD
    if resource not in card["rate"]:
        raise KeyError(f"{card['sku']} has no rate.{resource}: its rates are {sorted(card['rate'])} and dram")
    if fmt is not None and (resource != "tensor" or fmt not in card["weight"]):
        raise KeyError(f"{fmt!r} weighs tensor MACs only, and {card['sku']}'s formats are {sorted(card['weight'])}")
    clocks = card["clock"]
    tick = clocks["tensor"] if resource == "tensor" and "tensor" in clocks else clocks["sm"]
    clock = number(tick) if mhz is None else Fraction(str(mhz))
    units = number(card["sm"]["count"]) * number(card["rate"][resource]) * clock * 10 ** 6
    return units if fmt is None else units / number(card["weight"][fmt])


READINGS = ("tensor", "sm")
"""The readings of the tensor's clock a GPU card states, the card's own first: `clock.tensor`, the tensor's clock
ceiling, and `clock.sm`, the highest clock of the SMs its tensor cores are in. They are one clock on a card that gives
the tensor none of its own."""


def readings(card: dict, *, fmt: str | None = None) -> dict[str, Fraction]:
    """The tensor peak of the whole GPU under each reading of its clock (``READINGS``), in that order: ``rate(card,
    fmt=fmt)``, the card's own, then the same at ``clock.sm``. A figure quoted under every reading of the data sheet
    is one of these a reading, so no reader names or picks a clock; ``rate`` refuses what it refuses."""
    return {r: rate(card, fmt=fmt, mhz=None if r == "tensor" else card["clock"][r]["value"]) for r in READINGS}


def check_card(table: dict, where: str) -> None:
    """Refuse ``table``, the file ``where``'s, unless every number in it, at any depth, is an entry with its unit, mark
    and source: the rule of a card file, and of every file that states numbers as the cards do."""
    for key, item in table.items():
        at = f"{where}: {key}"
        if isinstance(item, dict) and "value" in item:
            _check_entry(item, at)
        elif isinstance(item, dict):
            check_card(item, at)
        elif isinstance(item, (int, float)):
            raise InvalidArtifact(f"{at} is a bare number; write it as {{ value, unit, mark, source }}")


def _check_entry(entry: dict, at: str) -> None:
    if set(entry) != ENTRY:
        raise InvalidArtifact(f"{at} has keys {sorted(entry)}; an entry has exactly {sorted(ENTRY)}")
    if isinstance(entry["value"], bool) or not isinstance(entry["value"], (int, float)):
        raise InvalidArtifact(f"{at}: value {entry['value']!r} is not a number")
    if entry["mark"] not in MARKS:
        raise InvalidArtifact(f"{at}: mark {entry['mark']!r}; mark it one of {MARKS}")
    for field in ("unit", "source"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            raise InvalidArtifact(f"{at}: give it a {field}")


NODES = _PACKAGE.parent / "hardware" / "nodes"
CARD_KIND = {"device": "gpu", "host": "host"}
"""The kind of card each of a node's tables names by ``sku``."""
TIER = {"dram": ("memory", "bandwidth_max"), "disk": ("disk", "read_max")}
"""A memory leaf's tiers, each with the entry of the host card that bounds the host's leaves of the tier together."""
STORAGE = "storage"
"""The storage path's table, and the name the path registers under, ``HOST/storage``: no leaf and no interface of a
file takes it."""


class Leaf(NamedTuple):
    """A memory leaf of a host's own, as its node file states it."""

    id: str
    """``HOST/NAME``, the id the leaf registers under."""
    tier: str
    bytes: int
    bw: int
    """The most the leaf's bytes are read back at, in whole bits a second, rounded up."""


class Way(NamedTuple):
    """A way out of a node, as its node file states it: an interface, or the storage path."""

    id: str
    """``HOST/NAME``, the id its link registers under: an interface's name, or ``storage``."""
    bw: int | None
    """What it carries each way, in whole bits a second, rounded up; ``None`` where the file states no rate."""
    split: bool
    """Whether a network certifier may split it: an interface, yes; the storage path, no."""


def node_kinds(*, directory: Path | None = None) -> list[str]:
    """The kinds of node with a file in ``directory`` (``NODES`` when ``None``)."""
    return sorted(p.stem for p in (NODES if directory is None else directory).glob("*.toml"))


def load_node(kind: str, *, directory: Path | None = None, cards_directory: Path | None = None) -> dict:
    """The node file of ``kind`` in ``directory`` (``NODES`` when ``None``) as parsed TOML, refused unless every number
    in it is an entry and each card it names is a card of its kind in ``cards_directory`` (``CARDS`` when
    ``None``)."""
    directory = NODES if directory is None else directory
    path = directory / f"{kind}.toml"
    if not path.is_file():
        raise KeyError(f"no node {kind!r}: the nodes are {node_kinds(directory=directory)} ({directory}/<kind>.toml)")
    node = tomllib.loads(path.read_text(encoding="utf-8"))
    if node.get("node") != kind:
        raise InvalidArtifact(f"{path.name}: node = {node.get('node')!r}; set it to the file's name, {kind!r}")
    check_card(node, path.name)
    for table, card in CARD_KIND.items():
        sku, named = node.get(table, {}).get("sku"), skus(card, directory=cards_directory)
        if sku not in named:
            raise InvalidArtifact(f"{path.name}: {table}.sku = {sku!r}; name a {card} card, one of {named}")
    return node


def _ceiling(entry: Any, what: str) -> int:
    """A rate entry in whole bits a second, rounded up, refused unless it is a ceiling in ``B/s`` above 0."""
    fix = 'state it as { value, unit = "B/s", mark = "ceiling", source }, a bound that holds whatever the node runs'
    if not isinstance(entry, Mapping) or set(entry) != ENTRY:
        raise InvalidArtifact(f"{what} is {entry!r}, no entry; {fix}")
    if entry["unit"] != "B/s" or entry["mark"] != "ceiling":
        raise InvalidArtifact(f"{what} is in {entry['unit']!r}, marked {entry['mark']!r}; {fix}: a measured rate and a "
                              "limit somebody sets are no ceiling")
    rate = number(entry)
    if rate <= 0:
        raise InvalidArtifact(f"{what} is {entry['value']!r}; {fix}, above 0: a leaf that holds bytes is read")
    return math.ceil(8 * rate)


def _bytes(entry: Any, what: str) -> int:
    """A size entry's bytes, refused unless it is a whole number of them in ``B``."""
    if not isinstance(entry, Mapping) or set(entry) != ENTRY or entry["unit"] != "B":
        raise InvalidArtifact(f"{what} is {entry!r}; state it as an entry in B")
    size = number(entry)
    if size < 0 or size.denominator != 1:
        raise InvalidArtifact(f"{what} is {entry['value']!r}; state a whole number of bytes")
    return int(size)


def leaves(node: Mapping[str, Any], host: str, card: Mapping[str, Any] | None = None, *,
           cards_directory: Path | None = None) -> tuple[tuple[Leaf, ...], dict[str, int]]:
    """**A node's memory leaves and their read rates.** For a node file's table (``load_node(kind)``) registered as host
    ``host``: each ``memory.NAME`` as a ``Leaf`` (its id ``host/NAME``, its tier, its bytes, and ``bw``, its ``rate``
    in whole bits a second, rounded up), in the file's order; and, by tier, the rate that bounds the host's leaves of
    that tier together, in the same unit: the host card's ``memory.bandwidth_max`` for ``dram`` and its
    ``disk.read_max`` for ``disk``. The host reads a tier at no more than the smaller of its leaves' rates summed and
    the tier's.

    ``card`` is the host card's table: the card the file names by ``host.sku``, in ``cards_directory``
    (``CARDS`` when ``None``), when none is given. A leaf is refused
    when it states no ``rate``, a rate that is no ceiling in ``B/s`` above 0, a tier that is neither, or a rate above
    its tier's, since no leaf is read faster than the host's leaves of its tier together. A ``dram`` leaf's rate is
    its tier's exactly: DRAM has no interface narrower than the server's channels."""
    try:
        card = load_card(node["host"]["sku"], directory=cards_directory) if card is None else card
        stated = [(name, leaf["tier"], leaf["bytes"], leaf.get("rate"))
                  for name, leaf in node.get("memory", {}).items()]
        together = {tier: _ceiling(card[table][key], f"the host card's {table}.{key}")
                    for tier, (table, key) in TIER.items()}
    except (AttributeError, KeyError, TypeError) as e:
        raise InvalidArtifact(f"a node file's table or its host card's: {e!r}; the file has host.sku and, for each "
                              "memory.NAME, a tier, bytes and a rate, and the host card has memory.bandwidth_max and "
                              "disk.read_max") from None
    out = []
    for name, tier, size, rate in stated:
        at = f"memory.{name}"
        if tier not in TIER:
            raise InvalidArtifact(f"a node's {at}.tier is {tier!r}; a host's own leaf is one of {sorted(TIER)}")
        bound = "the host card's " + ".".join(TIER[tier])
        if rate is None:
            raise InvalidArtifact(f"a node's {at} states no rate, so nothing bounds how fast its bytes are read back; "
                                  f"state {at}.rate, a ceiling in B/s: {bound}, or for a drive of the server's own "
                                  "its link's lanes at their rate, less the coding")
        bw = _ceiling(rate, f"a node's {at}.rate")
        if bw > together[tier] or (tier == "dram" and bw != together[tier]):
            raise InvalidArtifact(f"a node's {at}.rate is {bw} bit/s, and {bound}, which bounds the host's {tier} "
                                  f"leaves together, is {together[tier]}; state "
                                  + ("that figure" if tier == "dram" else "a rate at or under it"))
        out.append(Leaf(f"{host}/{name}", tier, _bytes(size, f"a node's {at}.bytes"), bw))
    return tuple(out), together


def _line_rate(entry: Any, what: str) -> int:
    """An interface's rate entry in whole bits a second, rounded up, refused unless it is in ``bit/s`` above 0."""
    fix = 'state the line rate each way as { value, unit = "bit/s", mark, source }'
    if not isinstance(entry, Mapping) or set(entry) != ENTRY:
        raise InvalidArtifact(f"{what} is {entry!r}, not an entry; {fix}")
    rate = number(entry)
    if entry["unit"] != "bit/s" or rate <= 0:
        raise InvalidArtifact(f"{what} is {entry['value']!r} {entry['unit']}; {fix}, above 0, in bit/s")
    return math.ceil(rate)


def ways_out(node: Mapping[str, Any], host: str, card: Mapping[str, Any] | None = None, *,
             cards_directory: Path | None = None) -> tuple[Way, ...]:
    """**A node's ways out.** For a node file's table (``load_node(kind)``) registered as host ``host``: each
    ``interface.NAME`` as a ``Way`` a network certifier may split (its id ``host/NAME``, and ``bw``, its ``rate`` in
    whole bits a second, rounded up, or ``None`` where the table states its ``driver`` and the ``source`` of a read
    that found no rate), in the file's order; then, where the file states a ``storage`` table, the storage path as a
    ``Way`` no certifier may split, ``host/storage``, at the rate that bounds the node's disks together
    (``leaves(node, host, card, cards_directory=cards_directory)``'s for ``disk``, where ``card`` is the host card's
    table as there). The file writes
    that rate once, on the host card, and nowhere else.

    An interface is refused when it states neither a rate in ``bit/s`` above 0 nor what was read of it, the storage
    path when it does not name, each once, disk leaves of the file's, and a leaf or an interface named ``storage``,
    the path's own name. What registers each way as a link is ``Registration.node_entry``."""
    ports, memory, path = node.get("interface", {}), node.get("memory", {}), node.get(STORAGE)
    if not isinstance(ports, Mapping) or not isinstance(memory, Mapping):
        raise InvalidArtifact("a node file's interface and memory are tables, one table a name")
    taken = [table for table, names in (("memory", memory), ("interface", ports)) if STORAGE in names]
    if taken:
        raise InvalidArtifact(f"a node's {taken[0]}.{STORAGE} would register as {host}/{STORAGE}, the storage path's "
                              "id; give it another name")
    out = []
    for name, port in ports.items():
        at = f"a node's interface.{name}"
        if isinstance(port, Mapping) and set(port) == {"rate"}:
            out.append(Way(f"{host}/{name}", _line_rate(port["rate"], f"{at}.rate"), True))
        elif isinstance(port, Mapping) and set(port) == {"driver", "source"} and all(
                isinstance(port[key], str) and port[key].strip() for key in port):
            out.append(Way(f"{host}/{name}", None, True))
        else:
            raise InvalidArtifact(f"{at} is {port!r}; state its rate alone, the line rate each way as an entry in "
                                  "bit/s, or, where the interface reads none, its driver and the source of that read "
                                  "alone, each a string, and no rate")
    if path is None:
        return tuple(out)
    disks = [name for name, leaf in memory.items() if isinstance(leaf, Mapping) and leaf.get("tier") == "disk"]
    fix = ("state the storage path as leaves, the names of the disk leaves that ride it, each once, and source, why "
           f"a byte written to them has left the node; the file's disk leaves are {disks}, and a server whose drives "
           "are all its own has no storage table")
    if not isinstance(path, Mapping) or set(path) != {"leaves", "source"}:
        raise InvalidArtifact(f"a node's {STORAGE} is {path!r}; {fix}")
    ride, source = path["leaves"], path["source"]
    if (not isinstance(ride, list) or not ride or not all(isinstance(name, str) for name in ride)
            or len(set(ride)) != len(ride) or not set(ride) <= set(disks)):
        raise InvalidArtifact(f"a node's {STORAGE}.leaves is {ride!r}; {fix}")
    if not isinstance(source, str) or not source.strip():
        raise InvalidArtifact(f"a node's {STORAGE}.source is {source!r}; {fix}")
    return (*out, Way(f"{host}/{STORAGE}", leaves(node, host, card, cards_directory=cards_directory)[1]["disk"], False))
