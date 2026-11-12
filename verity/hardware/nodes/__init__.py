"""The node files: one TOML file per kind of node (``<kind>.toml``, beside this file), holding what a node's
registration states that no card does.

A node is a host and the GPUs it carries. Its file names the GPU card and how many of it the node carries
(``device.sku``, ``device.count``), the host card (``host.sku``), the host's own memory, one table a leaf
(``memory.NAME``: its ``tier``, ``dram`` or ``disk``, its ``bytes`` and its ``rate``, the most its bytes are read back
at, in ``B/s``, a ceiling), and the node's ways out: the interfaces that carry traffic off the node, one table each
(``interface.NAME``: its ``rate``, the line rate each way in ``bit/s``, or, where the interface reads none, its
``driver`` and the ``source`` of that read, and no rate), and the storage path, once (``storage``, below). A card's
own memory is its card's ``store.dram``, whole, and no clock or peak of a card is here: those are the cards'. Every
number is an entry ``{value, unit, mark, source}`` under the cards' rules (``verity.hardware.cards``), so what is not
a number is not written as one, and a leaf or an interface with no read on record has no table. A kind is the card
and its count, never a host's name, and a file names no machine, no address and no provider: whoever registers a node
gives its host's id (``verity.hardware.site.Registration.node_entry``, which gives no entry of a file with no ``dram``
or no ``disk`` leaf, and none of a file with no interface).
Stdlib and the cards only.

**A leaf's rate** is the tightest bound on record that holds whatever the node runs. The DRAM leaf's is the host
card's ``memory.bandwidth_max``, the server's channels. A disk's is its own interface's where the disk states one (a
drive of the server's own: its link's lanes at their rate, less the coding), and else the host card's
``disk.read_max``, the lanes into the server's sockets, which every byte of a disk enters by. That figure of the host
card bounds the host's disks together too, so ``leaves`` gives it beside each leaf's own, once a tier: a reader that
adds the leaves' rates of a tier takes the smaller of the sum and the tier's. Neither a limit a provider sets on a
disk nor a measured rate is a ceiling, so neither is a leaf's rate.

**The ways out** are what a byte leaves the node by, and ``ways_out`` is their one reader. An interface is one, and a
network certifier may split it. **The storage path** is the other: a disk that is not the server's own (a network
disk, which can be attached to another machine or copied through whoever provides it) takes what is written to it
off the node past every interface, so no certifier can split it. A file states it once, as ``storage``: ``leaves``,
the names of the disk leaves that ride it, and ``source``, why they leave. It states no rate of its own: its rate is
the one that bounds the node's disks together, ``leaves``'s for ``disk``. A server whose drives are all its own has no
``storage`` table and no such way out.

~~~python
from verity.hardware import cards, nodes
from verity.hardware.site import Auditor, Registration, check
node = nodes.load("rtx-pro-6000-server-x8")
gpu = cards.load(node["device"]["sku"])
log = Registration("remote")
entry = log.node_entry(node, gpu, host="node0", ways=nodes.ways_out(node, "node0"))
site = check(log.append(entry), Auditor.of_files([gpu]))
site.units                          # ('node0/gpu0', ..., 'node0/gpu7')
site.tiers("node0")                 # {'hbm': 824633720832, 'dram': 1872605741056, 'disk': 6266358333440}
[(k.id, k.beta) for k in site.boundary("node0")]   # [('node0/eth0', 400000000000), ('node0/storage', 5584738461544)]
each, tier = nodes.leaves(node, "node0")
[(leaf.id, leaf.bw) for leaf in each]   # [('node0/dram', 9011200000000), ('node0/vda', 5584738461544), ...], bit/s
tier                                # {'dram': 9011200000000, 'disk': 5584738461544}, each tier's leaves together
nodes.ways_out(node, "node0")       # (Way('node0/eth0', 400000000000, True), Way('node0/storage', 5584738461544, False))
~~~
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from verity.errors import InvalidArtifact
from verity.hardware import cards

NODES = Path(__file__).parent
CARD = {"device": "gpu", "host": "host"}
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


def kinds() -> list[str]:
    """The kinds of node with a file."""
    return sorted(p.stem for p in NODES.glob("*.toml"))


def load(kind: str) -> dict:
    """The node file of ``kind`` as parsed TOML, refused unless every number in it is an entry and each card it names
    is a card of its kind."""
    path = NODES / f"{kind}.toml"
    if not path.is_file():
        raise KeyError(f"no node {kind!r}: the nodes are {kinds()} (verity/hardware/nodes/<kind>.toml)")
    node = tomllib.loads(path.read_text(encoding="utf-8"))
    if node.get("node") != kind:
        raise InvalidArtifact(f"{path.name}: node = {node.get('node')!r}; set it to the file's name, {kind!r}")
    cards.check(node, path.name)
    for table, card in CARD.items():
        sku = node.get(table, {}).get("sku")
        if sku not in cards.skus(card):
            raise InvalidArtifact(f"{path.name}: {table}.sku = {sku!r}; name a {card} card, one of {cards.skus(card)}")
    return node


def _ceiling(entry: Any, what: str) -> int:
    """A rate entry in whole bits a second, rounded up, refused unless it is a ceiling in ``B/s`` above 0."""
    fix = 'state it as { value, unit = "B/s", mark = "ceiling", source }, a bound that holds whatever the node runs'
    if not isinstance(entry, Mapping) or set(entry) != cards.ENTRY:
        raise InvalidArtifact(f"{what} is {entry!r}, no entry; {fix}")
    if entry["unit"] != "B/s" or entry["mark"] != "ceiling":
        raise InvalidArtifact(f"{what} is in {entry['unit']!r}, marked {entry['mark']!r}; {fix}: a measured rate and a "
                              "limit somebody sets are no ceiling")
    rate = cards.number(entry)
    if rate <= 0:
        raise InvalidArtifact(f"{what} is {entry['value']!r}; {fix}, above 0: a leaf that holds bytes is read")
    return math.ceil(8 * rate)


def _bytes(entry: Any, what: str) -> int:
    """A size entry's bytes, refused unless it is a whole number of them in ``B``."""
    if not isinstance(entry, Mapping) or set(entry) != cards.ENTRY or entry["unit"] != "B":
        raise InvalidArtifact(f"{what} is {entry!r}; state it as an entry in B")
    size = cards.number(entry)
    if size < 0 or size.denominator != 1:
        raise InvalidArtifact(f"{what} is {entry['value']!r}; state a whole number of bytes")
    return int(size)


def leaves(node: Mapping[str, Any], host: str, card: Mapping[str, Any] | None = None
           ) -> tuple[tuple[Leaf, ...], dict[str, int]]:
    """**A node's memory leaves and their read rates.** For a node file's table (``load(kind)``) registered as host
    ``host``: each ``memory.NAME`` as a ``Leaf`` (its id ``host/NAME``, its tier, its bytes, and ``bw``, its ``rate``
    in whole bits a second, rounded up), in the file's order; and, by tier, the rate that bounds the host's leaves of
    that tier together, in the same unit: the host card's ``memory.bandwidth_max`` for ``dram`` and its
    ``disk.read_max`` for ``disk``. The host reads a tier at no more than the smaller of its leaves' rates summed and
    the tier's.

    ``card`` is the host card's table: the card the file names by ``host.sku`` when none is given. A leaf is refused
    when it states no ``rate``, a rate that is no ceiling in ``B/s`` above 0, a tier that is neither, or a rate above
    its tier's, since no leaf is read faster than the host's leaves of its tier together. A ``dram`` leaf's rate is
    its tier's exactly: DRAM has no interface narrower than the server's channels."""
    try:
        card = cards.load(node["host"]["sku"]) if card is None else card
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
    if not isinstance(entry, Mapping) or set(entry) != cards.ENTRY:
        raise InvalidArtifact(f"{what} is {entry!r}, not an entry; {fix}")
    rate = cards.number(entry)
    if entry["unit"] != "bit/s" or rate <= 0:
        raise InvalidArtifact(f"{what} is {entry['value']!r} {entry['unit']}; {fix}, above 0, in bit/s")
    return math.ceil(rate)


def ways_out(node: Mapping[str, Any], host: str, card: Mapping[str, Any] | None = None) -> tuple[Way, ...]:
    """**A node's ways out.** For a node file's table (``load(kind)``) registered as host ``host``: each
    ``interface.NAME`` as a ``Way`` a network certifier may split (its id ``host/NAME``, and ``bw``, its ``rate`` in
    whole bits a second, rounded up, or ``None`` where the table states its ``driver`` and the ``source`` of a read
    that found no rate), in the file's order; then, where the file states a ``storage`` table, the storage path as a
    ``Way`` no certifier may split, ``host/storage``, at the rate that bounds the node's disks together
    (``leaves(node, host, card)``'s for ``disk``, where ``card`` is the host card's table as there). The file writes
    that rate once, on the host card, and nowhere else.

    An interface is refused when it states neither a rate in ``bit/s`` above 0 nor what was read of it, the storage
    path when it does not name, each once, disk leaves of the file's, and a leaf or an interface named ``storage``,
    the path's own name. What registers each way as a link is ``verity.hardware.site.Registration.node_entry``."""
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
    return (*out, Way(f"{host}/{STORAGE}", leaves(node, host, card)[1]["disk"], False))
