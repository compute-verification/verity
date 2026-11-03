"""A node's remote-bandwidth figures, read from its cards: the value of the Site's ``node h``.

A node is its devices and its host. A device's figures are its GPU card's: device memory's measured read
(``memory.read``) and its ceiling (``memory.bandwidth``, which bounds a write too), its on-chip storage
(``store.chip``), its SHA-512 compressions a second (``hash.sha512``) and its squaring ceiling at each codec width
(``squaring``). The host's are its host card's: DRAM's measured read (``memory.read``), its ceiling
(``memory.bandwidth``) and its write ceiling (``memory.write``), its on-chip storage (``store.chip``) and one link
window (``link.window``). Each of the node's figures is the exact sum over its parts (``math.fsum``), whatever their
order.

A node reads ceilings and measurements. A figure its cards mark ``placeholder`` refuses it, unless the caller names
that figure (``placeholders``, each ``"<sku> <table>.<key>"``), and then the node carries the name, so whatever reads
the node can say which of its figures are open. A name the node doesn't read as a placeholder is refused too, so the
list goes when the card's mark does.

~~~python
from verity.hardware.node import VY_NEBIUS, node
rtx = node(*VY_NEBIUS)  # eight rtx-pro-6000-server cards on an xeon-6776p-2s host
rtx.read_bytes_per_s    # 12918800000000.0: eight cards' measured reads and host DRAM's
rtx.placeholders        # ('xeon-6776p-2s memory.bandwidth', 'xeon-6776p-2s memory.write')
~~~
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from verity.hardware import load

VY_NEBIUS = (("rtx-pro-6000-server",) * 8, "xeon-6776p-2s",
             ("xeon-6776p-2s memory.bandwidth", "xeon-6776p-2s memory.write"))
"""A vy-nebius node's devices, host and the placeholders it is read with: the remote audit's node (``node(*VY_NEBIUS)``)
until the Site registers it. Its host DRAM's ceiling is DDR5-6400's, which the remote audit has always taken; the
installed DIMMs aren't on record, and the part takes MRDIMMs up to 8800 MT/s (the host card's ``memory.speed``)."""


@dataclass(frozen=True, slots=True)
class GpuFigures:
    """One device's figures (bytes a second, bytes, compressions and squarings a second)."""

    sku: str
    read_bytes_per_s: float
    spec_bytes_per_s: float
    chip_bytes: int
    sha512_per_s: float
    squarings_per_s: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class HostFigures:
    """The host's figures: its DRAM's measured read, ceiling and write ceiling (bytes a second), its on-chip storage
    and one link window (bytes)."""

    sku: str
    read_bytes_per_s: float
    spec_bytes_per_s: float
    write_bytes_per_s: float
    chip_bytes: int
    link_window_bytes: int


@dataclass(frozen=True, slots=True)
class NodeFigures:
    """A node's devices and its host, and the sums the remote bandwidth audit reads: R measured and at spec over every
    tier, S_chip, the devices' SHA-512 rate and squaring ceiling, one device's and host DRAM's write ceilings. Its
    ``placeholders`` are the open figures it was read with."""

    devices: tuple[GpuFigures, ...]
    host: HostFigures
    placeholders: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("a node has one or more devices")

    @property
    def device_read_bytes_per_s(self) -> float:
        """The devices' measured reads, a store in device memory alone."""
        return math.fsum(d.read_bytes_per_s for d in self.devices)

    @property
    def read_bytes_per_s(self) -> float:
        """R measured: every tier at its measured read, the devices' memories and host DRAM."""
        return math.fsum([*(d.read_bytes_per_s for d in self.devices), self.host.read_bytes_per_s])

    @property
    def spec_bytes_per_s(self) -> float:
        """R at spec: every tier at its ceiling."""
        return math.fsum([*(d.spec_bytes_per_s for d in self.devices), self.host.spec_bytes_per_s])

    @property
    def chip_bytes(self) -> int:
        """S_chip: every device's on-chip storage and the host's."""
        return sum(d.chip_bytes for d in self.devices) + self.host.chip_bytes

    @property
    def sha512_per_s(self) -> float:
        """The devices' SHA-512 compressions a second (the host's cores' are not measured)."""
        return math.fsum(d.sha512_per_s for d in self.devices)

    @property
    def squarings_per_s(self) -> Mapping[str, float]:
        """The squaring ceiling at each width every device states, the sum over them."""
        widths = set.intersection(*(set(d.squarings_per_s) for d in self.devices))
        return MappingProxyType({w: math.fsum(d.squarings_per_s[w] for d in self.devices) for w in sorted(widths)})

    @property
    def device_write_bytes_per_s(self) -> float:
        """One device's write ceiling, the fastest device's."""
        return max(d.spec_bytes_per_s for d in self.devices)

    @property
    def host_write_bytes_per_s(self) -> float:
        """Host DRAM's write ceiling."""
        return self.host.write_bytes_per_s


class _Reader:
    def __init__(self, placeholders: Sequence[str]) -> None:
        self.named = frozenset(placeholders)
        self.read: set[str] = set()

    def card(self, sku: str, kind: str) -> dict:
        card = load(sku)
        if card["kind"] != kind:
            raise ValueError(f"{sku} is a {card['kind']} card; a node's {'devices' if kind == 'gpu' else 'host'} "
                             f"are {kind} cards")
        return card

    def figure(self, card: dict, table: str, key: str) -> float:
        entry, name = card[table][key], f"{card['sku']} {table}.{key}"
        if entry["mark"] == "placeholder":
            if name not in self.named:
                raise ValueError(f"{name} is a placeholder ({entry['source']}); a node reads ceilings and "
                                 f"measurements, so replace it in verity/hardware/cards/{card['sku']}.toml, or name "
                                 f"{name!r} in the node's placeholders to read it as open")
            self.read.add(name)
        return entry["value"]

    def done(self) -> tuple[str, ...]:
        if stale := sorted(self.named - self.read):
            raise ValueError(f"{stale} name no placeholder the node reads; drop them from its placeholders")
        return tuple(sorted(self.read))


def _device(r: _Reader, sku: str) -> GpuFigures:
    card = r.card(sku, "gpu")
    return GpuFigures(sku, float(r.figure(card, "memory", "read")), float(r.figure(card, "memory", "bandwidth")),
                  int(r.figure(card, "store", "chip")), float(r.figure(card, "hash", "sha512")),
                  MappingProxyType({w: float(r.figure(card, "squaring", w)) for w in card["squaring"]}))


def _host(r: _Reader, sku: str) -> HostFigures:
    card = r.card(sku, "host")
    return HostFigures(sku, float(r.figure(card, "memory", "read")), float(r.figure(card, "memory", "bandwidth")),
                float(r.figure(card, "memory", "write")), int(r.figure(card, "store", "chip")),
                int(r.figure(card, "link", "window")))


def device_of(sku: str, placeholders: Sequence[str] = ()) -> GpuFigures:
    """The device figures of GPU card ``sku``."""
    r = _Reader(placeholders)
    d = _device(r, sku)
    r.done()
    return d


def host_of(sku: str, placeholders: Sequence[str] = ()) -> HostFigures:
    """The host figures of host card ``sku``."""
    r = _Reader(placeholders)
    h = _host(r, sku)
    r.done()
    return h


def node(devices: Sequence[str], host: str, placeholders: Sequence[str] = ()) -> NodeFigures:
    """The node of a device of each GPU card in ``devices`` on the host of host card ``host``, reading the
    placeholders ``placeholders`` names as open figures."""
    r = _Reader(placeholders)
    ds, h = tuple(_device(r, sku) for sku in devices), _host(r, host)
    return NodeFigures(ds, h, r.done())
