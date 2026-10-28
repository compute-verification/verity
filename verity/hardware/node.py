"""A node's remote-bandwidth figures, read from its cards: the value of the Site's ``node h``.

A node is its devices and its host. A device's figures are its card's: device memory's measured read
(``memory.read``) and its ceiling (``memory.bandwidth``, which bounds a write too), its on-chip storage
(``store.chip``), its SHA-512 compressions a second (``hash.sha512``) and its squaring ceiling at each codec width
(``squaring``). The host's are its card's ``host`` table: host DRAM's measured read (``dram_read``) and its ceiling
(``dram_bandwidth``, which bounds a write too), its on-chip storage (``chip``) and one link window (``link_window``).
Each of the node's figures is the exact sum over its parts (``math.fsum``), whatever their order.

A node reads only ceilings and measurements: a figure its cards mark ``placeholder`` refuses it.

~~~python
from verity.hardware.node import VY_NEBIUS, node
rtx = node(*VY_NEBIUS)  # node(["rtx-pro-6000-server"] * 8, host="rtx-pro-6000-server")
rtx.read_bytes_per_s    # 12918800000000.0: eight cards' measured reads and host DRAM's
~~~
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from verity.hardware import load

VY_NEBIUS = (("rtx-pro-6000-server",) * 8, "rtx-pro-6000-server")
"""A vy-nebius node's devices and host, the remote audit's node (``node(*VY_NEBIUS)``) until the Site registers it:
eight RTX PRO 6000s on a host of two Xeon 6776P sockets (the card's ``nodes``)."""


@dataclass(frozen=True, slots=True)
class Device:
    """One device's figures (bytes a second, bytes, compressions and squarings a second)."""

    sku: str
    read_bytes_per_s: float
    spec_bytes_per_s: float
    chip_bytes: int
    sha512_per_s: float
    squarings_per_s: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Host:
    """The host's figures: its DRAM's measured read and ceiling (bytes a second), its on-chip storage and one link
    window (bytes)."""

    sku: str
    read_bytes_per_s: float
    spec_bytes_per_s: float
    chip_bytes: int
    link_window_bytes: int


@dataclass(frozen=True, slots=True)
class Node:
    """A node's devices and its host, and the sums the remote bandwidth audit reads: R measured and at spec over every
    tier, S_chip, the devices' SHA-512 rate and squaring ceiling, one device's and host DRAM's write ceilings."""

    devices: tuple[Device, ...]
    host: Host

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
        return self.host.spec_bytes_per_s


def _figure(card: dict, table: str, key: str) -> float:
    entry = card[table][key]
    if entry["mark"] == "placeholder":
        raise ValueError(f"{card['sku']}: {table}.{key} is a placeholder ({entry['source']}); a node reads only "
                         f"ceilings and measurements, so replace it in verity/hardware/cards/{card['sku']}.toml first")
    return entry["value"]


def device_of(sku: str) -> Device:
    """The device figures of card ``sku``."""
    card = load(sku)
    return Device(sku, float(_figure(card, "memory", "read")), float(_figure(card, "memory", "bandwidth")),
                  int(_figure(card, "store", "chip")), float(_figure(card, "hash", "sha512")),
                  MappingProxyType({w: float(_figure(card, "squaring", w)) for w in card["squaring"]}))


def host_of(sku: str) -> Host:
    """The host figures of card ``sku``'s ``host`` table."""
    card = load(sku)
    return Host(sku, float(_figure(card, "host", "dram_read")), float(_figure(card, "host", "dram_bandwidth")),
                int(_figure(card, "host", "chip")), int(_figure(card, "host", "link_window")))


def node(devices: Sequence[str], host: str) -> Node:
    """The node of a device of each card in ``devices`` on the host of card ``host``."""
    return Node(tuple(device_of(sku) for sku in devices), host_of(host))
