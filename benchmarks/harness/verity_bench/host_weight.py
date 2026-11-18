"""A host CPU core-second's weight in SM-seconds on a kind of node, from its node file and the two cards it names: one
core's share of the host card's peak AMX rate (`rate.ops` over `cpu.sockets` x `cpu.cores`, the whole server's cores at
its turbo, Priority Core Turbo included where the card counts it) over one SM's peak tensor rate at `clock.sm` (twice
the GPU card's s8 MACs a second at that reading, `verity.device.readings`, over `sm.count`). It is a property of the
node, its host and its GPU, and no constant of a card.

~~~python
from verity_bench.host_weight import cpu_sm_s_per_core_s
float(cpu_sm_s_per_core_s("h200-sxm-141gb-x1"))   # 0.4798: 7.471e14 op/s over 96 cores, over 2 x 4096 x 1980 MHz
~~~
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from verity.device import load_card, load_node, number, readings

ROOT = Path(__file__).resolve().parents[3]
CARD_FILES, NODE_FILES = ROOT / "calibration/hardware/cards", ROOT / "calibration/hardware/nodes"


def cpu_sm_s_per_core_s(kind: str, *, nodes: Path = NODE_FILES, cards: Path = CARD_FILES) -> Fraction:
    """The SM-seconds a host CPU core-second of node `kind` weighs, exactly."""
    node = load_node(kind, directory=nodes, cards_directory=cards)
    host, gpu = (load_card(node[table]["sku"], directory=cards) for table in ("host", "device"))
    core = number(host["rate"]["ops"]) / (number(host["cpu"]["sockets"]) * number(host["cpu"]["cores"]))
    sm = 2 * readings(gpu)["sm"] / number(gpu["sm"]["count"])
    return core / sm
