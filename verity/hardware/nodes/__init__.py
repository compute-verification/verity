"""The node files: one TOML file per kind of node (``<kind>.toml``, beside this file), holding what a node's
registration states that no card does.

A node is a host and the GPUs it carries. Its file names the GPU card and how many of it the node carries
(``device.sku``, ``device.count``), the host card (``host.sku``), and the host's own memory, one table a leaf
(``memory.NAME``: its ``tier``, ``dram`` or ``disk``, and its ``bytes``). A card's own memory is its card's
``store.dram_installed``, whole, and no rate, clock or peak is here: those are the cards'. Every number is an entry
``{value, unit, mark, source}`` under the cards' rules (``verity.hardware.cards``), and a leaf with no read on record
has no entry. A kind is the card and its count, never a host's name, and a file names no machine, no address and no
provider: whoever registers a node gives its host's id (``verity.hardware.site.Registration.node_entry``, which gives
no entry of a file with no ``dram`` or no ``disk`` leaf). Stdlib and the cards only.

~~~python
from verity.hardware import cards, nodes
from verity.hardware.site import Auditor, Registration, check
node = nodes.load("rtx-pro-6000-server-x8")
gpu = cards.load(node["device"]["sku"])
log = Registration("remote")
site = check(log.append(log.node_entry(node, gpu, host="node0")), Auditor.of_files([gpu]))
site.units                          # ('node0/gpu0', ..., 'node0/gpu7')
site.tiers("node0")                 # {'hbm': 824633720832, 'dram': 1872605741056, 'disk': 6266358333440}
~~~
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from verity.errors import InvalidArtifact
from verity.hardware import cards

NODES = Path(__file__).parent
CARD = {"device": "gpu", "host": "host"}
"""The kind of card each of a node's tables names by ``sku``."""


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
