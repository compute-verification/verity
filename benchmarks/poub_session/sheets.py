"""The session harness's node figures, read from the node's data sheets: its node file and the GPU and host cards it
names (`verity.device`'s `load_node` and `load_card`), through `timing.Figures.of_sheets`."""

from __future__ import annotations

from pathlib import Path

from verity.device import load_card, load_node
from verity.protocols.resource_accounting.memory.poub.verifier import timing

CARD_FILES = Path(__file__).resolve().parents[2] / "hardware/cards"
NODE_FILES = Path(__file__).resolve().parents[2] / "hardware/nodes"

NODE = "rtx-pro-6000-server-x8"


def figures(node: str = NODE) -> timing.Figures:
    """The figures of a node of kind `node`, from its sheets."""
    n = load_node(node, directory=NODE_FILES, cards_directory=CARD_FILES)
    return timing.Figures.of_sheets(n, load_card(n["device"]["sku"], directory=CARD_FILES), load_card(n["host"]["sku"], directory=CARD_FILES))


__all__ = ["NODE", "figures"]
