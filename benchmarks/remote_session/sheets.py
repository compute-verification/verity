"""The session harness's node figures, read from the node's data sheets: its node file and the GPU and host cards it
names (`verity.device`'s `load_node` and `load_card`), through `timing.Figures.of_sheets`."""

from __future__ import annotations

from verity.device import load_card, load_node
from verity.protocols.resource_accounting.memory.poub.verifier import timing

NODE = "rtx-pro-6000-server-x8"


def figures(node: str = NODE) -> timing.Figures:
    """The figures of a node of kind `node`, from its sheets."""
    n = load_node(node)
    return timing.Figures.of_sheets(n, load_card(n["device"]["sku"]), load_card(n["host"]["sku"]))


__all__ = ["NODE", "figures"]
