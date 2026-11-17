"""The session harness's node figures are its data sheets', value for value the ones `timing` states."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[3] / "benchmarks" / "poub_session")))

import sheets  # noqa: E402
from verity.device import load_card, load_node  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import timing  # noqa: E402


def test_the_sheets_figures_are_timings():
    assert sheets.figures() == timing.RTX_PRO_6000_NODE


def test_a_card_the_node_does_not_name_is_refused():
    node = load_node(sheets.NODE)
    with pytest.raises(ValueError, match="hand in the device card the node names"):
        timing.Figures.of_sheets(node, load_card("h200-sxm-141gb"), load_card(node["host"]["sku"]))
