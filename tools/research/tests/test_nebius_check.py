"""The bootstrap check (`pods/nebius/check.py`) takes the node's figures from its node file and card, and says which file it lacks."""
from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("nebius_check", REPO / "tools/research/src/research/pods/nebius/check.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_the_checks_figures_are_the_node_files_and_its_cards():
    node = tomllib.loads((REPO / "hardware/nodes" / f"{C.NODE}.toml").read_text())
    card = tomllib.loads((REPO / "hardware/cards" / f"{node['device']['sku']}.toml").read_text())
    assert C.HARDWARE == REPO / "hardware"
    assert C.WANT["gpus"] == node["device"]["count"]["value"]
    assert C.WANT["sms"] == card["sm"]["count"]["value"]
    assert C.WANT["ram_gib"] * 2**30 == node["memory"]["dram"]["bytes"]["value"]
    assert "sm_%d%d" % C.WANT["cc"] == card["arch"].removesuffix("a")


def test_a_missing_file_stops_the_check_naming_its_path(tmp_path):
    with pytest.raises(SystemExit, match=re.escape(str(tmp_path / "hardware/nodes" / f"{C.NODE}.toml"))):
        C.figures(tmp_path / "hardware")
