"""The DGX H200's card and the inference-only policy's data state one figure, and neither changes alone.

The policy's ``h200-hgx`` location is one DGX H200, and its ``links``, the line rate across the location's boundary,
counts the server's twelve ConnectX-7 ports: the card's ``link.cluster`` and ``link.storage``. Its ``low`` is the eight
cluster ports alone. The policy counts no management port, so the card's ``link.rate``, every port, is above it.
"""

from __future__ import annotations

import tomllib
from fractions import Fraction
from pathlib import Path

from verity.hardware.cards import load, number

ROOT = Path(__file__).resolve().parents[3]
POLICY = "verity/protocols/inference_only/rho/data/hardware.toml"


def test_the_policys_links_are_the_cards_connectx_ports() -> None:
    links = tomllib.loads((ROOT / POLICY).read_text(encoding="utf-8"))["location"]["h200-hgx"]["links"]
    link = load("dgx-h200")["link"]
    cluster, storage = number(link["cluster"]), number(link["storage"])
    assert links["unit"] == link["cluster"]["unit"] == link["storage"]["unit"]
    stated = {key: Fraction(str(links[key])) for key in ("low", "value", "high")}
    assert stated == {"low": cluster, "value": cluster + storage, "high": cluster + storage}, (
        f"verity/hardware/cards/dgx-h200.toml (link.cluster, link.storage) and {POLICY} (location.h200-hgx.links) "
        "state one figure, the line rate of the DGX H200's twelve ConnectX-7 ports and of its eight cluster ports "
        "alone: change both files or neither")
