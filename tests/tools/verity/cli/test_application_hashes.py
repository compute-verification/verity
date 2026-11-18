"""The applications' configuration hashes (`verity config hash applications/NAME.toml`), pinned: moving a setting from
the table's parameters to a scheme's declaration (`verity.protocols.resource_accounting.scheme.settings`) moves none,
and neither does stating a setting at its default. A change to an application's file that is meant to move its hash
re-pins it here in the same commit."""
from __future__ import annotations

from pathlib import Path

import pytest

from verity import config as K
from verity_cli import run as R

ROOT = Path(R.ROOT)
REG = R.registry()
HASHES = {
    "onsite_compute": "2c2330966a3620c084e93aeb80942784beebf6b0c2440aa759c0683acae8755c"
                      "f16fec85faca64fcf4c4b04d5dd48deff7a1dd928a91afd3967aafa678d75b54",
    "onsite_full": "49293ebb4c6c8a25bfcc69a4c0f8e15bb933f75b70c8f40663fb3745e608c701"
                   "0a3b6d659591881478d5e02b167804e81687a97b76c140834e519fa6ebe0e01a",
    "onsite_network": "f8f66f5c9d295011814ce369dd00883491d532b88d3575f05b8e200f9a30ae98"
                      "2a9eba6dda54c3d03174571ec82c45e69539ff48b5398c59587f301d1aafcd98",
    "onsite_network_front": "58ffdb693cc5c0da4abaf3021a97910859859c141ca527dc79ff1470cc26a19a"
                            "16f3e68c0284a4127aeff3a5362eea529290ea15a5f135bceb20795120241334",
    "remote": "0b0611e7cbff2cd20514e26a4fe3feb1af746b33d061f3be43ff4679657429ed"
              "ef66a8d47b5c59600dc835865cb92442eac0c6f278b27481eef742782b5762ed",
}


@pytest.mark.parametrize("name", sorted(HASHES))
def test_an_applications_hash_is_pinned(name: str) -> None:
    assert K.digest(R.load(ROOT / "applications" / f"{name}.toml", REG)) == HASHES[name], (
        f"applications/{name}.toml's hash moved: re-pin it only if the change is meant to")


@pytest.mark.parametrize("name", sorted(HASHES))
def test_a_setting_stated_at_its_default_moves_no_hash(name: str) -> None:
    raw = K.parse((ROOT / "applications" / f"{name}.toml").read_text())
    listed = [REG.components[n] for n in raw["protocols"]]
    # what the file states for a subprotocol that declares no default (Pearl+'s class and tie) stays as stated
    stated = raw.get("params", {}) | {c.id: {**c.default_params, **raw.get("params", {}).get(c.id, {})}
                                      for c in listed if c.default_params}
    assert K.digest(K.check({**raw, "params": stated}, REG.components, REG.workloads)) == HASHES[name]
