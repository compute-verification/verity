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
    "onsite_compute": "4ccd5a22e5f5ba7af0bcdd27c3a02d4728816499f8e8083ea4b5d65186f5e940"
                      "8be9cb19d41dbdc9ef584846212dbdc61e3f206a12c89c7833ede7d347f88ec8",
    "onsite_full": "9d2e6c3d79940c85994d185ba9c16a59cdd800d01102c7ec3fdf7befb4a6cde3"
                   "6bc3bb0d9254e98c6650e473c14b52f5616fdfee8736d9bf0266d2c7124d0be9",
    "onsite_network": "41676c2b8ed781e6e8f4fcc85d8b708dce2b6d3a825edbc42a7ce354ed8cec12"
                      "cdf15a3f66e7156a39a3eecd99c725dd09f857218a4a09e19a3bb37765b212bb",
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
    stated = {c.id: {**c.default_params, **raw.get("params", {}).get(c.id, {})} for c in listed if c.default_params}
    assert K.digest(K.check({**raw, "params": stated}, REG.components, REG.workloads)) == HASHES[name]
