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
    "onsite_compute": "94e39b782f3a59b7d12e74efbb7279061df2b0dafb55c547bb901bec98f983e3"
                      "ffdc40c529c2143f859a4bd0d6dde61f87251c64568a52c56b77a8f28fcecfc3",
    "onsite_full": "78a31992cc4f87122f5c04ff18823bcdd7eefea7385de600baa723deb06f82ff"
                   "84520bd0601c076f567459cb3803d1d3ed87a3a5eb85e5bd3a5b60028771bd75",
    "onsite_network": "d7573dd065d5d692918ded81d936c2dd9bf736e4e52b10d6fe56344b838b5564"
                      "f0a08fbd7bd3ae9bb26ab70619465f9bb9a048513db90d0b3d2be63dff676679",
    "remote": "dac8166b62d4569e8c1259e1a71df0da959009e5200ac292fef50780e199ab08"
              "154e1dd01a184f2e5e569980ba0b1edd7ca99174ac490923d641456925514820",
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
