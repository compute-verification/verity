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
    "onsite_compute": "f5ca90b1a0f6c0b201537a257c2c7070050a8f94629b4356ca437cca08d9b627"
                      "e3fbb2da763263e35fccb143ae171e908f3f7de57f120d206a17dda2c1554322",
    "onsite_full": "29ddc537eb839e0115f3cbd66f5842a9a534eef5f64f389bad552b6df2ca7436"
                   "024e372e061d5e818eb65f459b2ef6b91820597f21489fe55837a7f84773dffe",
    "onsite_network": "9d9c1d03935955f17207629a34cc2001c04b90185587a00fe6a08bb88f342c59"
                      "cde7036cc73eef8d75abb2b9e14295857d7ab94207f3b2d7572fc78ef519f00e",
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
