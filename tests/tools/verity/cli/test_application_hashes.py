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
    "onsite_compute": "1863f59b308eb6c9915db83fedcdb97d4a4f6f1e0bfc2034529cd4d11cbd5e33"
                      "799452d2ca0c523f56767882a349d42ac4ce09c939728519e75d11163a7018f0",
    "onsite_full": "80320484d775e12e55e3590e50375cadc36a5029bd092bea8577c8500b6ece26"
                   "8b2685aee0b3524453a322d34c8bc95fbe23f99f7e5d12668e1ea373405ea4ea",
    "onsite_network": "2f2088711c4d9b1177888e52dfb326b658b94b87b27c5d727a711d518cb90ca8"
                      "1b23da142a7de84a6429a31a9bf663cdcdfc01f33c309f6312ebe23d9ceef37a",
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
