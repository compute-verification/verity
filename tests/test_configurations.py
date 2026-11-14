"""The committed configurations' hashes (`demos/configurations.toml`): a line for each demo's configuration
(`demos/NAME/config.toml`) and none for another file, each the hash `verity config hash` prints of it."""
from __future__ import annotations

import tomllib
from pathlib import Path

from verity import config as K
from verity_cli import run as R

ROOT = Path(__file__).resolve().parents[1]
HASHES = ROOT / "demos" / "configurations.toml"


def test_each_demo_configuration_has_its_hash_and_nothing_else_does():
    listed = tomllib.loads(HASHES.read_text())
    found = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("demos/*/config.toml"))
    assert sorted(listed) == found, f"{HASHES.relative_to(ROOT)} lists {sorted(listed)}, the demos' configurations are {found}"
    reg = R.registry()
    got = {p: K.digest(R.load(ROOT / p, reg)) for p in found}
    wrong = {p: h for p, h in got.items() if h != listed[p]}
    assert not wrong, (f"set each line of {HASHES.relative_to(ROOT)} to what `verity config hash PATH` prints: "
                       + ", ".join(f'"{p}" = "{h}"' for p, h in wrong.items()))
