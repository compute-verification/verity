"""The constant text a proved bundle's summary carries, pinned by name against main's bytes.

A word change alters no recorded text: the tampered session's arms name its ``negatives`` keys
(``tampered/<arm>``) and its session directories, so they keep their bytes. A fault message is no site.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "audit" / "prove_bundle.py"

#: name -> a regex over prove_bundle.py whose first group is the text it writes
SITES = {
    "tampered.arm.zk_server": r'arms = \[\("([a-z]+)", True\)\]',
    "tampered.arm.plain": r'\[\*arms, \("([a-z]+)", False\)\]',
    "negatives.tampered": r'negatives\[f"([a-z]+)/\{arm\}"\] = ',
    "negatives.flipped": r'negatives\[f"([a-z]+)/\{name\}"\] = ',
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "negatives.flipped": "bf6e7814c35d7231fb53f7ec0c85745df062bf65e96742df48cbb966b7129b71",
    "negatives.tampered": "d121be3103007b41edf96f8262925f8c7d61894afe9a041843b631f69445bc57",
    "tampered.arm.plain": "c4cdb23c9c04b010ec7a3b09b4e66103d93c36c38fd71b75102bff4a32677a1a",
    "tampered.arm.zk_server": "9450fcf8ce62d14e18042e1a5e9b5c1729254b69a4449e11bbc299ea61cfe38c",
}


def test_every_site_is_pinned():
    assert sorted(SITES) == sorted(PINS), "a site and its pin are added together, the pin from main's text"


@pytest.mark.parametrize("name", sorted(PINS))
def test_a_recorded_text_keeps_its_bytes(name):
    found = re.findall(SITES[name], SOURCE.read_text())
    assert found, f"{name}: no site in {SOURCE.name}; a moved site moves its entry here"
    assert hashlib.sha256(found[0].encode()).hexdigest() == PINS[name], (
        f"{name}: the text prove_bundle.py writes into its summary changed. A word change alters no recorded text: put "
        f"main's bytes back (`git show origin/main:benchmarks/one_stage/prove_bundle.py`). Only a new record format "
        f"re-pins, with its format id bumped.")
