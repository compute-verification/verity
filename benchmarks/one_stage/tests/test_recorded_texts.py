"""The constant text the one-stage audit's stand-ins and a proved bundle's summary write into their records, pinned by
name against main's bytes.

A word change alters no recorded text: a5_network_attester's session root is hashed under its tag, and a8's window
states what it is in its registration record (``registration.json``, which its receipt binds), so both keep their bytes;
a proved bundle's tampered session's arms name its ``negatives`` keys (``tampered/<arm>``) and its session directories.
replay's era paths (`ERA_PARTS`) and A4's layout file are paths into history, read at a recorded run's commit: a move
adds its new path at the end of a part's list and keeps these. A provenance naming a file follows the file. A fault
message is no site.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]

#: name -> (the producing file, a regex whose first group is the text it writes)
SITES = {
    "a5_network_attester.root.tag": ("a5_network_attester.py", r'ROOT = hashlib\.sha512\(b"([^"]+)"\)'),
    "a8.window.of": ("a8.py", r'win = \{"kind": "stand-in", "of": "([^"]+)"\s+f"'),
    "tampered.arm.zk_server": ("prove_bundle.py", r'arms = \[\("([a-z]+)", True\)\]'),
    "tampered.arm.plain": ("prove_bundle.py", r'\[\*arms, \("([a-z]+)", False\)\]'),
    "negatives.tampered": ("prove_bundle.py", r'negatives\[f"([a-z]+)/\{arm\}"\] = '),
    "negatives.flipped": ("prove_bundle.py", r'negatives\[f"([a-z]+)/\{name\}"\] = '),
    "replay.era.served_zk.0": ("replay.py", r'"served_zk": \("([^"]+)",'),
    "replay.era.served_zk.1": ("replay.py", r'"served_zk": \("[^"]+", "([^"]+)"'),
    "replay.era.catalog": ("replay.py", r'"catalog": \("([^"]+)",\)'),
    "replay.era.kernels": ("replay.py", r'"kernels": \("([^"]+)",\)'),
    "replay.a4_layout": ("replay.py", r'"show", f"\{commit\}:([^"{]+a4_layout\.json)"'),
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "a5_network_attester.root.tag": "3a658528806e0bd73c2b1882f1cd46eccef975622bbed37ad204554818dde5b3",
    "a8.window.of": "8dbf69c2a1bcdcb8a73dfa922bbd0d3a538f945469c18a164fae8705dd9781c8",
    "negatives.flipped": "bf6e7814c35d7231fb53f7ec0c85745df062bf65e96742df48cbb966b7129b71",
    "negatives.tampered": "d121be3103007b41edf96f8262925f8c7d61894afe9a041843b631f69445bc57",
    "tampered.arm.plain": "c4cdb23c9c04b010ec7a3b09b4e66103d93c36c38fd71b75102bff4a32677a1a",
    "tampered.arm.zk_server": "9450fcf8ce62d14e18042e1a5e9b5c1729254b69a4449e11bbc299ea61cfe38c",
    "replay.era.served_zk.0": "9038ae134f8e3daf6bbd04132adc7d6e5802fb7f74633696a91ef5b79f296131",
    "replay.era.served_zk.1": "d52f6833e8fd3cda7828282d4e742e4ffb79f4239dd1b993ba27fafadcfe5659",
    "replay.era.catalog": "dc2b61a59d76056d6175ccfc532cb06d38205a2b57c99a276ffa1e0155a71130",
    "replay.era.kernels": "5417057d372861189467521a8754102c0b33170e4ee2a88529907d7d89d34589",
    "replay.a4_layout": "1dac3402f7ecaba733ab8311a27af19265690bf828a7c27f76dfe0b57531870e",
}


def test_every_site_is_pinned():
    assert sorted(SITES) == sorted(PINS), "a site and its pin are added together, the pin from main's text"


@pytest.mark.parametrize("name", sorted(PINS))
def test_a_recorded_text_keeps_its_bytes(name):
    path, rx = SITES[name]
    found = re.findall(rx, (HERE / path).read_text())
    assert found, f"{name}: no site in {path}; a moved site moves its entry here"
    assert hashlib.sha256(found[0].encode()).hexdigest() == PINS[name], (
        f"{name}: the text {path} writes into its records changed. A word change alters no recorded text: put main's "
        f"bytes back (`git show origin/main:<file>`, following its move). Only a new record format re-pins, with its "
        f"format id bumped.")
