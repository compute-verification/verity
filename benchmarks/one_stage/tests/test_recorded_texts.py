"""The constant text the one-stage audit's stand-ins write into their records, pinned by name against main's bytes.

A word change or a file's move alters no recorded text: a5_network_certifier's docket names its stand-in and committer
as main's dockets do, which its `check` compares, and its session root's tag is hashed. A fault message is no site.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]

#: name -> (the producing file, a regex whose first group is the text it writes)
SITES = {
    "a5_network_certifier.stand-in": ("a5_network_certifier.py", r'return \{"stand-in": "([^"]+)", "connection"'),
    "a5_network_certifier.committer": ("a5_network_certifier.py", r'"committer": \{"module": "([^"]+)"\}'),
    "a5_network_certifier.root.tag": ("a5_network_certifier.py", r'ROOT = hashlib\.sha512\(b"([^"]+)"\)'),
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "a5_network_certifier.committer": "24a93b89afe10f7eccb0fce6159b5a2a8395332605e904244ca527cd7ccb8308",
    "a5_network_certifier.root.tag": "3a658528806e0bd73c2b1882f1cd46eccef975622bbed37ad204554818dde5b3",
    "a5_network_certifier.stand-in": "80a86713c42e5dca93769acb66c605b27cbc309c7ddee35ea07d491fce02c02d",
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
