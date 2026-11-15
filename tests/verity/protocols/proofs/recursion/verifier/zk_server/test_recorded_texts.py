"""The constant text the ZK server's Lean loop writes into its records, pinned by name against main's bytes.

A word change alters no recorded text: the loop's ``FIREWALL-TIMES`` line and the refusals its record carries keep their
bytes, so a record made on main reads the same here (their Python twins are pinned in Flock's prover suite). A fault or
refusal message a failing run writes is no site.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proofs" / "recursion" / "verifier" / "zk_server" / "ZkServer"
LOOP = "Loop.lean"
REC = "Record.lean"

#: name -> (the producing file, a regex whose first group is the text it writes)
SITES = {
    "loop.times.tag": (LOOP, r'IO\.eprintln s!"([A-Z_-]+) \{Lean\.Json\.compress'),
    "record.stopped": (REC, r'refuse s!"recursive session \{s\} stopped at \{p\}: ([^"]+)" evs'),
    "record.released": (REC, r'refuse s!"recursive session \{s\} released all its statements: ([^"]+)" evs'),
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "loop.times.tag": "9781714dad0719b3ce747886515db2e89aa3f749cb91c5ea64c9d8d4528b891e",
    "record.released": "4f91ae0fb283b2b18e04ad068e0476db4d27be0dfd1c0d626cfbffdc4c3cee36",
    "record.stopped": "4f91ae0fb283b2b18e04ad068e0476db4d27be0dfd1c0d626cfbffdc4c3cee36",
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
        f"bytes back (`git show origin/main:<file>`). Only a new record format re-pins, with its schema id bumped.")
