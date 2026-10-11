"""The code reproduces every pinned vector (`vectors.json`, written by `bandwidth_vectors.py`)."""

from __future__ import annotations

import json

import bandwidth_vectors as bv


def test_vectors_are_reproduced() -> None:
    pinned = json.loads(bv.PATH.read_text())
    assert bv.generate() == pinned


def test_vectors_cover_every_tag_and_a_short_last_leaf() -> None:
    pinned = json.loads(bv.PATH.read_text())
    assert {v["tag"] for v in pinned["tags"]} == {"ro-blake3/v1", "ro-sha512/v1", "gf31-poly/v1"}
    for name in ("gf31-poly/v1", "ro-blake3/v1"):
        sixteen = next(a for a in pinned["audits"] if a["tag"] == name)
        assert sixteen["leaves"] == 3 and sixteen["blocks"] % 16 == 5
