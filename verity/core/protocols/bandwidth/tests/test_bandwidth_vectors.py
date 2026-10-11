"""The code reproduces every pinned vector (`vectors.json`, written by `bandwidth_vectors.py`)."""

from __future__ import annotations

import json

import bandwidth_vectors as bv


def test_vectors_are_reproduced() -> None:
    pinned = json.loads(bv.PATH.read_text())
    assert bv.generate() == pinned


def test_vectors_cover_both_tags_and_a_short_last_leaf() -> None:
    pinned = json.loads(bv.PATH.read_text())
    assert {v["tag"] for v in pinned["tags"]} == {"ro-sha512/v1", "gf31-poly/v1"}
    poly = next(a for a in pinned["audits"] if a["tag"] == "gf31-poly/v1")
    assert poly["leaves"] == 3 and poly["blocks"] % 16 == 5
