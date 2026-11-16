"""The constant text a block record carries, pinned by name against main's bytes.

A word change alters no recorded text: `check` compares a record made on main against these, so the joint guarantee's
name and file and the record's subprotocol key keep their bytes. A fault message is no site.
"""
from __future__ import annotations

import hashlib
import inspect
import re

import pytest

from verity_e2e_demo import block_record as B

#: name -> the text, as this tree writes it
SITES = {
    "JOINT_HORIZON.name": lambda: B.JOINT_HORIZON["name"],
    "JOINT_HORIZON.file": lambda: B.JOINT_HORIZON["file"],
    "record.subprotocols.key": lambda: re.findall(r'"([a-z_]+)": subprotocols, "composed"', inspect.getsource(B))[0],
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "JOINT_HORIZON.file": "2b9c14dfa922e564c15b5c4f382ee13e55dfa2b2c0a9d2ec939b8e6becce7803",
    "JOINT_HORIZON.name": "c444b6302a9c91a2de0d4466809cd6a86022829a9465d24d87d9fab08f34c176",
    "record.subprotocols.key": "14e32f3cbdd3b4c414be5f81c8c6b602bc9e814c0cad7290a07fe109ea9830a7",
}


def test_every_site_is_pinned():
    assert sorted(SITES) == sorted(PINS), "a site and its pin are added together, the pin from main's text"


@pytest.mark.parametrize("name", sorted(PINS))
def test_a_recorded_text_keeps_its_bytes(name):
    assert hashlib.sha256(SITES[name]().encode()).hexdigest() == PINS[name], (
        f"{name}: the text a block record carries changed. A word change alters no recorded text: put main's bytes back "
        f"(`git show origin/main:benchmarks/one_stage/interfaces/verity_e2e_demo/block_record.py`). Only a new record "
        f"format re-pins, with its format id bumped.")
