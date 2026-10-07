"""``verity_flock.tables``: the ``flock-tables`` file ``flock_live::tables::parse`` reads (its Rust tests prove one)."""
from __future__ import annotations

import json

import pytest

from verity_flock import tables as TB


def _net(name: str) -> str:
    return f"flock-ir-unit/v2 {name} 2 1 1 0 1 1 1 1 1 1 1 1\n1 0 1 0\n1 1 1 1\n"


def test_a_tables_file_carries_its_tables_glue_and_circuits():
    glue = [{"dst": TB.region("c", {5: 0}, [0, 1, 2, 3, 4]), "src": TB.region("p", {5: 1}, [0, 1, 2, 3, 4]), "map": "id"}]
    text = TB.compose("demo", [{"name": "p", "circuit": "a", "blocks": 8}, {"name": "c", "circuit": "b", "blocks": 8}],
                      {"a": _net("a"), "b": _net("b")}, glue)
    lines = text.split("\n")
    assert lines[0] == "flock-tables demo"
    meta = json.loads(lines[1][5:])
    assert meta["format"] == TB.FORMAT and meta["statement"] == TB.STATEMENT and meta["glue"] == glue
    assert meta["glue"][0]["dst"]["fixed"] == {"5": 0}
    assert lines[2] == "CIRCUIT a 3" and lines[3].startswith("flock-ir-unit/v2 a")
    assert len(TB.digest(text)) == 128 and TB.digest(text) != TB.digest(text.replace('"map":"id"', '"map":{"perm":[1,0,2,3,4]}'))
    with pytest.raises(ValueError, match="does not carry"):
        TB.compose("demo", [{"name": "p", "circuit": "missing", "blocks": 8}], {"a": _net("a")}, [])
