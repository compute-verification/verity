"""The overnight dashboard's reading of `superseded_by` labels."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ovd", Path(__file__).parents[1] / "console" / "ovd.py")
ovd = sys.modules["ovd"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovd)


class Idx:
    def __init__(self, labels: list[tuple[str, str, str]]):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE labels (target TEXT, key TEXT, value TEXT)")
        self.db.executemany("INSERT INTO labels VALUES (?, ?, ?)", labels)

    def sql(self, q: str) -> list[tuple]:
        return self.db.execute(q).fetchall()


def test_a_run_superseded_by_itself_stays_shown():
    idx = Idx([("r-old", "superseded_by", "r-new"), ("r-cap", "superseded_by", "r-cap"), ("r-new", "ov.ws", "build")])
    assert ovd.superseded(idx) == {"r-old"}
