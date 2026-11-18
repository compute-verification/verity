"""Writers of a machine record (machines.d/NAME.toml): a mark (`pods stopped`, fleet.py's) changes one top-level key and no
table, and `register` keeps an unchanged table's text (tests/test_machine_record_writers.py holds every writer to the helpers)."""
from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

import pytest

from research.pods import registry
from research.pods.nebius import fleet

T = dt.datetime(2026, 11, 12, 19, 15, tzinfo=dt.UTC)
MARK = "2026-11-12T19:15:00Z"
TABLE = ('\n# read on the machine by MACHREC\n[properties]\ncards = { value = 8, read = "nvidia-smi -L | wc -l" }\n'
         'root = { value = true, read = "id -u" }  # a container\n')
RECORD = '# machine vy-spot\nprovider = "ssh"\nhost = "192.0.2.7"\nproject = "verity"\n' + TABLE


@pytest.fixture
def rec(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "notes" / "machines.d"
    d.mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(d))
    monkeypatch.setenv("RESEARCH_NOTES", str(d.parent))
    monkeypatch.setenv("RESEARCH_NOTES_SYNC", "0")
    (d / "vy-spot.toml").write_text(RECORD)
    return d / "vy-spot.toml"


def test_a_stop_mark_on_a_record_with_a_table_is_top_level_and_back_takes_it_out(rec):
    """Rule: the mark placement reads is the top-level key; weakening: appending it, which puts it in [properties]."""
    assert registry.stopped("vy-spot", now=T) == (rec, f"is marked stopped since {MARK}")
    doc = tomllib.loads(rec.read_text())
    assert doc["stopped_since"] == MARK and "stopped_since" not in doc["properties"]
    assert registry.left_out(doc, T.timestamp()) == f"stopped since {MARK}"
    assert rec.read_text().endswith(TABLE), "the table and its comments are as they were"
    assert registry.stopped("vy-spot", back=True) == (rec, f"is no longer marked stopped (it was since {MARK})")
    assert rec.read_text() == RECORD


def test_back_takes_out_a_stray_mark_an_older_writer_put_in_the_table_and_says_so(rec):
    stray = RECORD + f'stopped_since = "{MARK}"\n'
    rec.write_text(stray)
    assert registry.stopped("vy-spot", back=True) == (rec, "wasn't marked stopped; took out the stopped_since an older "
                                                           "`pods stopped` wrote inside [properties]")
    assert rec.read_text() == RECORD
    rec.write_text(stray)
    assert registry.stopped("vy-spot", now=T)[1].endswith("inside [properties]")
    doc = tomllib.loads(rec.read_text())
    assert doc["stopped_since"] == MARK and "stopped_since" not in doc["properties"]


def test_fleets_mark_goes_through_the_same_helper(rec, monkeypatch):
    monkeypatch.setattr(fleet, "MACHINES_D", rec.parent)
    fleet.set_key("vy-spot", "unreachable_since", MARK)
    assert tomllib.loads(rec.read_text())["unreachable_since"] == MARK and rec.read_text().endswith(TABLE)
    fleet.set_key("vy-spot", "unreachable_since", None)
    assert rec.read_text() == RECORD


def test_a_whole_record_written_again_keeps_an_unchanged_tables_text(rec):
    entry = tomllib.loads(RECORD)
    registry.write_entry(rec.parent, "vy-spot", {**entry, "lane": "infra"})
    assert rec.read_text().endswith(TABLE) and tomllib.loads(rec.read_text())["lane"] == "infra"
    changed = {**entry, "properties": {**entry["properties"], "root": {"value": False, "read": "id -u"}}}
    registry.write_entry(rec.parent, "vy-spot", changed)
    assert tomllib.loads(rec.read_text()) == changed and "# a container" not in rec.read_text()
