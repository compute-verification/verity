"""Writers of a machine record (machines.d/NAME.toml): a mark (`pods stopped`, fleet.py's) changes one top-level key and no
table, `register` keeps an unchanged table's text, and every function that writes under machines.d is one known to go
through `registry.write_key` or `registry.write_entry`."""
from __future__ import annotations

import ast
import datetime as dt
import tomllib
from pathlib import Path

import pytest

from research.pods import registry
from research.pods.nebius import fleet

REPO = Path(__file__).resolve().parents[3]
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


#: every function under tools/research and infra/ that writes a file and names machines.d or a record writer, and why it
#: may: each writes a record only through `write_key` (one key) or `write_entry` (a whole record), or writes no record
KNOWN = {
    "tools/research/src/research/pods/registry.py::write_entry": "the whole-record writer",
    "tools/research/src/research/pods/registry.py::write_key": "the one-key writer",
    "tools/research/src/research/pods/registry.py::_write_checked": "register: write_entry",
    "tools/research/src/research/pods/registry.py::main": "import: write_entry; the families file is no record",
    "tools/research/src/research/pods/registry.py::stopped": "write_key's text (with_key) through _replace",
    "tools/research/src/research/pods/nebius/fleet.py::set_key": "write_key",
    "tools/research/src/research/pods/nebius/fleet.py::watch_unreachable": "set_key; its own write is fleet's state file",
    "infra/control/vm_stop.py::one_pass": "writes its own state file, no record",
}
_NAMES = ("MACHINES_D", "machines.d", "write_dir", "registry_dir", "REGISTRY_ENV", "write_entry", "write_key", "with_key")
_WRITES = {"write_text", "write_bytes", "replace", "rename", "dump", "write_entry", "write_key", "_replace"}


def _writers() -> set[str]:
    found = set()
    for root in ("tools/research", "infra"):
        for f in sorted((REPO / root).rglob("*.py")):
            text = f.read_text()
            if not any(n in text for n in _NAMES):
                continue
            for fn in ast.walk(ast.parse(text)):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name in ("with_key", "_replace"):
                    continue
                if not any(n in (ast.get_source_segment(text, fn) or "") for n in _NAMES):
                    continue
                for c in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                    called = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
                    mode = [a.value for a in [*c.args[1:2], *(k.value for k in c.keywords if k.arg == "mode")]
                            if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                    if called in _WRITES or (called == "open" and any(m[:1] in "wax" for m in mode)):
                        found.add(f"{f.relative_to(REPO)}::{fn.name}")
                        break
    return found


def test_every_writer_under_machines_d_is_a_known_one():
    """Rule: a new writer of a machine record is seen; weakening: a third writer that appends a mark into a table, unseen."""
    found = _writers()
    assert found - KNOWN.keys() == set(), ("a new function writes under machines.d: write a record's key through "
                                           "research.pods.registry.write_key (or a whole record through write_entry), then "
                                           "add it to KNOWN with why")
    assert KNOWN.keys() - found == set(), "KNOWN names a writer that is gone: drop it"
