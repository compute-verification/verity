"""The console's store panel: the last day's `check` Attempts, each with the main commit its `landed` label names."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from research.store import bucket_index as BI
from research.store.local import LocalStore

_spec = importlib.util.spec_from_file_location("verity_console", Path(__file__).parents[1] / "console" / "verity_console.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def test_a_check_shows_the_commit_it_landed_as_from_its_landed_label_and_an_old_one_is_left_out():
    st = LocalStore()
    idx = st.index
    for run, start, rc in (("r20261011-100000-aaaa", "2026-10-11T10:00:00Z", 0), ("r20261011-110000-bbbb", "2026-10-11T11:00:00Z", 1),
                           ("r20261009-100000-cccc", "2026-10-09T10:00:00Z", 0)):
        idx.db.execute("INSERT INTO attempts(id, tool, start_utc, wall_s, rc, state, source_commit, manifest_json) VALUES (?,?,?,?,?,?,?,?)",
                       (run, "check", start, 600, rc, "done", "ab" * 20, json.dumps({"execution": {"machine": "vy-nebius-1"}})))
    idx.db.commit()
    st.add_label("r20261011-100000-aaaa", "landed", "cd" * 20, "research-merge")
    st.add_label("r20261011-110000-bbbb", "landed", "ef" * 20, "someone-else")
    p = vc.check_panel(idx, datetime(2026, 10, 11, 12, tzinfo=timezone.utc))
    assert vc.problems(p) == []
    assert [(r[0], r[3], r[5], r[6]) for r in p["rows"]] == [
        ("r20261011-110000-bbbb", "fail (rc 1)", None, "vy-nebius-1"),
        ("r20261011-100000-aaaa", "pass", "cd" * 4 + "c", "vy-nebius-1")]


def derived_index(path: Path) -> BI.SqliteIndex:
    idx = BI.SqliteIndex(path)
    rows = []
    for run, start, rc in (("r20261011-100000-aaaa", "2026-10-11T10:00:00Z", 0),
                            ("r20261011-110000-bbbb", "2026-10-11T11:00:00Z", 1),
                            ("r20261009-100000-cccc", "2026-10-09T10:00:00Z", 0)):
        rows.append(BI.Row(run, "run", "check", start, {
            "id": run, "tool": {"name": "check"}, "execution": {"start_utc": start, "wall_s": 600, "machine": "vy-nebius-1"},
            "status": {"state": "done", "rc": rc}, "source": {"commit": "ab" * 20}}))
    rows += [
        BI.Row("label:" + "1" * 64, "label", "landed", "2026-10-11T11:30:00Z",
               {"target": "r20261011-100000-aaaa", "key": "landed", "value": "cd" * 20, "by": "research-merge"}),
        BI.Row("label:" + "2" * 64, "label", "landed", "2026-10-11T11:31:00Z",
               {"target": "r20261011-110000-bbbb", "key": "landed", "value": "ef" * 20, "by": "someone-else"})]
    idx.add(rows)
    return idx


def test_the_maintained_index_produces_the_same_check_panel(tmp_path):
    idx = derived_index(tmp_path / "index.sqlite")
    p = vc.check_index_panel(idx, datetime(2026, 10, 11, 12, tzinfo=timezone.utc))
    assert vc.problems(p) == []
    assert [(r[0], r[3], r[5], r[6]) for r in p["rows"]] == [
        ("r20261011-110000-bbbb", "fail (rc 1)", None, "vy-nebius-1"),
        ("r20261011-100000-aaaa", "pass", "cd" * 4 + "c", "vy-nebius-1")]
    idx.close()


def test_the_store_group_reads_the_maintained_index_without_refreshing_r2(tmp_path, monkeypatch):
    idx = derived_index(tmp_path / "index.sqlite")
    idx.close()
    monkeypatch.setenv("STORE_INDEXER_DATABASE_URL", "postgresql://dummy@db.example/store")
    monkeypatch.setattr(BI, "PostgresIndex", lambda *args, **kwargs: BI.SqliteIndex(tmp_path / "index.sqlite"))
    monkeypatch.setattr(LocalStore, "refresh", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("listed R2")))
    ns = SimpleNamespace(research_src=Path(__file__).parents[1] / "src", store=None, no_refresh=False)
    panels, info = vc.store_panels(ns, datetime(2026, 10, 11, 12, tzinfo=timezone.utc), [])
    assert [p["id"] for p in panels] == ["verity/merge-checks"]
    assert info["refresh"].startswith("maintained index")
