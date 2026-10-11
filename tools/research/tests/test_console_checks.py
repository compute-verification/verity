"""The console's store panel: the last day's `check` Attempts, each with the main commit its `landed` label names."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

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
