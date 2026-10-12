"""`research usage` is `cluster usage` from the same tree's infra/cluster, run as a subprocess (research never imports cluster),
with its output and exit code."""
from __future__ import annotations

import json

from research import cli

ROW = {"schema": "cluster/usage-hour/v1", "hour": "2026-10-03T08:00Z", "node": "vy-nebius-2", "owner": "memory-accounting",
       "kind": "fill", "held_s": 1000.0, "busy_s": 100.0, "idle_s": 900.0, "unmeasured_s": 0.0, "useful_s": 100.0,
       "minutes_read": 60}
DAY = [{**ROW, "hour": f"2026-10-03T{h:02d}:00Z", "node": n, "owner": None, "kind": None, "held_s": 0.0, "busy_s": 0.0,
        "idle_s": 28800.0, "useful_s": 0.0} for h in range(24) for n in ("vy-nebius-1", "vy-nebius-2")]


def test_research_usage_runs_cluster_usage_from_this_tree(tmp_path, capfd):
    rec = tmp_path / "usage-hourly.jsonl"
    rec.write_text("".join(json.dumps(r) + "\n" for r in [*DAY, ROW]))      # the whole day, whatever the clock says
    assert cli.main(["usage", "wasters", "--day", "2026-10-03", "--file", str(rec), "--json"]) == 0
    rep = json.loads(capfd.readouterr().out)
    assert rep["schema"] == "cluster/usage-wasters/v1" and [w["owner"] for w in rep["wasters"]] == ["memory-accounting"]
    assert rep["hours_covered"] == {"vy-nebius-1": 24, "vy-nebius-2": 24}
    assert cli.main(["usage", "wasters", "--day", "2026-10-01", "--file", str(rec)]) == 1
    assert cli.main(["usage", "hourly", "--hours", "x"]) == 2


def test_without_tools_cluster_beside_it_research_usage_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "CLUSTER_SRC", tmp_path)
    assert cli.main(["usage", "hourly"]) == 2
    assert "no infra/cluster" in capsys.readouterr().err
