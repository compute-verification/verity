"""The release side as `rec_profile` reads it from 85's RELEASE=1 records (oprove's release.json and release-wall.tsv, the
inner step's schedule and coin-tree audit), and the session steps' times the result carries."""
from __future__ import annotations

import json
from pathlib import Path

import rec_profile as RP
from test_recursion_e2e import job

PARTS = ["L0", "L1", "alg-p0", "cap"]


def _release(o: Path, ok: bool = True, held: bool = True) -> None:
    (o / "params.txt").write_text((o / "params.txt").read_text() + "release=1 release_budget=\n")
    op = o / "steps" / "oprove" / "r1" / "out" / "oprove"
    rows = [{"part": p, "window_ms": 1000, "release_ms": 2000 + 1000 * j, "ready_ms": 1500 + 1000 * j, "decided": "go",
             "record": ["release"], "slack_ms": 500, "loop_release_ms": 2000 + 1000 * j} for j, p in enumerate(PARTS)]
    (op / "release.json").write_text(json.dumps({"session": "r0000", "stream": "circuit/rep0", "start_ms": 1000, "parts": rows,
                                                 "released": PARTS, "stops": [], "inner_finished": True, "ok": ok,
                                                 "schedule_ms": 4000}))
    (op / "release-wall.tsv").write_text("".join(f"{p}\t{p}\t{1000 * j}\t{1000 * j + 400}\t0\n" for j, p in enumerate(PARTS)))
    i = o / "steps" / "inner" / "r1" / "out" / "inner"
    (i / "release.json").write_text(json.dumps({"parts": PARTS, "windows_ms": [1000] * 4}))
    (i / "tree-source.txt").write_text("local 127.0.0.1:9\n")
    (i / "tree-held.json").write_text(json.dumps({"ok": held}))


def test_the_release_side_is_read_and_holds_the_headline(tmp_path):
    o = job(tmp_path)
    _release(o)
    doc = RP.read(o)
    rel = doc["release"]
    assert rel["ok"] and rel["trees_held"] and rel["tree_source"].startswith("local")
    assert [p["prove_wall_s"] for p in rel["parts"]] == [0.4] * 4 and rel["schedule"]["parts"] == PARTS
    assert doc["headline"]["released"] is True
    md = RP.markdown(doc)
    assert "**Release side.**" in md and "| L0 | 1.0 | 1.0 | 0.5 | 0.5 | 0.4 | go | release |" in md


def test_the_result_carries_each_session_steps_time(tmp_path):
    """Each session step's first invocation's time; the job has no vcap step, which has none."""
    doc = RP.read(job(tmp_path))
    times = doc["session_times"]
    assert tuple(times) == RP.SESSION_STEPS == RP.E2E_STEPS[1:]
    assert times["vcap"] is None
    assert times["inner"] == {"wall_s": 70.0, "peak_rss_gb": 2.0, "cpu_s": 2.0}
    assert all({"wall_s", "cpu_s"} <= set(t) for s, t in times.items() if s != "vcap")
    assert json.loads(json.dumps(doc))["session_times"] == times


def test_a_release_not_held_or_a_tree_not_the_challengers_is_no_accepted_run(tmp_path):
    for ok, held in ((False, True), (True, False)):
        o = job(tmp_path / f"{ok}{held}")
        _release(o, ok, held)
        doc = RP.read(o)
        assert doc["release"]["ok"] is False and doc["headline"]["released"] is False and doc["headline"]["ok"] is False


def test_release_1_without_a_report_is_no_accepted_run(tmp_path):
    o = job(tmp_path)
    (o / "params.txt").write_text((o / "params.txt").read_text() + "release=1\n")
    doc = RP.read(o)
    assert "release" not in doc and doc["headline"]["ok"] is False
