"""`research run-times`: each run's wait to start, setup and run time from its own records, on fixture run directories."""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
from datetime import UTC, datetime

import pytest

from research import run_records, run_times as R
from research.store.local import LocalStore

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
POOLED = ["env", "CHECK_SLOTS=/workspace/research/locks/research/slots", "python3", "tools/verity/check/slot.py", "--by", "proofs",
          "--wait-max", "1800", "--", "nice", "-n", "19"]
QUICK = [*POOLED, "bash", "-lc", "uv sync -q && uv run tools/verity/check/suites.py --changed abc --quick"]
RECORDS = [*POOLED, "python3", "tools/verity/lean/lean_changed.py", "--records", "security"]
SHARD = ["python3", "tools/verity/check/slot.py", "--priority", "--rank", "3", "--", "uv", "run", "--locked", "python",
         "tools/verity/check/check.py", "--keep-going"]
DIRECT = ["bash", "-lc", "python3 measure.py"]
WAITED, AT_ONCE, REFUSED, CANCELLED = ("r20261028-100000-aaaa", "r20261028-100100-bbbb", "r20261028-100200-cccc",
                                       "r20261028-100300-dddd")


def _run(root, rid, command=None, *, status=None, slot=None, refused=None, events=(), stderr=None, live=None, claimed=None):
    d = root / rid
    d.mkdir(parents=True)
    docs = {"job.json": command and {"execution": {"command": command}, "run": {"remote": {"machine": "n1"}}},
            "status.json": status, "slot.json": slot, "refused.json": refused, "check-live.json": live,
            "launch.json": claimed and {"claimed_utc": claimed, "argv": ["runner", "--", "TYPED-TOKEN"], "hostname": "instance-7"}}
    for name, doc in docs.items():
        if doc:
            (d / name).write_text(json.dumps(doc))
    if events:
        (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    if stderr is not None:
        (d / "stderr.log").write_text(stderr)
    return d


def _done(started, ended, state="done", **k):
    return {"state": state, "started_utc": started, "ended_utc": ended, "command": ["TYPED-TOKEN"], **k}


@pytest.fixture
def runs(tmp_path):
    """Four launches: one that waited 12 min for its slot, one that started at once, one the machine refused, and one cancelled
    after 20 min in line."""
    root = tmp_path / "runs"
    _run(root, WAITED, QUICK, claimed="2026-10-28T10:00:30.000Z",
         status=_done("2026-10-28T10:00:31.000Z", "2026-10-28T10:42:00.000Z"),
         slot={"state": "held", "since": "2026-10-28T10:00:32Z", "took": "2026-10-28T10:12:00Z", "slot": "r1"},
         stderr="slot: 2 check(s) ahead in line; retrying every 20 s\nslot:   r1  held by pid 7 (run r1, by x): bash -lc OTHERS-TOKEN\n"
                "slot: every check slot is taken (r1: busy); retrying every 20 s\nslot: check slot r1 (cores 0-7)\n")
    _run(root, AT_ONCE, DIRECT, claimed="2026-10-28T10:01:04.000Z",
         status=_done("2026-10-28T10:01:06.000Z", "2026-10-28T10:03:06.000Z"))
    _run(root, REFUSED, refused={"at_utc": "2026-10-28T10:02:05.000Z", "fault": "MemoryNotAvailable", "reason": "asked 300 GB",
                                 "run_id": REFUSED})
    _run(root, CANCELLED, RECORDS, claimed="2026-10-28T10:03:10.000Z",
         status=_done("2026-10-28T10:03:11.000Z", "2026-10-28T10:23:00.000Z", state="failed", rc=143),
         slot={"state": "waiting", "since": "2026-10-28T10:03:12Z", "why": "3 check(s) ahead in line"},
         events=[{"kind": "started", "t_utc": "2026-10-28T10:03:11.000Z"},
                 {"kind": "cancel_intent", "actor": "manual", "reason": "ran elsewhere", "t_utc": "2026-10-28T10:22:59.000Z",
                  "target": {"members": [{"comm": "TYPED-TOKEN"}]}}])
    return root


def _rows(root, now=NOW) -> dict[str, dict]:
    return {r["run"]: r for r in (R.times(run_records.record(d), now) for d in run_records.window(root, "r", "s"))}


def test_a_run_that_waited_counts_from_its_launch_to_its_slot_and_says_what_it_waited_behind(runs):
    r = _rows(runs)[WAITED]
    assert (r["outcome"], r["kind"], r["pool"], r["by"], r["machine"]) == ("started", "quick suites", "research", "proofs", "n1")
    assert r["wait_s"] == 720.0, "from the run id's time, the launch, to slot.json's took: the 30 s of shipping are in it"
    assert (r["ship_s"], r["line_s"]) == (30.0, 688.0), "its two parts: to the machine's claim, and in the line"
    assert r["setup_s"] is None, "no record says when a suites job's own command started"
    assert r["run_s"] == 1800.0, "so its run counts from the slot"
    assert r["behind"] == "every check slot is taken (r1: busy)", "the held slot.json has no why: slot.py's last headline in stderr.log"


def test_a_run_through_no_slot_started_when_its_command_did(runs):
    r = _rows(runs)[AT_ONCE]
    assert (r["outcome"], r["kind"], r["pool"], r["wait_s"], r["line_s"], r["run_s"]) == ("started", "other", "-", 6.0, None, 120.0)


def test_a_refused_launch_is_counted_with_its_reason_and_is_no_wait(runs):
    rows = _rows(runs)
    r = rows[REFUSED]
    assert (r["outcome"], r["kind"], r["pool"], r["wait_s"]) == ("refused", R.UNKNOWN, R.UNKNOWN, 5.0)
    assert r["behind"] == "MemoryNotAvailable: asked 300 GB"
    s = R.summary(list(rows.values()))
    assert s["all"]["refused"] == 1 and s["all"]["wait"]["n"] == 3, "its 5 s to the refusal is in no wait percentile"
    assert s["refused"] == {"faults": {"MemoryNotAvailable": 1}, "told": {"n": 1, "p50_min": 0.1, "p90_min": 0.1, "p95_min": 0.1}}


def test_a_run_cancelled_in_line_counts_with_the_time_it_waited(runs):
    rows = _rows(runs)
    r = rows[CANCELLED]
    assert (r["outcome"], r["kind"], r["wait_s"], r["run_s"]) == ("cancelled", "Lean records", 1200.0, None)
    assert (r["behind"], r["cancel"]) == ("3 check(s) ahead in line", "manual: ran elsewhere")
    s = R.summary(list(rows.values()))
    assert s["all"]["over"] == 2 and [x["run"] for x in R.slow(list(rows.values()))] == [CANCELLED, WAITED], "the longest first"
    assert s["kind"]["Lean records"]["wait"] == {"n": 1, "p50_min": 20.0, "p90_min": 20.0, "p95_min": 20.0}
    assert s["behind"] == {"others ahead in line": 1, "slots busy": 1}


def test_no_command_line_leaves_the_machine(runs):
    out = json.dumps([run_records.record(d) for d in run_records.window(runs, "r", "s")])
    assert "TYPED-TOKEN" not in out and "OTHERS-TOKEN" not in out and "suites.py" not in out


def test_a_run_still_in_line_has_waited_until_now_and_one_in_its_slot_has_no_run_time_yet(tmp_path):
    _run(tmp_path, WAITED, QUICK, status={"state": "running", "started_utc": "2026-10-28T10:00:05.000Z"},
         slot={"state": "waiting", "since": "2026-10-28T10:00:06Z", "why": "every check slot is taken (b0: 5 GB of memory free, under "
               "the 60 GB a check here may peak at; b1: busy)"})
    _run(tmp_path, AT_ONCE, QUICK, status={"state": "running", "started_utc": "2026-10-28T10:01:05.000Z"},
         slot={"state": "held", "since": "2026-10-28T10:01:06Z", "took": "2026-10-28T10:01:06Z"})
    rows = _rows(tmp_path)
    assert (rows[WAITED]["outcome"], rows[WAITED]["wait_s"]) == ("waiting", 7200.0)
    assert R.cause(rows[WAITED]["behind"]) == "slots busy + memory short"
    assert (rows[AT_ONCE]["outcome"], rows[AT_ONCE]["wait_s"], rows[AT_ONCE]["run_s"]) == ("started", 6.0, None)


def test_setup_is_from_the_slot_to_check_py_starting_where_its_record_says(tmp_path):
    _run(tmp_path, WAITED, SHARD, status=_done("2026-10-28T10:00:40.000Z", "2026-10-28T11:02:10.000Z"),
         slot={"state": "held", "since": "2026-10-28T10:00:41Z", "took": "2026-10-28T10:01:00Z", "priority": True},
         live={"started_utc": "2026-10-28T10:02:10Z"})
    r = _rows(tmp_path)[WAITED]
    assert (r["kind"], r["pool"], r["wait_s"], r["setup_s"], r["run_s"]) == ("check shards", "check", 60.0, 70.0, 3600.0)


def test_a_run_that_left_the_line_waited_and_a_twin_or_one_that_never_joined_did_not(tmp_path):
    _run(tmp_path, WAITED, QUICK, status=_done("2026-10-28T10:00:05.000Z", "2026-10-28T10:10:06.000Z", state="failed", rc=75),
         slot={"state": "left", "since": "2026-10-28T10:00:06Z", "left": "2026-10-28T10:10:06Z",
               "why": "no check slot within --wait-max 600 s (6 check(s) ahead in line)"})
    _run(tmp_path, AT_ONCE, SHARD, status=_done("2026-10-28T10:01:05.000Z", "2026-10-28T10:04:00.000Z", state="failed", rc=75),
         slot={"state": "left", "since": "2026-10-28T10:01:06Z", "left": "2026-10-28T10:04:00Z",
               "why": "its twin took a slot first (run r1 on n2)"})
    _run(tmp_path, REFUSED, QUICK, status=_done("2026-10-28T10:02:05.000Z", "2026-10-28T10:02:06.000Z", state="failed", rc=75))
    _run(tmp_path, CANCELLED)
    rows = _rows(tmp_path)
    assert [rows[k]["outcome"] for k in (WAITED, AT_ONCE, REFUSED, CANCELLED)] == ["left", "twin", "no slot", "no record"]
    s = R.summary(list(rows.values()))
    assert (s["all"]["wait"]["n"], s["all"]["over"], s["behind"]) == (1, 1, {"others ahead in line": 1})


def test_a_command_is_read_for_its_kind_its_pool_and_whom_it_is_for():
    assert [run_records.kind(c) for c in (QUICK, RECORDS, SHARD, DIRECT)] == ["quick suites", "Lean records", "check shards", "other"]
    assert run_records.kind([*POOLED, "sh", "-c", 'exec python3 "$RESEARCH_RUN_DIR/inputs/quick_tier.py" --lean Security']) == "quick suites"
    assert run_records.kind(["gpu-lease", "1", "--wait", "--mem-gb", "0", "--", "python3", "bench.py"]) == "GPU jobs"
    assert run_records.kind([*POOLED, "uv", "run", "tools/verity/check/suites.py", "verity/core"]) == "other", "not the quick tier"
    assert [run_records.pool(c) for c in (QUICK, SHARD, DIRECT)] == ["research", "check", "-"]
    assert (run_records.asked_by(QUICK), run_records.asked_by(DIRECT), run_records.asked_by(["x", "--by"])) == ("proofs", None, None)


def test_percentiles_are_by_kind_machine_and_pool():
    def row(i, kind, machine, pool, wait_min):
        return {"run": f"r{i}", "kind": kind, "machine": machine, "pool": pool, "outcome": "started", "wait_s": wait_min * 60.0,
                "ship_s": 6.0, "line_s": wait_min * 60.0 - 6, "setup_s": None, "run_s": 600.0, "behind": None}

    rows = [row(i, "quick suites", "cpu-2" if i % 2 else "cpu-10", "research", float(i)) for i in range(1, 21)]
    rows.append(row(21, "other", "cpu-2", "-", 0.1))
    s = R.summary(rows)
    assert list(s["kind"]) == ["quick suites", "other"] and list(s["machine"]) == ["cpu-2", "cpu-10"] and list(s["pool"]) == ["-", "research"]
    assert s["kind"]["quick suites"]["wait"] == {"n": 20, "p50_min": 11.0, "p90_min": 18.0, "p95_min": 19.0}
    assert (s["kind"]["quick suites"]["over"], s["machine"]["cpu-2"]["runs"], s["pool"]["research"]["started"]) == (10, 11, 20)
    assert s["all"]["run"] == {"n": 21, "p50_min": 10.0, "p90_min": 10.0, "p95_min": 10.0} and s["all"]["setup"] == {"n": 0}
    assert s["parts"]["ship"]["p95_min"] == 0.1 and s["parts"]["line"]["n"] == 21


def _event(event, task, at, sub="s1", **k):
    return {"event": event, "task": task, "submission": sub, "ts": at, **k}


def test_a_graph_task_waits_from_when_it_could_first_run_not_from_its_submission():
    events = [_event("queued", "test-file:a", "2026-10-28T09:00:00Z"),
              _event("queued", "lean-module:b", "2026-10-28T10:30:00Z"),
              _event("started", "lean-module:b", "2026-10-28T10:45:00Z", wait_s=6300.0, worker="cpu-7/c1"),
              _event("finished", "lean-module:b", "2026-10-28T10:46:00Z", seconds=60.0, status="pass"),
              _event("started", "test-file:a", "2026-10-28T11:00:00Z", wait_s=7200.0, worker="cpu-1/c2"),
              _event("errored", "test-file:a", "2026-10-28T11:01:00Z", seconds=60.0, status="error"),
              _event("started", "test-file:a", "2026-10-28T11:05:00Z", wait_s=7500.0, worker="cpu-1/c2"),
              _event("queued", "test-file:c", "2026-10-28T11:10:00Z"),
              _event("queued", "test-file:d", "2026-10-28T11:10:00Z"), _event("hit", "test-file:d", "2026-10-28T11:10:01Z"),
              _event("started", "test-file:c", "2026-10-28T12:30:00Z", worker="cpu-1/c2")]
    g = R.graph_tasks(events, datetime(2026, 10, 28, 10, 0, tzinfo=UTC), NOW)
    assert [(t["task"], t["machine"], t["wait_s"], t["run_s"], t["status"]) for t in g["tasks"]] == [
        ("lean-module:b", "cpu-7", 900.0, 60.0, "pass"), ("test-file:a", "cpu-1", 7200.0, 60.0, "error"),
        ("test-file:a", "cpu-1", None, None, None)], "b by its queued event, not its wait_s; a's retry has no wait"
    assert g["silent"] == [{"task": "test-file:c", "submission": "s1"}], "queued, and no event by the window's end; d was a hit"
    s = R.graph_summary(g)
    assert (s["all"]["tasks"], s["all"]["wait"]["n"], s["all"]["over"]) == (3, 2, 2) and list(s["kind"]) == ["lean-module", "test-file"]
    assert s["submissions_over"] == {"s1": {"tasks": 2, "longest_min": 120.0}} and s["silent"] == {"tasks": 1, "submissions": 1}


def _machine(root):
    """`ssh MACHINE python3 - RUNS FIRST LAST` with run_records.py on its stdin, answered here from ``root``."""
    def run(argv, script):
        assert script == open(run_records.__file__, "rb").read() and argv[:2] == ["ssh", "n1"]
        py, dash, *args = argv[2].split()
        assert (py, dash, args[0]) == ("python3", "-", "/w/runs")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = run_records.main([str(root), *args[1:]])
        return subprocess.CompletedProcess(argv, rc, out.getvalue().encode(), b"")
    return run


def _store(tmp_path, events=()):
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    part = tmp_path / "vault" / "events" / "v1" / "s1" / "part-000000.jsonl"
    part.parent.mkdir(parents=True)
    part.write_text("".join(json.dumps(e) + "\n" for e in events))
    return LocalStore(tmp_path / "reader", config_path=cfg)


def test_it_reads_each_machine_and_names_one_that_did_not_answer(runs, tmp_path, capsys, monkeypatch):
    def reach(name):
        return ["ssh", name], "/w/runs"

    def run(argv, script):
        if argv[1] == "n2":
            raise subprocess.TimeoutExpired(argv, 1)
        if argv[1] == "n3":
            return subprocess.CompletedProcess(argv, 255, b"", b"ssh: connect to host 203.0.113.7 port 22: Connection timed out\n")
        if argv[1] == "n4":
            return subprocess.CompletedProcess(argv, 0, json.dumps({"run": WAITED}).encode() + b"\n", b"")
        return _machine(runs)(argv, script)

    monkeypatch.setattr(R, "clock", lambda: NOW)
    store = _store(tmp_path, [_event("queued", "test-file:a", "2026-10-28T10:00:00Z"),
                              _event("started", "test-file:a", "2026-10-28T10:01:00Z", worker="n1/c1")])
    assert R.main(["--hours", "2", "--on", "n1,n2,n3,n4", "--json"], store=store, reach=reach, run=run) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["window"] == {"start": "2026-10-28T10:00:00Z", "end": "2026-10-28T12:00:00Z"} and out["read"] == ["n1"]
    assert out["unread"] == {"n2": f"no answer in {R.READ_TIMEOUT_S} s", "n3": "exit 255: ssh: connect to host n3 port 22: Connection "
                             "timed out", "n4": "its answer was cut short: 1 run(s) and no closing count that matches"}
    assert [(r["run"], r["machine"], r["outcome"]) for r in out["runs"]] == [
        (WAITED, "n1", "started"), (AT_ONCE, "n1", "started"), (REFUSED, "n1", "refused"), (CANCELLED, "n1", "cancelled")]
    assert out["summary"]["machine"]["n1"]["runs"] == 4 and out["graph"]["all"]["wait"] == {"n": 1, "p50_min": 1.0, "p90_min": 1.0,
                                                                                            "p95_min": 1.0}


def test_the_table_names_each_run_over_ten_minutes_and_a_window_takes_only_its_launches(runs, capsys, monkeypatch):
    monkeypatch.setattr(R, "clock", lambda: NOW)
    assert R.main(["--runs", str(runs), "--until", "2026-10-28T10:02:30Z", "--hours", "1"]) == 0
    out = capsys.readouterr().out
    assert "runs launched 2026-10-28 09:02Z to 2026-10-28 10:02Z: 3 on 1 machine(s)" in out and CANCELLED not in out
    assert "runs that waited over 10 min: 1 of the 2 that asked to start; behind: slots busy 1" in out
    named = out.split("runs that waited over")[1]
    assert WAITED in named and "every check slot is taken (r1: busy)" in named and AT_ONCE not in named and "graph2" not in out
    assert R.main(["--runs", str(runs), "--until", "2026-10-28T11:00:00Z", "--hours", "1"]) == 0
    assert "cancelled 3 check(s) ahead in line; cancelled (manual: ran elsewhere)" in " ".join(capsys.readouterr().out.split())
    assert "refused: 1 (MemoryNotAvailable 1)" in out


def test_a_record_that_is_there_and_unreadable_is_named_never_read_as_absent(tmp_path, capsys, monkeypatch):
    d = _run(tmp_path, WAITED, QUICK, status=_done("2026-10-28T10:00:05.000Z", "2026-10-28T10:30:00.000Z"),
             slot={"state": "held", "since": "2026-10-28T10:00:06Z", "took": "2026-10-28T10:12:00Z"})
    (d / "slot.json").write_text('{"state": "held", "since": "2026-10-28T10:00:06Z", "to')
    (d / "events.jsonl").write_text('{"kind": "cancel_intent", "actor": "man\n')
    _run(tmp_path, AT_ONCE, DIRECT, status=_done("2026-10-28T10:01:06.000Z", "2026-10-28T10:03:06.000Z"))
    _run(tmp_path, CANCELLED, DIRECT, status=_done("the day before", "2026-10-28T10:03:06.000Z"))
    rec = run_records.record(d)
    assert set(rec["unread"]) == {"slot.json", "events.jsonl"} and rec["unread"]["slot.json"].startswith("not JSON: ")
    assert "unread" not in run_records.record(tmp_path / AT_ONCE), "a record file a run doesn't have is not one it couldn't read"
    rows, left_out = R.rows_of((run_records.record(p) for p in run_records.window(tmp_path, "r", "s")), NOW)
    assert [(r["run"], r["outcome"], bool(r["unread"])) for r in rows] == [(WAITED, "no slot", True), (AT_ONCE, "started", False)]
    assert list(left_out) == [CANCELLED] and "a time in its records is no time" in left_out[CANCELLED]
    monkeypatch.setattr(R, "clock", lambda: NOW)
    assert R.main(["--runs", str(tmp_path), "--hours", "3"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert f"2 on 1 machine(s); LEFT OUT, in no number here: {CANCELLED} (" in out
    assert f"1 run(s) have a record file that is there and couldn't be read, and their rows are made without it: {WAITED} (" in out


def test_an_events_part_that_is_not_json_leaves_the_graph_unread_by_name(runs, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(R, "clock", lambda: NOW)
    store = _store(tmp_path, [_event("queued", "test-file:a", "2026-10-28T10:00:00Z")])
    part = tmp_path / "vault" / "events" / "v1" / "s1" / "part-000000.jsonl"
    part.write_text(part.read_text() + "\n" + json.dumps(_event("started", "test-file:a", "2026-10-28T10:01:00Z", worker="n1/c1")) + "\n")

    def read() -> dict:
        assert R.main(["--hours", "2", "--on", "n1", "--json"], store=store, reach=lambda name: (["ssh", name], "/w/runs"),
                      run=_machine(runs)) == 0
        return json.loads(capsys.readouterr().out)

    assert read()["graph"]["all"]["tasks"] == 1, "a blank line is no line"
    part.write_text(part.read_text() + '{"event": "started", "task": "test-fi\n')
    out = read()
    assert len(out["runs"]) == 4 and out["left_out"] == {}, "the runs are read all the same"
    assert out["graph"].startswith("ValueError: events/v1/s1/part-000000.jsonl has a line that is not JSON")


def test_with_no_machine_read_it_answers_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(R, "clock", lambda: NOW)

    def reach(name):
        raise RuntimeError(f"machine {name!r} is not listed")

    assert R.main(["--on", "nowhere"], store=LocalStore(tmp_path / "s"), reach=reach) == 1
    out, err = capsys.readouterr()
    assert "UNREAD, in no number here: nowhere (machine 'nowhere' is not listed)" in out and "graph2's tasks: UNREAD" in out
    assert "no machine was read" in err
