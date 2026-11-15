"""A run's memory mark (research.memmark): the figure from a series of the sampler's cgroup records, each source of the ask
and of the limit, when a run is over its limit, the runs whose figure can't be computed, the line `research status RUN`
prints, and that the harness writes the mark in the same write as the run's end. The records are written here as the
sampler writes them, with their own times, so nothing reads a clock or a cgroup."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from research import cli, memmark, remote
from research.cli import main
from research.telemetry import run as trun

GB, GIB = 10 ** 9, 2 ** 30
PY = sys.executable
SCOPE = "/user.slice/user-1001.slice/session-7.scope"


def _utc(i: int) -> str:
    return f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}.500Z"


def _rec(i: int, current: float, file: float = 0.0, shmem: float = 0.0) -> dict:
    """The sampler's cgroup record `i` seconds in; the sizes are GB."""
    return {"schema": "telemetry/resources/v0.1", "kind": "cgroup", "scope": "shared_pod_cgroup", "t_utc": _utc(i), "t_mono": 50.0 + i,
            "cgroup": {"version": 2, "path": SCOPE, "memory_current_bytes": int(current * GB), "memory_peak_bytes": int(99 * GB),
                       "memory_stat": {"anon": 1, "file": int(file * GB), "shmem": int(shmem * GB), "kernel": 2}}}


def _samples(run_dir: Path, recs: list[dict], tail: str = "") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / memmark.SAMPLES).write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs) + tail)


def test_the_peak_is_the_most_a_sample_held_with_the_file_cache_out_and_shared_memory_in():
    """A cgroup at its limit loses its page cache before the kernel stops anything, so the cache is not held memory; shared
    memory is `file` to the kernel and can't be reclaimed, so it is."""
    recs = [_rec(1, 10, file=6, shmem=1),            # holds 5
            _rec(2, 30, file=28, shmem=1),           # the most memory.current, nearly all cache: holds 3
            _rec(3, 12, file=6, shmem=2),            # holds 8: 4 of them not cache, the 2 of shared memory, and 2 more
            _rec(4, 9, file=2)]                      # holds 7
    assert [memmark.held(r["cgroup"]) for r in recs] == [5 * GB, 3 * GB, 8 * GB, 7 * GB]
    no_stat = {**_rec(5, 99), "cgroup": {"path": SCOPE, "memory_current_bytes": 99 * GB, "memory_stat": None}}
    got = memmark.peak([*recs, no_stat], 100 * GB)
    assert got == {"peak_held_bytes": 8 * GB, "peak_utc": _utc(3), "samples": 4, "over_limit": False, "over_limit_utc": None,
                   "cgroup": SCOPE}, "a record without memory.stat is no sample: its cache can't be taken out"


def test_the_ask_is_the_slot_s_booking_then_the_queue_job_s_memory_then_the_run_s_mem_gb():
    held = {"state": "held", "slot": "p0", "took": _utc(0), "reserves_gib": 193}
    pool = {"env": {memmark.RUN_MEM_ENV: "24"}}
    assert memmark.ask(pool, held) == (193 * GIB, "slot"), "what slot.py reserved for the run, not the run's --mem-gb"
    assert memmark.ask(pool, {"state": "held", "slot": "p0", "reserves_gib": 0}) == (24 * GB, "--mem-gb"), "a slot that reserves nothing"
    assert memmark.ask(pool, {**held, "state": "left"}) == (24 * GB, "--mem-gb"), "a slot it never took booked nothing"
    assert memmark.ask({"mem_gb": 24.5}, {}) == (int(24.5 * GB), "--mem-gb")
    assert memmark.ask({"mem_gb": None, "queue": {"job": {"memory_gib": 16}}}, {}) == (16 * GIB, "queue")
    assert memmark.ask({"queue": {"job": {}}}, {}) == (GIB, "queue"), "a queue job's default memory"
    assert memmark.ask({"queue": {"job": {"memory_gib": 16}}}, held) == (193 * GIB, "slot")
    assert memmark.ask({}, {}) == (None, "none") and memmark.ask({"mem_gb": None, "env": {}}, {}) == (None, "none")


def test_the_limit_is_the_ask_and_128_gb_or_128_gb_or_what_was_stated_and_never_over_the_machine_s_bound():
    assert memmark.limit(24 * GB, None, None) == {"limit_bytes": 152 * GB, "limit_from": "ask+128GB", "bound_bytes": None}
    assert memmark.limit(None, None, 600 * GB) == {"limit_bytes": 128 * GB, "limit_from": "default", "bound_bytes": 600 * GB}
    assert memmark.limit(24 * GB, {"gb": 300, "by": "proofs"}, 600 * GB) == {
        "limit_bytes": 300 * GB, "limit_from": "stated", "limit_by": "proofs", "bound_bytes": 600 * GB}
    assert memmark.limit(500 * GB, None, 600 * GB) == {"limit_bytes": 600 * GB, "limit_from": "bound", "bound_bytes": 600 * GB}
    assert memmark.limit(472 * GB, None, 600 * GB)["limit_from"] == "ask+128GB", "at the bound is not over it"
    assert memmark.limit(None, {"gb": 900, "by": "proofs"}, 600 * GB) == {
        "limit_bytes": 600 * GB, "limit_from": "bound", "limit_by": "proofs", "bound_bytes": 600 * GB}
    session = {"memory_max_bytes": 650 * GB, "effective_memory_max_bytes": 600 * GB}
    assert memmark.bound(session) == 600 * GB, "the least over the cgroup and its ancestors"
    assert memmark.bound({"memory_max_bytes": 650 * GB}) == 650 * GB
    assert memmark.bound({"memory_max_bytes": None, "effective_memory_max_bytes": None}) is None and memmark.bound({}) is None


def test_a_run_is_over_its_limit_from_the_first_sample_that_held_more():
    recs = [_rec(1, 100), _rec(2, 152), _rec(3, 200, file=47), _rec(4, 260), _rec(5, 90)]
    got = memmark.mark({"mem_gb": 24}, {}, {"effective_memory_max_bytes": 600 * GB}, recs)
    assert (got["ask_bytes"], got["ask_from"], got["limit_bytes"], got["limit_from"]) == (24 * GB, "--mem-gb", 152 * GB, "ask+128GB")
    assert (got["over_limit"], got["over_limit_utc"]) == (True, _utc(3)), "152 held is at the limit, 153 is the first over it"
    assert (got["peak_held_bytes"], got["peak_utc"], got["samples"], got["reason"]) == (260 * GB, _utc(4), 5, None)
    cache = memmark.mark({"mem_gb": 24}, {}, {}, [_rec(1, 400, file=300)])
    assert (cache["peak_held_bytes"], cache["over_limit"], cache["over_limit_utc"]) == (100 * GB, False, None), \
        "memory.current over the limit with its cache, under it without: squeezed, not stopped"


def test_a_slot_run_s_peak_counts_from_when_it_took_its_slot():
    slot = {"state": "held", "slot": "p0", "took": "2026-01-01T00:00:03Z", "reserves_gib": 24}
    got = memmark.mark({}, slot, {}, [_rec(1, 500), _rec(2, 400), _rec(3, 30), _rec(4, 20)])
    assert (got["peak_held_bytes"], got["peak_utc"], got["samples"], got["over_limit"]) == (30 * GB, _utc(3), 2, False)
    none = memmark.mark({}, slot, {}, [_rec(1, 500)])
    assert none["peak_held_bytes"] is None and none["samples"] == 0 and none["over_limit"] is None
    assert none["reason"].endswith("since the run took its slot at 2026-01-01T00:00:03Z")


def test_a_run_whose_figure_can_t_be_computed_says_why_and_holds_no_zero():
    cases = {
        "the run has no resources.jsonl: its sampler wrote no sample": memmark.mark({"mem_gb": 24}, {}, {}, None),
        "no cgroup sample in resources.jsonl holds memory.current and memory.stat's file and shmem": memmark.mark({"mem_gb": 24}, {}, {}, []),
        "the run took no check slot (its slot record says left)": memmark.mark({"mem_gb": 24}, {"state": "left"}, {}, [_rec(1, 500)]),
        "the workload ran in a GPU lease's scope (gpu-lease), and the samples are of the runner's cgroup":
            memmark.mark({"mem_gb": 24}, {}, {}, [_rec(1, 500)], elsewhere="a GPU lease's scope (gpu-lease)"),
    }
    for why, got in cases.items():
        assert got["reason"] == why
        assert (got["peak_held_bytes"], got["peak_utc"], got["samples"], got["over_limit"], got["over_limit_utc"]) == (None, None, 0, None, None)
        assert (got["ask_bytes"], got["limit_bytes"]) == (24 * GB, 152 * GB), "what it asked and its limit are known all the same"


def test_a_run_whose_slot_py_predates_the_mark_is_marked_from_neither_its_own_mem_gb_nor_its_samples():
    """Such a slot.py wrote neither what its slot reserved nor the scope a check left for: the run's own --mem-gb is not known
    to be its ask, and its samples may be of the runner alone."""
    old = {"state": "held", "slot": "t", "cores": "0-31", "took": _utc(0)[:19] + "Z", "copy": {"of": "machine"}}
    got = memmark.mark({"env": {memmark.RUN_MEM_ENV: "24"}}, old, {}, [_rec(1, 0.1)])
    assert got == {"reason": memmark.OLD_SLOT}
    assert memmark.line(got) == f"peak not recorded: {memmark.OLD_SLOT}"
    now = memmark.mark({"env": {memmark.RUN_MEM_ENV: "24"}}, {**old, "reserves_gib": 0}, {}, [_rec(1, 5)])
    assert (now["peak_held_bytes"], now["ask_from"]) == (5 * GB, "--mem-gb"), "a slot that reserves nothing, by a slot.py that says so"


def test_a_check_in_its_own_scope_has_no_held_figure_and_gets_its_scope_s_peak_with_the_cache(tmp_path):
    """The samples are of the runner's cgroup, which a `scope=1` check left: they hold the runner and its sampler alone."""
    slot = {"state": "held", "slot": "b", "took": _utc(0)[:19] + "Z", "reserves_gib": 300, "scope": "check-b-r1.scope"}
    (tmp_path / "slot.json").write_text(json.dumps(slot))
    _samples(tmp_path, [_rec(1, 0.1), _rec(2, 0.1)])
    spans = [{"span_id": "root", "parent_span_id": None, "name": "check", "attributes": {}},
             {"span_id": "s1", "parent_span_id": "root", "name": "pytest", "attributes": {"step": "pytest", "mem_peak_gb": 180.5}},
             {"span_id": "root", "parent_span_id": None, "name": "check", "end_utc": _utc(9), "attributes": {"mem_peak_gb": 217.3}}]
    (tmp_path / "spans.jsonl").write_text("".join(json.dumps(s) + "\n" for s in spans))
    got = memmark.of_run(tmp_path, {}, {})[memmark.KEY]
    assert got["reason"] == "the workload ran in its own scope check-b-r1.scope, and the samples are of the runner's cgroup"
    assert (got["peak_held_bytes"], got["samples"], got["scope_peak_bytes"]) == (None, 0, 217_300_000_000)
    assert (got["ask_bytes"], got["ask_from"], got["limit_bytes"]) == (300 * GIB, "slot", 300 * GIB + 128 * GB)
    assert got["over_limit"] is False, "a peak with the cache under the limit: what it held was under it too"
    (tmp_path / "slot.json").write_text(json.dumps({**slot, "reserves_gib": 64}))
    over = memmark.of_run(tmp_path, {}, {})[memmark.KEY]
    assert over["scope_peak_bytes"] > over["limit_bytes"] and over["over_limit"] is None, "over it with the cache says nothing of without"
    (tmp_path / "spans.jsonl").unlink()
    assert "scope_peak_bytes" not in memmark.of_run(tmp_path, {}, {})[memmark.KEY], "a lane run in such a slot writes no spans"


def test_a_run_dir_s_mark_is_made_from_its_slot_record_and_its_samples_and_never_raises(tmp_path):
    run = tmp_path / "run"
    process = {"kind": "process", "t_utc": _utc(2), "t_mono": 52.0, "processes": [{"pid": 1, "rss_bytes": 900 * GB}]}
    _samples(run, [_rec(1, 1), process, _rec(3, 40, file=10, shmem=2), _rec(4, 20)], tail='{"kind":"cgroup","t_mo')
    (run / "slot.json").write_text(json.dumps({"state": "held", "slot": "p0", "took": "2026-01-01T00:00:02Z", "reserves_gib": 23}))
    request = {"env": {memmark.RUN_MEM_ENV: "24"}, "mem_limit": {"gb": 30, "by": "proofs"}}
    got = memmark.of_run(run, request, {"effective_memory_max_bytes": 600 * GB}, stated=request["mem_limit"])
    assert got == {"memory": {"ask_bytes": 23 * GIB, "ask_from": "slot", "limit_bytes": 30 * GB, "limit_from": "stated", "limit_by": "proofs",
                              "bound_bytes": 600 * GB, "peak_held_bytes": 32 * GB, "peak_utc": _utc(3), "samples": 2, "over_limit": True,
                              "over_limit_utc": _utc(3), "cgroup": SCOPE, "reason": None}}
    unread = memmark.of_run(run, request, {}, elsewhere="its own unit research-job-r1.service (--job-user)")["memory"]
    assert (unread["ask_bytes"], unread["ask_from"], unread["peak_held_bytes"]) == (24 * GB, "--mem-gb", None), \
        "nothing in the run dir is read: the job could have left anything there"
    (run / "slot.json").unlink()
    broken = memmark.of_run(run, {"mem_gb": "a lot"}, {})["memory"]
    assert broken["reason"].startswith("the mark was not computed: ValueError") and "peak_held_bytes" not in broken


LINES = {
    "under": ({"mem_gb": 24}, [_rec(1, 41.26), _rec(2, 3)],
              "peak held 41.3 GB at 2026-01-01T00:00:01Z (2 samples); asked 24.0 GB (--mem-gb); limit 152.0 GB (ask+128GB)"),
    "over": ({"mem_gb": 24}, [_rec(1, 41), _rec(2, 269)],
             "peak held 269.0 GB at 2026-01-01T00:00:02Z (2 samples); asked 24.0 GB (--mem-gb); limit 152.0 GB (ask+128GB): would have "
             "been stopped at its memory limit of 152.0 GB (asked 24.0 GB) at 2026-01-01T00:00:02Z"),
    "no ask": ({}, [_rec(1, 130)],
               "peak held 130.0 GB at 2026-01-01T00:00:01Z (1 sample); asked none; limit 128.0 GB (no ask: 128.0 GB): would have "
               "been stopped at its memory limit of 128.0 GB (asked none) at 2026-01-01T00:00:01Z"),
    "no samples": ({"mem_gb": 24}, None,
                   "peak held not recorded (the run has no resources.jsonl: its sampler wrote no sample); asked 24.0 GB (--mem-gb); "
                   "limit 152.0 GB (ask+128GB)"),
}


@pytest.mark.parametrize("case", sorted(LINES))
def test_the_status_line_says_the_peak_the_ask_and_the_limit_and_when_the_run_would_have_been_stopped(case):
    request, recs, want = LINES[case]
    assert memmark.line(memmark.mark(request, {}, {}, recs)) == want


def test_the_status_line_says_who_stated_a_limit_what_bounds_it_and_what_a_scope_peaked_at():
    stated = memmark.mark({"mem_gb": 24}, {}, {}, [_rec(1, 5)], stated={"gb": 300, "by": "proofs"})
    assert memmark.line(stated).endswith("asked 24.0 GB (--mem-gb); limit 300.0 GB (stated by proofs)")
    clipped = memmark.mark({}, {}, {"memory_max_bytes": 200 * GB}, [_rec(1, 5)], stated={"gb": 300, "by": "proofs"})
    assert memmark.line(clipped).endswith("limit 200.0 GB (the machine's bound, under what proofs stated)")
    bound = memmark.mark({"mem_gb": 500}, {}, {"memory_max_bytes": 600 * GB}, [_rec(1, 5)])
    assert memmark.line(bound).endswith("asked 500.0 GB (--mem-gb); limit 600.0 GB (the machine's bound)")
    slot = {"state": "held", "took": "2026-01-01T00:00:00Z", "reserves_gib": 64, "scope": "check-b-r1.scope"}
    scoped = memmark.mark({}, slot, {}, [], scope_peak=217_300_000_000)
    assert memmark.line(scoped) == (
        "peak held not recorded (the workload ran in its own scope check-b-r1.scope, and the samples are of the runner's cgroup); "
        "its own scope's memory.peak, page cache included, 217.3 GB; asked 68.7 GB (slot); limit 196.7 GB (ask+128GB): that peak "
        "is over the limit, and whether what it held was is not recorded")
    assert memmark.line(None) == ("peak not recorded: the run's record holds no memory mark (its runner predates the mark, or never "
                                  "wrote the run's end)")
    assert memmark.line({"reason": "the mark was not computed: ValueError: x"}) == "peak not recorded: the mark was not computed: ValueError: x"


def _harness_args(d: Path) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    trun.add_args(ap)
    return ap.parse_args(["--attempt-dir", str(d), "--stage", "t", "--no-sampler", "--no-classify"])


def test_the_harness_writes_the_mark_in_the_write_that_ends_the_run_however_it_ends(tmp_path, monkeypatch):
    """A reader that finds a run ended finds its mark: `research status` fetches an ended run's records once."""
    for name, cmd, rc in (("done", [PY, "-c", "pass"], 0), ("failed", [PY, "-c", "raise SystemExit(3)"], 3),
                          ("unspawned", [str(tmp_path / "no-such-program")], 127)):
        d = tmp_path / name
        _samples(d, [_rec(1, 7), _rec(2, 9, file=4)])
        assert trun.run(_harness_args(d), cmd, None, lambda d=d: memmark.of_run(d, {"mem_gb": 1}, {})) == rc
        status = json.loads((d / "status.json").read_text())
        end = status["transitions"][-1]
        assert end["state"] == status["state"] == ("done" if rc == 0 else "failed") and end["memory"] == status["memory"]
        assert (status["memory"]["peak_held_bytes"], status["memory"]["peak_utc"], status["memory"]["samples"]) == (7 * GB, _utc(1), 2)
        assert all("memory" not in t for t in status["transitions"][:-1])
    d = tmp_path / "raises"
    assert trun.run(_harness_args(d), [PY, "-c", "pass"], None, lambda: 1 / 0) == 0, "a mark that fails never fails the run"
    status = json.loads((d / "status.json").read_text())
    assert status["state"] == "done" and "memory" not in status
    marks = [json.loads(ln) for ln in (d / "events.jsonl").read_text().splitlines()]
    assert [m["error"] for m in marks if m.get("name") == "ended_failed"] == ["ZeroDivisionError: division by zero"]
    monkeypatch.setattr(trun, "uncapped", lambda limits, session: "its login got no scope of its own")
    d = tmp_path / "refused"
    assert trun.run(_harness_args(d), [PY, "-c", "pass"], None, lambda: memmark.of_run(d, {"mem_gb": 1}, {})) == 75
    status = json.loads((d / "status.json").read_text())
    assert status["state"] == "failed" and status["memory"] == status["transitions"][-1]["memory"]
    assert (status["memory"]["ask_bytes"], status["memory"]["peak_held_bytes"]) == (GB, None) and status["memory"]["reason"]


WRITES_SAMPLES = ("import json, os; f = open(os.path.join(os.environ['RESEARCH_RUN_DIR'], 'resources.jsonl'), 'w'); "
                  "[f.write(json.dumps(r, separators=(',', ':')) + '\\n') for r in json.loads(os.environ['SAMPLES'])]")


def test_research_run_records_the_mark_and_research_status_prints_its_line(tmp_path, capsys, monkeypatch):
    """The workload stands in for the sampler: it writes the cgroup records the run's mark is made from."""
    runs = tmp_path / "runs"
    monkeypatch.setenv("SAMPLES", json.dumps([_rec(1, 0.25), _rec(2, 0.75, file=0.125), _rec(3, 0.5)]))
    base = ["run", "--runs-dir", str(runs), "--project", "t", "--no-sampler"]
    assert main([*base, "--mem-limit-gb", "0.5", "--", PY, "-c", "pass"]) == 2
    assert "--mem-limit-gb N states the run's memory limit" in capsys.readouterr().err and not runs.exists(), "nobody stated it"
    assert main([*base, "--mem-limit-gb", "0", "--as", "infra", "--", PY, "-c", "pass"]) == 2
    assert main([*base, "--mem-limit-gb", "0.5", "--as", "infra", "--", PY, "-c", WRITES_SAMPLES]) == 0
    [run] = runs.iterdir()
    mark = json.loads((run / "status.json").read_text())["memory"]
    assert (mark["ask_bytes"], mark["ask_from"], mark["limit_bytes"], mark["limit_from"], mark["limit_by"]) == (
        None, "none", GB // 2, "stated", "infra")
    assert (mark["peak_held_bytes"], mark["peak_utc"], mark["samples"], mark["over_limit"], mark["over_limit_utc"]) == (
        625_000_000, _utc(2), 3, True, _utc(2))
    job = json.loads((run / "job.json").read_text())
    assert mark["bound_bytes"] == memmark.bound(job["environment"]["cgroup"]), "the bound is what job.json recorded at the start"
    capsys.readouterr()
    assert main(["status", run.name, "--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert f"mem  {memmark.line(mark)}\n" in out and "would have been stopped at its memory limit of 0.5 GB (asked none) at" in out
    assert main(["status", run.name, "--runs-dir", str(runs), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["memory"] == mark
    assert main(["status", "--runs-dir", str(runs)]) == 0
    assert "mem  " not in capsys.readouterr().out, "the line is `research status RUN`'s, not the listing's"
    status = json.loads((run / "status.json").read_text())
    del status["memory"]
    (run / "status.json").write_text(json.dumps(status))
    assert main(["status", run.name, "--runs-dir", str(runs)]) == 0
    assert "mem  peak not recorded: the run's record holds no memory mark" in capsys.readouterr().out, "an older runner's record"


def test_the_runner_takes_the_ask_and_a_stated_limit_from_its_launch_request_and_reads_no_run_dir_of_a_job_user(tmp_path):
    run = tmp_path / "runs" / "r1"
    _samples(run, [_rec(1, 40), _rec(2, 70)])
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"mem_gb": 24, "mem_limit": {"gb": 50, "by": "proofs"}, "env": {}}))
    job = {"environment": {"cgroup": {"memory_max_bytes": 650 * GB, "effective_memory_max_bytes": 600 * GB}}}
    ns = argparse.Namespace(claimed=True, remote_json=json.dumps({"request": str(request)}), job_user=None, mem_limit_gb=None, lane=None)
    got = cli._memory_mark(ns, "r1", run, job)()["memory"]
    assert (got["ask_bytes"], got["ask_from"], got["limit_bytes"], got["limit_from"], got["limit_by"], got["bound_bytes"]) == (
        24 * GB, "--mem-gb", 50 * GB, "stated", "proofs", 600 * GB)
    assert (got["peak_held_bytes"], got["over_limit"], got["over_limit_utc"]) == (70 * GB, True, _utc(2))
    request.write_text(json.dumps({"mem_gb": 24, "gpus": 1}))
    leased = cli._memory_mark(ns, "r1", run, job)()["memory"]
    assert leased["peak_held_bytes"] is None and leased["reason"].startswith("the workload ran in a GPU lease's scope (gpu-lease)")
    request.write_text(json.dumps({"queue": {"job": {"memory_gib": 96}}, remote.QUEUE_WRAP: ["systemd-run"], "gpus": 1,
                                   remote.QUEUE_WORKLOAD_WRAP: []}))
    queued = cli._memory_mark(ns, "r1", run, {"environment": {"cgroup": {"memory_max_bytes": 96 * GIB}}})()["memory"]
    assert (queued["ask_from"], queued["limit_bytes"], queued["limit_from"], queued["peak_held_bytes"]) == ("queue", 96 * GIB, "bound", 70 * GB), \
        "a queue job's cpu phase runs in the scope its runner is in, which already holds it to what it asked"
    (run / "slot.json").write_text(json.dumps({"state": "held", "slot": "p0", "took": "2026-01-01T00:00:00Z", "reserves_gib": 193}))
    local = argparse.Namespace(claimed=False, remote_json=None, job_user="vyjob", mem_limit_gb=8.0, lane="infra")
    job_user = cli._memory_mark(local, "r1", run, {})()["memory"]
    assert job_user["reason"] == ("the workload ran in its own unit research-job-r1.service (--job-user), and the samples are of the "
                                  "runner's cgroup")
    assert (job_user["ask_from"], job_user["limit_bytes"], job_user["limit_by"], job_user["samples"]) == ("none", 8 * GB, "infra", 0)


def test_a_stated_limit_goes_in_the_launch_request_and_a_tree_that_predates_the_mark_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "resolve_machine", lambda name, project: {"root": "/workspace/research", "python": "python3"})
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "user": "research", "pod_name": "n1",
                                                         "cost_per_hr": 0, "rate_observed_utc": "t"})
    monkeypatch.setattr(remote, "wake", lambda m, ssh: None)
    monkeypatch.setattr(remote, "ship_tool", lambda *a: "/r/tool/x")
    staged = []

    def stage(ssh, root, run_id, request, sends, private=None):
        staged.append(request)
        raise remote.Refused("staged")

    monkeypatch.setattr(remote, "stage_request", stage)
    launch = dict(machine_name="n1", project="verity", command=["true"], harness_args=["--stage", "t"], sends=[], cwd=None, env={},
                  exclusive=False)
    for run_id, stated in (("m1", {"gb": 300.0, "by": "proofs"}), ("m2", None)):
        with pytest.raises(remote.Refused, match="staged"):
            remote.launch(run_id=run_id, local_dir=tmp_path / "runs" / run_id, source=None, mem_limit=stated, **launch)
    assert staged[0]["mem_limit"] == {"gb": 300.0, "by": "proofs"} and "mem_limit" not in staged[1]
    ns = argparse.Namespace(mem_limit_gb=300.0, lane="proofs")
    assert cli._stated_limit(ns) == staged[0]["mem_limit"] and cli._stated_limit(argparse.Namespace(mem_limit_gb=None, lane="proofs")) is None
    old = tmp_path / "tree" / "tools" / "research" / "src" / "research"
    old.mkdir(parents=True)
    (old / "__init__.py").write_text("")
    (old / "remote.py").write_text("def tool_identity(): pass\n")
    monkeypatch.setattr(remote, "ship_tool", lambda *a: pytest.fail("nothing ships"))
    with pytest.raises(remote.Refused, match="--mem-limit-gb: the research package in .* predates the memory mark"):
        remote.launch(run_id="m3", local_dir=tmp_path / "runs" / "m3", source=tmp_path / "tree", mem_limit={"gb": 300.0, "by": "proofs"},
                      **launch)
