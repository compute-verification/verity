"""Stage 1's dispatcher (`research.jobs.dispatch`) against the Job queue's reference model, on a fake clock, with a fake runner
standing in for ssh and `research run --on`; and the launch child against a local git repository."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.jobs import dispatch as D
from research.jobs import model
from research.jobs.cli import main as jobs_main
from research.jobs.client import ModelClient

GOOD = {"avx512": True, "mem_gb": 240, "cpus": 64, "disk_gb": 70, "busy": 0, "tools": ["uv", "lake", "cargo"], "lease_s": 6 * 3600,
        "setup_done": True, "setup_at": "2026-09-29T00:00:00Z"}
C1, C2, BASE = "1" * 40, "2" * 40, "b" * 40
RUN = "r20260929-000005-aaaa"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeRunner:
    def __init__(self):
        self.caps: dict[str, dict] = {}
        self.launches: dict[str, tuple | None] = {}
        self.runs: dict[str, D.Observation] = {}
        self.started, self.stopped, self.killed, self.probes = [], [], [], []
        self.local: str | None = None

    def local_problem(self):
        return self.local

    def probe(self, pod):
        self.probes.append(pod)
        return dict(self.caps.get(pod, GOOD))

    def start(self, job, attempt, pod):
        self.started.append((job["id"], attempt, pod))
        self.launches[f"L{job['id']}-{attempt}"] = None
        return f"L{job['id']}-{attempt}"

    def poll(self, launch):
        return self.launches[launch]

    def kill(self, launch):
        self.killed.append(launch)

    def observe(self, job, run_id):
        return self.runs[run_id]

    def stop(self, run_id):
        self.stopped.append(run_id)


class World:
    def __init__(self, pods=("t1",), state_path=None):
        self.svc = model.JobService()
        self.svc.mint("tok-rc", "coordinator", ["jobs:enqueue", "events:read"])
        self.svc.mint("tok-d", "dispatch-rc", ["jobs:work"])
        self.svc.register("dispatch:rc", "dispatcher", "dispatch-rc", list(D.KINDS))
        self.clock = Clock()
        self.coord = ModelClient(self.svc, "tok-rc", self.clock)
        self.worker = ModelClient(self.svc, "tok-d", self.clock)
        self.runner = FakeRunner()
        self.lines: list[str] = []
        self.pods, self.state_path = list(pods), state_path
        self.d = self.dispatcher()

    def dispatcher(self) -> D.Dispatcher:
        return D.Dispatcher(self.worker, self.runner, self.pods, clock=self.clock, out=self.lines.append, state_path=self.state_path)

    def add(self, commit=C1, **kw) -> int:
        status, body = self.coord.call("POST", "/api/jobs", {"kind": "merge-check", "payload": {"commit": commit, "base": BASE}, **kw})
        assert status == 201, body
        return body["id"]

    def at(self, t: float) -> None:
        self.clock.t = t
        self.d.step()

    def job(self, i: int) -> model.Job:
        return self.svc.jobs[i]


def test_a_merge_check_runs_on_a_free_pod_and_completes_with_its_run_id():
    w = World()
    j = w.add()
    w.at(0)
    assert w.runner.started == [(j, 1, "t1")]
    assert (w.job(j).state, w.job(j).lease_worker, w.job(j).lease_slot) == ("running", "dispatch:rc", "t1")
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("running", "running")
    w.at(65)
    assert w.job(j).progress == {"pod": "t1", "run_id": RUN, "step": "running"}
    w.runner.runs[RUN] = D.Observation("passed", "check passed: done rc=0")
    w.at(130)
    assert w.job(j).state == "done"
    assert w.job(j).result == {"outcome": "passed", "run_id": RUN, "summary": "check passed: done rc=0"}
    assert w.d.slots[0].job is None
    assert [e["kind"] for e in w.svc.events] == ["job.added", "job.claimed", "job.finished"]


@pytest.mark.parametrize("tail", [
    "research run --on t1: refused: the machine failed the preflight tools/verity/check/preflight.py (exit 1): uv is not on the PATH",
    "research: run r20260929-000001-abcd refused on t1 (PreflightFailed): cpu: no avx512f  (recorded in /x)",
])
def test_a_pod_that_fails_checks_preflight_is_benched_and_another_pod_takes_the_job(tail):
    """Only preflight.py (#440) says whether a pod is prepared; the dispatcher's probe measures, and it believes the preflight."""
    w = World(pods=("t1", "t2"))
    w.runner.caps["t1"] = {**GOOD, "tools": [], "disk_gb": 3}           # measured, never judged
    j = w.add()
    w.at(0)
    assert w.runner.started == [(j, 1, "t1")]
    w.runner.launches["L1-1"] = (3, None, tail)
    w.at(5)
    assert (w.job(j).state, w.job(j).attempts) == ("queued", 1) and "t1 failed check's preflight" in w.job(j).result["summary"]
    assert any("t1: failed check's preflight; benched for 1800 s" in line for line in w.lines)
    w.at(5 + D.CLAIM_S)
    assert w.runner.started[-1] == (j, 2, "t2")
    k = w.add(C2)
    w.at(5 + D.BENCH_S - 1)
    assert [x for x in w.runner.started if x[0] == k] == []
    w.at(5 + D.BENCH_S + D.CLAIM_S)
    assert w.runner.started[-1] == (k, 1, "t1")


def test_the_probe_judges_only_whether_a_pod_is_free():
    assert D.unavailable({**GOOD, "busy": 1}) == "1 live research run(s) on it"
    assert D.unavailable({**GOOD, "setup_done": False}).startswith("pod_setup.sh hasn't reported done")
    assert D.unavailable({**GOOD, "tools": [], "disk_gb": 1, "avx512": False}) is None


def test_no_job_starts_on_a_pod_until_its_setup_reports_done():
    """A launch chain that started a check while pod_setup.sh still ran cost train TVC an hour: the dispatcher can't do that."""
    w = World()
    j = w.add()
    w.runner.caps["t1"] = {**GOOD, "setup_done": False, "setup_at": None}
    w.at(0)
    assert w.runner.started == [] and w.job(j).state == "queued"
    assert any("t1: skipped: pod_setup.sh hasn't reported done" in line for line in w.lines)
    w.runner.caps["t1"] = GOOD
    w.at(D.PREFLIGHT_S + D.CLAIM_S)
    assert w.runner.started == [(j, 1, "t1")]


def test_a_pod_that_cant_be_measured_asks_for_nothing_and_the_dispatcher_goes_on():
    w = World()
    w.add()
    w.runner.probe = lambda pod: (_ for _ in ()).throw(RuntimeError("probe exited 255: ssh: connect refused"))
    w.at(0)
    assert w.runner.started == []
    assert any("t1: skipped: can't be measured: RuntimeError: probe exited 255" in line for line in w.lines)


def test_a_job_goes_only_to_a_pod_that_meets_its_requirements():
    """An idle pass stops at the first empty claim, and the pods take turns asking first, so each pod's capabilities are offered."""
    w = World(pods=("t1", "t2"))
    w.runner.caps["t1"] = {**GOOD, "avx512": False}
    j = w.add(requirements={"avx512": True})
    w.at(0)
    assert w.runner.started == []
    w.at(D.CLAIM_S)
    assert w.runner.started == [(j, 1, "t2")]


def test_every_free_pod_takes_a_job_in_one_pass():
    w = World(pods=("t1", "t2", "t3"))
    a, b = w.add(C1), w.add(C2)
    w.at(0)
    assert w.runner.started == [(a, 1, "t1"), (b, 1, "t2")]


def test_a_failed_check_is_final_and_an_error_is_retried():
    w = World()
    a = w.add(C1)
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("failed", "check failed: failed rc=1")
    w.at(65)
    assert w.job(a).state == "failed"
    b = w.add(C2)
    w.at(65 + D.CLAIM_S)
    assert w.runner.started[-1] == (b, 1, "t1")
    w.runner.launches["L2-1"] = (3, None, "research run --on t1: refused: pod is EXITED")
    w.at(65 + D.CLAIM_S + 5)
    assert (w.job(b).state, w.job(b).attempts) == ("queued", 1)
    assert "refused: pod is EXITED" in w.job(b).result["summary"]
    w.at(65 + 2 * D.CLAIM_S)
    assert w.runner.started[-1] == (b, 2, "t1")


def test_cancel_stops_the_run_at_the_next_renewal():
    w = World()
    j = w.add()
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("running", "running")
    assert w.coord.call("POST", f"/api/jobs/{j}/cancel", {})[1] == {"state": "running", "cancel_requested": True}
    w.at(60)
    assert w.runner.stopped == [RUN]
    assert w.job(j).state == "cancelled" and w.d.slots[0].job is None


def test_a_lost_lease_stops_the_run_and_drops_the_job():
    w = World()
    j = w.add(lease_s=60)
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("running", "running")
    w.worker.down = True
    w.at(60)
    assert any("renew unreachable" in line for line in w.lines)
    assert w.svc.tick(61) == [j]
    w.worker.down = False
    w.at(62)
    assert w.runner.stopped == [RUN]
    assert any("the lease is gone (409" in line for line in w.lines)
    assert w.runner.started == [(j, 1, "t1"), (j, 2, "t1")]


def test_a_restarted_dispatcher_keeps_tending_its_runs(tmp_path):
    w = World(state_path=tmp_path / "dispatch-rc.json")
    j = w.add()
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    assert stat.S_IMODE((tmp_path / "dispatch-rc.json").stat().st_mode) == 0o600
    w.d = w.dispatcher()
    assert (w.d.slots[0].job["id"], w.d.slots[0].run_id) == (j, RUN)
    w.runner.runs[RUN] = D.Observation("passed", "check passed")
    w.at(70)
    assert w.job(j).state == "done" and w.runner.started == [(j, 1, "t1")]
    assert json.loads((tmp_path / "dispatch-rc.json").read_text())["slots"] == []


def test_a_run_past_its_timeout_is_stopped_and_retried():
    w = World()
    j = w.add(payload={"commit": C1, "base": BASE, "timeout_s": 100})
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("running", "running")
    w.at(65)
    assert w.job(j).state == "running"
    w.at(130)
    assert w.runner.stopped == [RUN]
    first = w.svc.attempts[0]
    assert first.outcome == "error" and "timeout_s (100 s)" in first.detail
    assert w.runner.started == [(j, 1, "t1"), (j, 2, "t1")]


def test_a_launch_that_hangs_is_killed():
    w = World()
    j = w.add()
    w.at(0)
    w.at(D.LAUNCH_TIMEOUT_S + 1)
    assert w.runner.killed == ["L1-1"]
    assert w.job(j).result["summary"] == f"the launch took over {D.LAUNCH_TIMEOUT_S} s"


def test_a_complete_that_cant_be_sent_is_sent_later():
    w = World()
    j = w.add()
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("passed", "check passed")
    w.worker.down = True
    w.at(65)
    assert w.job(j).state == "running" and w.d.slots[0].outcome == ["passed", "check passed"]
    w.worker.down = False
    w.at(70)
    assert w.job(j).state == "done"


def test_a_refused_claim_stops_the_dispatcher():
    w = World()
    w.add()
    w.svc.revoke("dispatch-rc")
    with pytest.raises(D.DispatchError, match="claim refused: 401"):
        w.at(0)


@pytest.mark.parametrize("kind,summary,want", [
    ("merge-check", {"state": "done", "rc": 0, "validation": "passed", "result_valid": True}, "passed"),
    ("merge-check", {"state": "failed", "rc": 1, "validation": "failed", "result_valid": True}, "failed"),
    ("merge-check", {"state": "failed", "rc": 1, "validation": None, "result_valid": False, "result": "absent"}, "error"),
    ("merge-check", {"state": "failed", "rc": None, "signal": 9, "class": "oom"}, "error"),
    ("merge-check", {"state": "done", "rc": 0, "validation": None, "result": "absent"}, "error"),
    ("merge-check", {"state": "refused"}, "error"),
    ("merge-check", {"state": "running"}, "running"),
    ("merge-check", {"state": "submitted"}, "running"),
    ("lean-regen", {"state": "done", "rc": 0}, "passed"),
    ("lean-regen", {"state": "failed", "rc": 1}, "failed"),
    ("lean-regen", {"state": "failed", "rc": None, "signal": 15}, "error"),
    ("lean-regen", {"state": "cancelled", "rc": None}, "error"),
])
def test_only_a_verdict_is_a_failure(kind, summary, want):
    assert D.classify(kind, summary).state == want


def test_the_commands_each_kind_runs(tmp_path):
    job = {"kind": "merge-check", "payload": {"commit": C1, "base": BASE}}
    assert D.command(job, "t1", "py", None) == ["py", "tools/verity/check/check.py", "--record", "--on", "t1", "--base", BASE]
    assert D.command(job, "t1", "py", tmp_path / "v.tar.gz")[-2:] == ["--verdicts-in", str(tmp_path / "v.tar.gz")]
    regen = D.command({"kind": "lean-regen", "payload": {"packages": ["verity/ml/flock/verifier/soundness"]}}, "t1", "py", None)
    assert regen[:15] == ["py", "-m", "research", "run", "--on", "t1", "--project", "verity", "--source", ".", "--cwd", "source",
                          "--preflight", "check", "--"]
    assert "--preflight" not in D.command({"kind": "lean-regen", "payload": {"packages": ["p"]}}, "t1", "py", None, preflight=False)
    assert D.command({"kind": "audit-main", "payload": {"commit": C1}}, "t1", "py", None) == [
        "py", "tools/verity/check/check.py", "--record", "--on", "t1", "--base", D.EMPTY_TREE, "--no-cache"]
    assert "check.py --build --update verity/ml/flock/verifier/soundness" in regen[-1]
    assert '$RESEARCH_RUN_DIR/records/$p' in regen[-1] and regen[-1].endswith("exit $rc")


@pytest.mark.parametrize("payload,why", [
    ({"commit": "abc", "base": BASE}, "full commit sha"),
    ({"commit": C1}, "payload.base"),
    ({"commit": C1, "base": BASE, "repo": "someone/else"}, "this dispatcher checks"),
])
def test_a_payload_the_dispatcher_cant_run_is_an_error(payload, why):
    with pytest.raises(ValueError, match=why):
        D.validate({"kind": "merge-check", "payload": payload}, "danielreuter/verity")
    with pytest.raises(ValueError, match="package directories"):
        D.validate({"kind": "lean-regen", "payload": {"commit": C1, "packages": ["../etc"]}}, "danielreuter/verity")


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.fsmonitor=false", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True, env=env).stdout.strip()


def test_the_launch_child_checks_the_commit_in_its_own_worktree_without_the_token(tmp_path):
    origin, seen = tmp_path / "origin", tmp_path / "seen.json"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "README.md").write_text("base\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-qm", "base")
    base = _git(origin, "rev-parse", "HEAD")
    (origin / "tools" / "verity" / "check").mkdir(parents=True)
    (origin / "tools" / "verity" / "check" / "check.py").write_text(
        "import json, os, subprocess, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'cwd': os.getcwd(), 'token': os.environ.get('JOBS_TOKEN'),\n"
        "           'head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()},\n"
        f"          open({str(seen)!r}, 'w'))\n"
        "print('research: run r20260929-000000-abcd launched on t1 (podid) pid=1 -> /workspace/research/runs/r20260929-000000-abcd',\n"
        "      file=sys.stderr)\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-qm", "head")
    head = _git(origin, "rev-parse", "HEAD")
    runner = D.ResearchRunner(workdir=tmp_path / "jobs", url=str(origin), environ={**os.environ, "JOBS_TOKEN": "not-for-pr-code"})
    job = {"id": 7, "kind": "merge-check", "payload": {"commit": head, "base": base}}
    launch = runner.start(job, 1, "t1")
    runner.children[launch].wait()
    assert runner.poll(launch) == (0, "r20260929-000000-abcd", runner.poll(launch)[2])
    got = json.loads(seen.read_text())
    assert got["argv"] == ["--record", "--on", "t1", "--base", base]
    assert got["token"] is None and got["head"] == head
    assert got["cwd"] == str(tmp_path / "jobs" / "worktrees" / "job-7-1")
    assert not (tmp_path / "jobs" / "worktrees" / "job-7-1").exists()
    assert "launch done: 0 r20260929-000000-abcd" in (tmp_path / "jobs" / "logs" / "job-7-1.log").read_text()
    assert D.launcher_pid(Path(launch)) is None


def test_a_launch_of_a_commit_origin_lacks_is_an_error(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "README.md").write_text("x\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-qm", "x")
    runner = D.ResearchRunner(workdir=tmp_path / "jobs", url=str(origin), environ=dict(os.environ))
    launch = runner.start({"id": 8, "kind": "merge-check", "payload": {"commit": C1, "base": _git(origin, "rev-parse", "HEAD")}}, 1, "t1")
    runner.children[launch].wait()
    rc, run_id, tail = runner.poll(launch)
    assert (rc, run_id) == (-1, None) and f"commit {C1} is not on" in tail


def test_research_jobs_adds_lists_shows_cancels_and_reads_events(capsys):
    w = World()
    args = ["add", "merge-check", "--commit", C1, "--base", BASE]
    assert jobs_main(args, client=w.coord) == 0
    assert jobs_main(args, client=w.coord) == 0
    assert capsys.readouterr().out.splitlines() == ["job 1 added (queued)", "job 1 already there (queued)", f"    added for commit {C1}"]
    assert (w.job(1).dedupe_key, w.job(1).payload["base"], w.job(1).payload["timeout_s"]) == (f"merge-check:{C1}", BASE, D.DEFAULT_TIMEOUT_S)
    assert w.job(1).requirements == {"min_lease_s": D.DEFAULT_TIMEOUT_S + 900}
    assert jobs_main(["list"], client=w.coord) == 0
    assert "merge-check  queued" in capsys.readouterr().out
    assert jobs_main(["cancel", "1"], client=w.coord) == 0
    assert capsys.readouterr().out.strip() == "job 1: cancelled"
    assert jobs_main(["show", "1"], client=w.coord) == 0
    assert "cancelled" in capsys.readouterr().out
    assert jobs_main(["events", "--after", "1"], client=w.coord) == 0
    out = capsys.readouterr().out.splitlines()
    assert "job.cancelled" in out[0] and out[-1] == "next 2"
    assert jobs_main(["show", "9"], client=w.coord) == 1
    assert "404 no such job" in capsys.readouterr().err
    w.coord.down = True
    assert jobs_main(["list"], client=w.coord) == 4


def test_the_child_environment_never_holds_the_token():
    assert "JOBS_TOKEN" not in D.ResearchRunner(environ={"JOBS_TOKEN": "x", "PATH": "/bin"}).env()


def test_research_worker_is_wired(capsys):
    from research.cli import main

    assert main(["worker", "--pods", "t1"]) == 2
    assert "only --dispatch exists" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["jobs", "--help"])
    assert e.value.code == 0 and "research jobs" in capsys.readouterr().out


def test_no_claim_while_the_host_is_under_the_disk_floor():
    w = World()
    j = w.add()
    w.runner.local = "0.90 GB free on the filesystem of /home/rc/.research/runs, under the 1.5 GB floor"
    w.at(0)
    w.at(D.CLAIM_S)
    assert w.runner.started == [] and w.runner.probes == []
    assert [line.split(" ", 1)[1] for line in w.lines] == ["claiming stopped: " + w.runner.local]
    w.runner.local = None
    w.at(2 * D.CLAIM_S)
    assert w.runner.started == [(j, 1, "t1")]
    assert "claiming again: the host has room" in w.lines[1]


def test_a_running_job_is_tended_under_the_disk_floor():
    w = World()
    j = w.add()
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.runner.local = "under the floor"
    w.runner.runs[RUN] = D.Observation("passed", "check passed")
    w.at(5)
    w.at(65)
    assert w.job(j).state == "done"


def test_the_disk_floor_is_research_runs_own(tmp_path):
    r = D.ResearchRunner(workdir=tmp_path / "jobs", environ={"RESEARCH_RUNS": str(tmp_path / "runs"), "RESEARCH_DISK_FLOOR_GB": "1e9"})
    assert "under the 1e+09 GB floor" in r.local_problem()
    r.environ["RESEARCH_DISK_FLOOR_GB"] = "0"
    assert r.local_problem() is None


def test_a_job_is_claimed_only_for_a_pod_whose_lease_covers_it(capsys):
    w = World(pods=("t1", "t2"))
    w.runner.caps["t1"] = {**GOOD, "lease_s": D.lease_needed(3600) - 1}
    assert jobs_main(["add", "merge-check", "--commit", C1, "--base", BASE, "--timeout-s", "3600"], client=w.coord) == 0
    assert w.job(1).requirements["min_lease_s"] == 3600 + 900
    w.at(0)
    assert w.runner.started == []
    w.at(D.CLAIM_S)
    assert w.runner.started == [(1, 1, "t2")]
    assert any("t1: free: lease 1.2 h, 70 GB free, 240 GB, 64 cpus, avx512" in line for line in w.lines)


def test_a_job_without_min_lease_s_on_a_short_lease_is_named():
    w = World()
    w.runner.caps["t1"] = {**GOOD, "lease_s": 600}
    j = w.add()
    w.at(0)
    assert w.runner.started == [(j, 1, "t1")]
    assert any(f"job {j} was added without min_lease_s" in line and "research pods extend t1" in line for line in w.lines)


def test_a_pod_without_a_lease_says_so():
    w = World()
    w.runner.caps["t1"] = {k: v for k, v in GOOD.items() if k != "lease_s"}
    assert jobs_main(["add", "merge-check", "--commit", C1, "--base", BASE], client=w.coord) == 0
    w.at(0)
    assert w.runner.started == []
    assert any("NO lease (a job with min_lease_s won't come)" in line for line in w.lines)


def test_the_probe_reads_the_lease_the_tools_and_live_runs(tmp_path):
    root = tmp_path / "research"
    for name, status in (("r1", {"state": "done", "pid": os.getpid()}), ("r2", {"state": "running", "pid": os.getpid()}),
                         ("r3", {"state": "running", "pid": 2 ** 22 + 1})):
        (root / "runs" / name).mkdir(parents=True)
        (root / "runs" / name / "status.json").write_text(json.dumps(status))
    lease = tmp_path / "lease"
    lease.write_text("4102444800 2100-01-01T00:00:00Z pods-create\n")
    (tmp_path / ".research").mkdir()
    (tmp_path / ".research" / "pod-setup.json").write_text(json.dumps({"done_utc": "2026-09-29T22:00:00Z", "uv": "0.12.20"}))
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}

    def probe(r, e=env):
        return json.loads(subprocess.run([sys.executable, "-c", D.PROBE, str(r), str(lease), D.SETUP_MARKER], capture_output=True,
                                         text=True, check=True, env=e).stdout)

    caps = probe(root)
    assert caps["busy"] == 1 and caps["lease_s"] > 2 * 10 ** 9
    assert (caps["setup_done"], caps["setup_at"]) == (True, "2026-09-29T22:00:00Z")
    assert set(caps) == {"avx512", "mem_gb", "cpus", "disk_gb", "busy", "tools", "lease_s", "setup_done", "setup_at"}
    lease.unlink()
    (tmp_path / ".research" / "pod-setup.json").unlink()
    caps = probe(root / "missing")
    assert "lease_s" not in caps and caps["busy"] == 0 and caps["setup_done"] is False
    caps = probe(root, {**env, "RESEARCH_LEASE_EXPIRES": "4102444800"})
    assert caps["lease_s"] > 2 * 10 ** 9


def _lean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    files = {
        "a/lean-audit.json": "{}", "a/lakefile.toml": "", "a/A.lean": "a",
        "a/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": "../b"}, {"type": "git", "name": "mathlib"}]}),
        "a/nested/lakefile.toml": "", "a/nested/N.lean": "n",
        "b/lakefile.toml": "", "b/B.lean": "b", "c/lean-audit.json": "{}", "c/C.lean": "c",
        "tools/verity/lean/check.py": "audit", "tools/verity/check/check.py": "check", "README.md": "readme",
    }
    for f, text in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(text)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "one")
    return repo


def test_a_lean_regen_is_keyed_on_the_files_the_audit_reads(tmp_path):
    from research.jobs.cli import lean_inputs, lean_key

    repo = _lean_repo(tmp_path)
    one = _git(repo, "rev-parse", "HEAD")
    assert sorted(lean_inputs(one, ["a"], repo)) == ["a/A.lean", "a/lake-manifest.json", "a/lakefile.toml", "a/lean-audit.json",
                                                    "b/B.lean", "b/lakefile.toml", "tools/verity/lean/check.py"]
    keys = {}
    for name, path, text in (("readme", "README.md", "x"), ("nested", "a/nested/N.lean", "x"), ("check", "tools/verity/check/check.py", "x"),
                             ("dep", "b/B.lean", "x"), ("tool", "tools/verity/lean/check.py", "x"), ("other", "c/C.lean", "x")):
        (repo / path).write_text(text)
        _git(repo, "commit", "-qam", name)
        keys[name] = lean_key(_git(repo, "rev-parse", "HEAD"), ["a"], repo)
    base = lean_key(one, ["a"], repo)
    assert keys["readme"] == keys["nested"] == keys["check"] == base
    assert len({base, keys["dep"], keys["tool"]}) == 3 and keys["other"] == keys["tool"]
    assert lean_key(one, ["a/"], repo) == base != lean_key(one, ["a", "c"], repo)
    with pytest.raises(SystemExit, match="b has no lean-audit.json"):
        lean_key(one, ["b"], repo)


def test_a_lean_regen_is_keyed_on_its_provers_and_everything_they_require(tmp_path):
    """A package whose guarantees are proved in a nested package (`proved_in`) is audited from that package's build, and so
    from every package it requires by path, at any depth."""
    from research.jobs.cli import lean_inputs, lean_key

    repo = _lean_repo(tmp_path)
    files = {"s/lean-audit.json": json.dumps({"proved_in": ["Proofs"]}), "s/lakefile.toml": "", "s/S.lean": "s",
             "s/Proofs/lakefile.toml": "", "s/Proofs/P.lean": "p",
             "s/Proofs/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": ".."}, {"type": "path", "dir": "../../a"}]})}
    for f, text in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "s")
    one = _git(repo, "rev-parse", "HEAD")
    assert {"s/S.lean", "s/Proofs/P.lean", "a/A.lean", "b/B.lean"} <= set(lean_inputs(one, ["s"], repo))
    assert not {"a/nested/N.lean", "c/C.lean"} & set(lean_inputs(one, ["s"], repo))
    base, keys = lean_key(one, ["s"], repo), set()
    for path in ("s/Proofs/P.lean", "a/A.lean", "b/B.lean"):
        (repo / path).write_text("x")
        _git(repo, "commit", "-qam", path)
        keys.add(lean_key(_git(repo, "rev-parse", "HEAD"), ["s"], repo))
    assert len(keys | {base}) == 4


def test_trains_with_the_same_lean_inputs_share_one_regeneration(tmp_path, monkeypatch, capsys):
    repo = _lean_repo(tmp_path)
    one = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("another train")
    _git(repo, "commit", "-qam", "two")
    two = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    w = World()
    assert jobs_main(["add", "lean-regen", "--commit", one, "--package", "a"], client=w.coord) == 0
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("passed", "done rc=0")
    w.at(65)
    capsys.readouterr()
    assert jobs_main(["add", "lean-regen", "--commit", two, "--package", "a"], client=w.coord) == 0
    assert capsys.readouterr().out.splitlines() == ["job 1 already there (done)", f"    added for commit {one}; run {RUN}"]
    assert w.job(1).dedupe_key == f"lean-regen:{w.job(1).payload['inputs']}" and len(w.svc.jobs) == 1


def _flock_repo(tmp_path: Path, rustflags: str, pattern: str | None = None) -> Path:
    repo = tmp_path / "flock"
    files = {"verity/ml/flock/verifier/upstream.json": json.dumps({"upstream": {"sha256": "0" * 64, "rustflags": rustflags}}),
             "verity/ml/flock/src.rs": "fn main() {}", "verity/ml/flock/README.md": "flock", "docs.md": "docs",
             "tools/verity/check/tool.py": 'CHECK = Tool(name="check", merge_requires={"lean-agreement": ["verity/ml/flock/", "!README.md"]})\n'}
    if pattern is not None:
        files["tools/verity/check/preflight.py"] = f"import re\nAVX512 = re.compile({pattern!r})\n"
    for f, text in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(text)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    return repo


def _change(repo: Path, path: str) -> tuple[str, str]:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / path).write_text("changed")
    _git(repo, "commit", "-qam", path)
    return _git(repo, "rev-parse", "HEAD"), base


@pytest.mark.parametrize("rustflags,pattern,path,want", [
    ("-C target-cpu=x86-64-v3", r"target-cpu=(native|x86-64-v4)\b|avx512", "verity/ml/flock/src.rs", False),   # today's pin
    ("-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512", "verity/ml/flock/src.rs", True),
    ("-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512", "docs.md", False),                # no agreement runs
    ("-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512", "verity/ml/flock/README.md", False),
    ("-C target-cpu=x86-64-v4", r"never", "verity/ml/flock/src.rs", False),                                     # the tree's own rule
    ("-C target-cpu=native", None, "verity/ml/flock/src.rs", True),                                            # a tree before #440
])
def test_a_check_asks_for_avx512_only_when_its_tree_says_so(tmp_path, rustflags, pattern, path, want):
    from research.jobs.cli import needs_avx512

    repo = _flock_repo(tmp_path, rustflags, pattern)
    commit, base = _change(repo, path)
    assert needs_avx512(commit, base, repo) is want


def test_a_flock_file_whose_name_git_quotes_still_asks_for_avx512(tmp_path):
    """`git diff --name-only` quotes a name with bytes outside ASCII (`"verity/ml/flock/\\303\\251.rs"`), which the agreement's
    rule never matched, so a change to such a file asked for no AVX-512 machine."""
    from research.jobs.cli import needs_avx512

    repo = _flock_repo(tmp_path, "-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "verity" / "ml" / "flock" / "é.rs").write_text("fn f() {}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "é")
    assert needs_avx512(_git(repo, "rev-parse", "HEAD"), base, repo) is True


def test_a_check_asks_for_avx512_for_a_file_renamed_out_of_the_agreements_paths(tmp_path):
    """git reports a rename by its destination alone unless asked not to, which would leave out the agreement the file's old
    path runs."""
    from research.jobs.cli import needs_avx512

    repo = _flock_repo(tmp_path, "-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "mv", "verity/ml/flock/src.rs", "src.rs")
    _git(repo, "commit", "-qm", "move")
    assert needs_avx512(_git(repo, "rev-parse", "HEAD"), base, repo) is True


def test_audit_main_is_a_cold_check_of_mains_tip_with_every_path_changed(tmp_path, monkeypatch, capsys):
    from research.jobs.cli import needs_avx512

    repo = _flock_repo(tmp_path, "-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512")
    tip = _git(repo, "rev-parse", "HEAD")
    assert needs_avx512(tip, D.EMPTY_TREE, repo) is True
    monkeypatch.chdir(repo)
    w = World()
    assert jobs_main(["add", "audit-main", "--commit", tip], client=w.coord) == 0
    j = w.job(1)
    assert (j.kind, j.priority, j.dedupe_key, j.payload["timeout_s"]) == ("audit-main", 5, f"audit-main:{tip}", D.AUDIT_TIMEOUT_S)
    assert j.requirements == {"min_lease_s": D.AUDIT_TIMEOUT_S + 900, "avx512": True}
    w.at(0)
    assert w.runner.started == [(1, 1, "t1")]
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("failed", "check failed: failed rc=1")
    w.at(65)
    assert w.job(1).state == "failed"


def test_the_worker_claims_with_its_own_token_even_beside_the_coordinators(tmp_path, monkeypatch):
    """Daniel's layout on the RC's VM: ~/.research/jobs/token is the coordinator's, ~/.research/jobs/dispatch-rc.token the
    dispatcher's. Each command takes its own, whatever else is set."""
    from research import remote
    from research.jobs import cli as J
    from research.jobs.client import WORKER_TOKEN_ENV, load_token, worker_token_file

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    for name, value in (("token", "coordinator-secret"), ("dispatch-rc.token", "dispatch-rc-secret")):
        (jobs / name).write_text(value + "\n")
        (jobs / name).chmod(0o600)
    monkeypatch.setattr("research.jobs.client.TOKEN_FILE", jobs / "token")
    assert worker_token_file("rc") == jobs / "dispatch-rc.token"
    assert load_token({}) == "coordinator-secret"
    seen = []
    monkeypatch.setattr(remote, "resolve_machine", lambda name, project: {"name": name})
    monkeypatch.setattr(J, "HTTPClient", lambda token=None: seen.append(token) or SimpleNamespace(url="u", token=token))
    monkeypatch.setattr(D.Dispatcher, "run", lambda self, once=False: None)
    monkeypatch.setenv("JOBS_TOKEN", "coordinator-secret")
    monkeypatch.delenv(WORKER_TOKEN_ENV, raising=False)
    assert J.worker_main(["--dispatch", "--pods", "t1", "--workdir", str(jobs)]) == 0
    monkeypatch.setenv(WORKER_TOKEN_ENV, "from-the-environment")
    assert J.worker_main(["--dispatch", "--pods", "t1", "--workdir", str(jobs)]) == 0
    (jobs / "other.token").write_text("given-file\n")
    (jobs / "other.token").chmod(0o600)
    assert J.worker_main(["--dispatch", "--pods", "t1", "--workdir", str(jobs), "--token-file", str(jobs / "other.token")]) == 0
    assert seen == ["dispatch-rc-secret", "from-the-environment", "given-file"]      # an explicit file wins over the variable
    monkeypatch.delenv(WORKER_TOKEN_ENV)
    (jobs / "dispatch-rc.token").unlink()
    with pytest.raises(SystemExit, match="dispatch-rc.token"):
        J.worker_main(["--dispatch", "--pods", "t1", "--workdir", str(jobs)])
    assert "JOBS_WORKER_TOKEN" not in D.ResearchRunner(environ={"JOBS_WORKER_TOKEN": "x", "JOBS_TOKEN": "y"}).env()


class Recording:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        return self.inner.call(method, path, body)


def test_a_complete_and_a_renewal_send_no_nulls():
    """The site takes run_id and summary only as strings: a launch that failed completes without a run_id."""
    w = World()
    rec = Recording(w.worker)
    w.d.client = rec
    j = w.add()
    w.at(0)
    w.at(D.RENEW_S)
    w.runner.launches["L1-1"] = (3, None, "research run --on t1: refused: pod is EXITED")
    w.at(D.RENEW_S + 5)
    renew = next(b for m, p, b in rec.calls if p.endswith("/renew"))
    complete = next(b for m, p, b in rec.calls if p.endswith("/complete"))
    assert renew["progress"] == {"pod": "t1", "step": "launching"} and "run_id" not in complete
    assert isinstance(complete["summary"], str) and w.svc.attempts[0].outcome == "error" and w.job(j).attempts == 2
