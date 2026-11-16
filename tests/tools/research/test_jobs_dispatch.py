"""Step 1's dispatcher (`research.jobs.dispatch`) against the Job queue's reference model, on a fake clock, with a fake runner
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

from research import settings
from research.jobs import dispatch as D
from research.jobs import model
from research.jobs.cli import main as jobs_main
from research.jobs.client import ModelClient
from research.pods.nebius.slot_only import POOL_WAIT_S

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
        self.capabilities: dict[str, dict] = {}
        self.launches: dict[str, tuple | None] = {}
        self.runs: dict[str, D.Observation] = {}
        self.started, self.stopped, self.killed, self.probes, self.cancelled = [], [], [], [], []
        self.local: str | None = None

    def local_problem(self):
        return self.local

    def probe(self, pod):
        self.probes.append(pod)
        return dict(self.capabilities.get(pod, GOOD))

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

    def cancel(self, run_id):
        self.cancelled.append(run_id)

    def restricted(self):
        return {"vy-auditor-1", "vy-control-verity"}


class World:
    def __init__(self, pods=("t1",), state_path=None, kinds=D.POD_KINDS, run_slots=0):
        self.svc = model.JobService()
        self.svc.mint("tok-rc", "coordinator", ["jobs:enqueue", "events:read"])
        self.svc.mint("tok-d", "dispatch-rc", ["jobs:work"])
        self.svc.register("dispatch:rc", "dispatcher", "dispatch-rc", list(D.KINDS))
        self.clock = Clock()
        self.coord = ModelClient(self.svc, "tok-rc", self.clock)
        self.worker = ModelClient(self.svc, "tok-d", self.clock)
        self.runner = FakeRunner()
        self.lines: list[str] = []
        self.pods, self.state_path, self.kinds, self.run_slots = list(pods), state_path, kinds, run_slots
        self.d = self.dispatcher()

    def dispatcher(self) -> D.Dispatcher:
        return D.Dispatcher(self.worker, self.runner, self.pods, clock=self.clock, out=self.lines.append, state_path=self.state_path,
                            kinds=self.kinds, run_slots=self.run_slots)

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
    "research run --on t1: refused: the machine failed the preflight tools/check/preflight.py (exit 1): uv is not on the PATH",
    "research: run r20260929-000001-abcd refused on t1 (PreflightFailed): cpu: no avx512f  (recorded in /x)",
])
def test_a_pod_that_fails_checks_preflight_is_benched_and_another_pod_takes_the_job(tail):
    """Only preflight.py says whether a pod is prepared; the dispatcher's probe measures, and it believes the preflight."""
    w = World(pods=("t1", "t2"))
    w.runner.capabilities["t1"] = {**GOOD, "tools": [], "disk_gb": 3}           # measured, never judged
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
    """A launch chain that started a check while pod_setup.sh still ran cost a candidate an hour: the dispatcher can't do that."""
    w = World()
    j = w.add()
    w.runner.capabilities["t1"] = {**GOOD, "setup_done": False, "setup_at": None}
    w.at(0)
    assert w.runner.started == [] and w.job(j).state == "queued"
    assert any("t1: skipped: pod_setup.sh hasn't reported done" in line for line in w.lines)
    w.runner.capabilities["t1"] = GOOD
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
    w.runner.capabilities["t1"] = {**GOOD, "avx512": False}
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
    assert D.command(job, "t1", "py", None) == ["py", "tools/check/check.py", "--record", "--on", "t1", "--base", BASE]
    assert D.command(job, "t1", "py", tmp_path / "v.tar.gz")[-2:] == ["--verdicts-in", str(tmp_path / "v.tar.gz")]
    regen = D.command({"kind": "lean-regen", "payload": {"packages": ["verity/protocols/proofs/flock/prover/agreement/soundness"]}}, "t1", "py", None)
    assert regen[:15] == ["py", "-m", "research", "run", "--on", "t1", "--project", "verity", "--source", ".", "--cwd", "source",
                          "--preflight", "check", "--"]
    assert "--preflight" not in D.command({"kind": "lean-regen", "payload": {"packages": ["p"]}}, "t1", "py", None, preflight=False)
    assert D.command({"kind": "audit-main", "payload": {"commit": C1}}, "t1", "py", None) == [
        "py", "tools/check/check.py", "--record", "--on", "t1", "--base", D.EMPTY_TREE, "--no-cache"]
    assert "check.py --build --update verity/protocols/proofs/flock/prover/agreement/soundness" in regen[-1]
    assert '$RESEARCH_RUN_DIR/records/$p' in regen[-1] and regen[-1].endswith("exit $rc")


def test_a_job_under_the_repositorys_old_name_is_the_same_repository():
    job = {"kind": "merge-check", "payload": {"commit": C1, "base": BASE, "repo": "danielreuter/verity"}}
    D.validate(job, "compute-verification/verity")
    D.validate({**job, "payload": {**job["payload"], "repo": "compute-verification/verity"}}, "danielreuter/verity")


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
    (origin / "tools" / "check").mkdir(parents=True)
    (origin / "tools" / "check" / "check.py").write_text(
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
    w.runner.capabilities["t1"] = {**GOOD, "lease_s": D.lease_needed(3600) - 1}
    assert jobs_main(["add", "merge-check", "--commit", C1, "--base", BASE, "--timeout-s", "3600"], client=w.coord) == 0
    assert w.job(1).requirements["min_lease_s"] == 3600 + 900
    w.at(0)
    assert w.runner.started == []
    w.at(D.CLAIM_S)
    assert w.runner.started == [(1, 1, "t2")]
    assert any("t1: free: lease 1.2 h, 70 GB free, 240 GB, 64 cpus, avx512" in line for line in w.lines)


def test_a_job_without_min_lease_s_on_a_short_lease_is_named():
    w = World()
    w.runner.capabilities["t1"] = {**GOOD, "lease_s": 600}
    j = w.add()
    w.at(0)
    assert w.runner.started == [(j, 1, "t1")]
    assert any(f"job {j} was added without min_lease_s" in line and "research pods extend t1" in line for line in w.lines)


def test_a_pod_without_a_lease_says_so():
    w = World()
    w.runner.capabilities["t1"] = {k: v for k, v in GOOD.items() if k != "lease_s"}
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
    env = {"PATH": settings.need("PATH"), "HOME": str(tmp_path)}

    def probe(r, e=env):
        return json.loads(subprocess.run([sys.executable, "-c", D.PROBE, str(r), str(lease), D.SETUP_MARKER], capture_output=True,
                                         text=True, check=True, env=e).stdout)

    capabilities = probe(root)
    assert capabilities["busy"] == 1 and capabilities["lease_s"] > 2 * 10 ** 9
    assert (capabilities["setup_done"], capabilities["setup_at"]) == (True, "2026-09-29T22:00:00Z")
    assert set(capabilities) == {"avx512", "mem_gb", "cpus", "disk_gb", "busy", "tools", "lease_s", "setup_done", "setup_at"}
    lease.unlink()
    (tmp_path / ".research" / "pod-setup.json").unlink()
    capabilities = probe(root / "missing")
    assert "lease_s" not in capabilities and capabilities["busy"] == 0 and capabilities["setup_done"] is False
    capabilities = probe(root, {**env, "RESEARCH_LEASE_EXPIRES": "4102444800"})
    assert capabilities["lease_s"] > 2 * 10 ** 9


#: the Lean check's lines that place its fixtures, as check.py writes them
CHECK = 'HERE = Path(__file__).resolve().parent\nROOT = HERE.parents[2]\nFIXTURES = HERE / "tests" / "fixtures"\n'


def _lean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    files = {
        "a/lean-audit.json": "{}", "a/lakefile.toml": "", "a/A.lean": "a",
        "a/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": "../b"}, {"type": "git", "name": "mathlib"}]}),
        "a/nested/lakefile.toml": "", "a/nested/N.lean": "n",
        "b/lakefile.toml": "", "b/B.lean": "b", "c/lean-audit.json": "{}", "c/C.lean": "c",
        "tools/verity/lean/check.py": CHECK, "tools/check/check.py": "check", "README.md": "readme",
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
    for name, path, text in (("readme", "README.md", "x"), ("nested", "a/nested/N.lean", "x"), ("check", "tools/check/check.py", "x"),
                             ("dep", "b/B.lean", "x"), ("tool", "tools/verity/lean/check.py", CHECK + "x = 1\n"),
                             ("other", "c/C.lean", "x")):
        (repo / path).write_text(text)
        _git(repo, "commit", "-qam", name)
        keys[name] = lean_key(_git(repo, "rev-parse", "HEAD"), ["a"], repo)
    base = lean_key(one, ["a"], repo)
    assert keys["readme"] == keys["nested"] == keys["check"] == base
    assert len({base, keys["dep"], keys["tool"]}) == 3 and keys["other"] == keys["tool"]
    assert lean_key(one, ["a/"], repo) == base != lean_key(one, ["a", "c"], repo)
    with pytest.raises(SystemExit, match="b has no lean-audit.json"):
        lean_key(one, ["b"], repo)


@pytest.mark.parametrize("fixtures, line", [
    ("tools/verity/lean/tests/fixtures", 'FIXTURES = HERE / "tests" / "fixtures"'),
    ("tests/tools/verity/lean/fixtures", 'FIXTURES = HERE.parents[2] / "tests" / "tools" / "verity" / "lean" / "fixtures"'),
    ("tests/tools/verity/lean/fixtures", 'FIXTURES = ROOT / "tests/tools/verity/lean/fixtures"')])
def test_a_lean_regen_is_keyed_on_the_controls_fixtures_wherever_check_puts_them(tmp_path, fixtures, line):
    """Every Lean check run builds the controls' packages (`check.FIXTURES`), so a change to one changes the key, also once
    they sit under the root's tests/, outside `LEAN_TOOL`."""
    from research.jobs.cli import lean_fixtures, lean_inputs, lean_key

    repo = _lean_repo(tmp_path)
    (repo / "tools/verity/lean/check.py").write_text(CHECK.replace('FIXTURES = HERE / "tests" / "fixtures"', line))
    (repo / fixtures / "control").mkdir(parents=True)
    (repo / fixtures / "control" / "Control.lean").write_text("theorem c : True := trivial\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixtures")
    one = _git(repo, "rev-parse", "HEAD")
    assert lean_fixtures(one, repo) == fixtures
    assert f"{fixtures}/control/Control.lean" in lean_inputs(one, ["a"], repo)
    (repo / fixtures / "control" / "Control.lean").write_text("theorem c : 1 = 1 := rfl\n")
    _git(repo, "commit", "-qam", "control")
    assert lean_key(_git(repo, "rev-parse", "HEAD"), ["a"], repo) != lean_key(one, ["a"], repo)


def test_a_lean_regen_refuses_a_check_whose_fixtures_it_cannot_place(tmp_path):
    from research.jobs.cli import lean_key

    repo = _lean_repo(tmp_path)
    (repo / "tools/verity/lean/check.py").write_text("FIXTURES = fixtures()\n")
    _git(repo, "commit", "-qam", "unplaced")
    with pytest.raises(SystemExit, match="sets no `FIXTURES`"):
        lean_key(_git(repo, "rev-parse", "HEAD"), ["a"], repo)


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
    files = {"verity/protocols/proofs/flock/prover/agreement/upstream.json": json.dumps({"upstream": {"sha256": "0" * 64, "rustflags": rustflags}}),
             "verity/protocols/proofs/flock/prover/src.rs": "fn main() {}", "verity/protocols/proofs/flock/prover/README.md": "flock", "docs.md": "docs",
             "tools/check/tool.py": 'CHECK = Tool(name="check", merge_requires={"lean-agreement": ["verity/protocols/proofs/flock/prover/", "!README.md"]})\n'}
    if pattern is not None:
        files["tools/check/preflight.py"] = f"import re\nAVX512 = re.compile({pattern!r})\n"
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
    ("-C target-cpu=x86-64-v3", r"target-cpu=(native|x86-64-v4)\b|avx512", "verity/protocols/proofs/flock/prover/src.rs", False),   # today's pin
    ("-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512", "verity/protocols/proofs/flock/prover/src.rs", True),
    ("-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512", "docs.md", False),                # no agreement runs
    ("-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512", "verity/protocols/proofs/flock/prover/README.md", False),
    ("-C target-cpu=x86-64-v4", r"never", "verity/protocols/proofs/flock/prover/src.rs", False),                                     # the tree's own rule
    ("-C target-cpu=native", None, "verity/protocols/proofs/flock/prover/src.rs", True),                                            # a tree before preflight.py's AVX512
])
def test_a_check_asks_for_avx512_only_when_its_tree_says_so(tmp_path, rustflags, pattern, path, want):
    from research.jobs.cli import needs_avx512

    repo = _flock_repo(tmp_path, rustflags, pattern)
    commit, base = _change(repo, path)
    assert needs_avx512(commit, base, repo) is want


def test_a_flock_file_whose_name_git_quotes_still_asks_for_avx512(tmp_path):
    """`git diff --name-only` quotes a name with bytes outside ASCII (`"verity/protocols/proofs/flock/prover/\\303\\251.rs"`), which the agreement's
    rule never matched, so a change to such a file asked for no AVX-512 machine."""
    from research.jobs.cli import needs_avx512

    repo = _flock_repo(tmp_path, "-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "verity" / "protocols" / "proofs" / "flock" / "prover" / "é.rs").write_text("fn f() {}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "é")
    assert needs_avx512(_git(repo, "rev-parse", "HEAD"), base, repo) is True


def test_a_check_asks_for_avx512_for_a_file_renamed_out_of_the_agreements_paths(tmp_path):
    """git reports a rename by its destination alone unless asked not to, which would leave out the agreement the file's old
    path runs."""
    from research.jobs.cli import needs_avx512

    repo = _flock_repo(tmp_path, "-C target-cpu=x86-64-v4", r"target-cpu=(native|x86-64-v4)\b|avx512")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "mv", "verity/protocols/proofs/flock/prover/src.rs", "src.rs")
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
    """The layout on the RC's VM: ~/.research/jobs/token is the coordinator's, ~/.research/jobs/dispatch-rc.token the
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


# ---- research-run: a lead's `research run` line ------------------------------------------------------------------------------

REPO = "compute-verification/verity"
LEAD = {"repo": REPO, "commit": C1, "lane": "proofs"}
TWIN = "r20260929-000105-bbbb"
CURSOR = "goes through a Cursor worker for now"


def plan(opts: str, restricted=lambda: {"vy-auditor-1"}) -> D.RunPlan:
    return D.run_plan({**LEAD, "command": f"research run {opts} -- uv run pytest -q"}, REPO, restricted)


@pytest.mark.parametrize("opts,want", [
    ("", ["--on", "auto"]),
    ("--on auto", ["--on", "auto"]),
    ("--on vy-nebius-cpu-3 --pool proofs --stay", ["--on", "vy-nebius-cpu-3", "--pool", "proofs", "--stay"]),
    ("--cpus 8 --mem-gb 31.5 --timeout 2h --cwd clone", ["--on", "auto", "--cpus", "8", "--mem-gb", "31.5", "--timeout", "2h", "--cwd", "clone"]),
    ("--max-min=30 --kind lean-audit --stage build --cwd source", ["--on", "auto", "--max-min", "30", "--kind", "lean-audit", "--stage",
                                                                    "build", "--cwd", "source"]),
    ("--expect tests=1 --expect lean=ok --require-result --declared-output 'out/*.json' --declared-output log --question 'does it pass?'",
     ["--on", "auto", "--expect", "tests=1", "--expect", "lean=ok", "--require-result", "--declared-output", "out/*.json",
      "--declared-output", "log", "--question", "does it pass?"]),
    ("--timeout 21600 --kind agent-test", ["--on", "auto", "--timeout", "21600", "--kind", "agent-test"]),
])
def test_a_research_run_line_the_first_week_allows(opts, want):
    p = plan(opts)
    assert (p.opts, p.argv, p.lane) == ([*want, "--gpus", "0"], ["uv", "run", "pytest", "-q"], "proofs")


@pytest.mark.parametrize("opts,why", [
    *[(o, f"takes {o.split()[0]}, which a research-run job may not carry: a GPU run, or a run that needs {o.split()[0]}, {CURSOR}")
      for o in ("--gpus 1", "--gpus 0", "--gpu-host h", "--phase gpu", "--queue gpu-h100", "--queue cpu-test", "--send f", "--source /",
                "--env-host h", "--runs-dir /x", "--local-only why", "--no-custody", "--no-custody-r2", "--custody-ttl 7d", "--id r1",
                "--claimed", "--store-read", "--env K=V", "--tool check", "--preflight check", "--as infra", "--project other",
                "--wait", "-h", "--made-up", "stray")],
    *[(f"--on {m}", f"gives --on '{m}', which isn't auto or a vy-nebius-cpu-N machine: a GPU run, or a run that needs another "
                    f"machine, {CURSOR}")
      for m in ("vy-nebius-2", "vy-control-verity", "vy-auditor-1", "vy-nebius-cpu-01", "vy-nebius-cpu-", "local")],
    ("--kind gpu-h100", "gives --kind 'gpu-h100', which isn't agent-test or lean-audit"),
    ("--pool train", "gives --pool 'train', which isn't network or proofs or research or soundness"),
    ("--cwd /tmp", "gives --cwd '/tmp', which isn't clone or source"),
    ("--cpus 0", "gives --cpus '0', which isn't above 0 and at most 64"),
    ("--cpus 65", "gives --cpus '65', which isn't above 0 and at most 64"),
    ("--cpus 1.5", "gives --cpus '1.5', which `research run` can't read"),
    ("--mem-gb nan", "gives --mem-gb 'nan', which isn't above 0 and at most 512 GB"),
    ("--mem-gb inf", "gives --mem-gb 'inf', which isn't above 0 and at most 512 GB"),
    ("--timeout nan", "gives --timeout 'nan', which isn't above 0 and at most 21600 s"),
    ("--timeout NaN", "gives --timeout 'NaN', which isn't above 0 and at most 21600 s"),
    ("--timeout 7h", "gives --timeout '7h', which isn't above 0 and at most 21600 s"),
    ("--timeout 1fortnight", "gives --timeout '1fortnight', which `research run` can't read"),
    ("--timeout -1", "gives --timeout '-1', which starts with '-', so `research run` would read it as an option"),
    ("--max-min 361", "gives --max-min '361', which isn't above 0 and at most 360 min"),
    ("--max-min ten", "gives --max-min 'ten', which `research run` can't read"),
    ("--on auto --on auto", "gives --on twice"),
    ("--require-result=yes", "gives the flag --require-result a value"),
    ("--stage", "gives --stage no value"),
    ("--expect nothing", "gives --expect 'nothing', which isn't of the form ^[A-Za-z0-9_.-]+="),
])
def test_a_research_run_line_the_first_week_refuses_names_the_option(opts, why):
    with pytest.raises(ValueError) as e:
        plan(opts)
    assert str(e.value) == f"its `research run` {why}"


def test_a_named_cpu_machine_that_is_an_auditor_device_or_cant_be_checked_is_refused():
    with pytest.raises(ValueError, match="gives --on 'vy-nebius-cpu-1', which is an auditor device, which no job reaches"):
        plan("--on vy-nebius-cpu-1", restricted=lambda: {"vy-nebius-cpu-1"})

    def unread():
        raise OSError("machines.toml: no such file")

    with pytest.raises(ValueError, match=r"whether it is an auditor device can't be read \(OSError: machines.toml: no such file\)"):
        plan("--on vy-nebius-cpu-1", restricted=unread)
    assert plan("", restricted=unread).opts[:2] == ["--on", "auto"]       # auto names no machine of its own to check


@pytest.mark.parametrize("payload,why", [
    ({k: v for k, v in LEAD.items() if k != "lane"} | {"command": "research run -- x"}, "must hold exactly command, commit, lane, repo"),
    ({**LEAD, "command": "research run -- x", "env": {"A": "1"}}, "must hold exactly command, commit, lane, repo"),
    ("research run -- x", "must hold exactly command, commit, lane, repo, and it holds str"),
    ({**LEAD, "repo": "someone/else", "command": "research run -- x"}, "the job is for 'someone/else'"),
    ({**LEAD, "commit": "abc", "command": "research run -- x"}, "payload.commit must be a full commit sha"),
    ({**LEAD, "commit": "A" * 40, "command": "research run -- x"}, "payload.commit must be a full commit sha"),
    ({**LEAD, "lane": "../infra", "command": "research run -- x"}, "payload.lane must be a lane's name"),
    ({**LEAD, "lane": 7, "command": "research run -- x"}, "payload.lane must be a lane's name"),
    ({**LEAD, "command": ["research", "run", "--", "x"]}, "payload.command must be a `research run` line of 1 to 16384"),
    ({**LEAD, "command": ""}, "payload.command must be a `research run` line"),
    ({**LEAD, "command": "research run -- " + "x" * 16384}, "payload.command must be a `research run` line"),
    ({**LEAD, "command": "research run -- a\0b"}, "without NUL"),
    ({**LEAD, "command": "research run -- 'unclosed"}, "payload.command doesn't split as a shell line"),
    ({**LEAD, "command": "uv run pytest"}, r"must be `research run \[OPTIONS\] -- COMMAND...`"),
    ({**LEAD, "command": "research run --cpus 2 uv run pytest"}, r"must be `research run \[OPTIONS\] -- COMMAND...`"),
    ({**LEAD, "command": "research run --cpus 2 --"}, r"must be `research run \[OPTIONS\] -- COMMAND...`"),
    ({**LEAD, "command": "research fetch -- x"}, r"must be `research run \[OPTIONS\] -- COMMAND...`"),
])
def test_a_research_run_payload_is_checked_whole(payload, why):
    with pytest.raises(ValueError, match=why):
        D.run_plan(payload, REPO, lambda: set())
    with pytest.raises(ValueError, match=why):
        D.validate({"kind": D.RUN_KIND, "payload": payload}, REPO)


def test_a_research_run_under_the_repositorys_old_name_and_its_workload_are_kept_as_given():
    p = D.run_plan({**LEAD, "repo": "danielreuter/verity", "command": "research run -- bash -lc 'a && b' -- --gpus 1"}, REPO)
    assert p.argv == ["bash", "-lc", "a && b", "--", "--gpus", "1"] and p.opts == ["--on", "auto", "--gpus", "0"]


def test_the_longest_timeout_is_the_custody_keys_default_life():
    from research.store.credentials import parse_ttl
    from research.store.custody import DEFAULT_TTL

    assert D.RUN_TIMEOUT_MAX_S == parse_ttl(DEFAULT_TTL)


def test_the_dispatcher_waits_out_the_runs_own_timeout_and_its_pools_wait():
    rest = POOL_WAIT_S + D.LAUNCH_TIMEOUT_S + D.RUN_MARGIN_S
    assert plan("--timeout 1h").limit_s == 3600 + rest
    assert plan("--max-min 90").limit_s == 5400 + rest
    assert plan("--max-min 90 --timeout 1h").limit_s == 5400 + rest
    assert plan("").limit_s == D.RUN_TIMEOUT_MAX_S + rest == 6 * 3600 + rest


def test_the_launcher_gets_the_pods_environment_less_the_queues_tokens_and_python_settings(tmp_path):
    environ = {"PATH": "/bin", "HOME": "/home/x", "AWS_SECRET_ACCESS_KEY": "not-to-be-copied", "JOBS_TOKEN": "dummy",
               "JOBS_WORKER_TOKEN": "dummy", "CURSOR_CONVERSATION_ID": "dummy", "PYTHONPATH": "/elsewhere", "PYTHONSTARTUP": "/x.py",
               "PYTHONHOME": "/y", "RESEARCH_LANE": "infra"}
    env = D.run_env(environ, tmp_path, "proofs")
    assert sorted(environ.keys() - env.keys()) == ["CURSOR_CONVERSATION_ID", "JOBS_TOKEN", "JOBS_WORKER_TOKEN", "PYTHONHOME", "PYTHONSTARTUP"]
    assert sorted(env.keys() - environ.keys()) == ["RESEARCH_AGENT_URL", "RESEARCH_AT_CHECKOUT", "RESEARCH_LAUNCHER_TREE"]
    assert sorted(k for k in env.keys() & environ.keys() if env[k] != environ[k]) == ["PYTHONPATH", "RESEARCH_LANE"]
    assert (env["PYTHONPATH"], env["RESEARCH_LAUNCHER_TREE"], env["RESEARCH_AT_CHECKOUT"]) == (
        str(tmp_path / "tools" / "research" / "src"), str(tmp_path), "1")
    assert env["RESEARCH_LANE"] == env["RESEARCH_AGENT_URL"] == "proofs"


def test_the_launch_runs_research_from_the_dispatchers_tree_with_its_checked_options():
    assert D.run_command(plan("--cpus 4"), "py") == [
        "py", "-P", "-s", "-m", "research", "run", "--source", ".", "--project", "verity", "--on", "auto", "--cpus", "4", "--gpus", "0",
        "--", "uv", "run", "pytest", "-q"]


def run_world(**kw) -> World:
    return World(pods=(), kinds=(D.RUN_KIND,), run_slots=2, **kw)


def add_run(w: World, opts: str = "--cpus 4", **kw) -> int:
    status, body = w.coord.call("POST", "/api/jobs", {"kind": D.RUN_KIND, "payload": {**LEAD, "command": f"research run {opts} -- uv run x"},
                                                      **kw})
    assert status == 201, body
    return body["id"]


def progress(run_id=RUN, state="running", tail="", rc=None) -> dict:
    return {"run_id": run_id, "machine": "vy-nebius-cpu-1", "state": state, "rc": rc, "tail": tail}


def test_a_research_run_takes_a_virtual_slot_reports_progress_and_completes_with_its_row():
    w = run_world()
    rec = Recording(w.worker)
    w.d.client = rec
    assert [s.pod for s in w.d.slots] == ["research-run-1", "research-run-2"]
    j = add_run(w)
    w.at(0)
    claim = next(b for m, p, b in rec.calls if p == "/api/jobs/claim")
    assert claim == {"worker": "dispatch:rc", "kinds": ["research-run"], "slot": "research-run-1"}     # no pod, no capabilities
    assert w.runner.started == [(j, 1, "research-run-1")] and w.runner.probes == []
    w.at(D.RENEW_S)
    assert w.job(j).progress == {"run_id": None, "machine": None, "state": "launching", "rc": None, "tail": ""}
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(D.RENEW_S + 5)
    w.runner.runs[RUN] = D.Observation("running", "running", RUN, progress(tail="collecting\n"))
    w.at(2 * D.RENEW_S + 5)
    assert w.job(j).progress == {"run_id": RUN, "machine": None, "state": "submitted", "rc": None, "tail": ""}
    w.at(3 * D.RENEW_S + 5)
    assert w.job(j).progress == progress(tail="collecting\n")
    w.runner.runs[RUN] = D.Observation("passed", "ROW\n--- tail ---\n" + "y" * 20000, RUN, progress("done", "y", 0))
    w.at(4 * D.RENEW_S + 5)
    assert w.job(j).state == "done"
    assert w.job(j).result == {"outcome": "passed", "run_id": RUN, "summary": ("ROW\n--- tail ---\n" + "y" * 20000)[:D.SUMMARY_CHARS]}


def test_research_run_slots_are_their_own_and_pods_keep_theirs():
    w = World(pods=("t1",), kinds=D.KINDS, run_slots=2)
    m = w.add()
    a, b, c = add_run(w), add_run(w), add_run(w)
    w.at(0)
    assert sorted(w.runner.started) == [(m, 1, "t1"), (a, 1, "research-run-1"), (b, 1, "research-run-2")]
    assert w.job(c).state == "queued"


def test_a_research_run_the_allowlist_refuses_fails_at_once_and_launches_nothing():
    w = run_world()
    j = add_run(w, "--gpus 1")
    w.at(0)
    assert w.runner.started == [] and (w.job(j).state, w.job(j).attempts) == ("failed", 1)
    assert w.job(j).result["summary"] == (f"refused: its `research run` takes --gpus, which a research-run job may not carry: a GPU run, or "
                                          f"a run that needs --gpus, {CURSOR}")


def test_a_research_run_whose_launch_fails_is_a_verdict_with_its_output():
    w = run_world()
    j = add_run(w, "--stay")
    w.at(0)
    w.runner.launches["L1-1"] = (2, None, "research run: --stay pins a run to the machine --on names, and auto names none\n")
    w.at(5)
    assert (w.job(j).state, w.job(j).attempts) == ("failed", 1)
    assert w.job(j).result["summary"] == "`research run` exited 2:\nresearch run: --stay pins a run to the machine --on names, and auto names none\n"


def test_a_research_run_is_followed_to_its_late_twin_and_a_cancel_reaches_the_twin():
    w = run_world()
    j = add_run(w)
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("running", "running", TWIN, progress(TWIN))
    w.at(65)
    assert w.d.slots[0].run_id == TWIN
    assert any(f"run {RUN} left its line unstarted; following its late twin {TWIN}" in line for line in w.lines)
    assert w.coord.call("POST", f"/api/jobs/{j}/cancel", {})[1] == {"state": "running", "cancel_requested": True}
    w.at(65 + D.RENEW_S)
    assert w.runner.cancelled == [TWIN] and w.runner.stopped == []
    assert w.job(j).state == "cancelled" and w.d.slots[0].job is None


def test_a_research_run_whose_lease_is_lost_is_left_running_and_untended():
    w = run_world()
    j = add_run(w, lease_s=60)
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("running", "running", RUN, progress())
    w.worker.down = True
    w.at(60)
    assert w.svc.tick(61) == [j]
    w.worker.down = False
    w.at(62)
    assert (w.runner.cancelled, w.runner.stopped, w.runner.killed) == ([], [], [])
    assert any("the lease is gone (409" in line for line in w.lines)
    assert any(f"run {RUN} is left running, and no longer tended here" in line for line in w.lines)
    assert all(s.job is None or s.job["id"] != j or s.attempt == 2 for s in w.d.slots)


def test_a_research_run_is_stopped_only_past_its_own_timeout_and_its_pools_wait():
    w = run_world()
    j = add_run(w, "--timeout 5h")
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.runner.runs[RUN] = D.Observation("running", "running", RUN, progress())
    limit = 5 * 3600 + POOL_WAIT_S + D.LAUNCH_TIMEOUT_S + D.RUN_MARGIN_S
    w.at(D.DEFAULT_TIMEOUT_S + D.POLL_S)
    w.at(limit - 1)
    assert w.runner.cancelled == [] and w.job(j).state == "running"
    w.at(limit + D.POLL_S)
    assert w.runner.cancelled == [RUN]
    assert w.svc.attempts[0].detail.startswith(f"stopped after its own --timeout, its pool's wait, the launch and the runner's margin "
                                               f"({limit:g} s)")


def test_a_restarted_dispatcher_keeps_tending_its_research_runs(tmp_path):
    w = run_world(state_path=tmp_path / "dispatch-rc.json")
    j = add_run(w, "--timeout 1h")
    w.at(0)
    w.runner.launches["L1-1"] = (0, RUN, "")
    w.at(5)
    w.d = w.dispatcher()
    s = w.d.slots[0]
    assert (s.pod, s.virtual, s.job["id"], s.run_id, s.limit_s) == ("research-run-1", True, j, RUN, plan("--timeout 1h").limit_s)
    assert s.progress == {"run_id": RUN, "machine": None, "state": "submitted", "rc": None, "tail": ""}
    w.runner.runs[RUN] = D.Observation("passed", "ROW", RUN, progress("done", rc=0))
    w.at(70)
    assert w.job(j).state == "done" and w.runner.started == [(j, 1, "research-run-1")]


def _run_dir(root: Path, run_id: str, status: dict | None, result: str | None = None) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "remote.json").write_text(json.dumps({"run_id": run_id, "machine": "vy-nebius-cpu-1", "project": "verity", "phase": "launched",
                                               "command": ["uv", "run", "x"], "launch_args": []}))
    if status is not None:
        (d / "status.json").write_text(json.dumps(status))
    if result is not None:
        (d / "result.json").write_text(result)
    return d


def test_a_research_runs_observation_is_its_progress_and_at_the_end_its_row_tail_and_result(tmp_path):
    d = _run_dir(tmp_path, RUN, {"state": "running"})
    tails = {RUN: b"\xff" + b"x" * 7000}
    obs = D.observe_run(d, lambda _d: {}, lambda rd: tails[rd.name], lambda _d: None)
    assert (obs.state, obs.run_id) == ("running", RUN)
    assert obs.progress == {"run_id": RUN, "machine": "vy-nebius-cpu-1", "state": "running", "rc": None, "tail": "x" * D.TAIL_BYTES}
    (d / "status.json").write_text(json.dumps({"state": "done", "rc": 0}))
    (d / "result.json").write_text('{"tests": 12}')
    tails[RUN] = b"12 passed\n"
    obs = D.observe_run(d, lambda _d: {}, lambda rd: tails[rd.name], lambda _d: None)
    assert (obs.state, obs.progress["state"], obs.progress["rc"]) == ("passed", "done", 0)
    row, rest = obs.summary.split("\n", 1)
    assert row.startswith(RUN) and "vy-nebius-cpu-1" in row
    assert rest == '--- the last 10 bytes of stdout.log ---\n12 passed\n\n--- result.json ---\n{"tests": 12}'
    (d / "result.json").write_text("y" * (D.RESULT_BYTES + 1))
    obs = D.observe_run(d, lambda _d: {}, lambda rd: b"", lambda _d: None)
    assert obs.summary.endswith(f"--- result.json: {D.RESULT_BYTES + 1} bytes, over the {D.RESULT_BYTES} a summary holds ---")


@pytest.mark.parametrize("status,want", [
    ({"state": "failed", "rc": 1}, "failed"),
    ({"state": "failed", "rc": None, "signal": 9}, "error"),
    ({"state": "cancelled", "rc": None}, "error"),
])
def test_a_research_run_that_ended_otherwise_is_failed_or_an_error(tmp_path, status, want):
    obs = D.observe_run(_run_dir(tmp_path, RUN, status), lambda _d: {}, lambda _d: b"", lambda _d: None)
    assert obs.state == want and obs.progress["state"] == status["state"]


def test_a_research_run_on_an_unreachable_machine_is_still_running(tmp_path):
    d = _run_dir(tmp_path, RUN, {"state": "running"})
    obs = D.observe_run(d, lambda _d: {"last_fetch_error": {"rc": 255}}, lambda _d: b"never read", lambda _d: None)
    assert (obs.state, obs.summary, obs.progress) == ("running", "machine unreachable (rc 255)", None)


def test_a_research_run_that_left_its_line_is_observed_as_its_late_twin(tmp_path):
    first = _run_dir(tmp_path, RUN, {"state": "failed", "rc": 75})
    twin = _run_dir(tmp_path, TWIN, {"state": "done", "rc": 0})
    fetched = []
    obs = D.observe_run(first, lambda d: fetched.append(d.name) or {}, lambda d: d.name.encode(), {first: twin}.get)
    assert (obs.state, obs.run_id, obs.progress["run_id"], obs.progress["tail"]) == ("passed", TWIN, TWIN, TWIN)
    assert fetched == [RUN, TWIN]
    started = _run_dir(tmp_path / "started", RUN, {"state": "failed", "rc": 1})
    (started / "slot.json").write_text(json.dumps({"state": "held"}))
    assert D.observe_run(started, lambda _d: {}, lambda _d: b"", lambda _d: twin).run_id == RUN    # it ran: its end is the run's


def test_a_research_run_launch_is_taken_only_for_a_run_dir_it_made_running_its_workload(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path))
    p = plan("")
    _run_dir(tmp_path, RUN, None)
    (tmp_path / RUN / "remote.json").write_text(json.dumps({"command": ["nice", "--", "uv", "run", "pytest", "-q"]}))
    assert D.launched_by(RUN, p, set())
    assert not D.launched_by(RUN, p, {RUN})                                         # there before the launch
    assert not D.launched_by(TWIN, p, set())                                        # no such run
    (tmp_path / RUN / "remote.json").write_text(json.dumps({"command": ["uv", "run", "other"]}))
    assert not D.launched_by(RUN, p, set())                                         # another workload


def _launcher_tree(tmp_path: Path, seen: Path, runs: Path) -> Path:
    """A stand-in for the dispatcher's tree: its `research` records how it was run, makes the run dir and prints the launch line,
    after two lines that name runs it didn't make."""
    pkg = tmp_path / "tree" / "tools" / "research" / "src" / "research"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(
        "import json, os, sys\n"
        "import research\n"
        f"runs, seen = {str(runs)!r}, {str(seen)!r}\n"
        "names = ('JOBS_TOKEN', 'JOBS_WORKER_TOKEN', 'PYTHONSTARTUP', 'AWS_SECRET_ACCESS_KEY', 'RESEARCH_LAUNCHER_TREE', 'RESEARCH_LANE',\n"
        "         'RESEARCH_AGENT_URL', 'RESEARCH_AT_CHECKOUT')\n"
        "json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'research': research.__file__, 'stdin': sys.stdin.read(),\n"
        "           'has': sorted(k for k in names if k in os.environ), 'tree': os.environ.get('RESEARCH_LAUNCHER_TREE'),\n"
        "           'lane': os.environ.get('RESEARCH_LANE')}, open(seen, 'w'))\n"
        "print('research: run r20260929-000000-0000 launched on vy-nebius-cpu-1 (a run that was there before)', flush=True)\n"
        "print('research: run r20260929-000000-9999 launched on vy-nebius-cpu-1 (no such run)', flush=True)\n"
        "rid = 'r20260929-000000-abcd'\n"
        "os.makedirs(os.path.join(runs, rid))\n"
        "json.dump({'command': sys.argv[sys.argv.index('--') + 1:]}, open(os.path.join(runs, rid, 'remote.json'), 'w'))\n"
        "print(f'research: run {rid} launched on vy-nebius-cpu-1 (podid) pid=1 -> /workspace/research/runs/{rid}', file=sys.stderr)\n")
    return _trusted(tmp_path / "tree")


def _trusted(tree: Path) -> Path:
    """``tree`` made a clean checkout of its own origin/main holding everything research.launcher's RUNS names: a tree the
    launcher trusts."""
    from research import launcher

    for f in launcher.RUNS:
        at = tree / f
        if not at.exists():
            (at.parent if Path(f).suffix else at).mkdir(parents=True, exist_ok=True)
            (at if Path(f).suffix else at / "keep").write_text("")
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "add", ".")
    _git(tree, "commit", "-qm", "main")
    _git(tree, "update-ref", "refs/remotes/origin/main", "HEAD")
    return tree


@pytest.mark.parametrize("case,problem", [
    ("blank", "is blank, and set"),
    ("relative", "is not an absolute path"),
    ("missing", "is not an existing directory"),
    ("incomplete", "lacks what the launcher runs from it: tools/research/src/research/launcher.py"),
    ("not a checkout", "is not a git checkout with an origin/main"),
    ("stale", "is stale: its HEAD"),
    ("modified", "is stale: what the launcher runs from it is changed or untracked there (tools/check/slot.py)"),
    ("deleted", "lacks what the launcher runs from it: tools/check/slot.py"),
    ("untracked", "is stale: what the launcher runs from it is changed or untracked there (infra/cluster/kinds/new.toml)"),
    ("around the source", "is around the --source tree"),
])
def test_a_tree_the_launcher_refuses_is_refused_in_the_launchers_own_words(tmp_path, case, problem):
    source = tmp_path / "worktree"
    source.mkdir()
    tree = tmp_path / "tree"
    if case not in ("missing", "incomplete", "around the source"):
        _trusted(tree)
    if case == "incomplete":
        tree.mkdir()
    if case == "not a checkout":
        subprocess.run(["rm", "-rf", str(tree / ".git")], check=True)
    if case == "stale":
        (tree / "README.md").write_text("ahead of origin/main\n")
        _git(tree, "add", ".")
        _git(tree, "commit", "-qm", "a branch")
    if case == "modified":
        (tree / "tools" / "check" / "slot.py").write_text("print('not main')\n")
    if case == "deleted":
        (tree / "tools" / "check" / "slot.py").unlink()
    if case == "untracked":
        (tree / "infra" / "cluster" / "kinds" / "new.toml").write_text("")
    if case == "around the source":
        tree = _trusted(tmp_path)
    value = {"blank": "", "relative": "tree", "missing": str(tmp_path / "gone")}.get(case, str(tree))
    why = D.launcher_refusal(value, source)
    assert why.startswith(f"research: $RESEARCH_LAUNCHER_TREE={value!r} {problem}"), why
    assert D.launcher_refusal(str(_trusted(tmp_path / "good")), source) is None


def test_a_launch_the_launcher_refuses_runs_nothing_and_fails_the_job_with_its_message(tmp_path):
    """A tree too broken to import research from: the launch child refuses it in the launcher's words, and runs nothing."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "README.md").write_text("x\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-qm", "x")
    tree = tmp_path / "tree"
    tree.mkdir()
    runner = D.ResearchRunner(workdir=tmp_path / "jobs", url=str(origin), environ=dict(os.environ), launcher_tree=tree)
    job = {"id": 11, "kind": D.RUN_KIND, "payload": {**LEAD, "commit": _git(origin, "rev-parse", "HEAD"), "command": "research run -- x"}}
    launch = runner.start(job, 1, "research-run-1")
    runner.children[launch].wait()
    rc, run_id, tail = runner.poll(launch)
    assert (rc, run_id) == (2, None)
    assert tail.startswith(f"research: $RESEARCH_LAUNCHER_TREE={str(tree)!r} lacks what the launcher runs from it: ")
    assert "dispatch: job 11 in" not in (tmp_path / "jobs" / "logs" / "job-11-1.log").read_text()     # nothing was started
    assert not (tmp_path / "jobs" / "worktrees" / "job-11-1").exists()
    w = run_world()
    j = add_run(w)
    w.at(0)
    w.runner.launches["L1-1"] = (rc, run_id, tail)
    w.at(5)
    w.at(5 + 2 * D.CLAIM_S)
    assert (w.job(j).state, w.job(j).attempts, w.runner.started) == ("failed", 1, [(j, 1, "research-run-1")])
    assert w.job(j).result["summary"] == f"`research run` exited 2:\n{tail}"


def test_the_launch_child_runs_the_dispatchers_research_in_the_commits_worktree_without_the_queues_tokens(tmp_path):
    """The commit's own `research` package is at the worktree's root, where `python -m` would find it without -P."""
    origin, seen, runs = tmp_path / "origin", tmp_path / "seen.json", tmp_path / "runs"
    (origin / "research").mkdir(parents=True)
    (origin / "research" / "__init__.py").write_text(f"open({str(tmp_path / 'pwned')!r}, 'w').write('ran the commit')\n")
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "add", ".")
    _git(origin, "commit", "-qm", "the lead's commit")
    head = _git(origin, "rev-parse", "HEAD")
    _run_dir(runs, "r20260929-000000-0000", None)
    (runs / "r20260929-000000-0000" / "remote.json").write_text(json.dumps({"command": ["uv", "run", "x"]}))
    tree = _launcher_tree(tmp_path, seen, runs)
    environ = {**os.environ, "JOBS_TOKEN": "dummy", "JOBS_WORKER_TOKEN": "dummy", "PYTHONSTARTUP": str(tmp_path / "nowhere.py"),
               "AWS_SECRET_ACCESS_KEY": "not-to-be-copied", "RESEARCH_RUNS": str(runs)}
    runner = D.ResearchRunner(workdir=tmp_path / "jobs", url=str(origin), environ=environ, launcher_tree=tree)
    job = {"id": 9, "kind": D.RUN_KIND, "payload": {**LEAD, "commit": head, "command": "research run --cpus 2 -- uv run x"}}
    launch = runner.start(job, 1, "research-run-1")
    runner.children[launch].wait()
    rc, run_id, _ = runner.poll(launch)
    assert (rc, run_id) == (0, "r20260929-000000-abcd")
    got = json.loads(seen.read_text())
    assert got["argv"] == ["run", "--source", ".", "--project", "verity", "--on", "auto", "--cpus", "2", "--gpus", "0", "--", "uv", "run", "x"]
    assert got["cwd"] == str(tmp_path / "jobs" / "worktrees" / "job-9-1") and got["stdin"] == ""
    assert got["research"] == str(tree / "tools" / "research" / "src" / "research" / "__init__.py")
    assert got["has"] == ["AWS_SECRET_ACCESS_KEY", "RESEARCH_AGENT_URL", "RESEARCH_AT_CHECKOUT", "RESEARCH_LANE", "RESEARCH_LAUNCHER_TREE"]
    assert (got["tree"], got["lane"]) == (str(tree), "proofs")
    assert not (tmp_path / "pwned").exists()
    assert not (tmp_path / "jobs" / "worktrees" / "job-9-1").exists()


def test_a_research_run_launch_from_no_checkout_is_refused(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "README.md").write_text("x\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-qm", "x")
    runner = D.ResearchRunner(workdir=tmp_path / "jobs", url=str(origin), environ=dict(os.environ), launcher_tree=None)
    job = {"id": 10, "kind": D.RUN_KIND, "payload": {**LEAD, "commit": _git(origin, "rev-parse", "HEAD"), "command": "research run -- x"}}
    launch = runner.start(job, 1, "research-run-1")
    runner.children[launch].wait()
    rc, run_id, tail = runner.poll(launch)
    assert (rc, run_id) == (-1, None) and "runs from no checkout" in tail


def test_research_worker_runs_research_run_slots_without_pods(tmp_path, monkeypatch, capsys):
    from research.jobs import cli as J

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "dispatch-control.token").write_text("dummy\n")
    (jobs / "dispatch-control.token").chmod(0o600)
    made = []
    monkeypatch.setattr(J, "HTTPClient", lambda token=None: SimpleNamespace(url="u", token=token))
    monkeypatch.setattr(D.Dispatcher, "run", lambda self, once=False: made.append([s.pod for s in self.slots]))
    monkeypatch.delenv("JOBS_WORKER_TOKEN", raising=False)
    base = ["--dispatch", "--name", "control", "--workdir", str(jobs)]
    assert J.worker_main([*base, "--kinds", "research-run", "--slots", "3"]) == 0
    assert made == [["research-run-1", "research-run-2", "research-run-3"]]
    assert J.worker_main([*base, "--kinds", "research-run", "--slots", "0"]) == 2
    assert "--slots must be at least 1 for research-run" in capsys.readouterr().err
    assert J.worker_main([*base, "--kinds", "research-run,merge-check"]) == 2
    assert "--pods names no pod for merge-check" in capsys.readouterr().err
    monkeypatch.setattr(J, "ResearchRunner", lambda **kw: D.ResearchRunner(**kw, launcher_tree=None))
    assert J.worker_main([*base, "--kinds", "research-run"]) == 2
    assert "runs from none" in capsys.readouterr().err
