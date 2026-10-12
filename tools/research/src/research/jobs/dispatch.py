"""`research worker --dispatch`: stage 1's worker.  It claims `merge-check`, `lean-regen` and `audit-main` jobs from the Job queue
and runs each with today's `research run --on` onto one of the pods it was given, one job per pod at a time.

    research worker --dispatch --pods vy-coord-t1,vy-coord-t2 [--name rc] [--kinds merge-check,lean-regen,audit-main] [--once]

It is one worker, `dispatch:{name}`, whose `jobs:work` token the Console mints: --token-file, else `$JOBS_WORKER_TOKEN`, else
~/.research/jobs/dispatch-{name}.token, and never `research jobs`' `$JOBS_TOKEN` or ~/.research/jobs/token.  It
runs where `research run --on` with custody works today, the research coordinator's VM.  Each pod it holds is a slot:

    host        no claim while this host's runs root or workdir has less free space than `research run`'s floor
                ($RESEARCH_DISK_FLOOR_GB): the launch writes a worktree and records there.  Running jobs are still tended.
    measure     before a free pod asks for work, and again every PREFLIGHT_S, over ssh: AVX-512, memory, cores, free disk,
                the tools on its PATH, the seconds left on its lease, whether a `research` run is live on it, and whether its
                `pod_setup.sh` has reported done (SETUP_MARKER, which the script removes first and writes last).  What it finds is
                the claim's capabilities, so a job that needs AVX-512 only goes to a pod that has it, and a job whose
                `min_lease_s` (timeout_s plus the lease margin, set by `research jobs add`) the pod's lease can't cover isn't
                claimed for it (`check.py --record` sets no --timeout, so its launch doesn't extend the lease).  Its rules of its
                own are about whether a pod is free, not whether it's prepared: no job for a pod whose setup hasn't reported done
                (a chain that launched a check while `pod_setup.sh` still ran cost train TVC an hour), nor for one with a live run.
    preflight   whether a pod is prepared is decided in one place, `tools/verity/check/preflight.py` (#440), which `research run --on`
                runs on the machine before claiming the run: a `--tool check` run gets it, and a `lean-regen` asks for it with
                `--preflight check`.  A refusal is an error, retried on another pod, and the pod is benched for BENCH_S.
    launch      in a child process (`python -m research.jobs.dispatch launch SPEC`), so renewals never wait on it: fetch the
                commit into the dispatcher's own clone with forced refspecs, a worktree of it, then
                  merge-check  `check.py --record --on POD --base BASE [--verdicts-in PACK]` (the pack fetched from the store)
                  audit-main   `check.py --record --on POD --base EMPTY_TREE --no-cache`: a cold check of `main`'s tip with every
                               path counted as changed, so `merge_requires` sends the Lean agreement's inputs too
                  lean-regen   `research run --on POD --source . --cwd source` of `check.py --build --update PACKAGE...`, each
                               regenerated lean-audit.json copied to the run dir's records/ (custody publishes it)
                and the worktree is removed once the source is shipped.  The job's token is not in the child's environment.
    run         every POLL_S, `research fetch` of the run's records.  merge-check, audit-main: passed on rc 0 with check's result.json
                passed, failed on a result.json that says failed, else error.  lean-regen: passed on rc 0, failed on
                another exit code, error on a signal or a lost run.  A run past the job's timeout_s is stopped: error.
    lease       renewed every RENEW_S with the run's progress.  A `cancel` answer stops the run by its process group on the pod
                and completes the job; a lost lease (409) stops the run and drops the job, since another worker has it.

A failure is a verdict and never retried; an error is, by the Job queue, until max_attempts.  The slots are saved to
`dispatch-{name}.json` after each change (mode 600: it holds lease tokens), so a restarted dispatcher keeps tending the runs it
launched.  Launch logs are under `logs/`.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..pods.lease import LEASE_FILE, RUN_MARGIN_S
from ..repo import SLUG, same
from .client import TOKEN_ENV, WORKER_TOKEN_ENV, Unreachable

KINDS = ("merge-check", "lean-regen", "audit-main")
CHECK_KINDS = ("merge-check", "audit-main")
PROJECT = "verity"
STEP_S = 5
CLAIM_S = 30                # how often an idle dispatcher asks for work
RENEW_S = 60
POLL_S = 60
PREFLIGHT_S = 600
BENCH_S = 1800             # a pod that failed check's preflight sits out this long
SETUP_MARKER = "~/.research/pod-setup.json"      # tools/verity/check/pod_setup.sh writes it last, on the pod
LAUNCH_TIMEOUT_S = 1800
DEFAULT_TIMEOUT_S = 2 * 3600       # a cold check took 41 min, a Lean regeneration 30-40
AUDIT_TIMEOUT_S = 3 * 3600         # a cold check with the Lean agreement
WORKDIR = Path.home() / ".research" / "jobs"
SHA = re.compile(r"[0-9a-f]{40}")
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"      # git's empty tree: a diff from it names every path
LAUNCHED = re.compile(r"research: run (r\d{8}-\d{6}-[0-9a-f]{4,}) launched")
#: `research run --on`'s two ways of saying the machine failed a tool's preflight (#440)
PREFLIGHT_REFUSED = re.compile(r"\(PreflightFailed\)|failed the preflight|preflight .+ did not finish")


class DispatchError(Exception):
    """The dispatcher can't go on (a refused token, a worker the Job queue doesn't know): fix the setup and restart it."""


@dataclass
class Observation:
    state: str                   # running | passed | failed | error
    summary: str


@dataclass
class Slot:
    pod: str
    job: dict | None = None      # the claim's `job`
    attempt: int = 0
    lease: str | None = None
    claimed: float = 0.0
    renewed: float = 0.0
    observed: float = 0.0
    run_id: str | None = None
    launch: str | None = None    # the launch spec's path while launching
    outcome: list | None = None  # [outcome, summary], until the Job queue has it
    phase: str = ""
    caps: dict | None = None
    probed: float | None = None
    problem: str | None = None
    bench_until: float = 0.0     # after a failed preflight


def lease_needed(timeout_s: int) -> int:
    """The lease a job needs left on its pod: its timeout plus what the runner needs after it (publish, custody push)."""
    return int(timeout_s) + RUN_MARGIN_S


def describe(caps: dict) -> str:
    lease = f"lease {caps['lease_s'] / 3600:.1f} h" if caps.get("lease_s") is not None else "NO lease (a job with min_lease_s won't come)"
    return (f"{lease}, {caps.get('disk_gb')} GB free, {caps.get('mem_gb')} GB, {caps.get('cpus')} cpus"
            + (", avx512" if caps.get("avx512") else ", no avx512"))


def unavailable(caps: dict) -> str | None:
    """Why a pod takes no job now: its setup hasn't reported done, or a run is live on it.  Whether a pod is prepared is
    preflight.py's to say."""
    if not caps.get("setup_done"):
        return f"pod_setup.sh hasn't reported done ({SETUP_MARKER}): run it there, and it's used once it finishes"
    return f"{caps['busy']} live research run(s) on it" if caps.get("busy") else None


@dataclass
class Dispatcher:
    client: Any                                  # call(method, path, body) -> (status, body)
    runner: Any                                  # probe(pod), start(job, attempt, pod), poll(launch), observe(job, run_id), stop(run_id)
    pods: list[str]
    name: str = "rc"
    kinds: tuple[str, ...] = KINDS
    state_path: Path | None = None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    out: Callable[[str], None] = print
    slots: list[Slot] = field(default_factory=list)
    next_claim: float = 0.0
    turn: int = 0
    host_problem: str | None = None

    def __post_init__(self):
        saved = self._load()
        self.slots = [saved.get(p) or Slot(p) for p in self.pods]
        for s in saved.values():
            if s.pod not in self.pods and s.job:
                self.log(f"{s.pod}: job {s.job['id']} was running there, but the pod isn't in --pods; its lease will lapse")

    @property
    def worker(self) -> str:
        return f"dispatch:{self.name}"

    def log(self, msg: str) -> None:
        self.out(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.clock()))} {msg}")

    # -- the loop

    def run(self, once: bool = False) -> None:
        while True:
            self.step()
            if once:
                return
            self.sleep(STEP_S)

    def step(self) -> None:
        for s in self.slots:
            if s.job:
                self._tend(s)
        if self.clock() >= self.next_claim:
            self._claim_free()

    # -- claiming

    def _claim_free(self) -> None:
        free = [s for s in self.slots if not s.job]
        if not free:
            return
        was, self.host_problem = self.host_problem, self.runner.local_problem()
        if self.host_problem != was:
            self.log(f"claiming stopped: {self.host_problem}" if self.host_problem else "claiming again: the host has room")
        if self.host_problem:
            self.next_claim = self.clock() + CLAIM_S
            return
        free = free[self.turn % len(free):] + free[:self.turn % len(free)]   # each pod gets the first ask in turn
        self.turn += 1
        for s in free:
            if self.clock() < s.bench_until:
                continue
            if s.probed is None or self.clock() - s.probed >= PREFLIGHT_S:
                self._measure(s)
            if s.problem:
                continue
            try:
                status, body = self.client.call("POST", "/api/jobs/claim", {"worker": self.worker, "kinds": list(self.kinds),
                                                                           "slot": s.pod, "capabilities": s.caps})
            except Unreachable as e:
                self.log(f"claim: the Job queue is unreachable ({e})")
                break
            if status == 204:
                break
            if status != 200:
                raise DispatchError(f"claim refused: {status} {body.get('error')}")
            self._start(s, body)
        self.next_claim = self.clock() + CLAIM_S

    def _measure(self, s: Slot) -> None:
        s.probed = self.clock()
        was = s.problem
        try:
            s.caps = self.runner.probe(s.pod)
            s.problem = unavailable(s.caps)
        except Exception as e:  # noqa: BLE001 -- an unreachable or missing pod asks for nothing, and the dispatcher goes on
            s.caps, s.problem = None, f"can't be measured: {type(e).__name__}: {e}"
        if s.problem and s.problem != was:
            self.log(f"{s.pod}: skipped: {s.problem}")
        elif not s.problem:
            self.log(f"{s.pod}: {'free again' if was else 'free'}: {describe(s.caps)}")

    def _start(self, s: Slot, claim: dict) -> None:
        now = self.clock()
        s.job, s.attempt, s.lease = claim["job"], claim["attempt"], claim["lease_token"]
        s.claimed = s.renewed = s.observed = now
        s.run_id, s.outcome, s.phase = None, None, "launching"
        self.log(f"{s.pod}: job {s.job['id']} ({s.job['kind']}) attempt {s.attempt}: launching")
        need = lease_needed(s.job["payload"].get("timeout_s") or DEFAULT_TIMEOUT_S)
        if (s.caps or {}).get("lease_s") is not None and s.caps["lease_s"] < need and "min_lease_s" not in s.job["requirements"]:
            self.log(f"{s.pod}: job {s.job['id']} was added without min_lease_s, and the pod's lease ({s.caps['lease_s']} s) is "
                     f"under the {need} s it needs; `research pods extend {s.pod}` before it ends the run")
        try:
            s.launch = self.runner.start(s.job, s.attempt, s.pod)
        except Exception as e:  # noqa: BLE001
            s.launch, s.outcome = None, ["error", f"launch: {type(e).__name__}: {e}"]
        self._save()
        if s.outcome:
            self._report(s)

    # -- tending a running job

    def _tend(self, s: Slot) -> None:
        if not s.outcome:
            self._renew(s)
        if s.job and not s.outcome and s.launch:
            self._launching(s)
        if s.job and not s.outcome and s.run_id and self.clock() - s.observed >= POLL_S:
            self._observe(s)
        if s.job and s.outcome:
            self._report(s)

    def _renew(self, s: Slot) -> None:
        if self.clock() - s.renewed < RENEW_S:
            return
        try:
            progress = {k: v for k, v in (("pod", s.pod), ("run_id", s.run_id), ("step", s.phase)) if v}
            status, body = self.client.call("POST", f"/api/jobs/{s.job['id']}/renew", {"lease_token": s.lease, "progress": progress})
        except Unreachable as e:
            self.log(f"{s.pod}: job {s.job['id']}: renew unreachable ({e}); the lease has {s.job.get('lease_s', 900)} s")
            return
        if status == 200:
            s.renewed = self.clock()
            if body.get("cancel"):
                self.log(f"{s.pod}: job {s.job['id']}: cancelled; stopping it")
                self._stop(s)
                s.outcome = ["error", "stopped: the job was cancelled"]
            self._save()
            return
        self.log(f"{s.pod}: job {s.job['id']}: the lease is gone ({status} {body.get('error')}); stopping the run and dropping the job")
        self._stop(s)
        self._free(s)

    def _launching(self, s: Slot) -> None:
        done = self.runner.poll(s.launch)
        if done is None:
            if self.clock() - s.claimed > LAUNCH_TIMEOUT_S:
                self.runner.kill(s.launch)
                s.launch, s.outcome = None, ["error", f"the launch took over {LAUNCH_TIMEOUT_S} s"]
                self._save()
            return
        rc, run_id, tail = done
        s.launch = None
        if rc == 0 and run_id:
            s.run_id, s.phase, s.observed = run_id, "running", self.clock()
            self.log(f"{s.pod}: job {s.job['id']}: run {run_id}")
        elif PREFLIGHT_REFUSED.search(tail):
            s.bench_until = self.clock() + BENCH_S
            s.outcome = ["error", f"{s.pod} failed check's preflight, so another pod takes the job: {tail[-400:]}"]
            self.log(f"{s.pod}: failed check's preflight; benched for {BENCH_S} s")
        else:
            s.outcome = ["error", f"the launch exited {rc}" + (f" (run {run_id})" if run_id else "") + f": {tail[-400:]}"]
        self._save()

    def _observe(self, s: Slot) -> None:
        s.observed = self.clock()
        try:
            obs = self.runner.observe(s.job, s.run_id)
        except Exception as e:  # noqa: BLE001 -- fetch trouble is retried at the next poll, within the timeout
            obs = Observation("running", f"observing: {type(e).__name__}: {e}")
        timeout = int(s.job["payload"].get("timeout_s") or DEFAULT_TIMEOUT_S)
        if obs.state == "running" and self.clock() - s.claimed > timeout:
            self._stop(s)
            obs = Observation("error", f"stopped after the job's timeout_s ({timeout} s); last seen: {obs.summary}")
        if obs.state != "running":
            s.outcome = [obs.state, obs.summary]
        s.phase = obs.summary[:80] if obs.state == "running" else s.phase
        self._save()

    def _report(self, s: Slot) -> None:
        outcome, summary = s.outcome
        try:
            body = {"lease_token": s.lease, "outcome": outcome, "summary": str(summary)[:2000]}
            if s.run_id:                                   # the site takes run_id only as a string
                body["run_id"] = s.run_id
            status, body = self.client.call("POST", f"/api/jobs/{s.job['id']}/complete", body)
        except Unreachable as e:
            self.log(f"{s.pod}: job {s.job['id']}: complete unreachable ({e}); trying again")
            return
        if status == 200:
            self.log(f"{s.pod}: job {s.job['id']}: {outcome} -> {body.get('state')}" + (f" (run {s.run_id})" if s.run_id else "")
                     + (f": {summary[:200]}" if outcome != "passed" else ""))
        else:
            self.log(f"{s.pod}: job {s.job['id']}: complete answered {status} {body.get('error')}; dropping it")
        self._free(s)

    def _stop(self, s: Slot) -> None:
        try:
            if s.launch:
                self.runner.kill(s.launch)
            if s.run_id:
                self.runner.stop(s.run_id)
        except Exception as e:  # noqa: BLE001 -- the pod's lease and guard still end a run nobody stopped
            self.log(f"{s.pod}: stopping job {s.job['id']}: {type(e).__name__}: {e}")
        s.launch = None

    def _free(self, s: Slot) -> None:
        s.job = s.lease = s.run_id = s.launch = s.outcome = None
        s.phase, s.attempt = "", 0
        s.probed = None                          # a fresh preflight before the next job
        self._save()

    # -- the state file

    def _load(self) -> dict[str, Slot]:
        if not self.state_path or not self.state_path.exists():
            return {}
        doc = json.loads(self.state_path.read_text())
        slots = {d["pod"]: Slot(**d) for d in doc.get("slots", [])}
        for s in slots.values():
            s.probed, s.problem = None, None
        return slots

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"worker": self.worker, "slots": [asdict(s) for s in self.slots if s.job]}, f, indent=1)
        os.replace(tmp, self.state_path)


# ---- the real runner: ssh probes, `research run --on`, `research fetch` ------------------------------------------------------------------

PROBE = r"""
import glob, json, os, shutil, sys, time
root, lease_file, marker = sys.argv[1], sys.argv[2], os.path.expanduser(sys.argv[3])
os.environ["PATH"] = os.path.expanduser("~/.elan/bin") + ":" + os.path.expanduser("~/.cargo/bin") + ":/usr/local/bin:" + os.environ.get("PATH", "")
def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
busy = 0
for d in glob.glob(root + "/runs/*/"):
    def read(n):
        try:
            return json.load(open(d + n))
        except Exception:
            return {}
    st, la = read("status.json"), read("launch.json")
    if st.get("state") not in ("done", "failed", "cancelled") and (alive(st.get("pid")) or alive(la.get("runner_pid"))):
        busy += 1
mem = int(open("/proc/meminfo").read().split("MemTotal:")[1].split()[0]) * 1024
try:
    mem = min(mem, int(open("/sys/fs/cgroup/memory.max").read()))
except (OSError, ValueError):
    pass
probe = root
while not os.path.exists(probe):
    probe = os.path.dirname(probe)
flags = open("/proc/cpuinfo").read()
try:
    expires = int(open(lease_file).read().split()[0])
except (OSError, ValueError, IndexError):
    expires = int(os.environ["RESEARCH_LEASE_EXPIRES"]) if os.environ.get("RESEARCH_LEASE_EXPIRES", "").isdigit() else None
caps = {"avx512": " avx512f" in flags, "mem_gb": mem // 2**30, "cpus": os.cpu_count(),
        "disk_gb": shutil.disk_usage(probe).free // 2**30, "busy": busy,
        "tools": [t for t in ("uv", "lake", "cargo", "nvidia-smi") if shutil.which(t)]}
if expires is not None:
    caps["lease_s"] = max(0, expires - int(time.time()))
try:
    caps["setup_at"] = json.load(open(marker))["done_utc"]
    caps["setup_done"] = True
except (OSError, ValueError, KeyError, TypeError):
    caps["setup_done"] = False
print(json.dumps(caps))
"""


@dataclass
class ResearchRunner:
    repo: str = SLUG
    workdir: Path = WORKDIR
    python: str = sys.executable
    environ: dict | None = None
    url: str | None = None                   # where to fetch from; default https://github.com/{repo}.git
    children: dict = field(default_factory=dict, repr=False)

    def env(self) -> dict:
        env = dict(os.environ if self.environ is None else self.environ)
        env.pop(TOKEN_ENV, None)
        env.pop(WORKER_TOKEN_ENV, None)
        return env

    def local_problem(self) -> str | None:
        from .. import runs as R

        env = os.environ if self.environ is None else self.environ
        for d in (R.runs_root(env.get("RESEARCH_RUNS")), self.workdir):
            if why := R.disk_floor_refusal(Path(d), env):
                return why
        return None

    def _ssh(self, pod: str) -> tuple[list[str], dict]:
        from .. import remote

        m = remote.resolve_machine(pod, PROJECT)
        t = remote.pod_target(m)
        return remote.ssh_argv(t["host"], t["port"]), m

    def probe(self, pod: str) -> dict:
        from .. import remote

        ssh, m = self._ssh(pod)
        r = remote.ssh_run(ssh, [m.get("python", "python3"), "-c", PROBE, m["root"], LEASE_FILE, SETUP_MARKER], timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"probe exited {r.returncode}: {r.stderr.decode(errors='replace')[-300:]}")
        return json.loads(r.stdout.decode().strip().splitlines()[-1])

    def start(self, job: dict, attempt: int, pod: str) -> str:
        stem = f"job-{job['id']}-{attempt}"
        spec = self.workdir / "launch" / f"{stem}.json"
        spec.parent.mkdir(parents=True, exist_ok=True)
        (self.workdir / "logs").mkdir(parents=True, exist_ok=True)
        spec.with_suffix(".result").unlink(missing_ok=True)
        spec.write_text(json.dumps({"job": job, "attempt": attempt, "pod": pod, "repo": self.repo, "workdir": str(self.workdir),
                                    "python": self.python, "url": self.url or f"https://github.com/{self.repo}.git"}, indent=1))
        with open(self.workdir / "logs" / f"{stem}.log", "ab") as log:
            p = subprocess.Popen([self.python, "-m", "research.jobs.dispatch", "launch", str(spec)], stdout=log, stderr=subprocess.STDOUT,
                                 env=self.env(), start_new_session=True)
        spec.with_suffix(".pid").write_text(str(p.pid))
        self.children[str(spec)] = p
        return str(spec)

    def poll(self, launch: str) -> tuple[int, str | None, str] | None:
        spec = Path(launch)
        p = self.children.get(launch)
        if p is not None and p.poll() is None:
            return None
        res = spec.with_suffix(".result")
        if res.exists():
            d = json.loads(res.read_text())
            return d["rc"], d.get("run_id"), d.get("tail", "")
        if p is None and launcher_pid(spec) is not None:
            return None                          # a launch a previous dispatcher started, still running
        return (p.returncode if p is not None else -1), None, "the launcher left no result"

    def kill(self, launch: str) -> None:
        if (pid := launcher_pid(Path(launch))) is not None:
            os.killpg(pid, signal.SIGTERM)       # its own session: the launcher, check.py and `research run`

    def _run_dir(self, run_id: str) -> Path:
        from .. import runs as R

        return R.resolve(R.runs_root(), run_id)

    def observe(self, job: dict, run_id: str) -> Observation:
        from .. import remote
        from .. import runs as R

        d = self._run_dir(run_id)
        rec = remote.fetch(d)
        if rec.get("last_fetch_error"):
            return Observation("running", f"machine unreachable (rc {rec['last_fetch_error'].get('rc')})")
        return classify(job["kind"], R.summarize(d))

    def stop(self, run_id: str) -> None:
        from .. import remote

        d = self._run_dir(run_id)
        remote.fetch(d)
        rec = json.loads((d / "remote.json").read_text())
        pgid = (json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}).get("pgid")
        if pgid:
            remote.ssh_run(remote.ssh_argv(rec["host"], rec["port"]), ["kill", "-TERM", f"-{int(pgid)}"], timeout=60)


def launcher_pid(spec: Path) -> int | None:
    """The pid of the launch child for `spec`, while it lives: a pid whose command line still names the spec, so a pid the
    system gave to another process since is never signalled."""
    try:
        pid = int(spec.with_suffix(".pid").read_text())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError):
        return None
    return pid if str(spec).encode() in cmdline else None


def classify(kind: str, s: dict) -> Observation:
    """A run's records as a job outcome.  Only a verdict is `failed`: check's own result.json saying so, or the Lean audit's exit
    code; a run that died, was killed or left no result is an `error`, which the Job queue retries."""
    state, rc = s.get("state"), s.get("rc")
    what = f"{state} rc={rc}" + (f" signal={s['signal']}" if s.get("signal") else "") + (f" class={s['class']}" if s.get("class") else "")
    if state not in ("done", "failed", "cancelled", "refused"):
        return Observation("running", str(state))
    if kind in CHECK_KINDS:
        if rc == 0 and s.get("validation") == "passed":
            return Observation("passed", f"check passed: {what}")
        if s.get("validation") == "failed" and s.get("result_valid"):
            return Observation("failed", f"check failed: {what}")
        return Observation("error", f"no verdict: {what}, result {s.get('result')}")
    if rc == 0 and state == "done":
        return Observation("passed", what)
    if isinstance(rc, int) and rc > 0 and not s.get("signal") and state in ("done", "failed"):
        return Observation("failed", what)
    return Observation("error", what)


# ---- the launch child ---------------------------------------------------------------------------------------------------------------

def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def fetch_commits(clone: Path, url: str, shas: list[str]) -> None:
    """Every sha into the dispatcher's clone, forced: by sha first (GitHub serves reachable commits), else every branch and PR head."""
    if not (clone / "HEAD").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--bare", "--quiet", url, str(clone)], check=True, capture_output=True, text=True)
    missing = [x for x in shas if _git(clone, "cat-file", "-e", f"{x}^{{commit}}", check=False).returncode]
    if missing and _git(clone, "fetch", "--force", "--quiet", url, *missing, check=False).returncode:
        _git(clone, "fetch", "--force", "--quiet", url, "+refs/heads/*:refs/remotes/origin/*", "+refs/pull/*/head:refs/remotes/pr/*")
    for x in shas:
        if _git(clone, "cat-file", "-e", f"{x}^{{commit}}", check=False).returncode:
            raise RuntimeError(f"commit {x} is not on {url}")


def command(job: dict, pod: str, python: str, pack: Path | None, preflight: bool = True) -> list[str]:
    """What the launch child runs in the commit's worktree.  `preflight`: the launcher's `research run` knows `--preflight` (#440)."""
    p = job["payload"]
    if job["kind"] == "merge-check":
        return [python, "tools/verity/check/check.py", "--record", "--on", pod, "--base", p["base"],
                *(["--verdicts-in", str(pack)] if pack else [])]
    if job["kind"] == "audit-main":
        return [python, "tools/verity/check/check.py", "--record", "--on", pod, "--base", EMPTY_TREE, "--no-cache"]
    if job["kind"] == "lean-regen":
        pkgs = " ".join(shlex.quote(x) for x in p["packages"])
        script = (f"python3 tools/verity/lean/check.py --build --update {pkgs}; rc=$?; for p in {pkgs}; do "
                  f"mkdir -p \"$RESEARCH_RUN_DIR/records/$p\" && cp \"$p/lean-audit.json\" \"$RESEARCH_RUN_DIR/records/$p/\"; done; exit $rc")
        return [python, "-m", "research", "run", "--on", pod, "--project", PROJECT, "--source", ".", "--cwd", "source",
                *(["--preflight", "check"] if preflight else []), "--", "sh", "-c", script]
    raise ValueError(f"this dispatcher doesn't run {job['kind']} jobs")


def validate(job: dict, repo: str) -> None:
    p = job["payload"]
    if not same(p.get("repo", repo), repo):
        raise ValueError(f"the job is for {p.get('repo')}, and this dispatcher checks {repo}")
    if not SHA.fullmatch(str(p.get("commit", ""))):
        raise ValueError("payload.commit must be a full commit sha")
    if job["kind"] not in KINDS:
        raise ValueError(f"this dispatcher doesn't run {job['kind']} jobs")
    if job["kind"] == "merge-check" and not SHA.fullmatch(str(p.get("base", ""))):
        raise ValueError("payload.base must be a full commit sha")
    if job["kind"] == "lean-regen" and not (p.get("packages") and all(re.fullmatch(r"[\w./-]+", x) and ".." not in x for x in p["packages"])):
        raise ValueError("payload.packages must name package directories")


def launch(spec_path: Path) -> dict:
    """The child: the commit's worktree, the launch, the run id from its output.  Returns what `.result` holds."""
    spec = json.loads(spec_path.read_text())
    job, pod, workdir = spec["job"], spec["pod"], Path(spec["workdir"])
    validate(job, spec["repo"])
    p = job["payload"]
    clone = workdir / "verity.git"
    with open(workdir / "clone.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fetch_commits(clone, spec["url"], [p["commit"], *([p["base"]] if p.get("base") else [])])
        wt = workdir / "worktrees" / spec_path.stem
        _git(clone, "worktree", "prune")
        if wt.exists():
            _git(clone, "worktree", "remove", "--force", str(wt), check=False)
        _git(clone, "worktree", "add", "--detach", "--quiet", str(wt), p["commit"])
    try:
        pack = None
        if p.get("verdict_pack"):
            from ..store.local import LocalStore

            pack = Path(LocalStore().fetch(p["verdict_pack"]))
        from .. import remote

        cmd = command(job, pod, spec["python"], pack, preflight=hasattr(remote, "run_preflight"))
        print(f"dispatch: job {job['id']} in {wt}: {' '.join(shlex.quote(c) for c in cmd[:8])} ...", flush=True)
        proc = subprocess.Popen(cmd, cwd=wt, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        run_id, tail = None, []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail = (tail + [line])[-20:]
            if m := LAUNCHED.search(line):
                run_id = m.group(1)
        rc = proc.wait()
    finally:
        with open(workdir / "clone.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _git(clone, "worktree", "remove", "--force", str(wt), check=False)
    return {"rc": rc, "run_id": run_id, "tail": "".join(tail)}


def _launch_main(argv: list[str]) -> int:
    spec = Path(argv[0])
    try:
        out = launch(spec)
    except Exception as e:  # noqa: BLE001 -- whatever stops the launch is the job's error summary
        out = {"rc": -1, "run_id": None, "tail": f"{type(e).__name__}: {e}"}
    tmp = spec.with_suffix(".result.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, spec.with_suffix(".result"))
    print(f"dispatch: launch done: {out['rc']} {out['run_id']}", flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["launch"]:
        sys.exit(_launch_main(sys.argv[2:]))
    sys.exit(2)
