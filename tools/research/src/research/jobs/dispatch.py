"""`research worker --dispatch`: step 1's worker.  It claims `merge-check`, `lean-regen` and `audit-main` jobs from the Job queue
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
                (a chain that launched a check while `pod_setup.sh` still ran cost a candidate an hour), nor for one with a live run.
    preflight   whether a pod is prepared is decided in one place, `tools/check/preflight.py`, which `research run --on`
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

`research-run` is a lead's own `research run` line, run by the control pod with the keys it holds:

    research worker --dispatch --kinds research-run [--slots 4] --name control

    payload     {repo, commit, command, lane}, nothing else: `command` is a `research run ... -- ARGV` line, `lane` the lane that
                sent it.  A payload `run_plan` refuses completes `failed` with why, and nothing is launched.
    allowlist   each option before `--` is one of `run_allowed`'s, with a value of its form (`runline`), and anything else is refused:
                CPU machines only (`--on auto`, the default, or `--on vy-nebius-cpu-N`), and never a path or key of the pod
                (`--send`, `--source`, `--env-host`, `--runs-dir`, `--store-read`), the run's machine-side settings (`--id`,
                `--claimed`, custody) or an auditor device (`remote.restricted`).  `--gpus 0` is always added, so a kind the
                commit registers can't book a GPU.
    slots       --slots virtual ones, `research-run-N`: no pod is measured, and the claim sends no capabilities.
    launch      `PYTHON -P -s -m research run --source . --project verity OPTS --gpus 0 -- ARGV` in the commit's worktree, with
                the dispatcher's own tree's research (`run_env`): nothing of the commit runs on the pod.
    observe     every POLL_S the run's records and the last TAIL_BYTES of its stdout.log, following a run that left its line for
                its late twin; they go out as `progress` {run_id, machine, state, rc, tail} at each renewal, and at the end as
                the summary: its `research status` row, the tail and a result.json of at most RESULT_BYTES.
    lease       a `cancel` answer cancels the run as `research cancel` does (SIGTERM to its process group on its machine); a
                lost lease leaves it running and stops tending it.  It is stopped only past its own --timeout (else
                RUN_TIMEOUT_MAX_S), its pool's wait, the launch and the runner's margin together.
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
from . import runline
from .client import TOKEN_ENV, WORKER_TOKEN_ENV, Unreachable

POD_KINDS = ("merge-check", "lean-regen", "audit-main")
RUN_KIND = "research-run"
KINDS = (*POD_KINDS, RUN_KIND)
CHECK_KINDS = ("merge-check", "audit-main")
PROJECT = "verity"
STEP_S = 5
CLAIM_S = 30                # how often an idle dispatcher asks for work
RENEW_S = 60
POLL_S = 60
PREFLIGHT_S = 600
BENCH_S = 1800             # a pod that failed check's preflight sits out this long
SETUP_MARKER = "~/.research/pod-setup.json"      # tools/check/pod_setup.sh writes it last, on the pod
LAUNCH_TIMEOUT_S = 1800
DEFAULT_TIMEOUT_S = 2 * 3600       # a cold check took 41 min, a Lean regeneration 30-40
AUDIT_TIMEOUT_S = 3 * 3600         # a cold check with the Lean agreement
WORKDIR = Path.home() / ".research" / "jobs"
SHA = re.compile(r"[0-9a-f]{40}")
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"      # git's empty tree: a diff from it names every path
LAUNCHED = re.compile(r"research: run (r\d{8}-\d{6}-[0-9a-f]{4,}) launched")
#: `research run --on`'s two ways of saying the machine failed a tool's preflight
PREFLIGHT_REFUSED = re.compile(r"\(PreflightFailed\)|failed the preflight|preflight .+ did not finish")

# ---- research-run: a lead's `research run` line, checked against the first week's allowlist (CPU machines only) ---------------

RUN_SLOTS = 4
RUN_SLOT = "research-run-"                 # a virtual slot's name, numbered from 1
PAYLOAD = frozenset({"repo", "commit", "command", "lane"})
LANE = re.compile(r"[a-z][a-z0-9-]{0,39}")
COMMAND_MAX = 16384                         # characters of the line
TAIL_BYTES = 6000                           # of stdout.log, in progress and the summary
RESULT_BYTES = 2000                         # a result.json larger than this is named, not copied, in the summary
SUMMARY_CHARS = 9000                        # the row, the tail and the result.json, under the site's 64 KiB body once escaped
#: the longest --timeout: what the run's custody key lives without --custody-ttl (store.custody.DEFAULT_TTL; a test holds the
#: two equal), which the list doesn't hold, and so what `research run` allows; a run that gives none is stopped as though it
#: gave this
RUN_TIMEOUT_MAX_S = 6 * 3600
RUN_CPUS_MAX, RUN_MEM_GB_MAX = 64, 512
CPU_MACHINE = re.compile(r"vy-nebius-cpu-[1-9][0-9]*")
#: the queue's kinds whose shape takes no GPU (infra/cluster/kinds/)
CPU_KINDS = frozenset({"agent-test", "lean-audit"})
WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _number(read: Callable[[str], float], low: float, most: float, unit: str = "") -> Callable[[str], str | None]:
    """A value `research run` reads with ``read``, above ``low`` and at most ``most``: NaN and an unreadable value are refused."""
    def why(value: str) -> str | None:
        try:
            n = read(value)
        except ValueError:
            return "which `research run` can't read"
        return None if low < n <= most else f"which isn't above {low:g} and at most {most:g}{unit}"
    return why


def _timeout_s(value: str) -> float:
    from ..remote import timeout_seconds

    return timeout_seconds(value)


def _named(value: str) -> str:
    return "a GPU run, or a run that needs " + value + ", goes through a Cursor worker for now"


def _on(restricted: Callable[[], set[str]]) -> Callable[[str], str | None]:
    def why(value: str) -> str | None:
        if value != "auto" and not CPU_MACHINE.fullmatch(value):
            return f"which isn't auto or a vy-nebius-cpu-N machine: {_named('another machine')}"
        if value != "auto":
            try:
                devices = restricted()
            except Exception as e:  # noqa: BLE001 -- unknown is refused
                return f"and whether it is an auditor device can't be read ({type(e).__name__}: {e})"
            if value in devices:
                return "which is an auditor device, which no job reaches"
        return None
    return why


def run_allowed(restricted: Callable[[], set[str]]) -> dict[str, runline.Form]:
    """Each option a research-run line may carry, and the form of its value; `--on`'s check reads the auditor devices from
    ``restricted``."""
    from ..pods.nebius.slot_only import POOLS

    return {
        "--on": _on(restricted),
        "--pool": frozenset(POOLS),
        "--cpus": _number(int, 0, RUN_CPUS_MAX),
        "--mem-gb": _number(float, 0, RUN_MEM_GB_MAX, " GB"),
        "--timeout": _number(_timeout_s, 0, RUN_TIMEOUT_MAX_S, " s"),
        "--max-min": _number(int, 0, RUN_TIMEOUT_MAX_S // 60, " min"),
        "--cwd": frozenset({"source", "clone"}),
        "--kind": frozenset(CPU_KINDS),
        "--stage": WORD,
        "--expect": re.compile(r"^[A-Za-z0-9_.-]+="),
        "--require-result": None,
        "--declared-output": re.compile(r"^."),
        "--question": re.compile(r"^."),
        "--stay": None,
    }


RUN_REPEATABLE = frozenset({"--expect", "--declared-output"})


def _restricted() -> set[str]:
    from .. import remote

    return remote.restricted(remote.load_machines())


@dataclass(frozen=True)
class RunPlan:
    opts: list[str]            # the checked options, `--on` first and `--gpus 0` last
    argv: list[str]            # the workload, after `--`
    lane: str
    limit_s: float             # the dispatcher stops the run past this, counted from its claim


def run_plan(payload: Any, repo: str, restricted: Callable[[], set[str]] = _restricted) -> RunPlan:
    """What a research-run job launches, or ValueError saying why it is refused (module doc)."""
    if not isinstance(payload, dict) or set(payload) != PAYLOAD:
        got = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"the payload must hold exactly {', '.join(sorted(PAYLOAD))}, and it holds {got}")
    if MOVED.get(payload["repo"], payload["repo"]) != MOVED.get(repo, repo):
        raise ValueError(f"the job is for {payload['repo']!r}, and this dispatcher runs {repo}")
    if not isinstance(payload["commit"], str) or not SHA.fullmatch(payload["commit"]):
        raise ValueError("payload.commit must be a full commit sha")
    if not isinstance(payload["lane"], str) or not LANE.fullmatch(payload["lane"]):
        raise ValueError(f"payload.lane must be a lane's name ({LANE.pattern})")
    line = payload["command"]
    if not isinstance(line, str) or not 0 < len(line) <= COMMAND_MAX or "\0" in line:
        raise ValueError(f"payload.command must be a `research run` line of 1 to {COMMAND_MAX} characters, without NUL")
    try:
        tokens = shlex.split(line)
    except ValueError as e:
        raise ValueError(f"payload.command doesn't split as a shell line ({e})") from None
    if tokens[:2] != ["research", "run"] or "--" not in tokens or not tokens[tokens.index("--") + 1:]:
        raise ValueError("payload.command must be `research run [OPTIONS] -- COMMAND...`")
    i = tokens.index("--")
    try:
        opts = runline.options(tokens[2:i], run_allowed(restricted), repeatable=RUN_REPEATABLE,
                               unknown=lambda name: f"takes {name}, which a research-run job may not carry: {_named(name)}")
    except runline.Refused as e:
        raise ValueError(f"its `research run` {e}") from None
    if not any(name == "--on" for name, _ in opts):
        opts.insert(0, ("--on", "auto"))
    given = dict(opts)
    run_s = max(_timeout_s(given.get("--timeout", "0")), int(given.get("--max-min", "0")) * 60) or RUN_TIMEOUT_MAX_S
    from ..pods.nebius.slot_only import POOL_WAIT_S

    return RunPlan([x for name, value in opts for x in (name,) + ((value,) if value is not None else ())] + ["--gpus", "0"],
                   tokens[i + 1:], payload["lane"], run_s + POOL_WAIT_S + LAUNCH_TIMEOUT_S + RUN_MARGIN_S)


def run_env(environ: dict[str, str], tree: Path, lane: str) -> dict[str, str]:
    """The `research run` launcher's environment: ``environ`` (the launch child's, which is the dispatcher's and holds the pod's
    keys, which the launcher needs to mint the run's custody) without the Job queue's tokens, CURSOR_CONVERSATION_ID or any
    PYTHON* variable, and with PYTHONPATH the research of ``tree`` (the dispatcher's own tree) alone, RESEARCH_LAUNCHER_TREE
    that tree, RESEARCH_AT_CHECKOUT=1 (that tree's research, never a cached main's) and RESEARCH_LANE and RESEARCH_AGENT_URL
    the lane. The workload gets none of it: `research run` sends no local variable to the machine but what --env names."""
    drop = {TOKEN_ENV, WORKER_TOKEN_ENV, "CURSOR_CONVERSATION_ID"}
    env = {k: v for k, v in environ.items() if k not in drop and not k.startswith("PYTHON")}
    env.update(PYTHONPATH=str(tree / "tools" / "research" / "src"), RESEARCH_LAUNCHER_TREE=str(tree), RESEARCH_AT_CHECKOUT="1",
               RESEARCH_LANE=lane, RESEARCH_AGENT_URL=lane)
    return env


def own_tree() -> Path | None:
    """The checkout this module runs from, which a research-run launch runs `research` from; None when it isn't in one."""
    here = Path(__file__).resolve()
    tree = here.parents[5]
    return tree if (tree / "tools" / "research" / "src" / "research" / "jobs" / "dispatch.py").resolve() == here else None


class DispatchError(Exception):
    """The dispatcher can't go on (a refused token, a worker the Job queue doesn't know): fix the setup and restart it."""


@dataclass
class Observation:
    state: str                   # running | passed | failed | error
    summary: str
    run_id: str | None = None    # research-run: the run now, its late twin once it has moved there
    progress: dict | None = None  # research-run: {run_id, machine, state, rc, tail}


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
    capabilities: dict | None = None
    probed: float | None = None
    problem: str | None = None
    bench_until: float = 0.0     # after a failed preflight
    virtual: bool = False        # a research-run slot: no pod behind it
    progress: dict | None = None  # research-run: what the last observation saw, sent at each renewal
    limit_s: float | None = None  # research-run: run_plan's limit_s


def lease_needed(timeout_s: int) -> int:
    """The lease a job needs left on its pod: its timeout plus what the runner needs after it (publish, custody push)."""
    return int(timeout_s) + RUN_MARGIN_S


def describe(capabilities: dict) -> str:
    lease = f"lease {capabilities['lease_s'] / 3600:.1f} h" if capabilities.get("lease_s") is not None else "NO lease (a job with min_lease_s won't come)"
    return (f"{lease}, {capabilities.get('disk_gb')} GB free, {capabilities.get('mem_gb')} GB, {capabilities.get('cpus')} cpus"
            + (", avx512" if capabilities.get("avx512") else ", no avx512"))


def unavailable(capabilities: dict) -> str | None:
    """Why a pod takes no job now: its setup hasn't reported done, or a run is live on it.  Whether a pod is prepared is
    preflight.py's to say."""
    if not capabilities.get("setup_done"):
        return f"pod_setup.sh hasn't reported done ({SETUP_MARKER}): run it there, and it's used once it finishes"
    return f"{capabilities['busy']} live research run(s) on it" if capabilities.get("busy") else None


@dataclass
class Dispatcher:
    client: Any                                  # call(method, path, body) -> (status, body)
    runner: Any                                  # probe(pod), start(job, attempt, pod), poll(launch), observe(job, run_id), stop(run_id),
                                                 # research-run's cancel(run_id) and restricted()
    pods: list[str]
    name: str = "rc"
    kinds: tuple[str, ...] = POD_KINDS
    run_slots: int = 0                           # research-run's virtual slots, when it is among the kinds
    repo: str = "compute-verification/verity"
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
        virtual = [f"{RUN_SLOT}{i}" for i in range(1, self.run_slots + 1)] if RUN_KIND in self.kinds else []
        self.slots = [saved.get(p) or Slot(p) for p in self.pods] + [saved.get(v) or Slot(v, virtual=True) for v in virtual]
        for s in saved.values():
            if s.pod not in self.pods + virtual and s.job:
                self.log(f"{s.pod}: job {s.job['id']} was running there, but the slot isn't in --pods or --slots; its lease will lapse")

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
        empty = set()                            # pods, virtual slots: a kind of slot whose claim came back empty this pass
        for s in free:
            if s.virtual in empty or self.clock() < s.bench_until:
                continue
            kinds = [k for k in self.kinds if (k == RUN_KIND) == s.virtual]
            if not s.virtual and (s.probed is None or self.clock() - s.probed >= PREFLIGHT_S):
                self._measure(s)
            if s.problem or not kinds:
                continue
            ask = {"worker": self.worker, "kinds": kinds, "slot": s.pod}
            if not s.virtual:
                ask["capabilities"] = s.capabilities
            try:
                status, body = self.client.call("POST", "/api/jobs/claim", ask)
            except Unreachable as e:
                self.log(f"claim: the Job queue is unreachable ({e})")
                break
            if status == 204:
                empty.add(s.virtual)
                continue
            if status != 200:
                raise DispatchError(f"claim refused: {status} {body.get('error')}")
            self._start(s, body)
        self.next_claim = self.clock() + CLAIM_S

    def _measure(self, s: Slot) -> None:
        s.probed = self.clock()
        was = s.problem
        try:
            s.capabilities = self.runner.probe(s.pod)
            s.problem = unavailable(s.capabilities)
        except Exception as e:  # noqa: BLE001 -- an unreachable or missing pod asks for nothing, and the dispatcher goes on
            s.capabilities, s.problem = None, f"can't be measured: {type(e).__name__}: {e}"
        if s.problem and s.problem != was:
            self.log(f"{s.pod}: skipped: {s.problem}")
        elif not s.problem:
            self.log(f"{s.pod}: {'free again' if was else 'free'}: {describe(s.capabilities)}")

    def _start(self, s: Slot, claim: dict) -> None:
        now = self.clock()
        s.job, s.attempt, s.lease = claim["job"], claim["attempt"], claim["lease_token"]
        s.claimed = s.renewed = s.observed = now
        s.run_id, s.outcome, s.phase = None, None, "launching"
        self.log(f"{s.pod}: job {s.job['id']} ({s.job['kind']}) attempt {s.attempt}: launching")
        if s.job["kind"] == RUN_KIND:
            try:
                s.limit_s = run_plan(s.job["payload"], self.repo, self.runner.restricted).limit_s
            except ValueError as e:      # a verdict on the payload: no attempt would launch it
                s.outcome = ["failed", f"refused: {e}"]
                self._save()
                self._report(s)
                return
            s.progress = {"run_id": None, "machine": None, "state": "launching", "rc": None, "tail": ""}
        else:
            need = lease_needed(s.job["payload"].get("timeout_s") or DEFAULT_TIMEOUT_S)
            if (s.capabilities or {}).get("lease_s") is not None and s.capabilities["lease_s"] < need and "min_lease_s" not in s.job["requirements"]:
                self.log(f"{s.pod}: job {s.job['id']} was added without min_lease_s, and the pod's lease ({s.capabilities['lease_s']} s) is "
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
            progress = (s.progress if s.job["kind"] == RUN_KIND else
                        {k: v for k, v in (("pod", s.pod), ("run_id", s.run_id), ("step", s.phase)) if v})
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
        if s.job["kind"] == RUN_KIND:
            what = f"run {s.run_id}" if s.run_id else f"its launch ({s.launch})" if s.launch else "nothing launched"
            self.log(f"{s.pod}: job {s.job['id']}: the lease is gone ({status} {body.get('error')}); {what} is left running, "
                     "and no longer tended here")
        else:
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
            if s.progress is not None:
                s.progress = {**s.progress, "run_id": run_id, "state": "submitted"}
            self.log(f"{s.pod}: job {s.job['id']}: run {run_id}")
        elif not s.virtual and PREFLIGHT_REFUSED.search(tail):
            s.bench_until = self.clock() + BENCH_S
            s.outcome = ["error", f"{s.pod} failed check's preflight, so another pod takes the job: {tail[-400:]}"]
            self.log(f"{s.pod}: failed check's preflight; benched for {BENCH_S} s")
        elif s.virtual:                          # a verdict: another attempt would launch the same line again
            s.progress = {**(s.progress or {}), "run_id": run_id, "state": "launch failed", "rc": rc, "tail": tail[-TAIL_BYTES:]}
            s.outcome = ["failed", f"`research run` exited {rc}" + (f" (run {run_id})" if run_id else "") + f":\n{tail[-TAIL_BYTES:]}"]
        else:
            s.outcome = ["error", f"the launch exited {rc}" + (f" (run {run_id})" if run_id else "") + f": {tail[-400:]}"]
        self._save()

    def _observe(self, s: Slot) -> None:
        s.observed = self.clock()
        try:
            obs = self.runner.observe(s.job, s.run_id)
        except Exception as e:  # noqa: BLE001 -- fetch trouble is retried at the next poll, within the timeout
            obs = Observation("running", f"observing: {type(e).__name__}: {e}")
        if obs.run_id and obs.run_id != s.run_id:
            self.log(f"{s.pod}: job {s.job['id']}: run {s.run_id} left its line unstarted; following its late twin {obs.run_id}")
            s.run_id = obs.run_id
        if obs.progress is not None:
            s.progress = obs.progress
        if s.limit_s is not None:
            timeout, what = s.limit_s, "its own --timeout, its pool's wait, the launch and the runner's margin"
        else:
            timeout, what = int(s.job["payload"].get("timeout_s") or DEFAULT_TIMEOUT_S), "the job's timeout_s"
        if obs.state == "running" and self.clock() - s.claimed > timeout:
            self._stop(s)
            obs = Observation("error", f"stopped after {what} ({timeout:g} s); last seen: {obs.summary}")
        if obs.state != "running":
            s.outcome = [obs.state, obs.summary]
        s.phase = obs.summary[:80] if obs.state == "running" else s.phase
        self._save()

    def _report(self, s: Slot) -> None:
        outcome, summary = s.outcome
        try:
            body = {"lease_token": s.lease, "outcome": outcome, "summary": str(summary)[:SUMMARY_CHARS if s.job["kind"] == RUN_KIND else 2000]}
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
            if s.run_id and s.job["kind"] == RUN_KIND:
                self.runner.cancel(s.run_id)
            elif s.run_id:
                self.runner.stop(s.run_id)
        except Exception as e:  # noqa: BLE001 -- the pod's lease and guard still end a run nobody stopped
            self.log(f"{s.pod}: stopping job {s.job['id']}: {type(e).__name__}: {e}")
        s.launch = None

    def _free(self, s: Slot) -> None:
        s.job = s.lease = s.run_id = s.launch = s.outcome = s.progress = s.limit_s = None
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
capabilities = {"avx512": " avx512f" in flags, "mem_gb": mem // 2**30, "cpus": os.cpu_count(),
        "disk_gb": shutil.disk_usage(probe).free // 2**30, "busy": busy,
        "tools": [t for t in ("uv", "lake", "cargo", "nvidia-smi") if shutil.which(t)]}
if expires is not None:
    capabilities["lease_s"] = max(0, expires - int(time.time()))
try:
    capabilities["setup_at"] = json.load(open(marker))["done_utc"]
    capabilities["setup_done"] = True
except (OSError, ValueError, KeyError, TypeError):
    capabilities["setup_done"] = False
print(json.dumps(capabilities))
"""


@dataclass
class ResearchRunner:
    repo: str = "compute-verification/verity"
    workdir: Path = WORKDIR
    python: str = sys.executable
    environ: dict | None = None
    url: str | None = None                   # where to fetch from; default https://github.com/{repo}.git
    launcher_tree: Path | None = field(default_factory=own_tree)   # research-run's launcher runs this tree's research
    children: dict = field(default_factory=dict, repr=False)

    def restricted(self) -> set[str]:
        return _restricted()

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
                                    "python": self.python, "url": self.url or f"https://github.com/{self.repo}.git",
                                    "launcher_tree": str(self.launcher_tree) if self.launcher_tree else None}, indent=1))
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
        if job["kind"] == RUN_KIND:
            from ..cli import _moved_to

            return observe_run(d, remote.fetch, stdout_tail, _moved_to)
        rec = remote.fetch(d)
        if rec.get("last_fetch_error"):
            return Observation("running", f"machine unreachable (rc {rec['last_fetch_error'].get('rc')})")
        return classify(job["kind"], R.summarize(d))

    def cancel(self, run_id: str) -> None:
        """The run cancelled as `research cancel` does it: SIGTERM to its process group on its machine, or withdrawn while it
        waits for its hold."""
        from .. import remote

        remote.cancel(self._run_dir(run_id), reason="its research-run job was cancelled", by="research worker --dispatch")

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


def stdout_tail(run_dir: Path) -> bytes:
    """The last TAIL_BYTES of the run's stdout.log on its machine; nothing while it has none, and why when it can't be read."""
    from .. import remote

    rec = remote._read_json(run_dir / "remote.json") or {}
    if not rec.get("remote_run_dir"):
        return b""
    try:
        r = remote.ssh_run(remote.ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root"),
                           ["tail", "-c", str(TAIL_BYTES), f"{rec['remote_run_dir']}/stdout.log"], timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"(stdout.log unread: {type(e).__name__}: {e})".encode()
    return r.stdout[-TAIL_BYTES:] if r.returncode == 0 else b""


#: a twin's twin is never launched (only the first's asker launches one), so a run is followed at most this far
TWIN_HOPS = 2


def observe_run(run_dir: Path, fetch: Callable[[Path], dict], read_tail: Callable[[Path], bytes],
                moved: Callable[[Path], Path | None]) -> Observation:
    """A research-run's run as its machine has it now (`fetch`), followed to its late twin when it left its line unstarted
    (`moved`, as `research run --wait` follows it), with its progress {run_id, machine, state, rc, tail}; once it has ended,
    its outcome and summary: its `research status` row, the tail and its result.json when that is at most RESULT_BYTES."""
    from .. import asker
    from .. import runs as R
    from ..cli import _fmt

    for _ in range(TWIN_HOPS + 1):
        rec = fetch(run_dir)
        if rec.get("last_fetch_error"):
            return Observation("running", f"machine unreachable (rc {rec['last_fetch_error'].get('rc')})", run_dir.name)
        s = R.summarize(run_dir)
        if not (s["state"] in R.ENDED and not asker.started(s) and (twin := moved(run_dir))):
            break
        run_dir = twin
    raw = read_tail(run_dir)[-TAIL_BYTES:]
    text = raw.decode(errors="replace")
    progress = {"run_id": run_dir.name, "machine": (s.get("remote") or {}).get("machine"), "state": s["state"], "rc": s["rc"],
                "tail": text}
    obs = classify(RUN_KIND, s)
    if obs.state == "running":
        return Observation("running", str(s["state"]), run_dir.name, progress)
    summary = f"{_fmt(s)}\n--- the last {len(raw)} bytes of stdout.log ---\n{text}"
    if (result := run_dir / "result.json").is_file():
        size = result.stat().st_size
        summary += (f"\n--- result.json ---\n{result.read_bytes().decode(errors='replace')}" if size <= RESULT_BYTES else
                    f"\n--- result.json: {size} bytes, over the {RESULT_BYTES} a summary holds ---")
    return Observation(obs.state, summary, run_dir.name, progress)


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
    """What the launch child runs in the commit's worktree.  `preflight`: the launcher's `research run` knows `--preflight`."""
    p = job["payload"]
    if job["kind"] == "merge-check":
        return [python, "tools/check/check.py", "--record", "--on", pod, "--base", p["base"],
                *(["--verdicts-in", str(pack)] if pack else [])]
    if job["kind"] == "audit-main":
        return [python, "tools/check/check.py", "--record", "--on", pod, "--base", EMPTY_TREE, "--no-cache"]
    if job["kind"] == "lean-regen":
        pkgs = " ".join(shlex.quote(x) for x in p["packages"])
        script = (f"python3 tools/verity/lean/check.py --build --update {pkgs}; rc=$?; for p in {pkgs}; do "
                  f"mkdir -p \"$RESEARCH_RUN_DIR/records/$p\" && cp \"$p/lean-audit.json\" \"$RESEARCH_RUN_DIR/records/$p/\"; done; exit $rc")
        return [python, "-m", "research", "run", "--on", pod, "--project", PROJECT, "--source", ".", "--cwd", "source",
                *(["--preflight", "check"] if preflight else []), "--", "sh", "-c", script]
    raise ValueError(f"this dispatcher doesn't run {job['kind']} jobs")


def run_command(plan: RunPlan, python: str) -> list[str]:
    """What the launch child runs in a research-run's worktree: `research run` from the dispatcher's tree, never the worktree's
    (`-P`: the working directory is not on the path; `-s`: no user site; `run_env` names the tree)."""
    return [python, "-P", "-s", "-m", "research", "run", "--source", ".", "--project", PROJECT, *plan.opts, "--", *plan.argv]


#: the repository's address before its move into the organization (GitHub redirects it); a job from a submitter still on the old
#: name is the same repository, so submitters and dispatchers don't have to change in lockstep
MOVED = {"danielreuter/verity": "compute-verification/verity"}


def validate(job: dict, repo: str) -> None:
    p = job["payload"]
    if job["kind"] == RUN_KIND:
        run_plan(p, repo)
        return
    if MOVED.get(p.get("repo", repo), p.get("repo", repo)) != MOVED.get(repo, repo):
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
        from .. import runs as R

        env, plan, before = None, None, set()
        if job["kind"] == RUN_KIND:
            if spec.get("launcher_tree") is None:
                raise RuntimeError("the dispatcher runs from no checkout, so it has no tree to launch `research run` from")
            plan = run_plan(p, spec["repo"])
            if why := launcher_refusal(spec["launcher_tree"], wt):
                print(why, flush=True)
                return {"rc": 2, "run_id": None, "tail": why + "\n"}
            cmd, env = run_command(plan, spec["python"]), run_env(dict(os.environ), Path(spec["launcher_tree"]), plan.lane)
            before = set(os.listdir(R.runs_root())) if R.runs_root().is_dir() else set()
        else:
            cmd = command(job, pod, spec["python"], pack, preflight=hasattr(remote, "run_preflight"))
        print(f"dispatch: job {job['id']} in {wt}: {' '.join(shlex.quote(c) for c in cmd[:8])} ...", flush=True)
        proc = subprocess.Popen(cmd, cwd=wt, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                stdin=subprocess.DEVNULL if plan else None)
        run_id, tail = None, []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail = (tail + [line])[-20:]
            if (m := LAUNCHED.search(line)) and (plan is None or (run_id is None and launched_by(m.group(1), plan, before))):
                run_id = m.group(1)
        rc = proc.wait()
    finally:
        with open(workdir / "clone.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _git(clone, "worktree", "remove", "--force", str(wt), check=False)
    return {"rc": rc, "run_id": run_id, "tail": "".join(tail)}


def launcher_refusal(tree: str, source: Path) -> str | None:
    """Why `research run` would refuse ``tree`` as the trusted tree of a launch from ``source`` (research.launcher's own
    message), asked here first with this process's launcher: `research run` imports itself from ``tree``, so a tree too broken
    for that would end in an import error, not the launcher's refusal. None when it would launch."""
    from .. import launcher, settings

    was = settings.get(launcher.ENV)
    os.environ[launcher.ENV] = tree
    try:
        return launcher.refusal(source)
    finally:
        if was is None:
            del os.environ[launcher.ENV]
        else:
            os.environ[launcher.ENV] = was


def launched_by(run_id: str, plan: RunPlan, before: set[str]) -> bool:
    """Whether ``run_id``, named in a research-run's launch output, is the run this launch made: a run dir new since the launch
    began, whose command ends in the plan's workload. The output can carry the lead's own text, and a run id taken from it is
    one the dispatcher cancels."""
    from .. import remote
    from .. import runs as R

    rec = remote._read_json(R.runs_root() / run_id / "remote.json") if run_id not in before else None
    command = (rec or {}).get("command")
    return isinstance(command, list) and len(command) >= len(plan.argv) and command[len(command) - len(plan.argv):] == plan.argv


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
