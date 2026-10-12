"""`research run --on <machine>`: the same harness and the same record, on an existing authorized machine.

    machines file   ~/.research/machines.toml ($RESEARCH_MACHINES) binds a name to a provider pod and a project:
                        [machines.vy-control]
                        provider = "runpod"
                        pod_id   = "abc123"
                        project  = "verity"            # or projects = ["verity", "other"]
                        root     = "/workspace/research"
                        python   = "python3"
                    or to a host reached over plain ssh (a cloud VM with a static address; nothing is looked up, nothing provisioned):
                        provider = "ssh"
                        host     = "203.0.113.7"       # port = 22, user = "root" by default
                        user     = "research"
                        rate_usd_per_hour = 14.40      # the attempt's cost rate (a runpod pod reports its own)
                    A machine not in the file, or bound to another project, is refused before anything is touched.
    tool snapshot   the research package source is shipped to <root>/tool/<content-sha>/ and run via PYTHONPATH; the run records
                    which snapshot ran.  No pip, no editable install, no credentials on the machine.
    version check   PYTHONPATH does not guarantee which `research` answers (a wrapper python, a `research/` earlier on sys.path, an
                    altered snapshot).  The launcher sends tool_identity() of the package it shipped: `launch-request --expect-tool ID`
                    refuses before the claim, and the runner again before the workload, with ResearchVersionMismatch naming the
                    running package's path and identity; a package that predates the flag fails closed on argparse and the
                    launcher names it the same way.  A --source tree whose own package predates the check runs unverified, and
                    the launcher says so (remote.json tool_checked=false).  job.json run.remote.tool_running = what ran.
    source          --source <git dir> ships `git archive HEAD` of a CLEAN checkout to <root>/src/<identity>/ (the run's cwd, tree_sha);
                    the identity is the commit sha.  A dirty tree is refused rather than mislabeled.  Untracked inputs go with --send.
                    Shipped as git objects (`ship_source`), the tree is a checkout: a clone of <root>/git/verity.git sharing its objects,
                    detached at the commit, so what asks git there gets a checkout's answers.  The archive stream ships no .git.
                    The ship is VERIFIED before it may execute (r20 incident: a stalled transfer left a half-tree that later runs used):
                    the archive lands in <root>/src/<identity>.partial-<nonce>/, is sha256'd on the machine and compared with the
                    launcher's, its extracted file set is compared with the archive's manifest, and only then is the directory
                    atomically renamed into place carrying READY.json (identity, archive sha256, file count, shipped_at, launcher).
                    A failed or truncated ship removes its partial directory and refuses the run (SourceNotReady); the machine side
                    (`launch-request`) refuses to execute from any source directory without a READY.json naming the requested identity.
                    The archive is STREAMED, never held in the launcher's memory (0.55-0.69 GB launchers were killed by the laptop's
                    guardian 2026-09-23): one `git archive` pass computes the digest and manifest, a second is piped through ssh
                    (compressed) in 1 MiB chunks and hashed again as it goes.  A launcher killed mid-ship leaves a short stream: the machine
                    finds the digest wrong, extracts nothing and removes the partial directory itself; partial directories untouched for
                    30 min are removed by the next ship.  The timeout grows with the archive (64 KiB/s floor, >= 600 s).
                    Legacy trees (the old `.complete` marker, no READY.json) are verified once per file against the archive; a tree that
                    matches is adopted (READY.json written), one that does not is moved to <root>/src/.quarantine/ and re-shipped.
    request         <root>/requests/<run-id>/request.json carries the workload argv as a JSON list, its env, cwd, and harness flags.
                    Nothing is rebuilt through a shell string.
    cwd             the request's `cwd` is the RUNNER's: the shipped tree (the store reads the closure and tool declarations there).  The
                    WORKLOAD runs in its run dir <root>/runs/<run-id>/ (WORKLOAD_CWD_DEFAULT): every run of one commit shares
                    <root>/src/<sha>/, so a relative `result.json` written there raced between concurrent runs.  `--cwd source` (or
                    `source/SUB`, or a relative SUB) runs it inside the tree instead, an explicit opt-in; `--cwd clone` (or `clone/SUB`)
                    in the run's own `git clone --shared` of the tree at <root>/scratch/<run-id>/tree, checked to be the shipped commit
                    and removed when the workload exits (research/clone.py), so nothing one run does there reaches another's; its
                    PYTHONPATH entries under the tree, and the tree's shim, move into the clone too (cli._clone_imports); job.json
                    records run.cwd and run.cwd_basis (and run.cwd_clone).  The launcher refuses `--cwd clone` for a tree shipped
                    without git or a package that predates it.  Python bytecode goes to <root>/cache/pycache (PYTHONPYCACHEPREFIX), not
                    into the tree; a clone's goes beside the clone.  A tree whose own research package predates this runs its workloads
                    in the tree; the launcher says so (remote.json workload_cwd).
    claim           `research launch-request` on the machine does mkdir(<root>/runs/<run-id>) as the claim, then starts the detached
                    runner.  Re-delivering the same request finds the directory and reports its state; it never starts a second workload.
                    Each delivery holds flock(<root>/requests/<run-id>/.launch.lock) until it has answered, so a re-delivery (an ssh drop,
                    or the launcher's deadline, mid-wait) reports the first one's settled answer, never a run still starting.
    first record    launch-request answers only once the runner has parsed its arguments and written job.json: the runner says so on
                    the FIFO <root>/requests/<run-id>/.launch-ack (`ack_launch`), found from the request path its --remote-json carries,
                    so no wrapper has to pass an fd or the environment through.  A runner (or a wrapper ahead of it) that exits first is
                    refused (RunnerFailed, refused.json with its rc and launcher.log's tail), so `research run --on` exits 3, never
                    `launched`.  `disk-hold` ahead of the runner acknowledges when it starts holding: the launch stays detached.  When
                    launch-request can't tell (job.json is there without an ack, the wrapper exited but its process group lives on, or the
                    wait itself failed), it answers started with `unconfirmed` and releases nothing: a live run is never called refused.
    exclusive       --exclusive takes <root>/runs/.exclusive.lock (O_EXCL) naming the run; a live holder refuses the launch with the reason.
                    A dead holder (terminal status or vanished pid) is reclaimed.  The runner releases its own lock at exit.
    memory          --mem-gb N: the machine side admits the run only if its memory cgroup's limit - in use - live reservations
                    (the fleet's /workspace/ramlock/*.json, machines.toml `ramlock`) >= N, and writes the run's reservation there;
                    else MemoryNotAvailable (refused.json).  --mem-wait S polls up to S seconds instead.  The runner removes the
                    reservation at exit; job.json run.remote.mem records the reading it was admitted on (research.mem).
    preflight       a `--tool` whose declaration in the shipped tree names a preflight (`Tool.preflight`: a Python script there and its
                    args) has it run by launch-request, in the tree, before anything else is admitted: exit 0 admits the run, anything
                    else refuses it (PreflightFailed, refused.json; nothing started, no attempt, so never a verdict), with the script's
                    stderr as the reason.  It gets the runner's environment plus RESEARCH_PREFLIGHT_INPUTS (the run's --send files) and,
                    with custody, RESEARCH_PREFLIGHT_CUSTODY (the staged key's dir).  A declaration that cannot be read refuses the run.
                    `--preflight TOOL` runs TOOL's preflight before a run of anything else (a Lean record regeneration on a check
                    pod), and refuses when the tree declares none.  job.json run.remote.preflight and the launch's answer carry its report.
    observe/fetch   `research fetch <id>` copies the small records (or --all files) into the local run dir over one ssh+tar, extracted
                    member by member off the pipe (a big run is never in the launcher's memory); remote.json keeps observed_utc.
                    Unreachable means "last observed at T; current status unknown", never "failed".
    adopt           a run another VM launched has no local dir: `research inspect|fetch|status RUN` (a full run id) finds
                    <root>/runs/<id> on `--on NAME`, else on each ssh machine (a pod belongs to the VM that made it), writes the
                    remote.json launch would have and fetches it as above (`adopt`).
    preservation    --all verifies every file's sha256 against the machine and writes preserved.json LAST.  Until then the only copy
                    is on the machine; this is stated, not hidden.  Then it writes <run>/.fetched ON THE MACHINE (research/fetched/v0.1:
                    verified_from = the machine's clock before hashing, file count, bytes, manifest sha256), which the guard requires.
    guard           machines.toml `guard = N` (idle minutes; `true` = 90) opts a pod in; a machine without it (a control pod, a live
                    verifier) never gets one.  launch-request then makes sure ONE guard daemon runs on the pod (ensure_guard ->
                    pods/sh/pod_guard.sh `daemon`, flock/pidfile <root>/guard/guard.pid): no live runner, no work process, no live verifier,
                    no GPU use (research.pods.workproc, the watcher's definition) for N minutes -> the pod terminates itself, except while
                    any <root>/runs/* lacks a valid .fetched or a valid .custody (refused, run ids logged, re-checked every minute).
    custody         --custody-r2: the launcher mints a delete-free R2 key (store.custody.mint_for_machine) and stages it with a secret-free
                    store.toml in <root>/requests/<run-id>/custody/ (mode 600, never in request.json or the workload's environment);
                    launch-request passes --custody-dir to the runner, which after the workload publishes the attempt with every run file
                    (outputs.run_record) but a sent credential, deleted first (store.custody.CREDENTIALS), verifies it on the remote,
                    deletes the key and writes <run>/.custody (store.custody).  A refused launch deletes the key and those credentials from
                    the request dir.  It terminates with the pod's OWN key (RUNPOD_API_KEY in /proc/1/environ); without one it is not
                    installed and the launch says so.  The account key is never sent.  <root>/guard/state.json: verdict, last activity,
                    idle minutes, refusal reasons.
    lease           a pod `research pods create` made carries a lease (research.pods.lease, pods/sh/lease.sh): it ends itself at the
                    lease's expiry.  The request's `lease_extend_s` (the run's --timeout + lease.RUN_MARGIN_S) makes launch-request
                    push the expiry to at least now + that, so the lease cannot end the run; the answer's `lease` says what it did.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import json
import os
import secrets
import select
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
from pathlib import Path, PurePosixPath
from typing import IO, Any, Callable

from . import clone
from .telemetry.events import utc_now_ms
from .telemetry.job import write_json_atomic

REQUEST_SCHEMA = "research/request/v0.1"
PRESERVED_SCHEMA = "research/preserved/v0.1"
READY_SCHEMA = "research/source-ready/v0.1"
READY_FILE = "READY.json"                     # the only marker that makes <root>/src/<identity>/ executable
LEGACY_COMPLETE = ".complete"                 # the pre-verification marker: never sufficient on its own
SHIP_SOURCE_TIMEOUT_S = 600                   # the floor; a big archive gets size / SHIP_MIN_BYTES_PER_S
SHIP_MIN_BYTES_PER_S = 64 * 1024              # the slowest link a ship is allowed to be (laptop -> pod measured 130-220 KB/s, 2026-09-24)
SHIP_CHUNK = 1 << 20                          # the launcher never holds more of the archive than this
STALE_PARTIAL_MIN = 30                        # a <root>/src/*.partial-* directory with nothing written for this long belongs to a dead ship
_RC_DIGEST = 75                               # the ship script's exit when the received archive is not the launcher's (nothing extracted)
WORKLOAD_CWD_DEFAULT = "run_dir"              # the machine-side runner's default workload cwd; a shipped package without this line predates it
SMALL_FILES = ("job.json", "status.json", "failure.json", "result.json", "launch.json", "refused.json", "events.jsonl", "slot.json")
TERMINAL = ("done", "failed", "cancelled")
FETCHED_FILE = ".fetched"                     # <run>/.fetched on the machine: written by `fetch --all` once every file verified
FETCHED_SCHEMA = "research/fetched/v0.1"
CUSTODY_DIR = "custody"                       # <root>/requests/<run-id>/custody/: --custody-r2's minted key + store.toml (store.custody)
LAUNCH_ACK = ".launch-ack"                    # <root>/requests/<run-id>/.launch-ack: the FIFO the runner names its first record on
LAUNCH_LOCK = ".launch.lock"                  # <root>/requests/<run-id>/.launch.lock: held by a delivery of the request until it answers
FIRST_RECORD = "job.json"
RUNNER_FAILED_LINES = 5                       # launcher.log lines in a RunnerFailed reason (refused.json keeps PREFLIGHT_TAIL)
GUARD_IDLE_MIN_DEFAULT = 90
GUARD_ENVIRON = "/proc/1/environ"             # RunPod injects RUNPOD_POD_ID and the pod-scoped RUNPOD_API_KEY into the container's first process
DIRECT_GPUS_FILE = "/etc/vy/direct-gpus"      # the Nebius cluster's split (pods/nebius/sky/cluster_up.sh): `none` once Kueue owns every GPU
DIRECT_CPUS_FILE = "/etc/vy/direct-cpus"      # the CPUs a direct run starts on (a cpu list, e.g. 0-95): the rest are pinned benchmarks'.
                                              # A command may still pin itself elsewhere (taskset -c 128-159 ...); absent: no default
RUNS_VIA_SLOT_FILE = "/etc/vy/runs-via-slot"  # present: a run starts only through a check slot (pods/nebius/slot_only.py); cpu-1's
GUARD_SCRIPT = Path(__file__).resolve().parent / "pods" / "sh" / "pod_guard.sh"
POD_TARGET_RETRIES = 6           # extra GET /pods/{id} reads when a RUNNING pod answers without an ssh mapping
POD_TARGET_RETRY_S = 2.0
PREFLIGHT_TIMEOUT_S = 180
PREFLIGHT_TAIL = 40              # lines of the preflight's stdout and stderr kept in its report


class Refused(Exception):
    """A launch or control action that must not proceed.  The message is the reason."""


class SourceNotReady(Refused):
    """A shipped source tree that is not verified READY (truncated transfer, digest mismatch, missing or foreign READY.json).
    Nothing executes from it."""


class ResearchVersionMismatch(Refused):
    """The research package answering on the machine is not the one the launcher shipped (its content identity differs, or it
    predates the check).  Nothing started."""


class MemoryNotAvailable(Refused):
    """`--mem-gb N` did not fit: limit - in use - live reservations < N on the machine (research.mem).  Nothing started."""


class PreflightFailed(Refused):
    """The tool's preflight (`Tool.preflight`) found the machine unprepared for the run, or could not run.  Nothing started."""

    def __init__(self, reason: str, report: dict[str, Any] | None = None):
        super().__init__(reason)
        self.report = report


class RunnerFailed(Refused):
    """The machine-side runner, or a wrapper ahead of it, exited before writing the run's first record (job.json): an argv it
    rejected, a store preparation that failed, a CPU scope that would not start.  No workload ran."""


# ---- machines ---------------------------------------------------------------------------------------------------------------------------

def machines_path() -> Path:
    return Path(os.environ.get("RESEARCH_MACHINES", Path.home() / ".research" / "machines.toml")).expanduser()


def load_machines(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """machines.toml's `[machines.*]`; without an explicit `path`, merged with the pod registry (research.pods.registry)."""
    p = path or machines_path()
    ms = tomllib.loads(p.read_text()).get("machines", {}) if p.exists() else {}
    if path is None:
        from .pods import registry

        ms = registry.merged(ms, p)
    return ms


#: a machine's runs root when its entry names none (a pod `research pods register` wrote without --root)
DEFAULT_ROOT = "/workspace/research"


def resolve_machine(name: str, project: str | None, machines: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    ms = load_machines() if machines is None else machines
    m = ms.get(name)
    if m is None:
        from .pods import registry

        if why := registry.problems.get(name):
            raise Refused(f"machine {name!r} refused: {why}")
        d = registry.registry_dir(machines_path())
        where = f"{machines_path()}" + (f" or {d}" if d else "")
        raise Refused(f"machine {name!r} is not in {where}; only listed machines may be used (research pods register)")
    if not project:
        raise Refused("--project is required for a remote run (the machine is bound to a project)")
    allowed = m.get("projects") or ([m["project"]] if m.get("project") else [])
    if project not in allowed:
        raise Refused(f"machine {name!r} belongs to project(s) {allowed}, not {project!r}; refusing to touch it")
    out = dict(m)
    out.setdefault("root", DEFAULT_ROOT)
    out.setdefault("python", "python3")
    out["name"] = name
    return out


def guard_idle_min(machine: dict[str, Any]) -> float | None:
    """machines.toml `guard = true` (GUARD_IDLE_MIN_DEFAULT minutes) or `guard = N` minutes; absent or false -> no guard."""
    g = machine.get("guard")
    if g is None or g is False:
        return None
    if g is True:
        return float(GUARD_IDLE_MIN_DEFAULT)
    if isinstance(g, (int, float)) and g > 0:
        return float(g)
    raise Refused(f"machine {machine.get('name')!r}: guard = {g!r} is neither true/false nor a positive number of idle minutes")


# ---- transport --------------------------------------------------------------------------------------------------------------------------

def ssh_host_target(machine: dict[str, Any]) -> dict[str, Any]:
    """host/port/user/rate for a `provider = "ssh"` machine, as its entry lists them."""
    who = f"machine {machine.get('name') or '?'}"
    host, port, user = machine.get("host"), machine.get("port", 22), machine.get("user", "root")
    if not isinstance(host, str) or not host:
        raise Refused(f"{who}: provider ssh needs host = \"...\"")
    if not isinstance(port, int) or not 0 < port < 65536:
        raise Refused(f"{who}: port {port!r} is not a TCP port")
    if not isinstance(user, str) or not user:
        raise Refused(f"{who}: user {user!r} is not a user name")
    return {"host": host, "port": port, "user": user, "pod_name": machine.get("name"), "cost_per_hr": machine.get("rate_usd_per_hour"),
            "rate_observed_utc": utc_now_ms(), "gpu": machine.get("gpu")}


def pod_target(machine: dict[str, Any]) -> dict[str, Any]:
    """host/port/user/rate for a runpod machine (read-only API call; not RUNNING -> refused) or an ssh one (`ssh_host_target`)."""
    if machine.get("provider") == "ssh":
        return ssh_host_target(machine)
    if machine.get("provider") != "runpod":
        raise Refused(f"provider {machine.get('provider')!r} not supported (runpod or ssh)")
    from .pods import runpod

    who = f"pod {machine['pod_id']} (machine {machine.get('name') or '?'})"
    try:
        pod = runpod.get(machine["pod_id"])
    except runpod.PodError as e:                # a dead / terminated pod, no API config, no network: a refusal with the pod's name, not a SystemExit
        if e.code == 404:
            raise Refused(f"{who} no longer exists on runpod (HTTP 404: terminated?); no implicit provisioning") from None
        raise Refused(f"{who}: runpod API unavailable: {e}") from None
    if not pod or (pod.get("desiredStatus") or "").upper() != "RUNNING":
        raise Refused(f"{who} is {pod.get('desiredStatus') if pod else 'not found'}; no implicit provisioning")
    tgt = runpod.ssh_target(pod)
    # The REST API intermittently answers a RUNNING pod's GET with publicIp="" / portMappings=None (observed 2026-09-22 on
    # s00kzu0nisihgr: alternating answers within seconds while ssh worked throughout).  Re-read a few times before refusing;
    # a mapping that never appears is still refused, by the same name.
    for _ in range(POD_TARGET_RETRIES):
        if tgt is not None:
            break
        time.sleep(POD_TARGET_RETRY_S)
        pod = runpod.get(machine["pod_id"]) or pod
        tgt = runpod.ssh_target(pod)
    if tgt is None:
        raise Refused(f"{who} has no ssh port mapped yet")
    return {"host": tgt[0], "port": tgt[1], "user": "root", "pod_name": pod.get("name"), "cost_per_hr": pod.get("costPerHr"),
            "rate_observed_utc": utc_now_ms(), "gpu": (pod.get("machine") or {}).get("gpuTypeId")}


def ssh_argv(host: str, port: int, *, user: str = "root", connect_timeout: int = 15) -> list[str]:
    from .pods import runpod

    base = runpod.ssh_command(host, port, user=user)
    return [*base[:-1], "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}", base[-1]]


def ssh_run(ssh: list[str], remote_argv: list[str], *, stdin: bytes | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    """Run one remote argv (quoted per-arg with shlex; no eval on the far side)."""
    cmd = [*ssh, " ".join(shlex.quote(a) for a in remote_argv)]
    return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)


def ssh_stream(ssh: list[str], remote_argv: list[str], feed: Callable[[IO[bytes]], None], *, timeout: float) -> subprocess.CompletedProcess:
    """`ssh_run` whose stdin is written by `feed(pipe)` while the remote side reads it, so no payload is held in memory.  A feed cut
    short (the remote exited, or the timeout killed ssh) just closes stdin: the remote side sees a short stream and must say so.
    TimeoutExpired after killing ssh; an exception of the feed's own (not a closed pipe) is raised once ssh has exited."""
    cmd = [*ssh, " ".join(shlex.quote(a) for a in remote_argv)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out: list[bytes] = []
    err: list[bytes] = []
    failed: list[BaseException] = []

    def pump() -> None:
        try:
            feed(proc.stdin)
        except (BrokenPipeError, ConnectionResetError, ValueError):     # the far side is gone (ValueError: stdin closed under us)
            pass
        except BaseException as e:  # noqa: BLE001
            failed.append(e)
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    threads = [threading.Thread(target=pump, daemon=True),
               threading.Thread(target=lambda: out.append(proc.stdout.read()), daemon=True),
               threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)]
    for t in threads:
        t.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        for t in threads:
            t.join(10)
        raise
    for t in threads:
        t.join()
    if failed:
        raise failed[0]
    return subprocess.CompletedProcess(cmd, proc.returncode, b"".join(out), b"".join(err))


def ssh_read(ssh: list[str], remote_argv: list[str], consume: Callable[[IO[bytes]], None], *, timeout: float) -> subprocess.CompletedProcess:
    """`ssh_run` whose stdout goes to `consume(pipe)` as it arrives (never held whole).  The timeout kills ssh, which ends the stream;
    whatever `consume` raises is re-raised after ssh exited.  The CompletedProcess carries stderr only."""
    cmd = [*ssh, " ".join(shlex.quote(a) for a in remote_argv)]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        try:
            consume(proc.stdout)
            while proc.stdout.read(SHIP_CHUNK):       # the tar record padding after the end-of-archive blocks
                pass
        except BaseException:
            proc.kill()
            raise
        finally:
            timer.cancel()
            proc.stdout.close()
            rc = proc.wait()
            err.seek(0)
            stderr = err.read()[-8000:]
    return subprocess.CompletedProcess(cmd, rc, b"", stderr)


# ---- snapshots --------------------------------------------------------------------------------------------------------------------------

def source_tool_package(source: Path | None) -> Path | None:
    """The `research` package a shipped source tree carries (tools/research/src/research), or None.  When present it is the tool the
    machine runs: the launcher's own install may be older (a venv elsewhere) and know neither the tree's tools nor its CLI verbs."""
    if source is None:
        return None
    cand = Path(source).expanduser() / "tools" / "research" / "src" / "research"
    return cand if (cand / "__init__.py").is_file() else None


def tool_tarball(src: Path | None = None) -> tuple[bytes, str]:
    """Deterministic tar of the research package source (no tests, no caches); sha256 of the tar is the tool identity."""
    src = Path(src).resolve() if src is not None else Path(__file__).resolve().parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in sorted(src.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix in (".pyc",):
                continue
            ti = tf.gettarinfo(str(p), arcname=str(Path("research") / p.relative_to(src)))
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            with open(p, "rb") as f:
                tf.addfile(ti, f)
    data = buf.getvalue()
    return data, hashlib.sha256(data).hexdigest()[:16]


def tool_identity(pkg: Path | str | None = None) -> str:
    """Content identity of a research package: sha256 over (relative path, sha256 of bytes) of every file tool_tarball ships.  File
    modes and mtimes do not enter (an extracted snapshot keeps its bytes, not necessarily its modes)."""
    src = Path(pkg).resolve() if pkg is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(src.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        h.update(f"{p.relative_to(src).as_posix()}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()[:16]


def knows_version_check(pkg: Path | None) -> bool:
    """Whether a shipped research package (a --source tree's) verifies `launch-request --expect-tool`; the launcher's own always does."""
    if pkg is None:
        return True
    try:
        return "def tool_identity" in (Path(pkg) / "remote.py").read_text()
    except OSError:
        return False


def knows_custody(pkg: Path | None) -> bool:
    """Whether the research package that will run on the machine (a --source tree's, else this launcher's) publishes --custody-r2."""
    if pkg is None:
        return True
    return (Path(pkg) / "store" / "custody.py").is_file() and "CUSTODY_DIR" in (Path(pkg) / "remote.py").read_text()


def knows_queue(pkg: Path | None) -> bool:
    """Whether the research package that will run on the machine runs its runner under a queue's wrapper (`--queue`)."""
    if pkg is None:
        return True
    try:
        return "QUEUE_WORKLOAD_WRAP" in (Path(pkg) / "remote.py").read_text() and (Path(pkg) / "disk.py").is_file()
    except OSError:
        return False


QUEUE_WRAP = "wrap"                       # the runner's, custody included: the job's CPU allocation
QUEUE_WORKLOAD_WRAP = "workload_wrap"     # the workload's alone: a GPU phase's lease


def queue_placement(source: Path | None, run_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    """`research run --queue`: the queue's placement for this job, `{node, machine, why, job, wrap}`, from `cluster submit` and the
    description in the --source tree (infra/cluster). Refused when the queue refuses the job."""
    from .store import tools_registry

    src = (Path(source) / (tools_registry.resolve("infra/cluster", lambda p: (Path(source) / p).is_dir()) or "infra/cluster")
           if source is not None else None)
    if src is None or not (src / "src" / "cluster" / "submit.py").is_file():
        raise Refused("--queue: the --source tree has no infra/cluster with `cluster submit` (the queue's description and client)")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    argv = [sys.executable, "-m", "cluster", "submit", str(src / "descriptions" / "nebius.toml"), "--run-id", run_id,
            "--now", str(int(time.time()))]
    for k, v in spec.items():
        if v is True:
            argv.append(f"--{k}")
        elif v not in (None, False):
            argv += [f"--{k}", str(v)]
    r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60)
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise Refused(f"--queue: `cluster submit` failed (exit {r.returncode}): {r.stderr.strip()[-400:]}") from None
    if "refused" in out:
        raise Refused(f"the queue refuses the job: {out['refused']}")
    return out


def knows_job_user(pkg: Path | None) -> bool:
    """Whether the research package that will run on the machine runs the workload as `--job-user` (one that predates it would
    run it as the machine's own user)."""
    if pkg is None:
        return True
    try:
        return (Path(pkg) / "jobuser.py").is_file() and '"job_user"' in (Path(pkg) / "remote.py").read_text()
    except OSError:
        return False


def knows_clone(pkg: Path | None) -> bool:
    """Whether the research package that will run on the machine gives `--cwd clone` a clone (one that predates it would look for a
    directory named `clone` in the shared tree)."""
    return pkg is None or (Path(pkg) / "clone.py").is_file()


def knows_preflight(pkg: Path | None) -> bool:
    """Whether the research package that will run on the machine runs `--preflight TOOL` (one that predates it would ignore it)."""
    if pkg is None:
        return True
    try:
        return "preflight_tool" in (Path(pkg) / "remote.py").read_text()
    except OSError:
        return False


def knows_pss_bound(pkg: Path | None) -> bool:
    """Whether the research package that will run on the machine knows `--pss-every 0` (PSS off) and `--pss-max-rss-gb`."""
    if pkg is None:
        return True
    try:
        return "pss_max_rss_gb" in (Path(pkg) / "telemetry" / "sample.py").read_text()
    except OSError:
        return False


def pss_bound_problem(harness_args: list[str], source: Path | None) -> str | None:
    """Why the machine must not get these sampler flags, or None: `--pss-every` <= 0 or `--pss-max-rss-gb` for a --source tree whose
    research package predates them. On that package `--pss-every 0` reads every process's smaps_rollup every round, and an unknown
    `--pss-max-rss-gb` fails the run on the machine."""
    asked = []
    every = flag_value(harness_args, "--pss-every")
    try:
        if every is not None and float(every) <= 0:
            asked.append(f"--pss-every {every}")
    except ValueError:
        pass                                    # the harness's own parser rejects it by name
    if (bound := flag_value(harness_args, "--pss-max-rss-gb")) is not None:
        asked.append(f"--pss-max-rss-gb {bound}")
    if not asked or knows_pss_bound(pkg := source_tool_package(source)):
        return None
    return (f"{' and '.join(asked)}: the research package in {source} predates them ({pkg}/telemetry/sample.py has no "
            f"--pss-max-rss-gb), and it is what the machine runs: there --pss-every 0 means a PSS round (smaps_rollup of every process, "
            f"under its mmap lock) every round, about once a second, and --pss-max-rss-gb fails the run.  Rebase the --source tree onto "
            f"main, or drop the flag")


def check_tool(expected: str | None) -> dict[str, str]:
    """The running package's identity and path; ResearchVersionMismatch when `expected` is given and differs."""
    here = str(Path(__file__).resolve().parent)
    got = tool_identity(here)
    if expected and got != expected:
        raise ResearchVersionMismatch(f"the research package running on this machine ({here}, identity {got}) is not the one the launcher "
                                      f"shipped (identity {expected}): another `research` shadows the snapshot on sys.path, or the snapshot "
                                      f"was altered; nothing started")
    return {"identity": got, "path": here}


def launch_request_failure(rc: int, stdout: bytes, stderr: bytes) -> Refused:
    """The named fault for a failed `research launch-request` (a refusal JSON on stdout, or an argparse error from an old package)."""
    err = stderr.decode(errors="replace")
    for line in reversed(stdout.decode(errors="replace").strip().splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and doc.get("refused"):
            cls = {"ResearchVersionMismatch": ResearchVersionMismatch, "PreflightFailed": PreflightFailed,
                   "RunnerFailed": RunnerFailed}.get(str(doc.get("fault")), Refused)
            return cls(str(doc["refused"]))
    if rc == 2 and "--expect-tool" in err:
        return ResearchVersionMismatch("the research package that answered `launch-request` on the machine predates the version check, so it "
                                       f"is not the snapshot the launcher shipped (another `research` shadows it on sys.path?): {err.strip()[-300:]}")
    return Refused(f"launch-request failed on the machine: {err[-400:]}")


def ship_tool(ssh: list[str], root: str, src: Path | None = None) -> str:
    data, sha = tool_tarball(src)
    dest = f"{root}/tool/{sha}"
    probe = ssh_run(ssh, ["test", "-f", f"{dest}/.complete"])
    if probe.returncode == 0:
        return dest
    r = ssh_run(ssh, ["sh", "-c", f"mkdir -p {shlex.quote(dest)} && tar -xf - -C {shlex.quote(dest)} && touch {shlex.quote(dest)}/.complete"], stdin=data)
    if r.returncode != 0:
        raise Refused(f"shipping the tool snapshot failed: {r.stderr.decode(errors='replace')[-400:]}")
    return dest


class _Hashing:
    """A read-only stream that sha256es and counts everything read through it."""

    def __init__(self, raw: IO[bytes]):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def read(self, size: int = -1) -> bytes:
        b = self.raw.read(size)
        self.h.update(b)
        self.n += len(b)
        return b


def _manifest_of(stream: IO[bytes]) -> dict[str, Any]:
    """archive_manifest over a tar read sequentially from `stream` (at most SHIP_CHUNK of it in memory at a time)."""
    src = _Hashing(stream)
    files: dict[str, str] = {}
    total = 0
    with tarfile.open(fileobj=src, mode="r|") as tf:
        for m in tf:
            if not m.isreg():
                continue
            h = hashlib.sha256()
            f = tf.extractfile(m)
            while f is not None and (chunk := f.read(SHIP_CHUNK)):
                h.update(chunk)
                total += len(chunk)
            files[m.name.removeprefix("./")] = h.hexdigest()
    while src.read(SHIP_CHUNK):                    # the end-of-archive blocks tarfile stops short of are part of the digest
        pass
    return {"archive_sha256": src.h.hexdigest(), "files": files, "bytes": total, "size": src.n}


def archive_manifest(tar_bytes: bytes) -> dict[str, Any]:
    """What a `git archive` tar promises: its sha256 and, per regular file, the path and content sha256 (`size` = archive bytes)."""
    return _manifest_of(io.BytesIO(tar_bytes))


# git's default delta-base cache (96 MiB) and unbounded pack windows put `git archive` itself at ~140 MiB on the laptop; these caps give
# the same bytes at the same speed in ~70 MiB (what remains is about the largest blob, which git holds whole)
GIT_ARCHIVE_CAPS = ("-c", "core.deltaBaseCacheLimit=8m", "-c", "core.packedGitWindowSize=4m", "-c", "core.packedGitLimit=16m")


def _git_archive(git_dir: Path) -> subprocess.Popen:
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(["git", *GIT_ARCHIVE_CAPS, "archive", "--format=tar", "HEAD"], cwd=git_dir, stdout=subprocess.PIPE, stderr=err)
    proc.archive_stderr = err          # Popen.stderr is None when stderr is a file, so the file is kept here
    return proc


def _git_archive_done(proc: subprocess.Popen) -> str | None:
    """None when `git archive` exited 0, else its stderr tail."""
    proc.stdout.close()
    if proc.wait() == 0:
        return None
    proc.archive_stderr.seek(0)
    return f"rc {proc.returncode}: {proc.archive_stderr.read().decode(errors='replace')[-300:].strip()}"


def archive_manifest_git(git_dir: Path) -> dict[str, Any]:
    """archive_manifest of `git archive HEAD`, streamed from git (the archive is never in memory)."""
    proc = _git_archive(git_dir)
    try:
        man = _manifest_of(proc.stdout)
    finally:
        err = _git_archive_done(proc)
    if err:
        raise Refused(f"git archive HEAD failed in {git_dir}: {err}")
    return man


def _archive_feed(git_dir: Path, sent: dict[str, Any]) -> Callable[[IO[bytes]], None]:
    """A `ssh_stream` feed: a fresh `git archive HEAD` copied to the pipe in SHIP_CHUNK pieces; `sent` gets what was written
    (sha256, bytes) and git's fault if it had one."""
    def feed(pipe: IO[bytes]) -> None:
        proc = _git_archive(git_dir)
        h, n = hashlib.sha256(), 0
        try:
            while chunk := proc.stdout.read(SHIP_CHUNK):
                pipe.write(chunk)
                h.update(chunk)
                n += len(chunk)
            pipe.flush()
        except (BrokenPipeError, ConnectionResetError):
            sent["cut"] = True                     # the far side went away first; git's SIGPIPE below is not git's fault
            raise
        finally:
            sent.update(sha256=h.hexdigest(), bytes=n, git_error=_git_archive_done(proc))
    return feed


def ship_timeout(archive_bytes: int) -> float:
    return max(float(SHIP_SOURCE_TIMEOUT_S), archive_bytes / SHIP_MIN_BYTES_PER_S)


def _compressed(ssh: list[str]) -> list[str]:
    """The same ssh with transport compression (a source archive halves; the laptop's uplink is the bottleneck)."""
    return [ssh[0], "-C", *ssh[1:]] if Path(ssh[0]).name == "ssh" else ssh


def _sha_lines(text: str) -> dict[str, str]:
    """`sha256sum` output -> {path: sha} with any leading `./` stripped."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        h, _, name = line.partition("  ")
        if h and name:
            out[name.removeprefix("./")] = h.strip()
    return out


def _rm_rf(ssh: list[str], path: str) -> None:
    """Best effort: a partial directory is never executable by name, so failing to remove it is cosmetic."""
    try:
        ssh_run(ssh, ["rm", "-rf", path], timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass


def read_ready(path: Path) -> dict[str, Any]:
    """The READY.json of a source directory, or SourceNotReady naming what is wrong with it."""
    p = Path(path) / READY_FILE
    try:
        doc = json.loads(p.read_text())
    except FileNotFoundError:
        legacy = (Path(path) / LEGACY_COMPLETE).exists()
        raise SourceNotReady(f"{path} has no {READY_FILE}" + (f" (only the legacy {LEGACY_COMPLETE} marker; ship it again to verify it)" if legacy
                                                              else " (never verified: a truncated or interrupted ship); ship it again")) from None
    except (OSError, ValueError) as e:
        raise SourceNotReady(f"{p} is unreadable: {e}") from None
    if not isinstance(doc, dict) or doc.get("schema") != READY_SCHEMA or not doc.get("identity"):
        raise SourceNotReady(f"{p} is not a {READY_SCHEMA} document")
    return doc


def require_ready_source(path: Path | str, identity: str) -> dict[str, Any]:
    """The gate on the machine: a source directory may execute only with a READY.json whose identity is the one requested."""
    doc = read_ready(Path(path))
    if doc["identity"] != identity:
        raise SourceNotReady(f"{path}/{READY_FILE} names identity {doc['identity']}, the run asked for {identity}; refusing to execute")
    return doc


def _ready_doc(identity: str, tree: str | None, man: dict[str, Any], verified: str) -> dict[str, Any]:
    return {"schema": READY_SCHEMA, "identity": identity, "commit": identity, "tree": tree, "archive_sha256": man["archive_sha256"],
            "files": len(man["files"]), "bytes": man["bytes"], "shipped_at_utc": utc_now_ms(), "launcher": os.uname().nodename, "verified": verified}


def _probe_source(ssh: list[str], dest: str) -> tuple[str, dict[str, Any] | None]:
    """(`ready` | `legacy` | `absent`, READY.json doc when ready)."""
    q = shlex.quote(dest)
    r = ssh_run(ssh, ["sh", "-c", f"if [ -f {q}/{READY_FILE} ]; then cat {q}/{READY_FILE}; elif [ -e {q} ]; then echo LEGACY; else echo ABSENT; fi"])
    out = r.stdout.decode(errors="replace").strip()
    if r.returncode != 0:
        raise Refused(f"probing {dest} failed: {r.stderr.decode(errors='replace')[-400:]}")
    if out == "ABSENT":
        return "absent", None
    if out == "LEGACY":
        return "legacy", None
    try:
        doc = json.loads(out)
    except ValueError:
        raise SourceNotReady(f"{dest}/{READY_FILE} is not JSON; inspect the machine") from None
    return "ready", doc


def _verify_legacy(ssh: list[str], dest: str, man: dict[str, Any]) -> list[str]:
    """Per-file sha256 of an existing (pre-READY) tree against the archive manifest.  Returns the mismatches (empty = identical)."""
    r = ssh_run(ssh, ["sh", "-c", f"cd {shlex.quote(dest)} && find . -type f ! -name {LEGACY_COMPLETE} -print0 | xargs -0 sha256sum"], timeout=SHIP_SOURCE_TIMEOUT_S)
    if r.returncode != 0:
        return [f"could not hash the existing tree: {r.stderr.decode(errors='replace')[-300:]}"]
    have = _sha_lines(r.stdout.decode(errors="replace"))
    want = man["files"]
    bad = [f"missing {p}" for p in sorted(set(want) - set(have))] + [f"extra {p}" for p in sorted(set(have) - set(want))]
    bad += [f"differs {p}" for p in sorted(set(want) & set(have)) if want[p] != have[p]]
    return bad


def _write_ready(ssh: list[str], dest: str, doc: dict[str, Any]) -> None:
    r = ssh_run(ssh, ["sh", "-c", f"cat > {shlex.quote(dest)}/{READY_FILE}"], stdin=(json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())
    if r.returncode != 0:
        raise SourceNotReady(f"could not write {dest}/{READY_FILE}: {r.stderr.decode(errors='replace')[-300:]}")


SHIP_ENV = "RESEARCH_SHIP"                     # `stream` forces the archive stream (the pre-git transport)
POD_GIT_REPO = "git/verity.git"                # <root>/git/verity.git: the machine's bare repository, fed by `git push`
_RC_NO_GIT = 69                                # the machine has no git: the archive stream ships instead
_RC_TREE = 76                                  # the pushed commit's tree on the machine is not the launcher's


def _ext_escape(arg: str) -> str:
    return arg.replace("%", "%%").replace(" ", "% ")


def git_push_url(ssh: list[str], repo: str) -> str:
    """A git `ext::` URL that runs `git receive-pack <repo>` through the very ssh argv the launcher already uses (its key, port,
    known_hosts, timeouts): no credential, no second connection setup to maintain.  Needs `-c protocol.ext.allow=always`."""
    return "ext::" + " ".join(_ext_escape(a) for a in ssh) + " " + _ext_escape(f"git receive-pack {shlex.quote(repo)}")


def _stale_partials_sweep(root: str) -> str:
    return (f"for d in {shlex.quote(root)}/src/*.partial-*; do [ -d \"$d\" ] && [ -z \"$(find \"$d\" -mmin -{STALE_PARTIAL_MIN} -print 2>/dev/null "
            f"| head -1)\" ] && rm -rf \"$d\"; done; ")


def _git_ship_script(root: str, repo: str, partial: str, sha: str) -> str:
    """The machine side of a git ship: the pushed commit's tree must be the launcher's ($1), then the partial directory becomes a
    checkout of it, a clone of the bare repository that borrows its objects (`clone --shared`: nothing is copied, and a clone,
    unlike a worktree, survives the rename into place), detached at the commit, with READY.json excluded from git's view.  Files
    get checkout-time mtimes, not the commit time.  Listed back: the sha256 of the machine's `git archive` of the commit, then
    every checked-out file's sha256, `.git` left out.  The partial removes itself on any exit until the listing starts."""
    q, g = shlex.quote(partial), shlex.quote(repo)
    return (_stale_partials_sweep(root) +
            f"trap 'rm -rf {q}' EXIT; trap 'exit 1' HUP INT TERM PIPE; set -e; "
            f"t=$(git --git-dir={g} rev-parse --verify -q {sha}^{{tree}}) || exit {_RC_TREE}; [ \"$t\" = \"$1\" ] || exit {_RC_TREE}; "
            f"git clone -q --shared --no-checkout {g} {q}; git -C {q} -c advice.detachedHead=false checkout -q --detach {sha}; "
            f"mkdir -p {q}/.git/info; echo /{READY_FILE} >> {q}/.git/info/exclude; "
            f"git --git-dir={g} archive --format=tar {sha} > {q}/.git/ship.tar; sha256sum {q}/.git/ship.tar | cut -d' ' -f1; "
            f"rm -f {q}/.git/ship.tar; trap - EXIT HUP INT TERM PIPE; "
            f"cd {q} && find . -path ./.git -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum")


def _ship_git(ssh: list[str], root: str, git_dir: Path, sha: str, tree: str | None, man: dict[str, Any], dest: str) -> tuple[str | None, str | None]:
    """Ship commit `sha` as git objects: `git push` into <root>/git/verity.git over `ssh` (only the objects the machine lacks travel),
    then check it out and verify it on the machine (tree hash, per-file sha256 against the launcher's archive manifest).  Returns
    (partial dir ready to be marked, None) or (None, why the archive stream must ship instead).  Never raises for the transport:
    every failure falls back, and nothing is marked READY by a failed git ship."""
    repo = f"{root}/{POD_GIT_REPO}"
    if not tree:
        return None, "no tree hash for HEAD"
    r = ssh_run(ssh, ["sh", "-c", f"command -v git >/dev/null 2>&1 || exit {_RC_NO_GIT}; mkdir -p {shlex.quote(str(Path(repo).parent))} && "
                                  f"{{ [ -d {shlex.quote(repo)} ] || git init -q --bare {shlex.quote(repo)}; }}"], timeout=120)
    if r.returncode == _RC_NO_GIT:
        return None, "the machine has no git"
    if r.returncode != 0:
        return None, f"could not create {repo}: {r.stderr.decode(errors='replace')[-200:].strip()}"
    try:
        push = subprocess.run(["git", "-c", "protocol.ext.allow=always", "push", "-q", git_push_url(ssh, repo), f"{sha}:refs/research/src/{sha}"],
                              cwd=git_dir, capture_output=True, text=True, timeout=ship_timeout(2 * man["size"]))
    except subprocess.TimeoutExpired:
        return None, "git push timed out"
    if push.returncode != 0:
        return None, f"git push failed: {push.stderr.strip()[-200:]}"
    partial = f"{dest}.partial-{secrets.token_hex(4)}"
    try:
        r = ssh_run(ssh, ["sh", "-c", _git_ship_script(root, repo, partial, sha), "sh", tree], timeout=ship_timeout(man["size"]))
    except subprocess.TimeoutExpired:
        _rm_rf(ssh, partial)
        return None, "extraction on the machine timed out"
    if r.returncode != 0:
        _rm_rf(ssh, partial)
        why = "the pushed commit's tree is not the launcher's" if r.returncode == _RC_TREE else r.stderr.decode(errors="replace")[-200:].strip()
        return None, f"extraction failed (rc {r.returncode}): {why}"
    lines = r.stdout.decode(errors="replace").splitlines()
    have = _sha_lines("\n".join(lines[1:]))
    want = man["files"]
    if have != want:
        _rm_rf(ssh, partial)
        bad = [f"missing {p}" for p in sorted(set(want) - set(have))] + [f"extra {p}" for p in sorted(set(have) - set(want))]
        bad += [f"differs {p}" for p in sorted(set(want) & set(have)) if want[p] != have[p]]
        return None, f"extracted tree does not match the archive manifest ({len(bad)}: {'; '.join(bad[:3])})"
    man["pod_archive_sha256"] = lines[0].strip() if lines else None
    return partial, None


def _mark_ready(ssh: list[str], partial: str, dest: str, sha: str, doc: dict[str, Any]) -> dict[str, Any]:
    """READY.json into the verified partial, one rename into place, the marker read back (step 3 of `ship_source`)."""
    q = shlex.quote(partial)
    _write_ready(ssh, partial, doc)
    r = ssh_run(ssh, ["sh", "-c", f"if [ -e {shlex.quote(dest)} ]; then echo EXISTS; else mv {q} {shlex.quote(dest)} && echo MOVED; fi"])
    outcome = r.stdout.decode(errors="replace").strip()
    if r.returncode != 0 or outcome not in ("MOVED", "EXISTS"):
        _rm_rf(ssh, partial)
        raise SourceNotReady(f"could not move {partial} into place: {r.stderr.decode(errors='replace')[-300:]}")
    if outcome == "EXISTS":                       # another launcher won the race: theirs stands if it is READY for this identity, ours goes
        _rm_rf(ssh, partial)
    state, have = _probe_source(ssh, dest)
    if state != "ready" or have.get("identity") != sha:
        raise SourceNotReady(f"{dest} appeared without a READY.json for {sha} (another writer?); refusing to execute from it")
    return have


def ship_source(ssh: list[str], root: str, git_dir: Path) -> tuple[str, str, dict[str, Any]]:
    """`git archive HEAD` of a clean checkout -> a VERIFIED <root>/src/<identity>/.  Returns (identity, remote dir, READY.json doc).

    Identity: the commit sha (a dirty tree is refused; there is no patch digest to name).  Protocol:
      1. probe <root>/src/<identity>/: READY.json naming this identity -> done; a foreign READY.json -> SourceNotReady (never touched);
         a directory without READY.json (legacy `.complete`, or an interrupted ship) -> per-file sha256 against the archive: identical
         -> adopted (READY.json written, `verified: legacy-sha256-per-file`); otherwise moved to <root>/src/.quarantine/<identity>-<nonce>/
         (kept for inspection, never executable by name) and shipped afresh;
      2. git first (`_ship_git`): `git push` of the commit into the machine's bare repo <root>/git/verity.git over the same ssh
         (only the objects it lacks travel; no credential anywhere), then on the machine the commit's tree hash must be HEAD's, the
         tree becomes a checkout of it (`_git_ship_script`: a shared clone of the bare repo, detached, READY.json excluded), so code
         there asks git what it asks in the launcher's checkout, and every file's sha256 must equal the launcher's archive manifest
         (`verified: git-push+tree+file-sha256`, `transport: git`, `checkout: true`).  A machine without git, or any failure of that
         path, prints one line and falls through to the archive stream below, a tree without .git; `RESEARCH_SHIP=stream` goes
         straight there.  A READY tree shipped before checkouts is reused as it is;
      2'. the archive stream into <root>/src/<identity>.partial-<nonce>/: the tar is staged as a file, sha256'd on the machine, extracted, and the
         extracted regular-file set listed back; the machine's archive sha256 must equal the launcher's and the file set must equal the
         archive's manifest.  The launcher streams it (`archive_manifest_git`, then `_archive_feed` through `ssh_stream`): its memory
         does not grow with the tree.  A short stream (a launcher killed mid-ship) makes the machine remove the partial itself;
      3. READY.json is written INTO the partial directory and the directory renamed into place (one rename, so a reader sees either
         nothing or a complete tree with its marker); the marker is read back and its identity checked.
    Any failure (transport rc != 0, the launcher's timeout, digest or manifest mismatch) removes the partial directory and raises
    SourceNotReady; `<identity>/` is never created by a failed ship.  The machine side (`launch_request`) enforces `require_ready_source`."""
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=git_dir, capture_output=True, text=True)
    if st.returncode != 0:
        raise Refused(f"{git_dir} is not a git checkout")
    if st.stdout.strip():
        raise Refused(f"{git_dir} has uncommitted changes; commit them (an unpushed commit is fine) or the snapshot would be mislabeled as HEAD")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=git_dir, capture_output=True, text=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=git_dir, capture_output=True, text=True).stdout.strip() or None
    dest = f"{root}/src/{sha}"
    state, doc = _probe_source(ssh, dest)
    if state == "ready":
        if doc.get("identity") != sha or doc.get("schema") != READY_SCHEMA:
            raise SourceNotReady(f"{dest}/{READY_FILE} names identity {doc.get('identity')!r}, not {sha}; inspect the machine before shipping over it")
        return sha, dest, doc
    man = archive_manifest_git(git_dir)
    if state == "legacy":
        bad = _verify_legacy(ssh, dest, man)
        if not bad:
            doc = _ready_doc(sha, tree, man, "legacy-sha256-per-file")
            _write_ready(ssh, dest, doc)
            return sha, dest, doc
        quarantine = f"{root}/src/.quarantine/{sha}-{secrets.token_hex(4)}"
        r = ssh_run(ssh, ["sh", "-c", f"mkdir -p {shlex.quote(root)}/src/.quarantine && mv {shlex.quote(dest)} {shlex.quote(quarantine)}"])
        if r.returncode != 0:
            raise SourceNotReady(f"{dest} is an unverified tree that does not match HEAD ({len(bad)} file(s): {'; '.join(bad[:3])}) and could not be moved aside: "
                                 f"{r.stderr.decode(errors='replace')[-300:]}")
        print(f"research: {dest} did not match the archive ({len(bad)} file(s): {'; '.join(bad[:3])}); moved to {quarantine}, shipping afresh", file=sys.stderr)
    if os.environ.get(SHIP_ENV, "").strip().lower() != "stream":
        gpartial, why = _ship_git(ssh, root, git_dir, sha, tree, man, dest)
        if gpartial is not None:
            doc = {**_ready_doc(sha, tree, man, "git-push+tree+file-sha256"), "transport": "git", "checkout": True,
                   "pod_archive_sha256": man.get("pod_archive_sha256")}
            return sha, dest, _mark_ready(ssh, gpartial, dest, sha, doc)
        print(f"research: git ship to {dest} not used ({why}); shipping the archive stream", file=sys.stderr)
    partial = f"{dest}.partial-{secrets.token_hex(4)}"
    q = shlex.quote(partial)
    # the archive is staged as a file and hashed AS RECEIVED; a digest that is not the launcher's exits 75 before anything is extracted
    # (GNU tar unpacks a truncated stream with a warning and rc 0 -- that is how the r20 half-tree came to be).  Until the file list is
    # printed the machine removes its own partial on any exit (a launcher killed mid-ship cannot); dead ships' partials go first.
    # `tar -m`: files get the extraction time, not the commit time `git archive` stamps.  Builds that share a target dir across
    # commits (Cargo's `/workspace/sp1-target`) judge freshness by mtime; with commit times an older commit shipped after a newer
    # build looks built, and runs the newer commit's binary.
    script = (_stale_partials_sweep(root) +
              f"trap 'rm -rf {q}' EXIT; trap 'exit 1' HUP INT TERM PIPE; set -e; mkdir -p {q}; cat > {q}/.archive.tar; "
              f"got=$({{ sha256sum {q}/.archive.tar 2>/dev/null || shasum -a 256 {q}/.archive.tar; }} | cut -d' ' -f1); echo \"$got\"; "
              f"[ \"$got\" = \"$1\" ] || exit {_RC_DIGEST}; "
              f"tar -xmf {q}/.archive.tar -C {q}; rm -f {q}/.archive.tar; trap - EXIT HUP INT TERM PIPE; cd {q} && find . -type f | LC_ALL=C sort")
    sent: dict[str, Any] = {}
    timeout = ship_timeout(man["size"])
    try:
        r = ssh_stream(_compressed(ssh), ["sh", "-c", script, "sh", man["archive_sha256"]], _archive_feed(git_dir, sent), timeout=timeout)
    except subprocess.TimeoutExpired:
        _rm_rf(ssh, partial)
        raise SourceNotReady(f"shipping the source snapshot to {dest} exceeded {timeout:.0f}s ({man['size']} bytes at >= {SHIP_MIN_BYTES_PER_S // 1024} KiB/s); "
                             f"the partial directory was removed, nothing is READY") from None
    lines = r.stdout.decode(errors="replace").splitlines()
    got_sha = lines[0].strip() if lines else ""
    if r.returncode == _RC_DIGEST or (r.returncode == 0 and got_sha != man["archive_sha256"]):
        _rm_rf(ssh, partial)
        why = "truncated transfer?"
        if sent.get("git_error") and not sent.get("cut"):
            why = f"git archive failed while shipping ({sent['git_error']})"
        elif sent.get("sha256") and sent["sha256"] != man["archive_sha256"] and sent.get("bytes") == man["size"]:
            why = f"git archive HEAD is not reproducible here (the second pass hashed {sent['sha256']})"
        raise SourceNotReady(f"source archive sha256 mismatch: the machine received {got_sha or '<nothing>'}, the launcher sent {man['archive_sha256']} "
                             f"({sent.get('bytes', 0)} of {man['size']} bytes written); {why} nothing was extracted, nothing is READY")
    if r.returncode != 0:
        _rm_rf(ssh, partial)
        raise SourceNotReady(f"shipping the source snapshot failed (rc {r.returncode}): {r.stderr.decode(errors='replace')[-400:]}; nothing is READY")
    got_files = {ln.strip().removeprefix("./") for ln in lines[1:] if ln.strip()}
    want_files = set(man["files"])
    if got_files != want_files:
        _rm_rf(ssh, partial)
        missing, extra = sorted(want_files - got_files), sorted(got_files - want_files)
        raise SourceNotReady(f"source tree manifest mismatch after extraction: {len(missing)} missing, {len(extra)} extra "
                             f"(e.g. {(missing or extra)[:3]}); nothing is READY")
    doc = {**_ready_doc(sha, tree, man, "archive-sha256+file-manifest"), "transport": "stream"}
    return sha, dest, _mark_ready(ssh, partial, dest, sha, doc)


def stage_request(ssh: list[str], root: str, run_id: str, request: dict[str, Any], sends: list[Path],
                  private: list[tuple[str, bytes]] | None = None) -> str:
    """Write request.json and any --send files under <root>/requests/<run-id>/ (unique per id; no claim involved).  `private` files
    (the --custody-r2 key) are extracted under umask 077 and chmod 600: readable by the machine's user only."""
    stage_dir = f"{root}/requests/{run_id}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        ti = tarfile.TarInfo("request.json")
        body = (json.dumps(request, indent=1) + "\n").encode()
        ti.size = len(body)
        tf.addfile(ti, io.BytesIO(body))
        for p in sends:
            if not p.is_file():
                raise Refused(f"--send {p}: not a file")
            tf.add(str(p), arcname=f"inputs/{p.name}")
        for name, data in private or []:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), 0o600
            tf.addfile(ti, io.BytesIO(data))
    q = shlex.quote(stage_dir)
    cmd = f"mkdir -p {q} && tar -xf - -C {q}"
    if private:
        cmd = f"umask 077 && {cmd} && chmod 600 " + " ".join(shlex.quote(f"{stage_dir}/{n}") for n, _ in private)
    r = ssh_run(ssh, ["sh", "-c", cmd], stdin=buf.getvalue(), timeout=600)
    if r.returncode != 0:
        raise Refused(f"staging the request failed: {r.stderr.decode(errors='replace')[-400:]}")
    return f"{stage_dir}/request.json"


# ---- the machine side: claim + detached start (runs under the shipped tool) -----------------------------------------------------------

def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _json_line(line: str) -> dict[str, Any]:
    try:
        d = json.loads(line)
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def _holder_live(runs_root: Path, holder: str) -> bool:
    d = runs_root / holder
    st = _read_json(d / "status.json")
    if st and st.get("state") in TERMINAL:
        return False
    launch = _read_json(d / "launch.json") or {}
    if st is None and not _pid_alive(launch.get("runner_pid")):
        return False
    if st and st.get("state") not in TERMINAL and not _pid_alive(st.get("pid")) and not _pid_alive(launch.get("runner_pid")):
        return False
    return True


def take_exclusive(runs_root: Path, run_id: str) -> None:
    lock = runs_root / ".exclusive.lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock.read_text().strip() if lock.exists() else ""
            if holder and holder != run_id and _holder_live(runs_root, holder):
                raise Refused(f"machine is exclusively held by run {holder} (still live); not starting {run_id}")
            try:
                lock.unlink()          # stale: terminal or vanished holder
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            f.write(run_id + "\n")
        return
    raise Refused("could not take the exclusive lock (contended twice)")


def exclusive_holder(runs_root: Path) -> str | None:
    """The run named in a LIVE exclusive lock, else None (a stale lock is removed)."""
    lock = runs_root / ".exclusive.lock"
    try:
        holder = lock.read_text().strip()
    except FileNotFoundError:
        return None
    if holder and _holder_live(runs_root, holder):
        return holder
    try:
        lock.unlink()
    except FileNotFoundError:
        pass
    return None


def release_exclusive(runs_root: Path, run_id: str) -> None:
    lock = runs_root / ".exclusive.lock"
    try:
        if lock.read_text().strip() == run_id:
            lock.unlink()
    except FileNotFoundError:
        pass


def active_runs(runs_root: Path, exclude: str | None = None) -> list[str]:
    out = []
    if not runs_root.is_dir():
        return out
    for d in runs_root.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name == exclude:
            continue
        st = _read_json(d / "status.json")
        launch = _read_json(d / "launch.json") or {}
        if st is None:
            if launch.get("runner_pid") and _pid_alive(launch["runner_pid"]) and not (d / "refused.json").exists():
                out.append(d.name)
        elif st.get("state") not in TERMINAL:
            out.append(d.name)
    return out


def _environ_of(path: str) -> dict[str, str]:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep:
            out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def _guard_pid(gdir: Path) -> int | None:
    """The pid in <root>/guard/guard.pid when it is a live guard daemon (not a recycled pid), else None."""
    try:
        pid = int((gdir / "guard.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    if not _pid_alive(pid):
        return None
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        cmd = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True).stdout
    return pid if "pod_guard.sh daemon" in cmd else None


def ensure_guard(root: Path, idle_min: float, *, pod_id: str | None = None, environ: str | None = None) -> dict[str, Any]:
    """One guard daemon per pod (pods/sh/pod_guard.sh `daemon`: terminates the pod after idle_min idle minutes once every run dir is
    fetched).  Installed only with a pod-scoped key, RunPod's RUNPOD_API_KEY in the container's first process; nothing is copied to the
    pod for it.  Without that key, or when this machine is not the pod the launcher resolved, nothing is started and `reason` says why."""
    environ = environ or GUARD_ENVIRON
    penv = _environ_of(environ)
    me = penv.get("RUNPOD_POD_ID")
    if not me:
        return {"installed": False, "reason": f"not a RunPod pod (no RUNPOD_POD_ID in {environ}): no guard"}
    if pod_id and me != pod_id:
        return {"installed": False, "reason": f"this machine is pod {me}, the request names pod {pod_id}: no guard"}
    if not penv.get("RUNPOD_API_KEY"):
        return {"installed": False, "reason": f"pod {me} has no pod-scoped RUNPOD_API_KEY in {environ}: guard NOT installed, the pod will "
                                              "not terminate itself (the account key is never copied to a pod)"}
    gdir = root / "guard"
    gdir.mkdir(parents=True, exist_ok=True)
    base = {"pod_id": me, "idle_min": idle_min, "state": str(gdir / "state.json"), "log": str(gdir / "guard.log")}
    pid = _guard_pid(gdir)
    if pid:
        return {"installed": True, "started": False, "pid": pid, **base}
    from .pods import workproc

    env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}     # the daemon reads the key from `environ` when it terminates
    env.update(workproc.guard_env(), GUARD_TERMINAL=" ".join(TERMINAL), GUARD_ENVIRON=environ)
    with open(gdir / "daemon.log", "ab") as log:
        proc = subprocess.Popen(["bash", str(GUARD_SCRIPT), "daemon", str(root), f"{idle_min:g}"], cwd=str(gdir), env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):                       # the daemon claims the pidfile within a moment, or exits because another guard holds it
        pid = _guard_pid(gdir)
        if pid or proc.poll() is not None:
            break
        time.sleep(0.05)
    pid = pid or _guard_pid(gdir)
    if not pid:
        return {"installed": False, "reason": f"the guard daemon did not start (exit {proc.poll()}; see {gdir / 'daemon.log'})", **base}
    return {"installed": True, "started": pid == proc.pid, "pid": pid, **base}


def tool_preflight(harness_args: list[str], tree: Path | str | None, asked: str | None = None) -> list[str] | None:
    """The preflight to run on `tree` before the run: the one `--preflight TOOL` (`asked`) names, else the one the run's `--tool`
    declares (`Tool.preflight`), else None (no --tool, no tree, or a --tool the tree does not declare or that names none).
    PreflightFailed when a declaration is there but cannot be read (it may name a preflight), or when `asked` has none to run."""
    spec = asked or flag_value(harness_args, "--tool")
    flag = "--preflight" if asked else "--tool"
    if not spec or tree is None:
        if asked:
            raise PreflightFailed(f"--preflight {asked}: the run ships no tree to run it from; nothing started")
        return None
    from .store import tools_registry

    if tools_registry.locate(spec, Path(tree)) is None:
        if asked:
            raise PreflightFailed(f"--preflight {asked}: {tree} does not declare that tool; nothing started")
        return None
    try:
        decl = tools_registry.from_tree(spec, Path(tree))
    except Exception as e:  # noqa: BLE001
        raise PreflightFailed(f"{flag} {spec}: its declaration in {tree} cannot be read ({type(e).__name__}: {e}), so its preflight is "
                              f"unknown; nothing started") from None
    argv = list(getattr(decl, "preflight", None) or [])
    if not argv and asked:
        raise PreflightFailed(f"--preflight {asked}: its declaration in {tree} names no preflight; nothing started")
    return argv or None


def _tail(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()][-PREFLIGHT_TAIL:]


def run_preflight(argv: list[str], tree: Path | str, env: dict[str, str]) -> dict[str, Any]:
    """Run a `Tool.preflight` (a Python script in `tree`, and its args) by this interpreter, in `tree`; its report when it exits 0,
    else PreflightFailed naming its stderr (its stdout when stderr is empty)."""
    cmd = [sys.executable, *argv]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, cwd=str(tree), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=PREFLIGHT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise PreflightFailed(f"preflight {shlex.join(argv)} did not finish in {PREFLIGHT_TIMEOUT_S} s; nothing started",
                              {"argv": argv, "rc": None, "seconds": PREFLIGHT_TIMEOUT_S}) from None
    rep = {"argv": argv, "rc": r.returncode, "seconds": round(time.monotonic() - t0, 1), "stdout": _tail(r.stdout), "stderr": _tail(r.stderr)}
    if r.returncode:
        why = "; ".join(rep["stderr"] or rep["stdout"][-5:]) or "no output"
        raise PreflightFailed(f"the machine failed the preflight {shlex.join(argv)} (exit {r.returncode}): {why}", rep)
    return rep


def launch_request(request_path: Path) -> dict[str, Any]:
    """Claim the run directory and start the detached runner (idempotent per run id); with the request's `guard`, make sure the pod's
    guard daemon runs (ensure_guard; its answer is the result's `guard`, and a guard that cannot start never fails the launch)."""
    request_path = Path(request_path).resolve()        # the runner finds the ack FIFO from this path, whatever its cwd
    with open(request_path.parent / LAUNCH_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = _launch_request(request_path)
    try:
        req = json.loads(request_path.read_text())
        g = req.get("guard")
    except (OSError, ValueError):
        req, g = {}, None
    if req.get("lease_extend_s") and out.get("started"):
        from .pods import lease

        try:
            out["lease"] = lease.machine_extend(float(req["lease_extend_s"]), f"run-{req.get('run_id')}")
        except Exception as e:  # noqa: BLE001  (the run is started; the answer says the lease could not be touched)
            out["lease"] = {"leased": None, "error": f"{type(e).__name__}: {e}"}
    if g:
        try:
            out["guard"] = ensure_guard(Path(req["runs_root"]).parent, float(g.get("idle_min") or GUARD_IDLE_MIN_DEFAULT),
                                        pod_id=(req.get("remote") or {}).get("pod_id"))
        except Exception as e:  # noqa: BLE001
            out["guard"] = {"installed": False, "reason": f"guard failed to start: {type(e).__name__}: {e}"}
    return out


def _launch_request(request_path: Path) -> dict[str, Any]:
    req = json.loads(request_path.read_text())
    if req.get("schema") != REQUEST_SCHEMA:
        raise Refused(f"unknown request schema {req.get('schema')!r}")
    run_id = req["run_id"]
    runs_root = Path(req["runs_root"])
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / run_id
    try:
        run_dir.mkdir(exist_ok=False)              # the claim
    except FileExistsError:
        launch = _read_json(run_dir / "launch.json") or {}
        st = _read_json(run_dir / "status.json") or {}
        refused = _read_json(run_dir / "refused.json")
        return {"claimed": False, "run_id": run_id, "run_dir": str(run_dir), "existing": True, "state": st.get("state"),
                "runner_pid": launch.get("runner_pid"), "refused": (refused or {}).get("reason"), "fault": (refused or {}).get("fault")}
    concurrent = active_runs(runs_root, exclude=run_id)
    src_dir = None
    mem_rec: dict[str, Any] | None = None
    preflight: dict[str, Any] | None = None
    remote = dict(req.get("remote") or {})
    remote["request"] = str(request_path)
    try:
        if (marker := Path(os.environ.get("RUNS_VIA_SLOT_FILE") or RUNS_VIA_SLOT_FILE)).exists():
            from .pods.nebius import slot_only

            if not slot_only.through_slot(req["command"]):
                raise Refused(slot_only.refusal(marker))
        if remote.get("source_sha"):              # a shipped tree executes only from a directory verified READY for the requested identity
            src_dir = remote.get("source_dir") or str(runs_root.parent / "src" / str(remote["source_sha"]))
            require_ready_source(src_dir, str(remote["source_sha"]))
        env = _runner_env(req, runs_root, src_dir, remote)
        if req.get("custody") and not (request_path.parent / CUSTODY_DIR / "cred.json").is_file():
            raise Refused(f"--custody-r2: no staged key under {request_path.parent / CUSTODY_DIR}; the run could not publish its custody")
        if req.get("job_user"):
            from . import jobuser

            try:
                jobuser.check_user(req["job_user"])
                jobuser.check_run_dir(run_dir, run_id)
            except jobuser.Refused as e:
                raise Refused(str(e)) from None
            remote["job_user"] = req["job_user"]
        holder = exclusive_holder(runs_root)
        if holder and holder != run_id:
            raise Refused(f"machine is exclusively held by run {holder} (still live); not starting {run_id}")
        if pf_argv := tool_preflight(req.get("harness_args") or [], src_dir, req.get("preflight_tool")):
            pf_env = {**env, "RESEARCH_PREFLIGHT_INPUTS": str(request_path.parent / "inputs")}
            if req.get("custody"):
                pf_env["RESEARCH_PREFLIGHT_CUSTODY"] = str(request_path.parent / CUSTODY_DIR)
            remote["preflight"] = preflight = run_preflight(pf_argv, src_dir, pf_env)
        if req.get("mem_gb"):
            from . import mem as M

            mem_rec = M.admit(float(req["mem_gb"]), ramlock_dir=Path(req.get("ramlock_dir") or M.RAMLOCK_DIR), run_id=run_id, run_dir=run_dir,
                              wait_s=float(req.get("mem_wait_s") or 0), note=f"research run {run_id}: {shlex.join(req['command'])}")
        if req.get("exclusive"):
            take_exclusive(runs_root, run_id)
            if concurrent:
                release_exclusive(runs_root, run_id)
                raise Refused(f"--exclusive: other run(s) are live on this machine: {', '.join(concurrent)}; not starting {run_id}")
        ack = request_path.parent / LAUNCH_ACK
        try:
            log = open(run_dir / "launcher.log", "ab")
            ack.unlink(missing_ok=True)
            os.mkfifo(ack, 0o600)
            ack_fd = os.open(ack, os.O_RDONLY | os.O_NONBLOCK)      # before the runner starts, so its non-blocking open finds a reader
        except OSError as e:
            raise Refused(f"could not prepare the launch ({type(e).__name__}: {e}); not starting {run_id}") from None
    except Refused as e:
        _release_launch(req, request_path, mem_rec)
        pf = {"preflight": e.report} if isinstance(e, PreflightFailed) and e.report else {}
        write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": str(e), "fault": type(e).__name__, "at_utc": utc_now_ms(), **pf})
        return {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": False, "refused": str(e), "fault": type(e).__name__, **pf}
    staged_inputs = request_path.parent / "inputs"
    if staged_inputs.is_dir():
        os.replace(staged_inputs, run_dir / "inputs")
    cost = dict(req.get("cost") or {})
    cost["shared"] = bool(concurrent)
    cost["concurrent_runs_at_launch"] = concurrent
    if mem_rec:
        remote["mem"] = mem_rec
        env["RESEARCH_MEM_RESERVATION"] = mem_rec["reservation"]      # the runner removes it at exit and does not pass it on
    cpus = direct_cpus()
    if cpus:
        remote["cpus"] = f"{format_cpus(cpus)}: {os.environ.get('DIRECT_CPUS_FILE') or DIRECT_CPUS_FILE} (a command may pin itself elsewhere)"
    argv = [req.get("python") or sys.executable, "-m", "research", "run", "--runs-dir", str(runs_root), "--id", run_id, "--claimed",
            "--cost-json", json.dumps(cost), "--remote-json", json.dumps(remote)]
    if req.get("exclusive"):
        argv += ["--exclusive-lock", str(runs_root / ".exclusive.lock")]
    if req.get("custody"):
        argv += ["--custody-dir", str(request_path.parent / CUSTODY_DIR)]
    if req.get("job_user"):
        argv += ["--job-user", req["job_user"]]
    argv += [*(req.get("harness_args") or []), "--", *(req.get(QUEUE_WORKLOAD_WRAP) or []), *req["command"]]
    if req.get(QUEUE_WRAP):                        # --queue: the runner, custody included, runs inside the job's CPU allocation
        argv = [*req[QUEUE_WRAP], *argv]
    if (req.get("queue") or {}).get("disk_gb"):    # held here, ahead of the allocation, while the disk would pass 80% (research.disk)
        argv = [req.get("python") or sys.executable, "-m", "research", "disk-hold", "--gb", str(req["queue"]["disk_gb"]),
                "--path", str(runs_root), "--ack", str(request_path.parent / LAUNCH_ACK), "--", *argv]
    started = {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": True, **({"preflight": preflight} if preflight else {})}
    try:
        try:
            proc = subprocess.Popen(argv, cwd=req.get("cwd") or None, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None)
        except OSError as e:
            _release_launch(req, request_path, mem_rec)
            reason = f"the runner could not be started: {type(e).__name__}: {e}"
            write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": reason, "fault": RunnerFailed.__name__, "at_utc": utc_now_ms()})
            return {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": False, "refused": reason, "fault": RunnerFailed.__name__}
        started["runner_pid"] = proc.pid
        try:
            if mem_rec:
                M.set_pid(mem_rec["reservation"], proc.pid)
            write_json_atomic(run_dir / "launch.json", {"run_id": run_id, "claimed_utc": utc_now_ms(), "runner_pid": proc.pid, "argv": argv,
                                                        "cwd": req.get("cwd"), "request": str(request_path), "hostname": os.uname().nodename})
            first = await_first_record(proc, ack_fd)
        except Exception as e:  # noqa: BLE001  (the runner is started: an error here leaves its launch unconfirmed, never refused)
            return {**started, "first_record": None, "unconfirmed": f"{type(e).__name__}: {e}"}
    finally:
        log.close()
        os.close(ack_fd)
        ack.unlink(missing_ok=True)
    if first is None:
        return _runner_failed(req, request_path, run_dir, proc, mem_rec, started)
    return {**started, "first_record": first}


def ack_launch(path: Path | str, what: str) -> None:
    """Tell the launch-request waiting on the FIFO `path` that `what` (the run's first record, or `disk-hold`) is in place.  Never
    blocks: no FIFO there, or nobody reading it any more (ENXIO), means nobody waits on this launch."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        if e.errno not in (errno.ENOENT, errno.ENXIO):
            print(f"research: could not acknowledge the launch on {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return
    try:
        if stat.S_ISFIFO(os.fstat(fd).st_mode):
            os.write(fd, what.encode() + b"\n")
    except OSError:
        pass
    finally:
        os.close(fd)


def await_first_record(proc: subprocess.Popen, ack_fd: int) -> str | None:
    """What the runner named on the ack FIFO (`ack_launch`), or None once `proc` (the runner, or the wrapper it runs under) exited
    without a word.  Waits on those two events (the FIFO and a pidfd of `proc`), never on a clock: a run that is up acknowledges
    in the time its runner takes to parse its arguments and write job.json, then detaches."""
    try:
        pid_fd = os.pidfd_open(proc.pid)
    except OSError:                        # no pidfd (kernel < 5.3, or a seccomp profile without it): a pipe closed at the exit
        pid_fd, w = os.pipe()
        threading.Thread(target=lambda: (proc.wait(), os.close(w)), daemon=True, name=f"exit-{proc.pid}").start()
    watch = [ack_fd, pid_fd]
    try:
        while True:
            ready, _, _ = select.select(watch, [], [])
            try:
                got = os.read(ack_fd, 256)
            except BlockingIOError:
                got = None
            if got:
                return got.decode(errors="replace").split()[0]
            if pid_fd in ready:
                proc.wait()
                return None
            if got == b"":                 # a writer opened and closed it without a word: only the exit is left to wait on
                watch = [pid_fd]
    finally:
        os.close(pid_fd)


def _release_launch(req: dict[str, Any], request_path: Path, mem_rec: dict[str, Any] | None) -> None:
    """What a launch took and a runner would release at its exit: the memory reservation, the exclusive lock, the custody key (nothing
    will publish: the key goes now, not at its expiry), and the sent URLs no run will take out of the staged inputs/.  Each release is
    idempotent and touches only this run's."""
    if mem_rec:
        from .mem import release

        release(mem_rec["reservation"], req["run_id"])
    if req.get("exclusive"):
        release_exclusive(Path(req["runs_root"]), req["run_id"])
    if req.get("custody"):
        (request_path.parent / CUSTODY_DIR / "cred.json").unlink(missing_ok=True)
    from .store.custody import drop_credentials

    drop_credentials(request_path.parent)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _runner_failed(req: dict[str, Any], request_path: Path, run_dir: Path, proc: subprocess.Popen, mem_rec: dict[str, Any] | None,
                   started: dict[str, Any]) -> dict[str, Any]:
    """The answer for a runner that exited before its first record: its own refusal when it wrote one (ResearchVersionMismatch), else
    RunnerFailed with its exit code and launcher.log's tail, recorded in refused.json, and what the launch took goes now.  Unless the
    run may be live: job.json is there (its ack went astray) or the process went but its session's group lives on (a wrap that forked
    the runner off), which answers started and `unconfirmed` and releases nothing."""
    run_id = req["run_id"]
    if (run_dir / FIRST_RECORD).exists():
        return {**started, "first_record": None, "rc": proc.returncode,
                "unconfirmed": f"no ack arrived, but the runner wrote {FIRST_RECORD} (exit {proc.returncode})"}
    if _group_alive(proc.pid):                  # start_new_session: the session's process group is the runner's pid
        return {**started, "first_record": None, "rc": proc.returncode,
                "unconfirmed": f"pid {proc.pid} exited {proc.returncode} before {FIRST_RECORD}, but its process group lives on"}
    if own := _read_json(run_dir / "refused.json"):
        reason, fault = str(own.get("reason")), str(own.get("fault") or "Refused")
    else:
        try:
            tail = _tail((run_dir / "launcher.log").read_text(errors="replace"))
        except OSError:
            tail = []
        reason = (f"the runner exited {proc.returncode} before writing {FIRST_RECORD}: "
                  + ("; ".join(tail[-RUNNER_FAILED_LINES:]) or "nothing in launcher.log"))
        fault = RunnerFailed.__name__
        write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": reason, "fault": fault, "rc": proc.returncode,
                                                     "log_tail": tail, "at_utc": utc_now_ms()})
    _release_launch(req, request_path, mem_rec)
    return {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": False, "refused": reason, "fault": fault,
            "runner_pid": proc.pid, "rc": proc.returncode}


def _runner_env(req: dict[str, Any], runs_root: Path, src_dir: str | None, remote: dict[str, Any]) -> dict[str, str]:
    """The runner's environment, which a preflight also runs in: the workload's --env, and with a shipped tree its PYTHONPATH behind
    the tool snapshot (recorded in `remote`)."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (req.get("env") or {}).items()})
    if remote.get("source_sha"):
        # [R19 tele] a tree the archive stream shipped has no .git: the workload's source-identity precheck reads the tree's sha from
        # here (research.result.ENV_SOURCE_SHA) when neither --sha nor EXPORT.json names it
        env["RESEARCH_SOURCE_SHA"] = str(remote["source_sha"])
        env.setdefault("PYTHONPYCACHEPREFIX", str(runs_root.parent / "cache" / "pycache"))   # imports from the shared tree do not write into it
        from .pythonpath import import_roots_dir, pythonpath

        derived = import_roots_dir(src_dir)                 # the tree's own workspace declaration, not a hand list (pythonpath.py)
        env["PYTHONPATH"] = pythonpath(src_dir, derived, extra=env.get("PYTHONPATH"), shim_base=runs_root.parent / "cache" / "pyshim")
        remote["pythonpath"] = [p for p in env["PYTHONPATH"].split(":") if p]
        remote["pythonpath_derived"] = {"roots": derived["roots"], "missing": derived["missing"], "source": derived["source"],
                                        **({"shim": derived["shim"]} if derived["shim"] else {})}
    if req.get("tool"):
        env["PYTHONPATH"] = req["tool"] + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["RESEARCH_TOOL_SNAPSHOT"] = Path(req["tool"]).name
    if _kueue_owns_every_gpu():
        # a direct run must not open CUDA beside the queue's pods, whatever its --env says: GPU work goes through Kueue
        env["CUDA_VISIBLE_DEVICES"] = ""
        remote["gpus"] = f"none: {DIRECT_GPUS_FILE} says every GPU is Kueue's (CUDA_VISIBLE_DEVICES='')"
    return env


def direct_cpus() -> set[int]:
    """The CPUs a direct run starts on, from DIRECT_CPUS_FILE (tests: $DIRECT_CPUS_FILE), intersected with this process's own; empty
    when the file is absent or names none of them."""
    try:
        text = Path(os.environ.get("DIRECT_CPUS_FILE") or DIRECT_CPUS_FILE).read_text()
    except OSError:
        return set()
    want: set[int] = set()
    for part in text.replace("\n", ",").split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit() and (not hi or hi.isdigit()):
            want |= set(range(int(lo), int(hi or lo) + 1))
    return want & os.sched_getaffinity(0)


def format_cpus(cpus: set[int]) -> str:
    """`0-31,64-95` for a CPU set."""
    out, run = [], []
    for c in sorted(cpus):
        if run and c != run[-1] + 1:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = []
        run.append(c)
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def _kueue_owns_every_gpu() -> bool:
    try:   # GPU_LEASE_ALLOWED_FILE: the same override gpu-lease reads (tests point it away from the host's file)
        return Path(os.environ.get("GPU_LEASE_ALLOWED_FILE") or DIRECT_GPUS_FILE).read_text().strip() == "none"
    except OSError:
        return False


# ---- the control side --------------------------------------------------------------------------------------------------------------------

def flag_value(args: list[str], flag: str) -> str | None:
    """The value of `flag X` / `flag=X` in an argv list (last one wins), else None."""
    val = None
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith(flag + "="):
            val = a.split("=", 1)[1]
    return val


def lease_extend_s(harness_args: list[str]) -> float | None:
    """How far the machine's lease must reach for this run: its --timeout plus the runner's publish and custody push
    (research.pods.lease.RUN_MARGIN_S); None without a --timeout."""
    from .pods.lease import RUN_MARGIN_S

    t = timeout_seconds(flag_value(harness_args, "--timeout"))
    return t + RUN_MARGIN_S if t and t > 0 else None


def timeout_seconds(text: str | None) -> float | None:
    """A --timeout (`1.5`, `900`, `40m`, `6h`, `7d`: seconds or credentials.parse_ttl's grammar) in seconds; None when absent.
    ValueError when malformed."""
    from .store.credentials import parse_ttl

    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return float(parse_ttl(text))


def lease_line(lz: dict[str, Any] | None, asked: float | None, machine: str) -> str:
    """The launcher's one line on the machine's lease after a launch."""
    if asked is None:
        return (f"{machine}'s lease was not extended: the run has no --timeout, so a pod lease (if any) may end it; "
                f"`research pods extend {machine} --hours H` extends it by hand")
    if lz is None:
        return f"{machine} answered without a lease report (its research package predates leases): the lease, if any, was not extended"
    if lz.get("leased") is False:
        return f"{machine} has no lease ({lz.get('reason')}): nothing ends it but its guards"
    if lz.get("extended"):
        return f"{machine}'s lease now ends no sooner than {lz.get('expires_utc')} (the run's --timeout plus its publish margin)"
    return f"{machine}'s lease could NOT be extended ({lz.get('error')}): it may end the run; `research pods extend {machine} --hours H`"


def workload_cwd(harness_args: list[str], tool_pkg: Path | None) -> str:
    """What the machine will use as the workload's cwd, for remote.json: the `--cwd` asked for, else the run dir -- unless the package
    that runs on the machine (the shipped tree's, `tool_pkg`) predates WORKLOAD_CWD_DEFAULT and so still runs in the shared tree."""
    asked = flag_value(harness_args, "--cwd")
    if asked is not None and clone.sub(asked) is not None:
        return f"--cwd {asked} (the run's own clone of the shipped tree, <root>/scratch/<run-id>/tree, removed when the workload exits)"
    if asked is not None:
        return f"--cwd {asked}"
    if tool_pkg is not None:
        try:
            knows = "WORKLOAD_CWD_DEFAULT" in (Path(tool_pkg) / "remote.py").read_text()
        except OSError:
            knows = False
        if not knows:
            return "source (the shipped tree's research package predates the per-run workload cwd: it runs in <root>/src/<sha>/)"
    return WORKLOAD_CWD_DEFAULT

def launch(*, run_id: str, local_dir: Path, machine_name: str, project: str, command: list[str], harness_args: list[str],
           source: Path | None, sends: list[Path], cwd: str | None, env: dict[str, str], exclusive: bool,
           mem_gb: float | None = None, mem_wait_s: float = 0.0, custody_ttl: str | None = None,
           preflight_tool: str | None = None, queue: dict[str, Any] | None = None, job_user: str | None = None) -> dict[str, Any]:
    """`custody_ttl` (--custody-r2): mint a delete-free key for that long and stage it for the machine's runner (store.custody).
    `preflight_tool` (--preflight): run that tool's declared preflight on the machine before this run, whatever the run's --tool.
    `queue` (--queue): the queue's placement (queue_placement); the machine runs the runner under its `wrap`, so the allocation
    is held until the runner, custody included, exits.  `job_user` (--job-user): the machine runs the workload as that user
    (research.jobuser), and refuses the run when it can't."""
    local_dir.mkdir(parents=True, exist_ok=False)
    rec: dict[str, Any] = {"schema": "research/remote/v0.1", "run_id": run_id, "machine": machine_name, "project": project, "phase": "resolving",
                           "created_utc": utc_now_ms(), "observed_utc": None, "last_observed": None, "command": command, "launch_args": harness_args,
                           **({"queue": queue} if queue else {})}
    write_json_atomic(local_dir / "remote.json", rec)
    try:
        m = resolve_machine(machine_name, project)
        guard_min = guard_idle_min(m)
        custody_pub: dict[str, Any] | None = None
        custody_files: list[tuple[str, bytes]] = []
        if custody_ttl:
            if not knows_custody(source_tool_package(source)):
                raise Refused(f"--custody-r2: the research package in {source} predates run custody (no research/store/custody.py); "
                              f"the machine would not publish it.  Commit a tree that has it, or pass --no-custody-r2")
            from .store import credentials as C
            from .store import custody as CU

            try:
                custody_pub, custody_files = CU.mint_for_machine(custody_ttl)
            except C.MintError as e:
                raise Refused(f"--custody-r2: cannot mint the machine's key here: {e} (the parent key: the Cursor secrets, or source "
                              f"~/.config/verity/r2.env; --no-custody-r2 launches without run custody)") from None
            rec["custody"] = custody_pub
        tgt = pod_target(m)
        user = tgt.get("user", "root")
        ssh = ssh_argv(tgt["host"], tgt["port"], user=user)
        rec.update(pod_id=m.get("pod_id"), host=tgt["host"], port=tgt["port"], user=user, root=m["root"], phase="shipping")
        write_json_atomic(local_dir / "remote.json", rec)
        tool_pkg = source_tool_package(source)      # --source <tree> with tools/research: the tree's own package runs the run
        if mem_gb and tool_pkg is not None and not (tool_pkg / "mem.py").exists():
            raise Refused(f"--mem-gb: the research package in {source} predates memory admission (no research/mem.py); the machine would "
                          f"ignore the reservation.  Commit a tree that has it, or drop --mem-gb")
        if preflight_tool and not knows_preflight(tool_pkg):
            raise Refused(f"--preflight: the research package in {source} predates it, so the machine would not run the preflight.  Commit "
                          f"a tree that has it, or drop --preflight")
        if queue and not knows_queue(tool_pkg):
            raise Refused(f"--queue: the research package in {source} predates it, so the machine would run the job outside its "
                          f"allocation.  Commit a tree that has it")
        if job_user and not knows_job_user(tool_pkg):
            raise Refused(f"--job-user: the research package in {source} predates it, so the machine would run the workload as its "
                          f"own user.  Commit a tree that has it")
        cwd_clone = clone.sub(flag_value(harness_args, "--cwd")) is not None
        if cwd_clone and not knows_clone(tool_pkg):
            raise Refused(f"--cwd clone: the research package in {source} predates it (no research/clone.py), so the machine would look "
                          f"for a directory named clone in the shared tree.  Commit a tree that has it, or pass --cwd source")
        tool_dir = ship_tool(ssh, m["root"], tool_pkg)
        rec["tool_from"] = str(tool_pkg) if tool_pkg else "launcher"
        rec["tool_identity"] = tool_identity(tool_pkg)
        rec["tool_checked"] = knows_version_check(tool_pkg)
        if not rec["tool_checked"]:
            print(f"research: the machine runs {tool_pkg}'s research package (identity {rec['tool_identity']}; this launcher's is "
                  f"{tool_identity()}), which predates the version check: it is not verified on the machine", file=sys.stderr)
        rec["workload_cwd"] = workload_cwd(harness_args, tool_pkg)
        if rec["workload_cwd"].startswith("source (") and source is not None:
            print(f"research: {rec['workload_cwd']}; concurrent runs of this commit share it", file=sys.stderr)
        tree_sha = None
        src_dir = None
        ready: dict[str, Any] = {}
        remote_cwd = cwd
        if source is not None:
            tree_sha, src_dir, ready = ship_source(ssh, m["root"], source)
            if cwd_clone and not ready.get("checkout"):
                raise Refused(f"--cwd clone: {src_dir} was shipped without git (transport {ready.get('transport') or 'unrecorded'}), so it "
                              f"has no .git to clone.  Pass --cwd source to run in it as it is, shared with other runs of the commit")
            remote_cwd = src_dir if cwd is None else str(Path(src_dir) / cwd)
            if "--tree-sha" not in harness_args:
                harness_args = [*harness_args, "--tree-sha", tree_sha]
        request = {"schema": REQUEST_SCHEMA, "run_id": run_id, "runs_root": f"{m['root']}/runs", "tool": tool_dir, "python": m["python"],
                   "cwd": remote_cwd, "env": env, "harness_args": harness_args, "command": command, "exclusive": exclusive,
                   "mem_gb": mem_gb, "mem_wait_s": mem_wait_s, "ramlock_dir": m.get("ramlock"),
                   "guard": {"idle_min": guard_min} if guard_min else None,
                   "lease_extend_s": lease_extend_s(harness_args),
                   **({"custody": custody_pub} if custody_pub else {}),
                   **({"submitted_by": by} if (by := os.environ.get("RESEARCH_AGENT_URL")) else {}),
                   **({"preflight_tool": preflight_tool} if preflight_tool else {}),
                   **({"job_user": job_user} if job_user else {}),
                   **({QUEUE_WRAP: queue["wrap"], QUEUE_WORKLOAD_WRAP: queue.get("workload_wrap") or [], "queue": queue} if queue else {}),
                   "cost": {"provider": m.get("provider", "runpod"), "pod_id": m.get("pod_id"), "machine": machine_name, "currency": "USD",
                            "rate_usd_per_hour": tgt["cost_per_hr"], "rate_observed_utc": tgt["rate_observed_utc"], "reason": None},
                   "remote": {"machine": machine_name, "pod_id": m.get("pod_id"), "pod_name": tgt["pod_name"], "host": tgt["host"], "port": tgt["port"],
                              "root": m["root"], "tool": tool_dir, "tool_identity": rec["tool_identity"] if rec["tool_checked"] else None,
                              "source_sha": tree_sha, "source_dir": src_dir,
                              "source_archive_sha256": ready.get("archive_sha256"), "source_files": ready.get("files"),
                              "source_verified": ready.get("verified"), "source_transport": ready.get("transport"),
                              "launched_from": os.uname().nodename}}
        req_path = stage_request(ssh, m["root"], run_id, request, sends, private=custody_files or None)
        rec.update(phase="launching", request=req_path, remote_run_dir=f"{m['root']}/runs/{run_id}", tool=tool_dir, source_sha=tree_sha,
                   source_archive_sha256=ready.get("archive_sha256"), rate_usd_per_hour=tgt["cost_per_hr"])
        write_json_atomic(local_dir / "remote.json", rec)
        last_err = None
        expect = ["--expect-tool", rec["tool_identity"]] if rec["tool_checked"] else []
        deadline = 120 + PREFLIGHT_TIMEOUT_S + (mem_wait_s or 0)
        for attempt in range(3):
            try:
                r = ssh_run(ssh, ["env", f"PYTHONPATH={tool_dir}", m["python"], "-m", "research", "launch-request", *expect, req_path],
                            timeout=deadline)
            except subprocess.TimeoutExpired:  # the request goes on on the machine: a re-delivery waits on its lock for the answer
                last_err = f"no answer from launch-request in {deadline:g} s"
                time.sleep(3 * (attempt + 1))
                continue
            if r.returncode == 0:
                out = json.loads(r.stdout.decode().strip().splitlines()[-1])
                break
            last_err = r.stderr.decode(errors="replace")[-400:]
            if r.returncode != 255:            # 255 = ssh transport failure: re-deliver the SAME request (idempotent claim)
                raise launch_request_failure(r.returncode, r.stdout, r.stderr)
            time.sleep(3 * (attempt + 1))
        else:
            rec.update(phase="unknown", note=f"3 deliveries of the staged request failed; it may or may not have started: {last_err}")
            write_json_atomic(local_dir / "remote.json", rec)
            raise Refused(rec["note"])
        rec.update(phase="refused" if out.get("refused") else "launched", launch=out, observed_utc=utc_now_ms())
        if pf := out.get("preflight"):
            rec["preflight"] = pf
            print(f"research: preflight {shlex.join(pf['argv'])} on {machine_name}: {'FAILED' if out.get('refused') else 'passed'} "
                  f"in {pf.get('seconds')} s", file=sys.stderr)
            for line in pf.get("stdout") or []:
                print(f"research: preflight: {line}", file=sys.stderr)
        if guard_min:
            g = out.get("guard") or {"installed": False, "reason": "the machine's research package predates the guard (no `guard` in its answer)"}
            rec["guard"] = g
            if g.get("installed"):
                print(f"research: guard on pod {g.get('pod_id')}: pid {g.get('pid')} ({'started' if g.get('started') else 'already running'}), "
                      f"terminates the pod after {g.get('idle_min'):g} idle min once every run is fetched (`research fetch --all`); "
                      f"state {g.get('state')}", file=sys.stderr)
            else:
                print(f"research: NO guard on {machine_name}: {g.get('reason')}", file=sys.stderr)
        if not out.get("refused"):
            rec["lease"] = lz = out.get("lease")
            print(f"research: {lease_line(lz, request['lease_extend_s'], machine_name)}", file=sys.stderr)
        write_json_atomic(local_dir / "remote.json", rec)
        return rec
    except Refused as e:
        rec.update(phase=rec["phase"] if rec.get("phase") in ("unknown",) else "refused", error=str(e))
        write_json_atomic(local_dir / "remote.json", rec)
        raise


def adopt(runs_root: Path, run_id: str, *, on: str | None = None, machines: dict[str, dict[str, Any]] | None = None,
          unique: bool = False) -> Path:
    """The local run dir of a run another VM launched: find <root>/runs/<run_id> on machine `on`, else on each ssh machine, write
    the remote.json `launch` would have (phase `adopted`), and fetch its records.  Refused when the id is not a full run id, when no
    machine holds it (each machine's answer in the reason), or when the local dir already exists; with `unique` (a cancel), every
    machine is asked and an id more than one holds is refused rather than taken from the first."""
    from .store import ids

    if not ids.is_run_id(run_id):
        raise Refused(f"{run_id!r} is not a full run id (rYYYYMMDD-HHMMSS-xxxx), so it is not looked up on the machines")
    local = runs_root / run_id
    if local.exists():
        raise Refused(f"{local} exists already")
    ms = load_machines() if machines is None else machines
    names = [on] if on else sorted(n for n, m in ms.items() if m.get("provider") == "ssh")
    answers, held = [], []
    for name in names:
        m = ms.get(name)
        if m is None:
            raise Refused(f"machine {name!r} is not registered")
        try:
            tgt = pod_target(m)
        except Refused as e:
            answers.append(f"{name}: {e}")
            continue
        m = {**m, "root": m.get("root") or DEFAULT_ROOT}
        rd = f"{m['root']}/runs/{run_id}"
        ssh = ssh_argv(tgt["host"], tgt["port"], user=tgt.get("user", "root"))
        try:
            r = ssh_run(ssh, ["test", "-d", rd], timeout=60)
        except subprocess.TimeoutExpired:
            answers.append(f"{name}: no answer in 60 s")
            continue
        if r.returncode != 0:
            answers.append(f"{name}: " + (f"no {rd}" if r.returncode == 1 else f"ssh rc {r.returncode}: {r.stderr.decode(errors='replace').strip()[-160:]}"))
            continue
        held.append((name, m, tgt, rd))
        if not unique:
            break
    if len(held) > 1:
        raise Refused(f"{' and '.join(h[0] for h in held)} each hold a run {run_id}: name one with --on")
    if held:
        name, m, tgt, rd = held[0]
        local.mkdir(parents=True, exist_ok=False)
        write_json_atomic(local / "remote.json", {
            "schema": "research/remote/v0.1", "run_id": run_id, "machine": name, "project": m.get("project"), "phase": "adopted",
            "created_utc": utc_now_ms(), "observed_utc": None, "last_observed": None, "command": None, "launch_args": [],
            "pod_id": m.get("pod_id"), "host": tgt["host"], "port": tgt["port"], "user": tgt.get("user", "root"), "root": m["root"],
            "remote_run_dir": rd})
        fetch(local)
        return local
    raise Refused(f"no machine holds run {run_id}" + (f" ({'; '.join(answers)})" if answers else " (no ssh machine is registered)"))


def _runner_gone(ssh: list[str], local_dir: Path, pid: int) -> str | None:
    """Why a run whose status isn't terminal has no runner on its machine, or None while it may have one: the machine rebooted
    since the runner started (job.json's boot id), or the pid is gone where the runner ran in the machine's own pid namespace
    (job.json's hostname is the machine's; a Kubernetes pod's runner isn't visible from the host). Unreadable is None."""
    env = (_read_json(local_dir / "job.json") or {}).get("environment") or {}
    if not env.get("boot_id"):
        return None
    probe = ssh_run(ssh, ["sh", "-c", f"cat /proc/sys/kernel/random/boot_id; hostname; kill -0 {pid} 2>/dev/null && echo alive "
                                      "|| echo gone"], timeout=120)
    lines = probe.stdout.decode(errors="replace").split() if probe.returncode == 0 else []
    if len(lines) != 3:
        return None
    boot, host, alive = lines
    if boot != env["boot_id"]:
        return f"the machine rebooted since its runner started (boot {env['boot_id']}, now {boot})"
    if host == env.get("hostname") and alive == "gone":
        return f"its runner (pid {pid}) is gone without recording an end"
    return None


def fetch(local_dir: Path, *, everything: bool = False, also: tuple[str, ...] = ()) -> dict[str, Any]:
    """Copy records (or everything) from the machine into the local run dir, with the run dir's files ``also`` names beside the
    records (plain names: the workload's own small progress files).  Unreachable -> observed_utc unchanged, reason recorded."""
    if bad := [n for n in also if not n or "/" in n or n.startswith(".")]:
        raise Refused(f"--also takes plain file names in the run dir, not {', '.join(map(repr, bad))}")
    rec = _read_json(local_dir / "remote.json")
    if not rec or not rec.get("remote_run_dir"):
        raise Refused(f"{local_dir} is not a launched remote run")
    ssh = ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root")
    rd = rec["remote_run_dir"]
    if everything:
        remote_cmd = ["sh", "-c", f"cd {shlex.quote(rd)} && tar -cf - ."]
    else:
        names = " ".join(shlex.quote(n) for n in (*SMALL_FILES, *also))
        remote_cmd = ["sh", "-c", f"cd {shlex.quote(rd)} && ls {names} 2>/dev/null | tar -cf - -T -"]

    def extract(stream: IO[bytes]) -> None:           # member by member off the ssh pipe: a big run is never in the launcher's memory
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for m in tf:
                name = m.name.removeprefix("./")
                if m.name.startswith("/") or ".." in Path(m.name).parts or name in ("", ".", "remote.json", FETCHED_FILE, FETCHED_FILE + ".tmp"):
                    continue
                tf.extract(m, local_dir, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))

    try:
        r = ssh_read(ssh, remote_cmd, extract, timeout=900)
    except (tarfile.TarError, OSError) as e:
        r = subprocess.CompletedProcess(remote_cmd, -1, b"", f"reading the run's tar stream: {type(e).__name__}: {e}".encode())
    if r.returncode != 0:
        rec["last_fetch_error"] = {"at_utc": utc_now_ms(), "rc": r.returncode, "stderr": r.stderr.decode(errors="replace")[-300:]}
        write_json_atomic(local_dir / "remote.json", rec)
        return rec
    rec["observed_utc"] = utc_now_ms()
    st = _read_json(local_dir / "status.json") or {}
    waits = st.get("state") == "running" and (_read_json(local_dir / "slot.json") or {}).get("state") == "waiting"
    rec["last_observed"] = {"state": "waiting" if waits else st.get("state"), "rc": st.get("rc"), "at_utc": rec["observed_utc"]}
    rec.pop("last_fetch_error", None)
    rec.pop("launcher_exited", None)
    rec.pop("runner_gone", None)
    if st.get("state") not in TERMINAL and isinstance(st.get("pid"), int):
        if why := _runner_gone(ssh, local_dir, st["pid"]):
            rec["runner_gone"] = {"at_utc": rec["observed_utc"], "runner_pid": st["pid"], "why": why}
    pid = (_read_json(local_dir / "launch.json") or {}).get("runner_pid")
    if not st and isinstance(pid, int):
        # launched, no status.json: the runner either has not written it yet or exited before it could (an argv it rejected)
        probe = ssh_run(ssh, ["sh", "-c", f"cd {shlex.quote(rd)} && if kill -0 {pid} 2>/dev/null; then echo alive; "
                                          f"else echo exited; tail -n 1 launcher.log 2>/dev/null; fi"], timeout=120)
        lines = probe.stdout.decode(errors="replace").splitlines() if probe.returncode == 0 else []
        if lines and lines[0] == "exited":
            rec["launcher_exited"] = {"at_utc": rec["observed_utc"], "runner_pid": pid, "last_line": (lines[1] if len(lines) > 1 else "")[-300:]}
    if everything:
        manifest = _verify_and_manifest(ssh, rd, local_dir, rec)
        rec["preserved"] = manifest is not None
        if manifest is not None:
            rec["fetched_marker"] = _mark_fetched(ssh, rd, local_dir, manifest)
    write_json_atomic(local_dir / "remote.json", rec)
    return rec


#: slot.py and lean_slot.py, beside this package in the caller's checkout, which NODE_PROGRAM loads on the machine (as `check.py
#: --slots` does) for the check and Lean slots a run holds or waits on; without them the reading has no slots
CHECK_DIR = Path(__file__).resolve().parents[3] / "verity" / "check"
#: what `node_readings` runs on a machine as `python3 - RUN_DIR...`, after `SRC = {...}`: research.scopemem's `node`
NODE_PROGRAM = '''
import json, sys, types


def load(name, src, file):
    m = types.ModuleType(name)
    m.__file__ = file
    sys.modules[name] = m
    exec(compile(src, file, "exec"), m.__dict__)
    return m


mem = load("research_scopemem", SRC["scopemem"], "research/scopemem.py")
mods = (load("check_slot_for_lean_slot", SRC["slot"], "tools/verity/check/slot.py"),
        load("check_lean_slot", SRC["lean"], "tools/verity/check/lean_slot.py")) if SRC.get("slot") and SRC.get("lean") else None
print(json.dumps(mem.node(sys.argv[1:], slot_mods=mods)))
'''


def node_payload(check_dir: Path = CHECK_DIR) -> bytes:
    from . import scopemem

    def src(p: Path) -> str | None:
        try:
            return p.read_text()
        except OSError:
            return None

    srcs = {"scopemem": Path(scopemem.__file__).read_text(), "slot": src(check_dir / "slot.py"), "lean": src(check_dir / "lean_slot.py")}
    return (f"SRC = {srcs!r}\n" + NODE_PROGRAM).encode()


def node_readings(local_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Each launched remote run's reading where it executes (research.scopemem's `node`: its memory and CPUs, and its machine's
    GPU leases and check and Lean slots), by local run dir: `{"run", "node", "machine"}`, or `{"error"}` when it can't be read.
    One ssh per machine, whatever the number of its runs."""
    by: dict[tuple, list[tuple[Path, dict[str, Any]]]] = {}
    out: dict[str, dict[str, Any]] = {}
    for d in local_dirs:
        rec = _read_json(d / "remote.json")
        if not rec or not rec.get("remote_run_dir"):
            out[str(d)] = {"error": f"{d} is not a launched remote run"}
            continue
        by.setdefault((rec["host"], rec["port"], rec.get("user") or "root"), []).append((d, rec))
    payload = node_payload() if by else b""

    def one(target: tuple, runs: list[tuple[Path, dict[str, Any]]]) -> None:
        machine = runs[0][1].get("machine")
        try:
            r = ssh_run(ssh_argv(target[0], target[1], user=target[2]), ["python3", "-", *(rec["remote_run_dir"] for _, rec in runs)],
                        stdin=payload, timeout=60)
            doc = json.loads(r.stdout) if r.returncode == 0 else None
            why = None if isinstance(doc, dict) else f"rc {r.returncode} {r.stderr.decode(errors='replace').strip()[-200:]}"
        except subprocess.TimeoutExpired:
            doc, why = None, "took over 60 s"
        except ValueError:
            doc, why = None, f"its answer is not JSON: {r.stdout[:120]!r}"
        except Exception as e:  # noqa: BLE001  (no ssh key here, ...): that machine's runs unread, the rest still shown
            doc, why = None, f"{type(e).__name__}: {e}"
        for d, rec in runs:
            run = (doc or {}).get("runs", {}).get(rec["remote_run_dir"])
            out[str(d)] = ({"run": run, "node": {k: v for k, v in doc.items() if k != "runs"}, "machine": machine} if run is not None
                           else {"error": f"reading {machine}: {why or 'no reading of this run'}"})

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, len(by))) as ex:
        list(ex.map(lambda kv: one(*kv), by.items()))
    return out


def py_spy(local_dir: Path) -> str:
    """`research inspect --py-spy`: a py-spy dump of the live workload's Python processes on its machine (research.telemetry.pyspy,
    sent as `python3 -` over fetch's ssh, which reads the machine's status.json)."""
    from .telemetry import pyspy

    rec = _read_json(local_dir / "remote.json")
    if not rec or not rec.get("remote_run_dir"):
        raise Refused(f"{local_dir} is not a launched remote run")
    limit = pyspy.MAX_PROCS * pyspy.DUMP_TIMEOUT_S + 60
    try:
        r = ssh_run(ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root"), ["python3", "-", rec["remote_run_dir"]],
                    stdin=Path(pyspy.__file__).read_bytes(), timeout=limit)
    except subprocess.TimeoutExpired:
        raise Refused(f"the dump on {rec.get('machine')} took over {limit:g} s") from None
    if r.returncode:
        raise Refused(f"no dump on {rec.get('machine')}: rc {r.returncode} {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout.decode(errors="replace")


def cancel(local_dir: Path, *, reason: str, by: str) -> dict[str, Any]:
    """`research cancel`: SIGTERM to the workload's process group on the machine (status.json `pgid`), its intent recorded first in the
    run's events.jsonl (`tele cancel-intent --actor manual`), so the run classifies CANCELLED_MANUAL.  Never the runner and never SIGKILL:
    the runner records the workload's exit and then publishes custody, and one killed in its upload leaves the run without custody.
    A run another VM launched (`adopt`) has no shipped tool in its remote.json; the machine's job.json names it (`run.remote.tool`), and
    it must be a snapshot under the machine's <root>/tool/.  The group is signalled only if one of its members is this run's workload
    (`--expect-run-dir`: RESEARCH_RUN_DIR in its environment), since a runner that died leaves status.json `running` and pids wrap; a
    run whose shipped tool predates that check is refused, not signalled unchecked."""
    rec = fetch(local_dir)
    if err := rec.get("last_fetch_error"):
        raise Refused(f"the machine is unreachable ({err['stderr'].strip() or 'rc ' + str(err['rc'])}); nothing was signalled")
    st = _read_json(local_dir / "status.json") or {}
    if st.get("state") != "running":
        raise Refused(f"{local_dir.name} is {st.get('state') or 'not running yet'}, so there is nothing to cancel")
    pgid = st.get("pgid")
    if not isinstance(pgid, int) or pgid <= 1:
        raise Refused(f"{local_dir.name}'s status.json names no workload process group (pgid {pgid!r})")
    tool = rec.get("tool") or ((((_read_json(local_dir / "job.json") or {}).get("run") or {}).get("remote") or {}).get("tool"))
    if not tool:
        raise Refused(f"neither {local_dir.name}'s remote.json nor its job.json names a shipped tool to record the cancel with")
    snap, root = PurePosixPath(tool), rec.get("root")
    if not root or snap.parent != PurePosixPath(root) / "tool" or ":" in tool or snap.name in ("", ".", ".."):
        raise Refused(f"{local_dir.name}'s tool {tool!r} is not a snapshot under {root}/tool/; nothing was signalled")
    rd = rec["remote_run_dir"]
    tele = ["env", f"PYTHONPATH={tool}", "python3", "-m", "research.telemetry", "cancel-intent", "--actor", "manual",
            "--reason", f"{reason} (research cancel by {by})", "--target-pgid", str(pgid), "--expect-run-dir", rd, "--attempt-dir", rd,
            "--deliver"]
    # every line appended since the cancel began: the runner records the workload's exit there as soon as the signal lands
    script = (f"f={shlex.quote(rd)}/events.jsonl; n=$(wc -l < \"$f\" 2>/dev/null || echo 0); {shlex.join(tele)}; rc=$?; "
              f"tail -n +$((n + 1)) \"$f\"; exit $rc")
    r = ssh_run(ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root"), ["sh", "-c", script], timeout=120)
    stderr = r.stderr.decode(errors="replace").strip()
    if r.returncode == 2 and "--expect-run-dir" in stderr:
        raise Refused(f"{local_dir.name}'s research package ({tool}) predates the workload check, so nothing was signalled; on the machine, "
                      f"`kill -TERM -{pgid}` once /proc/<pid>/environ of a group member names RESEARCH_RUN_DIR={rd}")
    res = next((x for x in map(_json_line, r.stdout.decode(errors="replace").splitlines())
                if x.get("kind") == "cancel_result" and (x.get("target") or {}).get("pgid") == pgid), None)
    if res is None:
        raise Refused(f"the cancel recorded no result on the machine (rc {r.returncode}): {stderr[-300:]}")
    tgt = res.get("target") or {}
    return {"pgid": pgid, "processes": tgt.get("members_total", len(tgt.get("members") or [])), "delivered": bool(res.get("delivered")),
            "why": res.get("reason")}


def _file_sha256(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _verify_and_manifest(ssh: list[str], remote_dir: str, local_dir: Path, rec: dict[str, Any]) -> dict[str, Any] | None:
    # T0 = the machine's clock before hashing: a file changed after it is newer than the verification, which the guard reads as unfetched
    r = ssh_run(ssh, ["sh", "-c", f"cd {shlex.quote(remote_dir)} && t=$(date +%s.%N 2>/dev/null); case $t in *N*|'') "
                                  f"t=$(python3 -c 'import time; print(time.time())' 2>/dev/null || date +%s);; esac; "
                                  f"echo \"T0 $t\" && find . -type f ! -name preserved.json ! -name {FETCHED_FILE} ! -name {FETCHED_FILE}.tmp "
                                  f"-print0 | xargs -0 sha256sum"], timeout=900)
    if r.returncode != 0:
        return None
    remote_sums, t0 = {}, None
    for line in r.stdout.decode().splitlines():
        if line.startswith("T0 "):
            t0 = float(line[3:])
            continue
        h, _, name = line.partition("  ")
        remote_sums[name.removeprefix("./")] = h
    files, mismatched = [], []
    for name, h in sorted(remote_sums.items()):
        p = local_dir / name
        if not p.is_file():
            mismatched.append(name)
            continue
        lh = _file_sha256(p)
        (files if lh == h else mismatched).append({"path": name, "bytes": p.stat().st_size, "sha256": lh} if lh == h else name)
    if mismatched:
        rec["preserve_error"] = {"at_utc": utc_now_ms(), "mismatched_or_missing": mismatched}
        return None
    manifest = {"schema": PRESERVED_SCHEMA, "run_id": rec["run_id"], "source": f"{rec['host']}:{remote_dir}", "destination": str(local_dir),
                "files": files, "completed_utc": utc_now_ms(), "verified_from_machine_epoch": t0,
                "note": "copy on the control machine; the machine's copy is the other one"}
    write_json_atomic(local_dir / "preserved.json", manifest)   # written last
    return manifest


def _mark_fetched(ssh: list[str], remote_dir: str, local_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """<run>/.fetched on the machine (atomic): this run dir, as of `verified_from` on the machine's clock, has a verified copy here.
    The pod's guard refuses to terminate while any run dir lacks one or has a file newer than `verified_from`."""
    files = manifest["files"]
    doc = {"schema": FETCHED_SCHEMA, "run_id": manifest["run_id"], "verified_from": manifest["verified_from_machine_epoch"],
           "files": len(files), "bytes": sum(f["bytes"] for f in files),
           "manifest_sha256": hashlib.sha256("".join(f"{f['sha256']}  {f['path']}\n" for f in files).encode()).hexdigest(),
           "fetched_utc": utc_now_ms(), "destination": f"{os.uname().nodename}:{local_dir}"}
    if doc["verified_from"] is None:
        return {"written": False, "reason": "the machine did not report its clock before hashing (no T0)"}
    q = shlex.quote(remote_dir)
    r = ssh_run(ssh, ["sh", "-c", f"cd {q} && cat > {FETCHED_FILE}.tmp && mv -f {FETCHED_FILE}.tmp {FETCHED_FILE}"],
                stdin=json.dumps(doc).encode(), timeout=120)
    if r.returncode != 0:
        return {"written": False, "reason": f"rc {r.returncode}: {r.stderr.decode(errors='replace')[-200:]}"}
    return {"written": True, "verified_from": doc["verified_from"], "files": doc["files"], "bytes": doc["bytes"]}
