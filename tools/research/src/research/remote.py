"""`research run --on <machine>`: the same harness and the same record, on an existing authorized machine.

    machines file   ~/.research/machines.toml ($RESEARCH_MACHINES) binds a name to a provider pod and a project:
                        [machines.vy-control]
                        provider = "runpod"
                        pod_id   = "abc123"
                        project  = "verity"            # or projects = ["verity", "other"]
                        root     = "/workspace/research"
                        python   = "python3"
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
                    `source/SUB`, or a relative SUB) runs it inside the tree instead, an explicit opt-in; job.json records run.cwd and
                    run.cwd_basis.  Python bytecode goes to <root>/cache/pycache (PYTHONPYCACHEPREFIX), not into the tree.  A tree whose
                    own research package predates this runs its workloads in the tree; the launcher says so (remote.json workload_cwd).
    claim           `research launch-request` on the machine does mkdir(<root>/runs/<run-id>) as the claim, then starts the detached
                    runner.  Re-delivering the same request finds the directory and reports its state; it never starts a second workload.
    exclusive       --exclusive takes <root>/runs/.exclusive.lock (O_EXCL) naming the run; a live holder refuses the launch with the reason.
                    A dead holder (terminal status or vanished pid) is reclaimed.  The runner releases its own lock at exit.
    memory          --mem-gb N: the machine side admits the run only if its memory cgroup's limit - in use - live reservations
                    (the fleet's /workspace/ramlock/*.json, machines.toml `ramlock`) >= N, and writes the run's reservation there;
                    else MemoryNotAvailable (refused.json).  --mem-wait S polls up to S seconds instead.  The runner removes the
                    reservation at exit; job.json run.remote.mem records the reading it was admitted on (research.mem).
    observe/fetch   `research fetch <id>` copies the small records (or --all files) into the local run dir over one ssh+tar, extracted
                    member by member off the pipe (a big run is never in the launcher's memory); remote.json keeps observed_utc.
                    Unreachable means "last observed at T; current status unknown", never "failed".
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
                    (outputs.run_record), verifies it on the remote, deletes the key and writes <run>/.custody (store.custody).  It terminates with the
                    pod's OWN key (RUNPOD_API_KEY in /proc/1/environ); without one it is not installed and the launch says so.  The account
                    key is never sent.  <root>/guard/state.json: verdict, last activity, idle minutes, refusal reasons.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from typing import IO, Any, Callable

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
SMALL_FILES = ("job.json", "status.json", "failure.json", "result.json", "launch.json", "refused.json", "events.jsonl")
TERMINAL = ("done", "failed", "cancelled")
FETCHED_FILE = ".fetched"                     # <run>/.fetched on the machine: written by `fetch --all` once every file verified
FETCHED_SCHEMA = "research/fetched/v0.1"
CUSTODY_DIR = "custody"                       # <root>/requests/<run-id>/custody/: --custody-r2's minted key + store.toml (store.custody)
GUARD_IDLE_MIN_DEFAULT = 90
GUARD_ENVIRON = "/proc/1/environ"             # RunPod injects RUNPOD_POD_ID and the pod-scoped RUNPOD_API_KEY into the container's first process
GUARD_SCRIPT = Path(__file__).resolve().parent / "pods" / "sh" / "pod_guard.sh"
POD_TARGET_RETRIES = 6           # extra GET /pods/{id} reads when a RUNNING pod answers without an ssh mapping
POD_TARGET_RETRY_S = 2.0


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
    out.setdefault("root", "/workspace/research")
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

def pod_target(machine: dict[str, Any]) -> dict[str, Any]:
    """host/port/rate for a runpod machine (read-only API call).  Not RUNNING -> refused."""
    if machine.get("provider") != "runpod":
        raise Refused(f"provider {machine.get('provider')!r} not supported (runpod only for now)")
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
    return {"host": tgt[0], "port": tgt[1], "pod_name": pod.get("name"), "cost_per_hr": pod.get("costPerHr"),
            "rate_observed_utc": utc_now_ms(), "gpu": (pod.get("machine") or {}).get("gpuTypeId")}


def ssh_argv(host: str, port: int, *, connect_timeout: int = 15) -> list[str]:
    from .pods import runpod

    base = runpod.ssh_command(host, port)
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
            cls = {"ResearchVersionMismatch": ResearchVersionMismatch}.get(str(doc.get("fault")), Refused)
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
    return subprocess.Popen(["git", *GIT_ARCHIVE_CAPS, "archive", "--format=tar", "HEAD"], cwd=git_dir, stdout=subprocess.PIPE,
                            stderr=tempfile.TemporaryFile())


def _git_archive_done(proc: subprocess.Popen) -> str | None:
    """None when `git archive` exited 0, else its stderr tail."""
    proc.stdout.close()
    if proc.wait() == 0:
        return None
    proc.stderr.seek(0)
    return f"rc {proc.returncode}: {proc.stderr.read().decode(errors='replace')[-300:].strip()}"


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
    """The machine side of a git ship: the pushed commit's tree must be the launcher's ($1), then `git archive` of it is extracted
    (`tar -m`, extraction mtimes) into the partial directory and every extracted file's sha256 is listed back.  The partial removes
    itself on any exit until the listing starts."""
    q, g = shlex.quote(partial), shlex.quote(repo)
    return (_stale_partials_sweep(root) +
            f"trap 'rm -rf {q}' EXIT; trap 'exit 1' HUP INT TERM PIPE; set -e; "
            f"t=$(git --git-dir={g} rev-parse --verify -q {sha}^{{tree}}) || exit {_RC_TREE}; [ \"$t\" = \"$1\" ] || exit {_RC_TREE}; "
            f"mkdir -p {q}; git --git-dir={g} archive --format=tar {sha} > {q}/.archive.tar; "
            f"sha256sum {q}/.archive.tar | cut -d' ' -f1; tar -xmf {q}/.archive.tar -C {q}; rm -f {q}/.archive.tar; "
            f"trap - EXIT HUP INT TERM PIPE; cd {q} && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum")


def _ship_git(ssh: list[str], root: str, git_dir: Path, sha: str, tree: str | None, man: dict[str, Any], dest: str) -> tuple[str | None, str | None]:
    """Ship commit `sha` as git objects: `git push` into <root>/git/verity.git over `ssh` (only the objects the machine lacks travel),
    then extract and verify on the machine (tree hash, per-file sha256 against the launcher's archive manifest).  Returns
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
         (only the objects it lacks travel; no credential anywhere), then on the machine the commit's tree hash must be HEAD's,
         `git archive` of it is extracted and every file's sha256 must equal the launcher's archive manifest (`verified:
         git-push+tree+file-sha256`, `transport: git`).  A machine without git, or any failure of that path, prints one line and
         falls through to the archive stream below; `RESEARCH_SHIP=stream` goes straight there;
      2'. the archive stream into <root>/src/<identity>.partial-<nonce>/: the tar is staged as a file, sha256'd on the machine, extracted, and the
         extracted regular-file set listed back; the machine's archive sha256 must equal the launcher's and the file set must equal the
         archive's manifest.  The launcher streams it (`archive_manifest_git`, then `_archive_feed` through `ssh_stream`): its memory
         does not grow with the tree.  A short stream (a launcher killed mid-ship) makes the machine remove the partial itself;
      3. READY.json is written INTO the partial directory and the directory renamed into place (one rename, so a reader sees either
         nothing or a complete tree with its marker); the marker is read back and its identity checked.
    Any failure (transport rc != 0, the launcher's timeout, digest or manifest mismatch) removes the partial directory and raises
    SourceNotReady; `<identity>/` is never created by a failed ship.  The machine side (`launch_request`) enforces `require_ready_source`."""
    st = subprocess.run(["git", "status", "--porcelain"], cwd=git_dir, capture_output=True, text=True)
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
            doc = {**_ready_doc(sha, tree, man, "git-push+tree+file-sha256"), "transport": "git",
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


def launch_request(request_path: Path) -> dict[str, Any]:
    """Claim the run directory and start the detached runner (idempotent per run id); with the request's `guard`, make sure the pod's
    guard daemon runs (ensure_guard; its answer is the result's `guard`, and a guard that cannot start never fails the launch)."""
    out = _launch_request(request_path)
    try:
        req = json.loads(request_path.read_text())
        g = req.get("guard")
    except (OSError, ValueError):
        g = None
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
                "runner_pid": launch.get("runner_pid"), "refused": (refused or {}).get("reason")}
    concurrent = active_runs(runs_root, exclude=run_id)
    src_dir = None
    mem_rec: dict[str, Any] | None = None
    try:
        remote_sec = req.get("remote") or {}
        if remote_sec.get("source_sha"):          # a shipped tree executes only from a directory verified READY for the requested identity
            src_dir = remote_sec.get("source_dir") or str(runs_root.parent / "src" / str(remote_sec["source_sha"]))
            require_ready_source(src_dir, str(remote_sec["source_sha"]))
        if req.get("custody") and not (request_path.parent / CUSTODY_DIR / "cred.json").is_file():
            raise Refused(f"--custody-r2: no staged key under {request_path.parent / CUSTODY_DIR}; the run could not publish its custody")
        holder = exclusive_holder(runs_root)
        if holder and holder != run_id:
            raise Refused(f"machine is exclusively held by run {holder} (still live); not starting {run_id}")
        if req.get("mem_gb"):
            from . import mem as M

            mem_rec = M.admit(float(req["mem_gb"]), ramlock_dir=Path(req.get("ramlock_dir") or M.RAMLOCK_DIR), run_id=run_id, run_dir=run_dir,
                              wait_s=float(req.get("mem_wait_s") or 0), note=f"research run {run_id}: {shlex.join(req['command'])}")
        if req.get("exclusive"):
            take_exclusive(runs_root, run_id)
            if concurrent:
                release_exclusive(runs_root, run_id)
                raise Refused(f"--exclusive: other run(s) are live on this machine: {', '.join(concurrent)}; not starting {run_id}")
    except Refused as e:
        if mem_rec:
            M.release(mem_rec["reservation"], run_id)
        if req.get("custody"):                    # nothing will publish: the key goes now, not at its expiry
            (request_path.parent / CUSTODY_DIR / "cred.json").unlink(missing_ok=True)
        write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": str(e), "fault": type(e).__name__, "at_utc": utc_now_ms()})
        return {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": False, "refused": str(e), "fault": type(e).__name__}
    staged_inputs = request_path.parent / "inputs"
    if staged_inputs.is_dir():
        os.replace(staged_inputs, run_dir / "inputs")
    cost = dict(req.get("cost") or {})
    cost["shared"] = bool(concurrent)
    cost["concurrent_runs_at_launch"] = concurrent
    remote = dict(req.get("remote") or {})
    remote["request"] = str(request_path)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (req.get("env") or {}).items()})
    if mem_rec:
        remote["mem"] = mem_rec
        env["RESEARCH_MEM_RESERVATION"] = mem_rec["reservation"]      # the runner removes it at exit and does not pass it on
    if remote.get("source_sha"):
        # [R19 tele] the shipped tree (<root>/src/<sha>/, a git archive) has no .git: the workload's source-identity precheck reads the
        # tree's sha from here (research.result.ENV_SOURCE_SHA) when neither --sha nor EXPORT.json names it
        env["RESEARCH_SOURCE_SHA"] = str(remote["source_sha"])
        env.setdefault("PYTHONPYCACHEPREFIX", str(runs_root.parent / "cache" / "pycache"))   # imports from the shared tree do not write into it
        from .pythonpath import import_roots_dir, pythonpath

        derived = import_roots_dir(src_dir)                 # the tree's own workspace declaration, not a hand list (pythonpath.py)
        env["PYTHONPATH"] = pythonpath(src_dir, derived, extra=env.get("PYTHONPATH"))
        remote["pythonpath"] = [p for p in env["PYTHONPATH"].split(":") if p]
        remote["pythonpath_derived"] = {"roots": derived["roots"], "missing": derived["missing"], "source": derived["source"]}
    if req.get("tool"):
        env["PYTHONPATH"] = req["tool"] + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["RESEARCH_TOOL_SNAPSHOT"] = Path(req["tool"]).name
    argv = [req.get("python") or sys.executable, "-m", "research", "run", "--runs-dir", str(runs_root), "--id", run_id, "--claimed",
            "--cost-json", json.dumps(cost), "--remote-json", json.dumps(remote)]
    if req.get("exclusive"):
        argv += ["--exclusive-lock", str(runs_root / ".exclusive.lock")]
    if req.get("custody"):
        argv += ["--custody-dir", str(request_path.parent / CUSTODY_DIR)]
    argv += [*(req.get("harness_args") or []), "--", *req["command"]]
    log = open(run_dir / "launcher.log", "ab")
    try:
        proc = subprocess.Popen(argv, cwd=req.get("cwd") or None, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        if mem_rec:
            M.release(mem_rec["reservation"], run_id)
        raise
    if mem_rec:
        M.set_pid(mem_rec["reservation"], proc.pid)
    write_json_atomic(run_dir / "launch.json", {"run_id": run_id, "claimed_utc": utc_now_ms(), "runner_pid": proc.pid, "argv": argv,
                                                "cwd": req.get("cwd"), "request": str(request_path), "hostname": os.uname().nodename})
    return {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": True, "runner_pid": proc.pid}


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


def workload_cwd(harness_args: list[str], tool_pkg: Path | None) -> str:
    """What the machine will use as the workload's cwd, for remote.json: the `--cwd` asked for, else the run dir -- unless the package
    that runs on the machine (the shipped tree's, `tool_pkg`) predates WORKLOAD_CWD_DEFAULT and so still runs in the shared tree."""
    asked = flag_value(harness_args, "--cwd")
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
           mem_gb: float | None = None, mem_wait_s: float = 0.0, custody_ttl: str | None = None) -> dict[str, Any]:
    """`custody_ttl` (--custody-r2): mint a delete-free key for that long and stage it for the machine's runner (store.custody)."""
    local_dir.mkdir(parents=True, exist_ok=False)
    rec: dict[str, Any] = {"schema": "research/remote/v0.1", "run_id": run_id, "machine": machine_name, "project": project, "phase": "resolving",
                           "created_utc": utc_now_ms(), "observed_utc": None, "last_observed": None, "command": command, "launch_args": harness_args}
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
        ssh = ssh_argv(tgt["host"], tgt["port"])
        rec.update(pod_id=m["pod_id"], host=tgt["host"], port=tgt["port"], root=m["root"], phase="shipping")
        write_json_atomic(local_dir / "remote.json", rec)
        tool_pkg = source_tool_package(source)      # --source <tree> with tools/research: the tree's own package runs the run
        if mem_gb and tool_pkg is not None and not (tool_pkg / "mem.py").exists():
            raise Refused(f"--mem-gb: the research package in {source} predates memory admission (no research/mem.py); the machine would "
                          f"ignore the reservation.  Commit a tree that has it, or drop --mem-gb")
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
            remote_cwd = src_dir if cwd is None else str(Path(src_dir) / cwd)
            if "--tree-sha" not in harness_args:
                harness_args = [*harness_args, "--tree-sha", tree_sha]
        request = {"schema": REQUEST_SCHEMA, "run_id": run_id, "runs_root": f"{m['root']}/runs", "tool": tool_dir, "python": m["python"],
                   "cwd": remote_cwd, "env": env, "harness_args": harness_args, "command": command, "exclusive": exclusive,
                   "mem_gb": mem_gb, "mem_wait_s": mem_wait_s, "ramlock_dir": m.get("ramlock"),
                   "guard": {"idle_min": guard_min} if guard_min else None,
                   **({"custody": custody_pub} if custody_pub else {}),
                   "cost": {"provider": "runpod", "pod_id": m["pod_id"], "machine": machine_name, "currency": "USD",
                            "rate_usd_per_hour": tgt["cost_per_hr"], "rate_observed_utc": tgt["rate_observed_utc"], "reason": None},
                   "remote": {"machine": machine_name, "pod_id": m["pod_id"], "pod_name": tgt["pod_name"], "host": tgt["host"], "port": tgt["port"],
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
        for attempt in range(3):
            r = ssh_run(ssh, ["env", f"PYTHONPATH={tool_dir}", m["python"], "-m", "research", "launch-request", *expect, req_path],
                        timeout=120 + (mem_wait_s or 0))
            if r.returncode == 0:
                out = json.loads(r.stdout.decode().strip().splitlines()[-1])
                break
            last_err = r.stderr.decode(errors="replace")[-400:]
            if r.returncode != 255:            # 255 = ssh transport failure: re-deliver the SAME request (idempotent claim)
                raise launch_request_failure(r.returncode, r.stdout, r.stderr)
            time.sleep(3 * (attempt + 1))
        else:
            rec.update(phase="unknown", note=f"ssh failed 3 times after staging; the request may or may not have started: {last_err}")
            write_json_atomic(local_dir / "remote.json", rec)
            raise Refused(rec["note"])
        rec.update(phase="refused" if out.get("refused") else "launched", launch=out, observed_utc=utc_now_ms())
        if guard_min:
            g = out.get("guard") or {"installed": False, "reason": "the machine's research package predates the guard (no `guard` in its answer)"}
            rec["guard"] = g
            if g.get("installed"):
                print(f"research: guard on pod {g.get('pod_id')}: pid {g.get('pid')} ({'started' if g.get('started') else 'already running'}), "
                      f"terminates the pod after {g.get('idle_min'):g} idle min once every run is fetched (`research fetch --all`); "
                      f"state {g.get('state')}", file=sys.stderr)
            else:
                print(f"research: NO guard on {machine_name}: {g.get('reason')}", file=sys.stderr)
        write_json_atomic(local_dir / "remote.json", rec)
        return rec
    except Refused as e:
        rec.update(phase=rec["phase"] if rec.get("phase") in ("unknown",) else "refused", error=str(e))
        write_json_atomic(local_dir / "remote.json", rec)
        raise


def fetch(local_dir: Path, *, everything: bool = False) -> dict[str, Any]:
    """Copy records (or everything) from the machine into the local run dir.  Unreachable -> observed_utc unchanged, reason recorded."""
    rec = _read_json(local_dir / "remote.json")
    if not rec or not rec.get("remote_run_dir"):
        raise Refused(f"{local_dir} is not a launched remote run")
    ssh = ssh_argv(rec["host"], rec["port"])
    rd = rec["remote_run_dir"]
    if everything:
        remote_cmd = ["sh", "-c", f"cd {shlex.quote(rd)} && tar -cf - ."]
    else:
        names = " ".join(shlex.quote(n) for n in SMALL_FILES)
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
    rec["last_observed"] = {"state": st.get("state"), "rc": st.get("rc"), "at_utc": rec["observed_utc"]}
    rec.pop("last_fetch_error", None)
    if everything:
        manifest = _verify_and_manifest(ssh, rd, local_dir, rec)
        rec["preserved"] = manifest is not None
        if manifest is not None:
            rec["fetched_marker"] = _mark_fetched(ssh, rd, local_dir, manifest)
    write_json_atomic(local_dir / "remote.json", rec)
    return rec


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
