"""The steward off the laptop (cloud migration step e): times any machine agrees on, one steward anywhere, and a service definition.

    research notes status|watch --liveness git [--fetch] ...     # (or RESEARCH_LIVENESS=git for every notes command)
    research notes watch --lease [store|PATH] [--standby] [--steward-id ID] ...
    research notes service systemd|launchd [--python P] [--workdir DIR] [--env-file F] [--log F] [-- WATCH ARGS...]

TIMES (`GitTimes`).  A git clone sets every file's mtime to the checkout time, so liveness read from mtimes differs by machine.
With `--liveness git`, a notes file's time is the author time of the newest notes commit that touched it (author, not committer:
`notes sync` rebases, which rewrites committer times); a file with uncommitted changes on this machine, or one git does not track,
keeps its mtime, since only this machine has that write.  Report age, which report is newest, the `--since-hours` window, the inbox
("new since last seen") and the anchor of a binding's `final` all read it.  A lane branch's last commit is origin's `lane/*` tip
(`--fetch`, and every watch pass, fetch them first); the local worktree signal (uncommitted edits, `dirty`) is not read, since the
push rule and the 20-minute checkpoint replace it.  `relaunch` bases the successor on origin's tip when the dead lane's worktree is
not on this machine.  The stale reaper's run-custody leg fails closed here: the local run directories it reads only exist where the
launcher ran (run custody on R2 is step f).

LEASE (`Lease`).  At most one steward reaps, routes, renders, keeps steward-state.json and commits the notes (I5).  The lease is one
JSON object, `steward/lease.json` in the store's bucket (`--lease` / `--lease store`; for an fs remote, the same key under its
directory), or a file (`--lease PATH`, for one host).  It records the owner (`--steward-id`, default the host name), host, pid, a
per-process token and an expiry (`ttl_s`, at least 15 min and 3 passes).  Every change is a compare-and-swap: create with
`If-None-Match: *`, renew or take over with `If-Match: <etag>` (a 412 is a lost race), then read back and require the token to be
ours, which also holds on a store that ignored the condition.  A lease is free when it is released, expired, or held by a dead pid
on this same host (a reboot).  A second watcher with `--lease` refuses (exit 2) and names the holder; with `--standby` it keeps
watching read-only and takes over when the lease frees.  A holder that fails to renew stops acting at once (standby, or exit 3).
Exit releases it.  The lease is the one mutable object under the bucket's `steward/` prefix; the store's records stay immutable.

SERVICE (`unit`).  The steward's service definition: a systemd user unit for an always-on Linux host, or a launchd agent (the laptop,
made reboot-safe: today's watcher was loaded with `launchctl submit` and has no plist).  Both restart the watcher when it exits and
after a reboot; both run `watch --lease --standby`, so a second host's copy waits instead of acting.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import plistlib
import secrets
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable

LIVENESS_ENV = "RESEARCH_LIVENESS"
LEASE_KEY = "steward/lease.json"
LEASE_SCHEMA = "research/steward-lease/v0.1"
LEASE_TTL_MIN_S = 15 * 60


def git_liveness(environ: dict | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(LIVENESS_ENV, "").strip().lower() == "git"


# ---- times from the notes history ------------------------------------------------------------------------------------------------------

class GitTimes:
    """File times from the notes repo's history (module docstring, TIMES).  `ok` is False when `root` is not in a git work tree:
    every time is then the mtime."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.committed: dict[str, float] = {}
        self.local: set[str] = set()
        self.ok = False
        prefix = self._git("rev-parse", "--show-prefix")
        if prefix is None:
            return
        prefix = prefix.strip()
        log = self._git("-c", "core.quotePath=false", "log", "--format=@%at", "--name-only", "--no-renames", "--relative", timeout=120)
        if log is None:
            return
        t = None
        for line in log.splitlines():
            if line.startswith("@") and line[1:].isdigit():
                t = float(line[1:])
            elif line and t is not None and t > self.committed.get(line, -1.0):
                self.committed[line] = t
        st = self._git("-c", "core.quotePath=false", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".") or ""
        toks, i = st.split("\0"), 0
        while i < len(toks):
            e = toks[i]
            i += 1
            if len(e) > 3:
                self.local.add(e[3:])
                if e[0] in "RC":
                    i += 1
        self.local = {p[len(prefix):] if prefix and p.startswith(prefix) else p for p in self.local}
        self.ok = True

    def _git(self, *args: str, timeout: float = 30) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    def time(self, p: Path) -> float:
        if self.ok:
            try:
                rel = Path(os.path.relpath(p, self.root)).as_posix()
            except ValueError:
                rel = None
            if rel is not None and rel not in self.local and rel in self.committed:
                return self.committed[rel]
        return p.stat().st_mtime


# ---- the lease ---------------------------------------------------------------------------------------------------------------------------

class LeaseConflict(Exception):
    """A conditional write lost: someone else changed the lease since it was read."""


class S3LeaseBackend:
    """The lease as one object in an S3-compatible bucket, changed only by conditional PUT."""

    def __init__(self, remote: Any, key: str = LEASE_KEY):
        self.remote, self.key = remote, key

    def describe(self) -> str:
        return f"{self.remote.describe()}/{self.key}"

    def read(self) -> tuple[dict, str] | None:
        try:
            with self.remote._request("GET", self.key) as r:  # noqa: SLF001
                body, etag = r.read(), r.headers.get("ETag")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        try:
            doc = json.loads(body)
        except ValueError:
            doc = {}
        return (doc if isinstance(doc, dict) else {}), etag or ""

    def write(self, doc: dict, etag: str | None) -> None:
        cond = {"if-none-match": "*"} if etag is None else {"if-match": etag}
        try:
            with self.remote._request("PUT", self.key, body=_encode(doc), headers=cond):  # noqa: SLF001
                pass
        except urllib.error.HTTPError as e:
            if e.code in (409, 412):
                raise LeaseConflict(f"{self.key}: HTTP {e.code}") from None
            if e.code in (400, 501):
                raise RuntimeError(f"{self.describe()}: the store refused a conditional PUT (HTTP {e.code}); no lease without it") from None
            raise


class FileLeaseBackend:
    """The lease as a file, changed under an exclusive lock on `<file>.lock`; the etag is the content's sha256."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()

    def describe(self) -> str:
        return str(self.path)

    def _locked(self, fn: Callable[[], Any]) -> Any:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{self.path}.lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                return fn()
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _read(self) -> tuple[dict, str] | None:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = {}
        return (doc if isinstance(doc, dict) else {}), hashlib.sha256(raw).hexdigest()

    def read(self) -> tuple[dict, str] | None:
        return self._locked(self._read)

    def write(self, doc: dict, etag: str | None) -> None:
        def cas() -> None:
            cur = self._read()
            if (cur is None) != (etag is None) or (cur is not None and cur[1] != etag):
                raise LeaseConflict(f"{self.path}: changed since read")
            tmp = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
            tmp.write_bytes(_encode(doc))
            os.replace(tmp, self.path)
        self._locked(cas)


def _encode(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()


def lease_backend(spec: str) -> S3LeaseBackend | FileLeaseBackend:
    """`store` (the store's configured remote: S3 -> the bucket's `steward/lease.json`, fs -> that file under its directory) or a path."""
    if spec != "store":
        return FileLeaseBackend(Path(spec))
    from .store.remote import FsRemote, load_remote

    r = load_remote()
    if r is None:
        raise RuntimeError("no store remote configured (store.toml or R2_ENDPOINT/R2_BUCKET): the lease lives there; or --lease PATH")
    if isinstance(r, FsRemote):
        return FileLeaseBackend(r.root / LEASE_KEY)
    return S3LeaseBackend(r)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lease:
    """One steward (module docstring, LEASE).  `ensure()` acquires or renews; -> (held, the lease document as last read)."""

    def __init__(self, backend: S3LeaseBackend | FileLeaseBackend, owner: str | None = None, ttl_s: float = LEASE_TTL_MIN_S,
                 now: Callable[[], float] = time.time, host: str | None = None, pid: int | None = None,
                 alive: Callable[[int], bool] = _pid_alive):
        self.backend, self.ttl_s, self.now, self.alive = backend, float(ttl_s), now, alive
        self.host = host or socket.gethostname()
        self.pid = os.getpid() if pid is None else pid
        self.owner = owner or self.host
        self.token = secrets.token_hex(8)
        self.held = False

    def free(self, doc: dict) -> str | None:
        """Why someone else's lease may be taken over, or None while it is live."""
        if doc.get("schema") != LEASE_SCHEMA:
            return "no valid lease document"
        if doc.get("released"):
            return "released"
        if self.now() >= float(doc.get("expires_at") or 0):
            return "expired"
        if doc.get("host") == self.host and isinstance(doc.get("pid"), int) and not self.alive(doc["pid"]):
            return f"its pid {doc['pid']} on this host is gone"
        return None

    def _doc(self, acquired_at: float | None, released: bool = False) -> dict:
        t = self.now()
        return {"schema": LEASE_SCHEMA, "owner": self.owner, "host": self.host, "pid": self.pid, "token": self.token,
                "acquired_at": acquired_at or t, "renewed_at": t, "expires_at": t if released else t + self.ttl_s,
                "ttl_s": self.ttl_s, "released": released}

    def ensure(self) -> tuple[bool, dict | None]:
        cur = self.backend.read()
        doc, etag = cur if cur else ({}, None)
        mine = cur is not None and doc.get("token") == self.token
        if cur is not None and not mine and self.free(doc) is None:
            self.held = False
            return False, doc
        try:
            self.backend.write(self._doc(doc.get("acquired_at") if mine else None), etag)
        except LeaseConflict:
            self.held = False
            back = self.backend.read()
            return False, back[0] if back else None
        back = self.backend.read()
        self.held = back is not None and back[0].get("token") == self.token
        return self.held, back[0] if back else None

    def release(self) -> None:
        if not self.held:
            return
        cur = self.backend.read()
        if cur and cur[0].get("token") == self.token:
            try:
                self.backend.write(self._doc(cur[0].get("acquired_at"), released=True), cur[1])
            except LeaseConflict:
                pass
        self.held = False


def holder(doc: dict | None, now: float | None = None) -> str:
    """`<owner> (host H, pid P) until HH:MMZ` for a lease document."""
    if not doc:
        return "nobody (no lease document)"
    until = dt.datetime.fromtimestamp(float(doc.get("expires_at") or 0), dt.timezone.utc)
    left = (float(doc.get("expires_at") or 0) - (now or time.time())) / 60
    return f"{doc.get('owner')} (host {doc.get('host')}, pid {doc.get('pid')}) until {until:%H:%MZ} ({left:.0f} min)"


# ---- the service definition ----------------------------------------------------------------------------------------------------------

DEFAULT_WATCH = ["--every", "2", "--pods", "--reap", "--lease", "--standby", "--liveness", "git", "--fetch", "--sync",
                 "--stale-min", "12", "--since-hours", "8", "--idle-min", "5", "--exclude", "vllm*"]
SYSTEMD_NAME = "research-steward.service"
LAUNCHD_LABEL = "com.research.steward"


def unit(kind: str, python: str, workdir: Path, watch_args: list[str] | None = None, env_file: Path | None = None,
         log: Path | None = None, extra_env: dict[str, str] | None = None) -> str:
    """The service definition text (module docstring, SERVICE).  `watch_args` default: DEFAULT_WATCH, plus `--guardian-log none`
    for systemd (the guardian kill routing stays on the laptop)."""
    args = list(watch_args) if watch_args else DEFAULT_WATCH + (["--guardian-log", "none"] if kind == "systemd" else [])
    argv = [python, "-m", "research", "notes", "watch", *args]
    env = {"PYTHONUNBUFFERED": "1", **(extra_env or {})}
    if kind == "systemd":
        lines = [
            f"# ~/.config/systemd/user/{SYSTEMD_NAME}  (research notes service systemd)",
            f"#   systemctl --user daemon-reload && systemctl --user enable --now {SYSTEMD_NAME}",
            "#   loginctl enable-linger \"$USER\"     # start at boot, not at login",
            f"#   journalctl --user -u {SYSTEMD_NAME} -f",
            "[Unit]",
            "Description=research steward: liveness, reaping, deadlines, budgets, daily render (research notes watch --lease)",
            "Wants=network-online.target",
            "After=network-online.target",
            "StartLimitIntervalSec=0",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={workdir}",
            *([f"EnvironmentFile=-{env_file}"] if env_file else []),
            *(f"Environment={shlex.quote(f'{k}={v}')}" for k, v in env.items()),
            "ExecStart=" + " ".join(shlex.quote(a).replace("%", "%%") for a in argv),
            "Restart=always",
            "RestartSec=60",
            "KillSignal=SIGTERM",
            "TimeoutStopSec=90",
            "",
            "[Install]",
            "WantedBy=default.target",
            ""]
        return "\n".join(lines)
    if kind == "launchd":
        log = log or Path.home() / ".research" / "log" / "steward.log"
        doc = {"Label": LAUNCHD_LABEL, "ProgramArguments": argv, "WorkingDirectory": str(workdir), "RunAtLoad": True,
               "KeepAlive": True, "ThrottleInterval": 60, "ProcessType": "Background", "EnvironmentVariables": env,
               "StandardOutPath": str(log), "StandardErrorPath": str(log)}
        head = (f"<!-- ~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist  (research notes service launchd)\n"
                f"     launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist\n"
                f"     R2 credentials: the watcher reads ~/.config/verity/r2.env itself (for reap custody and the lease). -->\n")
        xml = plistlib.dumps(doc).decode()
        first, rest = xml.split("\n", 1)
        return first + "\n" + head + rest
    raise ValueError(f"service kind {kind!r}: systemd | launchd")


def service_main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="research notes service", description="print the steward's service definition")
    ap.add_argument("kind", choices=["systemd", "launchd"])
    ap.add_argument("--python", default=sys.executable, help="the interpreter with `research` importable (default: this one)")
    ap.add_argument("--workdir", type=Path, default=Path.cwd(), help="a checkout of verity (the watcher's cwd; default: here)")
    ap.add_argument("--env-file", type=Path, default=Path.home() / ".config" / "verity" / "r2.env",
                    help="systemd: KEY=VALUE credentials file (R2, RunPod), optional at start")
    ap.add_argument("--log", type=Path, default=None, help="launchd: stdout/stderr file (default ~/.research/log/steward.log)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="extra environment (RESEARCH_NOTES, RESEARCH_REPO, ...)")
    ap.epilog = "-- then the watch arguments (default: the steward's)"
    own, args = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ns = ap.parse_args(own)
    extra = dict(kv.split("=", 1) for kv in ns.env if "=" in kv)
    print(unit(ns.kind, ns.python, ns.workdir.expanduser().resolve(), args or None, ns.env_file, ns.log, extra), end="")
    return 0
