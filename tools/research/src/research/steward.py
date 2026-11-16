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


class GitTimes:
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
        except (OSError, subprocess.SubprocessError) as e:
            print(f"steward: git {args[0] if args else ''} in {self.root} did not run ({type(e).__name__})", file=sys.stderr, flush=True)
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


class LeaseConflict(Exception):
    pass


class S3LeaseBackend:
    def __init__(self, remote: Any, key: str = LEASE_KEY):
        self.remote, self.key = remote, key

    def describe(self) -> str:
        return f"{self.remote.describe()}/{self.key}"

    def read(self) -> tuple[dict, str] | None:
        try:
            with self.remote._request("GET", self.key) as r:
                body, etag = r.read(), r.headers.get("ETag")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        try:
            doc = json.loads(body)
        except ValueError as e:
            print(f"steward: the lease's body is not JSON, read as empty ({e})", file=sys.stderr, flush=True)
            doc = {}
        return (doc if isinstance(doc, dict) else {}), etag or ""

    def write(self, doc: dict, etag: str | None) -> None:
        cond = {"if-none-match": "*"} if etag is None else {"if-match": etag}
        try:
            with self.remote._request("PUT", self.key, body=_encode(doc), headers=cond):
                pass
        except urllib.error.HTTPError as e:
            if e.code in (409, 412):
                raise LeaseConflict(f"{self.key}: HTTP {e.code}") from None
            if e.code in (400, 501):
                raise RuntimeError(f"{self.describe()}: the store refused a conditional PUT (HTTP {e.code}); no lease without it") from None
            raise


class FileLeaseBackend:
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
        except ValueError as e:
            print(f"steward: {self.path} is not JSON, read as empty ({e})", file=sys.stderr, flush=True)
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
            except LeaseConflict as e:
                print(f"steward: the lease changed before its release was written, so it is left as it is ({e})", file=sys.stderr, flush=True)
        self.held = False


def holder(doc: dict | None, now: float | None = None) -> str:
    if not doc:
        return "nobody (no lease document)"
    until = dt.datetime.fromtimestamp(float(doc.get("expires_at") or 0), dt.UTC)
    left = (float(doc.get("expires_at") or 0) - (now or time.time())) / 60
    return f"{doc.get('owner')} (host {doc.get('host')}, pid {doc.get('pid')}) until {until:%H:%MZ} ({left:.0f} min)"


DEFAULT_WATCH = ["--every", "2", "--pods", "--reap", "--custody-r2", "--lease", "--standby", "--liveness", "git", "--fetch", "--sync",
                 "--stale-min", "12", "--since-hours", "8", "--idle-min", "5", "--exclude", "vllm*"]
SYSTEMD_NAME = "research-steward.service"
LAUNCHD_LABEL = "com.research.steward"


def unit(kind: str, python: str, workdir: Path, watch_args: list[str] | None = None, env_file: Path | None = None,
         log: Path | None = None, extra_env: dict[str, str] | None = None) -> str:
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
