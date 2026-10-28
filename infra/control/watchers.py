#!/usr/bin/env python3
"""The control pod's watchers (INF-131): one supervisor for the loops that watch the system, which ran in tmux on agent VMs and
stopped whenever a VM was suspended: suspending any agent VM stops nothing that watches the system.

    watchers.py run [--config PATH]      supervise every loop (watchers.sh runs it; one at a time: it holds the config's `lock`)
    watchers.py check [--config PATH]    each loop resolved: its argv, the environment names it gets, what it waits for
    watchers.py exec NAME [-- CMD...]    CMD (else the loop's argv) once, in loop NAME's environment and directory: a seed or a
                                         dry run sees what the loop sees

The config (watchers.toml, deployed as /data/watchers/watchers.toml) lists the loops. Each [[loop]] runs as its own process group
with the config's [env], its own `env`, and the KEY=VALUE lines of its `env_files`, so a credential reaches only the loops that
name its file. Its output is appended to <dir>/log/<name>.log. A loop that exits is started again RESTART_S later. One whose
`needs` (a file, such as a credential not placed yet) is missing waits, and starts once it exists. Every executable `*.sh` in
<dir>/extra.d runs the same way, except that one exiting 0 is done (`<name>.done` beside it) and isn't started again: a one-off
timed job (restore a slot when a run ends) goes there instead of a tmux session on an agent VM. In argv and env values, `{dir}`
and `{src}` stand for the config's `dir` and `src`.

Every HEARTBEAT_S the supervisor replaces <dir>/status.json (`verity/watchers-status/v1`): its time `t`, the commit `src` holds,
each loop's state (running, waiting, restarting, done), pid, since, restarts and last exit, the cloud watch's last pass
(`cloud_watch_t`: the cloud watch can't watch itself), and `all_up_t`, the last time every loop was up (running for UP_S, waiting
on a need, or done), which the cloud watch reads as this supervisor's success marker. The same body is upserted into Neon's
`store_private.control_heartbeat` (one row per host, through psql with `heartbeat_env`'s STORE_INDEXER_DATABASE_URL), where the
dead-man's switch (infra/control/deadman/, a Vercel cron in console's team) reads it: when the pod stops, the row goes stale and
the cron posts in #agent-alerts. A failed upsert is printed at most once an hour and stops nothing.

Children die with the supervisor (PR_SET_PDEATHSIG), and SIGTERM stops each loop's group first, so a restarted supervisor never
leaves a second copy of a loop running.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SCHEMA = "verity/watchers-status/v1"
RESTART_S = 60.0
HEARTBEAT_S = 60.0
UP_S = 120.0
TICK_S = 5.0
SAY_EVERY_S = 3600.0
STAMP = "%Y-%m-%dT%H:%M:%SZ"
HEARTBEAT_DDL = "CREATE TABLE IF NOT EXISTS {table} (host text PRIMARY KEY, t timestamptz NOT NULL, body text NOT NULL);\n"
PR_SET_PDEATHSIG = 1


def iso(t: float) -> str:
    return time.strftime(STAMP, time.gmtime(t))


def env_file(path: Path) -> dict[str, str]:
    """A shell env file's KEY=VALUE lines (an `export ` prefix and one pair of surrounding quotes dropped), never sourced."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        k, sep, v = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


@dataclass
class Loop:
    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    env_files: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    once: bool = False                      # extra.d: an exit of 0 is done
    done_file: str | None = None
    proc: subprocess.Popen | None = None
    state: str = "new"
    since: float = 0.0
    restarts: int = 0
    last_exit: int | None = None
    next_start: float = 0.0

    def missing(self) -> list[str]:
        return [p for p in [*self.needs, *self.env_files] if not Path(p).exists()]

    def report(self) -> dict:
        return {"state": self.state, "pid": self.proc.pid if self.proc and self.state == "running" else None,
                "since": iso(self.since) if self.since else None, "restarts": self.restarts, "last_exit": self.last_exit,
                **({"waiting_for": self.missing()} if self.state == "waiting" else {})}


def loop_env(lp: Loop) -> dict[str, str]:
    env = {**lp.env}
    for f in lp.env_files:
        env.update({k: v for k, v in env_file(Path(f)).items() if k != "PYTHONPATH"})   # load() checked lp.env's
    return env


def expand(s: str, dirs: dict[str, str]) -> str:
    for k, v in dirs.items():
        s = s.replace("{" + k + "}", v)
    return s


def load(path: Path) -> tuple[dict, list[Loop]]:
    cfg = tomllib.loads(path.read_text())
    dirs = {"dir": cfg["dir"], "src": cfg["src"]}
    base = {k: expand(str(v), dirs) for k, v in (cfg.get("env") or {}).items()}
    loops = []
    for spec in cfg.get("loop", []):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", spec["name"]):
            raise ValueError(f"loop name {spec['name']!r}: lower-case letters, digits and dashes")
        loops.append(Loop(spec["name"], [expand(a, dirs) for a in spec["argv"]],
                          {**base, **{k: expand(str(v), dirs) for k, v in (spec.get("env") or {}).items()}},
                          [expand(p, dirs) for p in spec.get("env_files", [])], [expand(p, dirs) for p in spec.get("needs", [])]))
    names = [lp.name for lp in loops]
    if len(set(names)) != len(names):
        raise ValueError(f"a loop name repeats: {names}")
    src = cfg["src"].rstrip("/") + "/"
    for lp in loops:
        if bad := [p for p in lp.env.get("PYTHONPATH", "").split(":") if p and not p.startswith(src)]:
            raise ValueError(f"loop {lp.name}: PYTHONPATH {':'.join(bad)} is outside {cfg['src']}: a watcher imports only the deployed "
                             "tree; to run newer code, merge it into the branch and deploy that commit (README.md's deploy)")
    cfg["_base_env"] = base
    return cfg, loops


def extra_loops(extra: Path, base: dict[str, str], taken: set[str]) -> list[Loop]:
    """Each executable *.sh in extra.d not yet done, as a loop whose exit 0 is done."""
    out = []
    for p in sorted(extra.glob("*.sh")) if extra.is_dir() else []:
        name = "extra-" + re.sub(r"[^a-z0-9-]", "-", p.stem.lower())
        done = p.with_name(p.stem + ".done")
        if name in taken or done.exists() or not os.access(p, os.X_OK):
            continue
        out.append(Loop(name, [str(p)], dict(base), once=True, done_file=str(done)))
    return out


def _die_with_parent() -> None:
    os.setsid()
    try:
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, AttributeError):
        pass


class Supervisor:
    def __init__(self, cfg: dict, loops: list[Loop], *, clock: Callable[[], float] = time.time,
                 spawn: Callable[[Loop, Path], subprocess.Popen] | None = None,
                 publish: Callable[[bytes, float], None] | None = None):
        self.cfg, self.loops, self.clock = cfg, {lp.name: lp for lp in loops}, clock
        self.dir = Path(cfg["dir"])
        self.extra = Path(expand(cfg.get("extra", "{dir}/extra.d"), {"dir": cfg["dir"], "src": cfg["src"]}))
        self.spawn = spawn or self._spawn
        self.publish = publish
        self.all_up_t: float | None = None
        self.last_beat = -HEARTBEAT_S
        (self.dir / "log").mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        print(f"{iso(self.clock())} {msg}", flush=True)

    def _spawn(self, lp: Loop, logf: Path) -> subprocess.Popen:
        with open(logf, "ab") as out:
            return subprocess.Popen(lp.argv, env=loop_env(lp), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                    cwd=str(self.dir), preexec_fn=_die_with_parent)

    def step(self) -> None:
        now = self.clock()
        for lp in extra_loops(self.extra, self.cfg.get("_base_env", {}), set(self.loops)):
            self.loops[lp.name] = lp
            self.log(f"{lp.name}: found {lp.argv[0]}")
        for lp in self.loops.values():
            if lp.state == "running":
                rc = lp.proc.poll()
                if rc is None:
                    continue
                lp.last_exit, lp.since = rc, now
                if lp.once and rc == 0:
                    lp.state = "done"
                    Path(lp.done_file).write_text(f"{iso(now)} exited 0\n")
                    self.log(f"{lp.name}: done")
                    continue
                lp.state, lp.next_start, lp.restarts = "restarting", now + RESTART_S, lp.restarts + 1
                self.log(f"{lp.name}: exited rc={rc}; starting again in {RESTART_S:.0f} s")
            if lp.state == "done" or lp.state == "running":
                continue
            if lp.missing():
                if lp.state != "waiting":
                    lp.state, lp.since = "waiting", now
                    self.log(f"{lp.name}: waiting for {', '.join(lp.missing())}")
                continue
            if now >= lp.next_start:
                try:
                    lp.proc = self.spawn(lp, self.dir / "log" / f"{lp.name}.log")
                except OSError as e:
                    lp.state, lp.next_start, lp.last_exit = "restarting", now + RESTART_S, None
                    self.log(f"{lp.name}: could not start: {e}")
                    continue
                lp.state, lp.since = "running", now
                self.log(f"{lp.name}: started pid {lp.proc.pid}")
        if all(lp.state in ("waiting", "done") or (lp.state == "running" and now - lp.since >= UP_S) for lp in self.loops.values()):
            self.all_up_t = now
        if now - self.last_beat >= HEARTBEAT_S:
            self.last_beat = now
            self.heartbeat(now)

    def status(self, now: float) -> dict:
        commit = self.cfg["src"] + "/COMMIT"
        try:
            commit = Path(commit).read_text().strip()
        except OSError:
            commit = None
        cw = self.cfg.get("cloud_watch_state")
        try:
            cw_t = iso(Path(cw).stat().st_mtime) if cw else None
        except OSError:
            cw_t = None
        return {"schema": SCHEMA, "t": iso(now), "host": self.cfg.get("host") or socket.gethostname(), "commit": commit,
                "all_up_t": iso(self.all_up_t) if self.all_up_t else None, "cloud_watch_t": cw_t,
                "loops": {name: lp.report() for name, lp in sorted(self.loops.items())}}

    def heartbeat(self, now: float) -> None:
        body = (json.dumps(self.status(now), indent=1, sort_keys=True) + "\n").encode()
        tmp = self.dir / "status.json.tmp"
        tmp.write_bytes(body)
        tmp.replace(self.dir / "status.json")
        if self.publish:
            self.publish(body, now)

    def stop(self) -> None:
        for lp in self.loops.values():
            if lp.state == "running" and lp.proc.poll() is None:
                try:
                    os.killpg(lp.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for lp in self.loops.values():
            if lp.state == "running":
                try:
                    lp.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(lp.proc.pid, signal.SIGKILL)


class Publisher:
    """The status body upserted into Neon's heartbeat table at `host`, through psql; the table is created on first use."""

    def __init__(self, url: str, table: str, host: str, psql: str = "psql"):
        self.url, self.table, self.host, self.psql = url, table, host, psql
        self.created, self.said = False, -SAY_EVERY_S

    def script(self, body: bytes, t: float) -> str:
        lit = "'" + body.decode().replace("'", "''") + "'"
        host = "'" + self.host.replace("'", "''") + "'"
        return ((HEARTBEAT_DDL.format(table=self.table) if not self.created else "")
                + f"INSERT INTO {self.table} (host, t, body) VALUES ({host}, '{iso(t)}', {lit}) "
                  f"ON CONFLICT (host) DO UPDATE SET t = EXCLUDED.t, body = EXCLUDED.body;\n")

    def __call__(self, body: bytes, t: float) -> None:
        from research.store.bucket_index import pg_env

        try:
            r = subprocess.run([self.psql, "-X", "-q", "-1", "-v", "ON_ERROR_STOP=1", "-f", "-"], input=self.script(body, t),
                               env={**os.environ, **pg_env(self.url)}, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                raise RuntimeError(f"psql exited {r.returncode}: {r.stderr.strip()[-300:]}")
            self.created = True
        except Exception as e:  # noqa: BLE001 (a failed heartbeat stops no loop)
            if t - self.said >= SAY_EVERY_S:
                self.said = t
                print(f"{iso(t)} heartbeat not published to {self.table} (said at most once an hour): {type(e).__name__}: {e}",
                      flush=True)


def publisher(cfg: dict) -> Publisher | None:
    path = cfg.get("heartbeat_env")
    if not path or not Path(path).is_file():
        print(f"no {path}: the heartbeat goes only to status.json", flush=True)
        return None
    url = env_file(Path(path)).get("STORE_INDEXER_DATABASE_URL")
    if not url:
        return None
    return Publisher(url, cfg.get("heartbeat_table", "store_private.control_heartbeat"), cfg.get("host") or socket.gethostname())


def cmd_run(ns: argparse.Namespace) -> int:
    cfg, loops = load(ns.config)
    d = Path(cfg["dir"])
    d.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.get("lock", str(d / "supervisor.lock"))
    lock = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{iso(time.time())} another supervisor holds {lock_path}", flush=True)
        return 3
    (d / "supervisor.pid").write_text(f"{os.getpid()}\n")
    sup = Supervisor(cfg, loops, publish=publisher(cfg))
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(1))
    sup.log(f"supervising {len(loops)} loops from {ns.config}")
    try:
        while not stopping:
            sup.step()
            time.sleep(TICK_S)
    finally:
        sup.log("stopping every loop")
        sup.stop()
    return 0


def cmd_check(ns: argparse.Namespace) -> int:
    cfg, loops = load(ns.config)
    loops += extra_loops(Path(expand(cfg.get("extra", "{dir}/extra.d"), {"dir": cfg["dir"], "src": cfg["src"]})),
                         cfg["_base_env"], {lp.name for lp in loops})
    bad = 0
    for lp in loops:
        names = sorted(lp.env) + [f"{f}:*" for f in lp.env_files]
        miss = lp.missing()
        print(f"{lp.name}: {' '.join(lp.argv)}\n    env {', '.join(names)}" + (f"\n    waits for {', '.join(miss)}" if miss else ""))
        if not Path(lp.argv[0]).is_absolute() and not any((Path(p) / lp.argv[0]).exists()
                                                           for p in lp.env.get("PATH", os.environ["PATH"]).split(":")):
            print(f"    {lp.argv[0]} is not on its PATH")
            bad = 1
    return bad


def cmd_exec(ns: argparse.Namespace) -> int:
    cfg, loops = load(ns.config)
    lp = next((lp for lp in loops if lp.name == ns.loop), None)
    if lp is None:
        print(f"no loop {ns.loop!r} in {ns.config}: {', '.join(lp.name for lp in loops)}")
        return 2
    if miss := [f for f in lp.env_files if not Path(f).exists()]:
        print(f"{lp.name} has no {', '.join(miss)} yet")
        return 2
    return subprocess.run(ns.argv or lp.argv, env=loop_env(lp), cwd=cfg["dir"]).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="watchers.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check", "exec"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=Path("/data/watchers/watchers.toml"))
        if name == "exec":
            p.add_argument("loop")
            p.add_argument("argv", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    if ns.cmd == "exec" and ns.argv[:1] == ["--"]:
        ns.argv = ns.argv[1:]
    return {"run": cmd_run, "check": cmd_check, "exec": cmd_exec}[ns.cmd](ns)


if __name__ == "__main__":
    sys.exit(main())
