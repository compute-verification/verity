"""The node agent on vy-nebius-2: the one writer of the node's ledger and `state.json`.

`run` ticks until it is stopped, the stop file appears, or `until` passes:

- **Timed windows are left alone.** While the fill runner's `status.txt` says `timed True`, or the last observation held
  a timed lease, the agent reads nothing but `status.txt`, every `window_poll_s` seconds, and resumes once it says
  `timed False` and was written after the window began. With no `status.txt` (no fill runner) it reads the leases every
  `window_poll_s` until the timed lease is gone. It never touches NVML or `nvidia-smi` (an audit hook counts any
  process it starts and any library it loads that names them, and `summary.json` reports the count).
- **Shadow mode** (`mode="shadow"`) observes `gpu-lease` (`nebius2.read`, `preempted.log`) and plans beside it
  (`shadow.Shadow`). It never takes `agent.lock`, so `gpu-lease` keeps deciding.
- **Live mode** (`mode="live"`) holds `<lease dir>/agent.lock` for as long as it runs, so `gpu-lease` asks it
  (`gpu-lease`'s agent mode), and carries out its plan: a grant is `grant.<pid>` naming the GPUs, an eviction is SIGTERM
  to the lease's scope or pid, SIGKILL `kill_after_s` later, each logged to `preempted.log` in `gpu-lease`'s format. The
  kernel drops the lock when the agent exits, and `gpu-lease` falls back to its own rules.
- **It stops itself** on an exception, or when a plan would break a rule the observed state doesn't
  (`invariants.check`), writing `summary.json` with the reason, and exits nonzero. In live mode that is the fallback.

The clock and sleep are parameters, so tests drive it without wall-clock time.
"""
from __future__ import annotations

import fcntl
import json
import os
import resource
import signal
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import ledger, nebius2, shadow
from .model import Cluster
from .plan import Evict, Start, plan

STATUS = Path("/workspace/pouw/fill/status.txt")


@dataclass
class Footprint:
    """What the agent cost: CPU from rusage, and the processes it started and libraries it loaded, from an audit hook."""
    execs: int = 0
    nvml_calls: int = 0
    seen: list[str] = field(default_factory=list)

    def hook(self, event: str, args: tuple) -> None:
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn", "os.system", "ctypes.dlopen"):
            text = " ".join(map(str, args))[:200]
            self.execs += 1
            if "nvml" in text.lower() or "nvidia" in text.lower():
                self.nvml_calls += 1
                self.seen.append(f"{event}: {text}")

    def report(self, wall_s: float) -> dict[str, Any]:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ru.ru_utime + ru.ru_stime
        return {"cpu_s": round(cpu, 3), "wall_s": round(wall_s, 1), "cpu_share": round(cpu / wall_s, 5) if wall_s > 0 else None,
                "max_rss_kib": ru.ru_maxrss, "execs": self.execs, "nvml_calls": self.nvml_calls, "nvml_seen": self.seen[:5]}


def status_timed(path: Path) -> tuple[bool | None, float | None]:
    """`status.txt`'s first line says `timed True` or `timed False`; None when it is missing or says neither."""
    try:
        st = path.stat()
        with path.open() as f:
            first = f.readline()
    except FileNotFoundError:
        return None, None
    if "timed True" in first:
        return True, st.st_mtime
    if "timed False" in first:
        return False, st.st_mtime
    return None, st.st_mtime


class PreemptedLog:
    """`preempted.log` read from where the last read ended; the first read starts at its end (what came before the agent
    is history the agent didn't see as it happened)."""

    def __init__(self, path: Path, from_start: bool = False):
        self.path = path
        self.offset = 0 if from_start or not path.exists() else path.stat().st_size

    def new(self) -> list[dict[str, Any]]:
        try:
            with self.path.open() as f:
                f.seek(self.offset)
                text = f.read()
        except FileNotFoundError:
            return []
        keep = text.rfind("\n") + 1
        self.offset += keep
        return [p for p in map(nebius2.preemption, text[:keep].splitlines()) if p]


@dataclass
class Executor:
    """Live mode's hands on node 2: grants through `grant.<pid>` files and evictions by signal, logged as `gpu-lease` logs
    them. `kill` is `os.kill` (tests pass a recorder)."""
    lease_dir: Path
    kill: Callable[[int, int], None] = os.kill
    kill_after_s: int = 30
    termed: dict[str, tuple[int, int]] = field(default_factory=dict)

    def grant(self, pid: int, gpus: tuple[int, ...]) -> None:
        tmp = self.lease_dir / f".grant.{pid}"
        tmp.write_text(",".join(map(str, gpus)) + "\n")
        os.replace(tmp, self.lease_dir / f"grant.{pid}")

    def evict(self, t: int, alloc: str, gpus: tuple[int, ...], owner: str, by_pid: int, n: int, who: str) -> None:
        """SIGTERM the lease's scope, else its pid; SIGKILL once `kill_after_s` have passed; one log line per GPU."""
        pid = int(alloc.split("@")[0][3:])
        first = self.termed.setdefault(alloc, (t, pid))[0]
        if t == first:
            sig = signal.SIGTERM
        elif t - first >= self.kill_after_s:
            sig = signal.SIGKILL
        else:
            return
        try:
            self.kill(pid, sig)
        except ProcessLookupError:
            return
        with (self.lease_dir / "preempted.log").open("a") as log:
            for gpu in gpus:
                log.write(f"{stamp(t)} SIG{sig.name[3:]} GPU {gpu} for pid {by_pid} (n={n}, who={who}): {owner}\n")


def stamp(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(cluster: Cluster, out: Path, *, mode: str = "shadow", clock: Callable[[], float], sleep: Callable[[float], None],
        lease_dir: Path = nebius2.LEASE_DIR, proc: Path = Path("/proc"), status: Path = STATUS,
        stop_file: Path | None = None, until: float | None = None, tick_s: float = 1.0, window_poll_s: float = 5.0,
        footprint: Footprint | None = None, executor: Executor | None = None) -> dict[str, Any]:
    """Run the agent (module docstring) and return its summary, also written to `out/summary.json`."""
    if mode not in ("shadow", "live"):
        raise ValueError(f"unknown mode {mode!r}")
    out = Path(out)
    fp = footprint or Footprint()
    t0 = clock()
    sh = shadow.Shadow(cluster, out)
    pre = PreemptedLog(lease_dir / "preempted.log")
    lock_fd = None
    stop = {"why": None}
    exception = None
    in_window_since: float | None = None
    windows = ticks = window_polls = status_only = 0
    prev_handler = signal.getsignal(signal.SIGTERM)

    def on_term(signum, frame):
        stop["why"] = "SIGTERM"

    if mode == "live":
        executor = executor or Executor(lease_dir)
        lock_fd = os.open(lease_dir / "agent.lock", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        signal.signal(signal.SIGTERM, on_term)
    except ValueError:
        pass
    try:
        while stop["why"] is None:
            now = clock()
            if until is not None and now >= until:
                stop["why"] = "until"
                break
            if stop_file is not None and stop_file.exists():
                stop["why"] = f"stop file {stop_file}"
                break
            timed, mtime = status_timed(status)
            if in_window_since is not None and timed is not None and (timed or mtime < in_window_since):
                window_polls += 1
                sleep(window_poll_s)
                continue
            if timed:
                windows += 1
                status_only += 1
                in_window_since = now
                window_polls += 1
                sleep(window_poll_s)
                continue
            obs = nebius2.read(int(now), lease_dir, proc)
            sh.step(obs, pre.new())
            ticks += 1
            broke = [d for d in _tail(out / "divergences.jsonl", sh) if d.get("class") == "safety"
                     and d["msg"].startswith("the plan breaks a rule")]
            if broke:
                stop["why"] = f"invariant: {broke[0]['msg']}"
                break
            if mode == "live":
                _carry_out(cluster, sh, obs, executor, int(now))
            ledger.write_state(out / "state.json", sh.state)
            if nebius2.timed_now(cluster, obs):
                windows += 0 if in_window_since is not None else 1
                in_window_since = now
                sleep(window_poll_s)
                continue
            in_window_since = None
            sleep(tick_s)
    except Exception:
        exception = traceback.format_exc(limit=8)
        stop["why"] = "exception"
    finally:
        try:
            signal.signal(signal.SIGTERM, prev_handler)
        except ValueError:
            pass
        if lock_fd is not None:
            os.close(lock_fd)
    summary = {"mode": mode, "stopped": stop["why"], "exception": exception, "ticks_read": ticks,
               "window_polls": window_polls, "windows_seen": windows,
               "windows_from_status_only": status_only, "footprint": fp.report(clock() - t0)}
    return sh.close(int(clock()), summary)


def _tail(path: Path, sh: shadow.Shadow) -> list[dict[str, Any]]:
    """Divergences written since the last call (the agent reads its own output, never the node's)."""
    if not path.exists():
        return []
    size = path.stat().st_size
    start = getattr(sh, "_div_offset", 0)
    if size <= start:
        return []
    with path.open() as f:
        f.seek(start)
        text = f.read()
    sh._div_offset = size
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _carry_out(cluster: Cluster, sh: shadow.Shadow, obs: nebius2.Observation, ex: Executor, t: int) -> None:
    """Live mode: grant the requests the plan starts and evict what it evicts. The grants are `gpu-lease`'s to take; the
    ledger records them when their leases appear (the next observation), as in shadow mode."""
    observed = [a for a in sh.state.allocations.values()] + sh.unowned
    queue = [r.job for r in sh.requests.values() if r.granted is None and r.gone is None]
    if not queue:
        return
    actions = plan(cluster, sh.node, observed, queue, t)
    owners = {i: line for i, line in obs.leases.items()}
    for a in actions:
        if isinstance(a, Evict):
            victim = sh.state.allocations.get(a.alloc)
            if victim is None:
                continue
            req = sh.requests.get(a.by)
            ex.evict(t, a.alloc, victim.gpus, owners.get(victim.gpus[0], ""), int(a.by[3:]), req.job.gpus if req else 0,
                     req.job.submitter if req else "-")
    gone = {a.alloc for a in actions if isinstance(a, Evict)}
    for a in actions:
        if isinstance(a, Start) and not any(v.id in gone and set(v.gpus) & set(a.alloc.gpus) for v in observed):
            ex.grant(int(a.alloc.job.id[3:]), a.alloc.gpus)


def main_hook(fp: Footprint) -> None:
    sys.addaudithook(fp.hook)


__all__ = ["run", "Footprint", "Executor", "PreemptedLog", "status_timed", "STATUS", "main_hook"]
