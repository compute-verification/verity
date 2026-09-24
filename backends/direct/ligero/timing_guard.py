"""The bench timing guard: did anything compete with a timed run?

``run.main`` runs every ``bench`` / ``bench-vu`` under a :class:`Guard` (:func:`guarded`).  A daemon thread samples, every
:data:`INTERVAL` s and once more at the end,

* the GPU compute processes other than this one (``nvidia-smi --query-compute-apps``; the coordinator's rule is to time only
  when that list is empty),
* the GPU's utilisation and the 1-minute load average,
* the cgroup's CPU throttling counters (``cpu.stat`` ``nr_periods`` / ``nr_throttled``, cgroup v2 or v1).

When the command returns, the verdict is written into the JSON it wrote (``--out``) as ``contention`` (``bench_result.py``
carries it into ``result.json``) and printed::

    "contention": {"contended": true, "reasons": ["1 other GPU compute process (pids seen: 4711)"], "other_gpu_procs": 1,
                   "gpu_pids_seen": ["4711"], "throttled": 0.02, "busy_periods": 410, "load1_max": 3.1, "cpu_quota": 10.2,
                   "nproc": 96, "window_s": 41.0, "samples": 21}

Contended when another process held a GPU context at any sample; when the cgroup was throttled in more than :data:`THROTTLED`
of its scheduler periods while the GPU was busy (so an instance build or leaf hashing before the timed reps does not count);
or when the load average exceeded the host's cores.  ``verity_numerical.bench.summary`` shows the column, and its ``--best``
selection refuses contended rows.  ``VY_TIMING_GUARD=0`` turns the guard off.
"""
from __future__ import annotations

import functools
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

INTERVAL = 2.0
THROTTLED = 0.10
GPU_BUSY = 5.0
COMMANDS = ("bench", "bench-vu")
CGROUP = Path("/sys/fs/cgroup")


def cpu_quota(root: Path = CGROUP) -> float | None:
    """The cgroup's CPU quota in cores: v2 ``cpu.max``, else v1 ``cpu/cpu.cfs_quota_us`` / ``cpu.cfs_period_us``.  ``None`` when
    unlimited or unreadable."""
    try:
        q, p = (root / "cpu.max").read_text().split()[:2]
    except (OSError, ValueError):
        try:
            q, p = ((root / "cpu" / f).read_text().strip() for f in ("cpu.cfs_quota_us", "cpu.cfs_period_us"))
        except OSError:
            return None
    try:
        return int(q) / int(p) if q not in ("max", "-1") and int(p) > 0 else None
    except ValueError:
        return None


def cpu_stat(root: Path = CGROUP) -> dict[str, int]:
    """The cgroup's ``cpu.stat`` counters (v2 ``cpu.stat``, else v1 ``cpu/cpu.stat``); ``{}`` when unreadable."""
    for f in (root / "cpu.stat", root / "cpu" / "cpu.stat"):
        try:
            return {k: int(v) for k, v in (line.split() for line in f.read_text().splitlines() if line.strip())}
        except (OSError, ValueError):
            continue
    return {}


def _smi(query: str) -> list[str] | None:
    try:
        r = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [line.split(",")[0].strip() for line in r.stdout.splitlines() if line.strip()] if r.returncode == 0 else None


def _own_context() -> bool:
    torch = sys.modules.get("torch")
    try:
        return bool(torch is not None and torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001  (a half-imported torch)
        return False


def sample() -> dict[str, Any]:
    """One observation.  ``n_other`` is the count of GPU compute processes that are not this one: when NVML lists pids of another
    namespace (ours is not among them) and this process holds a context, one of the listed is ours."""
    pids, util = _smi("--query-compute-apps=pid"), _smi("--query-gpu=utilization.gpu")
    me = str(os.getpid())
    if pids is None:
        n_other = None
    elif me in pids:
        pids = [p for p in pids if p != me]
        n_other = len(pids)
    else:
        n_other = max(len(pids) - _own_context(), 0)
    try:
        load1: float | None = os.getloadavg()[0]
    except OSError:
        load1 = None
    busy = [float(u) for u in util or [] if u.replace(".", "", 1).isdigit()]
    stat = cpu_stat()
    return {"t": time.monotonic(), "n_other": n_other, "pids": pids, "util": max(busy) if busy else None, "load1": load1,
            **{k: stat[k] for k in ("nr_periods", "nr_throttled") if k in stat}}


def verdict(samples: Sequence[dict[str, Any]], quota: float | None, nproc: int | None) -> dict[str, Any]:
    """The ``contention`` record of a run's samples (see the module docstring for the rules)."""
    reasons: list[str] = []
    others = max((s["n_other"] for s in samples if s.get("n_other") is not None), default=None)
    pids = sorted({p for s in samples for p in s.get("pids") or []}, key=lambda p: (len(p), p))
    if others:
        reasons.append(f"{others} other GPU compute process{'es' * (others > 1)} (pids seen: {' '.join(pids)})")
    periods = throttled = 0
    for a, b in itertools.pairwise(samples):
        if max(a.get("util") or 0, b.get("util") or 0) >= GPU_BUSY and "nr_periods" in a and "nr_periods" in b:
            periods += b["nr_periods"] - a["nr_periods"]
            throttled += b["nr_throttled"] - a["nr_throttled"]
    frac = throttled / periods if periods > 0 else None
    if frac is not None and frac > THROTTLED:
        reasons.append(f"CPU throttled in {frac:.0%} of the cgroup's periods while the GPU was busy (quota "
                       f"{f'{quota:.1f} cores' if quota else 'unknown'})")
    load = max((s["load1"] for s in samples if s.get("load1") is not None), default=None)
    if load is not None and nproc and load > nproc:
        reasons.append(f"load average {load:.1f} > {nproc} cores")
    return {"contended": bool(reasons), "reasons": reasons, "other_gpu_procs": others, "gpu_pids_seen": pids,
            "throttled": None if frac is None else round(frac, 4), "busy_periods": periods, "load1_max": load, "cpu_quota": quota,
            "nproc": nproc, "window_s": round(samples[-1]["t"] - samples[0]["t"], 1) if samples else 0.0, "samples": len(samples)}


class Guard:
    """Samples in a daemon thread from :meth:`start` to :meth:`stop` (which returns the :func:`verdict`)."""

    def __init__(self, interval: float = INTERVAL, sampler: Callable[[], dict[str, Any]] | None = None):
        self.interval, self.sampler = interval, sampler or sample
        self.samples: list[dict[str, Any]] = []
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="timing-guard", daemon=True)

    def start(self) -> Guard:
        self.samples.append(self.sampler())
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._done.wait(self.interval):
            self.samples.append(self.sampler())

    def stop(self) -> dict[str, Any]:
        self._done.set()
        self._thread.join()
        self.samples.append(self.sampler())
        return verdict(self.samples, cpu_quota(), os.cpu_count())


def stamp(out: str | None, report: dict[str, Any], since: float) -> Path | None:
    """Put ``report`` into the JSON object at ``out`` as ``contention`` when the run wrote it (modified at or after ``since``)."""
    if not out:
        return None
    p = Path(os.path.expandvars(out))
    try:
        if p.stat().st_mtime < since:
            return None
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    doc["contention"] = report
    tmp = p.with_name(p.name + ".guard-tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    os.replace(tmp, p)
    return p


def _opt(argv: Sequence[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def guarded(main: Callable[..., int]) -> Callable[..., int]:
    """``main(argv)`` under a :class:`Guard` when argv names one of :data:`COMMANDS`; the verdict is stamped into its ``--out``."""

    @functools.wraps(main)
    def run(argv: Sequence[str] | None = None) -> int:
        args = list(sys.argv[1:] if argv is None else argv)
        if os.environ.get("VY_TIMING_GUARD") == "0" or not set(COMMANDS) & set(args):
            return main(argv)
        since = time.time()
        guard = Guard().start()
        try:
            return main(argv)
        finally:
            rep = guard.stop()
            where = stamp(_opt(args, "--out"), rep, since)
            state = "CONTENDED: " + "; ".join(rep["reasons"]) if rep["contended"] else "clean"
            print(f"timing guard: {state} ({rep['samples']} samples over {rep['window_s']} s"
                  f"{f'; recorded in {where}' if where else ''})", file=sys.stderr, flush=True)

    return run
