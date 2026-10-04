"""Level-0 sidecar: `tele sample --attempt-dir D --cadence 1.0 [--root-pid P | --pgid G] ...` (brief §4).

Writes scoped samples to `resources.jsonl` (schema telemetry/resources/v0.1) and a heartbeat `sampler.json`.  Every record has
`kind`, `scope`, `t_utc`, `t_mono`, `seq`, `stream`.  Bounded: fixed reads, --max-pids, --max-bytes (cadence doubles, never stops the
workload), errors rate-limited and counted; smaps_rollup, which holds a process's mmap lock while it is read, never for a process over
--pss-max-rss-gb and at most 1/PSS_HOLD_FACTOR of any process's time.  Survivable: SIGHUP ignored, SIGTERM/SIGINT -> final readings + `sampler_end`, workload death
-> final readings + linger.  It never signals the workload.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

from . import COLLECTOR_VERSION, RESOURCES_SCHEMA
from .cgroup import Cgroup, scope_for
from .events import Emitter, append, stream_id, utc_now_ms
from .job import write_json_atomic
from .procs import diskstats, gpu_sample, pss, snapshot_tree, statvfs

#: The clock (seconds) that decides which readings are due within a round and times the smaps_rollup reads; a test replaces it.
#: The loop's sleeps, gaps and linger, and the records' t_mono, stay on time.monotonic.
clock = time.monotonic
#: The RSS breakdown that /proc/<pid>/status gave the process snapshot: what a process whose smaps_rollup is not read records instead.
STATUS_RSS = ("rss_bytes", "rss_anon_bytes", "rss_file_bytes", "rss_shmem_bytes")
#: A process's smaps_rollup is read again no sooner than this many times its last read took, so that the reads hold its mmap lock
#: at most 1/PSS_HOLD_FACTOR of its time.
PSS_HOLD_FACTOR = 200


class Sampler:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.dir = Path(a.attempt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        suffix = a.stream_suffix if a.stream_suffix is not None else os.environ.get("TELEMETRY_STREAM_SUFFIX", "")
        self.path = self.dir / (f"resources.{suffix}.jsonl" if suffix else "resources.jsonl")
        self.emitter = Emitter(self.dir, suffix, proc="tele-sample")
        self.stream = stream_id()
        self.seq = 0
        self.bytes_written = 0
        self.cadence = float(a.cadence)
        self.base_cadence = self.cadence
        self.stop = False
        self.stop_reason: str | None = None
        self.errors: dict[str, int] = {}
        self.error_samples: list[str] = []
        self.gaps: list[dict[str, Any]] = []
        self.samples = 0
        self.skipped: dict[str, int] = {}
        self.cg = Cgroup()
        wl_cg = Cgroup(a.root_pid) if a.root_pid else None
        self.scope, self.scope_reason = scope_for(self.cg, wl_cg)
        self.gpu_state: str = "unknown"
        self.known_pids: dict[int, int] = {}      # pid -> starttime
        self.known_outsiders: dict[int, tuple[int, str]] = {}
        self.topology_changed_at = 0.0
        self.last_pss = 0.0
        self.pss_hold: dict[tuple[int, int | None], tuple[float, float]] = {}   # (pid, starttime) -> (no read before, last read's s)
        self.last_gpu = 0.0
        self.last_storage = 0.0
        self.last_self = 0.0
        self.mounts = [str(self.dir)] + list(a.mount or [])
        self.t0 = time.monotonic()
        self.cpu0 = resource.getrusage(resource.RUSAGE_SELF)

    # ---- writing --------------------------------------------------------------------------------------------------------------------
    def write(self, kind: str, scope: str, body: dict[str, Any]) -> None:
        self.seq += 1
        rec = {"schema": RESOURCES_SCHEMA, "kind": kind, "scope": scope, "t_utc": utc_now_ms(), "t_mono": time.monotonic(),
               "seq": self.seq, "stream": self.stream, **body}
        try:
            data = (json.dumps(rec, separators=(",", ":"), default=str) + "\n").encode()
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            self.bytes_written += len(data)
        except OSError as e:
            self.err("write", str(e))
        if self.bytes_written > self.a.max_bytes and self.cadence < 60:
            self.cadence = min(60.0, self.cadence * 2)
            self.emitter.emit("sampler_degraded", cadence_s=self.cadence, bytes_written=self.bytes_written, max_bytes=self.a.max_bytes,
                              reason="output cap reached: cadence doubled")
            self.a.max_bytes *= 2   # next doubling at the next cap

    def err(self, kind: str, msg: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1
        if len(self.error_samples) < 10:
            self.error_samples.append(f"{kind}: {msg}")

    # ---- one round ------------------------------------------------------------------------------------------------------------------
    def round(self, final: bool = False) -> bool:
        """Take one round of readings.  Returns False when the workload root has gone."""
        now = clock()
        alive = True
        # cgroup
        try:
            self.write("cgroup", self.scope, {"cgroup": self.cg.sample(), "scope_reason": self.scope_reason if self.seq < 3 else None})
        except Exception as e:  # noqa: BLE001 -- the sampler must not die on a reading
            self.err("cgroup", repr(e))
        # process tree
        snapped = True
        try:
            rows, truncated, total, outsiders = snapshot_tree(self.a.root_pid, self.a.pgid, self.a.max_pids)
            cur = {r["pid"]: r["starttime_ticks"] for r in rows}
            created = [p for p in cur if self.known_pids.get(p) != cur[p]]
            gone = [p for p in self.known_pids if p not in cur]
            if self.a.root_pid is not None and self.a.root_pid not in cur:
                alive = False
            out_cur = {o["pid"]: o["starttime_ticks"] for o in outsiders}
            out_gone = [{"pid": p, "comm": self.known_outsiders[p][1]} for p in self.known_outsiders if out_cur.get(p) != self.known_outsiders[p][0]]
            self.write("process", "process", {"processes": rows, "truncated": truncated, "selected": total, "created": created, "gone": gone,
                                              "root_pid": self.a.root_pid, "pgid": self.a.pgid,
                                              "outsiders": outsiders[: self.a.max_outsiders], "outsiders_total": len(outsiders), "outsiders_gone": out_gone,
                                              "outsiders_note": "other processes visible in the same pod cgroup (shared scope); needed for OOM attribution"})
            if created or gone:
                self.topology_changed_at = now
            self.known_pids = cur
            self.known_outsiders = {o["pid"]: (o["starttime_ticks"], o["comm"]) for o in outsiders}
        except Exception as e:  # noqa: BLE001
            self.err("process", repr(e))
            rows, snapped = [], False
        # PSS every pss_every s, plus a bounded extra round on topology change (at most one per 5 s); pss_every 0 turns it off
        if self.a.pss_every > 0 and (final or now - self.last_pss >= self.a.pss_every
                                     or (self.topology_changed_at and now - self.topology_changed_at < self.cadence and now - self.last_pss >= 5.0)):
            try:
                out = [self._pss(r) for r in rows[: self.a.max_pids]]
                if snapped:                       # a failed snapshot says nothing about which processes are gone
                    live = {(r["pid"], r.get("starttime_ticks")) for r in rows}
                    self.pss_hold = {k: v for k, v in self.pss_hold.items() if k in live}
                tot = sum(x["pss_bytes"] for x in out if x.get("pss_bytes") is not None)
                rss_sum = sum(r.get("rss_bytes") or 0 for r in rows)
                skipped = sum(1 for x in out if x.get("skipped"))
                self.write("pss", "process", {"processes": out, "pss_sum_bytes": tot, "rss_sum_bytes": rss_sum,
                                              "denied": sum(1 for x in out if x.get("pss_bytes") is None) - skipped, "skipped": skipped,
                                              "note": "pss shares pages proportionally; rss_sum counts shared pages once per process; pss_sum "
                                                      "leaves out the denied and the skipped (not read; their status RSS is recorded instead)"})
            except Exception as e:  # noqa: BLE001
                self.err("pss", repr(e))
            self.last_pss = now
        # GPU
        if self.gpu_state != "absent" and (final or now - self.last_gpu >= self.a.gpu_every):
            try:
                g = gpu_sample(self.a.gpu_timeout)
                if not g.get("available"):
                    if self.gpu_state != "absent":
                        self.write("gpu", "gpu", {"available": False, "reason": g.get("reason"), "note": "no further GPU samples will be attempted"})
                    self.gpu_state = "absent"
                else:
                    self.gpu_state = "present"
                    self.write("gpu", "gpu", g)
            except Exception as e:  # noqa: BLE001
                self.err("gpu", repr(e))
            self.last_gpu = now
        # storage
        if final or now - self.last_storage >= self.a.storage_every:
            try:
                self.write("storage", "host", {"mounts": [statvfs(m) for m in self.mounts], "diskstats": diskstats(),
                                               "note": "diskstats are cumulative device counters (sectors of 512 B); deltas are computed by the reader"})
            except Exception as e:  # noqa: BLE001
                self.err("storage", repr(e))
            self.last_storage = now
        # own overhead
        if final or now - self.last_self >= 30.0:
            self.write_self()
            self.last_self = now
        self.samples += 1
        return alive

    def _pss(self, row: dict[str, Any]) -> dict[str, Any]:
        """The process's smaps_rollup, unless its RSS is over --pss-max-rss-gb or its last read was less than PSS_HOLD_FACTOR times
        that read's length ago: the read walks the whole address space under the process's mmap lock, which stalls a large process
        for seconds."""
        rss = row.get("rss_bytes", row.get("rss_bytes_stat"))
        skipped = {"pid": row["pid"], "pss_bytes": None, "skipped": True, **{k: row.get(k) for k in STATUS_RSS},
                   **({} if "rss_bytes" in row else {"rss_bytes_stat": row.get("rss_bytes_stat")})}
        if rss is not None and rss > self.a.pss_max_rss_gb * 1e9:
            return {**skipped, "reason": f"skipped: rss {rss / 1e9:.2f} GB over --pss-max-rss-gb {self.a.pss_max_rss_gb:g} "
                                         "(smaps_rollup holds the process's mmap lock for seconds)"}
        key = (row["pid"], row.get("starttime_ticks"))
        t0 = clock()
        not_before, took = self.pss_hold.get(key, (t0, 0.0))
        if t0 < not_before:
            return {**skipped, "reason": f"skipped: its last smaps_rollup read took {took:.3f} s, so the next waits {PSS_HOLD_FACTOR}x "
                                         f"that ({not_before - t0:.1f} s more): reads hold its mmap lock at most 1/{PSS_HOLD_FACTOR} "
                                         "of its time"}
        out = pss(row["pid"])
        took = clock() - t0
        self.pss_hold[key] = (t0 + PSS_HOLD_FACTOR * took, took)
        out["read_s"] = round(took, 4)
        return out

    def write_self(self) -> None:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        body = {"cpu_s": (ru.ru_utime + ru.ru_stime), "maxrss_bytes": ru.ru_maxrss * (1024 if sys.platform != "darwin" else 1),
                "elapsed_s": time.monotonic() - self.t0, "samples": self.samples, "bytes_written": self.bytes_written, "cadence_s": self.cadence,
                "base_cadence_s": self.base_cadence, "gaps": len(self.gaps), "errors": dict(self.errors), "error_samples": self.error_samples[:10],
                "collector_version": COLLECTOR_VERSION}
        self.write("sampler", "process", body)
        try:
            write_json_atomic(self.dir / "sampler.json", {"schema": RESOURCES_SCHEMA, "pid": os.getpid(), "stream": self.stream, "last_utc": utc_now_ms(),
                                                          "last_mono": time.monotonic(), "scope": self.scope, "scope_reason": self.scope_reason,
                                                          "gaps": self.gaps[-20:], **body})
        except OSError as e:
            self.err("heartbeat", str(e))

    # ---- loop -----------------------------------------------------------------------------------------------------------------------
    def run(self) -> int:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, self._on_signal)
        self.emitter.emit("sampler_start", pid=os.getpid(), cadence_s=self.cadence, pss_every_s=self.a.pss_every,
                          pss_max_rss_gb=self.a.pss_max_rss_gb, gpu_every_s=self.a.gpu_every,
                          root_pid=self.a.root_pid, pgid=self.a.pgid, scope=self.scope, scope_reason=self.scope_reason, resources=str(self.path.name),
                          collector_version=COLLECTOR_VERSION)
        rc = 0
        reason = "workload_exit"
        try:
            next_t = time.monotonic()
            dead_since: float | None = None
            while not self.stop:
                start = time.monotonic()
                if start - next_t > 2 * self.cadence and self.samples:
                    gap = {"t_utc": utc_now_ms(), "late_s": start - next_t, "cadence_s": self.cadence}
                    self.gaps.append(gap)
                    if len(self.gaps) > 1000:
                        self.gaps = self.gaps[-500:]
                    self.emitter.emit("sampler_gap", **gap)
                alive = self.round()
                if not alive and self.a.root_pid is not None:
                    if dead_since is None:
                        dead_since = time.monotonic()
                    if time.monotonic() - dead_since >= self.a.linger:
                        break
                next_t = max(next_t + self.cadence, time.monotonic())
                slept = next_t - time.monotonic()
                if slept > 0:
                    time.sleep(slept)
            if self.stop:
                reason = self.stop_reason or "signal"
        except Exception as e:  # noqa: BLE001
            reason = f"exception: {e!r}"
            rc = 1
        finally:
            try:
                self.round(final=True)
            except Exception as e:  # noqa: BLE001
                self.err("final", repr(e))
            self.emitter.emit("sampler_end", pid=os.getpid(), reason=reason, samples=self.samples, gaps=len(self.gaps), errors=dict(self.errors),
                              bytes_written=self.bytes_written, elapsed_s=time.monotonic() - self.t0)
        return rc

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self.stop = True
        self.stop_reason = f"signal {signum}"


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--cadence", type=float, default=1.0, help="seconds between cheap rounds (default 1.0)")
    ap.add_argument("--root-pid", type=int, default=None, help="workload root; the tree = its descendants; exit after --linger once it is gone")
    ap.add_argument("--pgid", type=int, default=None, help="alternatively: every process of this process group")
    ap.add_argument("--pss-every", type=float, default=5.0, help="seconds between PSS rounds (smaps_rollup reads; default 5); 0 turns PSS off")
    ap.add_argument("--pss-max-rss-gb", type=float, default=8.0,
                    help="never read smaps_rollup of a process whose RSS (/proc/<pid>/status) is over this many GB (default 8): the read "
                         "holds the process's mmap lock for seconds. Its status RSS is recorded instead")
    ap.add_argument("--gpu-every", type=float, default=5.0)
    ap.add_argument("--gpu-timeout", type=float, default=3.0)
    ap.add_argument("--storage-every", type=float, default=5.0)
    ap.add_argument("--mount", action="append", default=[], help="additional mount paths to statvfs (scratch, evidence)")
    ap.add_argument("--max-pids", type=int, default=512)
    ap.add_argument("--max-outsiders", type=int, default=32, help="largest other processes of the pod cgroup recorded per sample")
    ap.add_argument("--max-bytes", type=int, default=200_000_000, help="output cap before the cadence doubles")
    ap.add_argument("--linger", type=float, default=5.0, help="seconds of sampling after the root pid is gone")
    ap.add_argument("--stream-suffix", default=None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tele sample", description=__doc__.splitlines()[0])
    add_args(ap)
    a = ap.parse_args(argv)
    return Sampler(a).run()


if __name__ == "__main__":
    sys.exit(main())
