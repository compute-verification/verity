"""On one CUDA device: how much of the serve-time commit (`slots_gpu`, this directory's call) hides behind the H200's
served step (`demos/h200`'s `Driver` at the frozen shape, resident weights), as JSON to ``--out``/overlap.json
and a summary to ``--out``/summary.txt.

Today's loop (`e2e_h200.driver.Driver.step_slots`) streams each group (the embedding, a layer, the head) to the commit's
side stream as it is written, at most `e2e_h200.commit.LAG` groups ahead of the side stream's reads, but it serializes
the rest: ``Committer.finish`` waits for the step's root before the step returns, so step t + 1 starts only after step
t's commit, and the driver's timing ticks (`Driver._tick`) synchronize the whole device, the side stream included, around
every micro-batch's attention. Each run here is ``--steps`` lockstep steps at positions 0, 1, ... on one `Driver`:

* ``serve``: the served step alone (its tables written and dropped, nothing committed);
* ``commit``: the commit alone, a step's groups added back to back from stand-in tables (one layer's tensors for every
  layer: the groups are alike), no forward;
* ``serial``: today's loop, unchanged;
* ``low``, ``high``, ``equal``: overlapped. The step returns once its forward ends and its last groups and root are in
  flight on the side stream, which trails the forward by at most ``LAG`` groups across step boundaries too, so the commit
  of step t's last groups runs while step t + 1 computes and no step's tables stay resident beyond those groups; the
  driver's ticks synchronize the forward's stream only. ``low``: the forward on a high-priority stream, the commit on a
  priority-0 side stream; ``high``: the commit's stream high, the forward's 0; ``equal``: both 0. With ``--host-lag`` the
  forward's host waits for those reads too before it allocates the next group's tables (`Pipelined.host_lag`).
* ``low-thread``, ``high-thread``: ``low`` and ``high`` with the commit's adds and closes issued by a host thread of
  their own, so the forward's launches never wait behind the commit's on the host (an ``add`` holds its caller while
  the launch queue is full, and on a pageable upload's synchronization).

A run is ``MODE:LAYOUT[:STEPS[:SMS]]``, LAYOUT ``shared`` (circuits' shared-row Layout JSON, ``--shared``) or ``families``
(today's, `e2e_h200.commit.layout`). SMS ``N`` puts the commit on a green context of N SMs (`slots_gpu.Plan`'s ``sms``),
the forward on every SM as before; ``Nr`` puts the forward on the green context of the other SMs too (``serve`` with
``Nr``: the served step alone there, against the served step on every SM). A step's time is
the median interval between step ends after the first; the hidden fraction is (serve + commit - step) / commit, commit
the commit alone at the run's SMS, the overhead step - serve. Salt keys are the fixture's (`fixture_salt`), so every
overlapped or capped step's root must equal ``serial``'s at that step, and a capped commit alone's the commit alone's on
every SM, or the exit is 1. An nvidia-smi thread samples the device through every run (SM clock, power, temperature,
memory, clock event reasons), each run's statistics skipping its first ``--skip`` seconds; HBM is torch's peak
allocation in each run, and each run records its allocator's counters (`ALLOCATOR`). ``--leaf-minb`` is `slots_gpu.Plan`'s ``leaf_minb`` (a tree whose Plan takes it). The summary
ends with a ``VERDICT`` line a run, each number beside its target and their ratio.

``--fused`` names the fusions the forward serves (`e2e_h200.plan.FUSIONS`), the layout's, and ``--uncommitted`` the
tables the forward writes but the layout doesn't hold (``softmax``: each layer's, for a layout fused past the served
map); each group's tables must be the layout's tensors (`committed`), or the exit is 1 (``--shared``'s before the
weights load).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import itertools
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from verity_config.settings import SHARED

HERE = Path(__file__).resolve().parent
E2E = HERE.parents[2] / "demos" / "h200"
sys.path[:0] = [str(HERE), str(E2E)]
import slots as S
import slots_gpu as G
from e2e_h200 import commit as CM
from e2e_h200.driver import Driver, Params
from e2e_h200.shape import SHAPES
from e2e_h200.store import GeneratorStore
from e2e_h200.weights import Weights

MODES = ("serve", "commit", "serial", "low", "high", "equal", "low-thread", "high-thread")
OVERLAPPED = ("low", "high", "equal", "low-thread", "high-thread")
RUNS = ("serve", "commit:shared", "serial:shared", "low:shared", "high:shared", "equal:shared", "low:families:2")
#: the target: the commit's share of the 3.78 s served step at 4,096 slots, at most 25%
TARGET_S = 0.95
#: an out-of-range priority: CUDA clamps it to the device's highest
HIGH = -(1 << 20)
SMI = ("clocks.sm", "power.draw.average", "power.draw.instant", "temperature.gpu", "memory.used",
       "clocks_event_reasons.active")
SMI_MIN = ("clocks.sm", "power.draw", "temperature.gpu", "memory.used")
#: torch's allocator counters a run records: a retry frees the cache, waiting on every stream a freed allocation was used on
ALLOCATOR = ("num_alloc_retries", "num_sync_all_streams", "num_device_alloc", "num_device_free", "num_ooms")
#: NVML's clock event reasons, by bit
REASONS = {0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
           0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}


class Smi:
    """nvidia-smi's samples of one device every ``ms`` ms on a thread, each stamped with `time.perf_counter`."""

    def __init__(self, device: str, ms: int = 100) -> None:
        self.samples: list[tuple[float, dict[str, Any]]] = []
        self.fields, self.error = SMI, None
        self.base = base = ["nvidia-smi", "-i", device, "--format=csv,noheader,nounits"]
        if subprocess.run(base + ["--query-gpu=" + ",".join(SMI)], check=False, capture_output=True).returncode:
            self.fields = SMI_MIN
        try:
            self.proc = subprocess.Popen(base + ["--query-gpu=" + ",".join(self.fields), "-lms", str(ms)],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError as e:
            self.proc, self.error = None, str(e)
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            vals = [v.strip() for v in line.split(",")]
            if len(vals) != len(self.fields):
                continue
            rec: dict[str, Any] = {}
            for f, v in zip(self.fields, vals, strict=True):
                if f.startswith("clocks_event"):
                    rec[f] = int(v, 16) if v.startswith("0x") else None
                else:
                    try:
                        rec[f] = float(v)
                    except ValueError:
                        rec[f] = None
            self.samples.append((time.perf_counter(), rec))

    def stats(self, t0: float, t1: float) -> dict[str, Any]:
        rs = [r for t, r in self.samples if t0 <= t <= t1]
        out: dict[str, Any] = {"samples": len(rs)}

        def col(f: str) -> list[float]:
            return [r[f] for r in rs if r.get(f) is not None]
        for f, name in (("clocks.sm", "sm_mhz"), ("power.draw.average", "power_w"), ("power.draw", "power_w"),
                        ("power.draw.instant", "power_instant_w"), ("temperature.gpu", "temp_c")):
            v = col(f)
            if v and name not in out:
                out[name] = round(statistics.median(v), 1)
                out[name + "_min" if name == "sm_mhz" else name + "_max"] = min(v) if name == "sm_mhz" else max(v)
        mem = col("memory.used")
        if mem:
            out["memory_used_mib_max"] = max(mem)
        bits = 0
        for r in rs:
            bits |= r.get("clocks_event_reasons.active") or 0
        if "clocks_event_reasons.active" in self.fields:
            out["reasons"] = [n for b, n in REASONS.items() if bits & b]
        return out

    def limits(self) -> dict[str, float]:
        """The device's power limit and highest SM clock: the power's and the clocks' targets."""
        try:
            p = subprocess.run(self.base + ["--query-gpu=power.limit,clocks.max.sm"], check=False, capture_output=True,
                               text=True)
            w, mhz = (float(v) for v in p.stdout.strip().split(","))
        except (OSError, ValueError):
            return {}
        return {"power_limit_w": w, "sm_mhz_max": mhz}

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()


def describe(layout: S.Layout) -> dict[str, Any]:
    return {"name": layout.name, "digest": S.layout_digest(layout).hex(), "groups": len(layout.groups),
            "distinct_groups": len({id(g) for g in layout.groups}), "units_a_slot": layout.unit_count,
            "rows_a_slot": layout.row_count, "bytes_a_slot": layout.slot_bytes}


class OverlapDriver(Driver):
    """`Driver` whose timing ticks synchronize the forward's stream only when ``device_sync`` is off."""
    device_sync = True

    def _sync(self) -> None:
        if self.device.type != "cuda":
            return
        if self.device_sync:
            torch.cuda.synchronize(self.device)
        else:
            torch.cuda.current_stream(self.device).synchronize()


class Null:
    """Serving alone: the step writes and drops its tables, and nothing is committed."""
    streams = True

    def __init__(self, fused: frozenset[str] = frozenset()) -> None:
        self.fused = fused

    def begin(self, st: CM.StepTables, salt_key: bytes | None = None, nonce0: int = 0) -> None:
        pass

    def group(self, st: CM.StepTables, i: int) -> None:
        pass

    def finish(self, st: CM.StepTables) -> CM.Roots:
        return CM.Roots(bytes(64))


class LayoutCommitter(CM.Committer):
    """`e2e_h200.commit.Committer`'s slots backend on a given plan (its layout), unchanged otherwise: today's loop."""

    def __init__(self, plan: G.Plan, context: bytes, key: bytes, shape: Any, salt: Any, fused: frozenset[str] = frozenset(),
                 uncommitted: frozenset[str] = frozenset()) -> None:
        self.backend, self.context, self.key, self.salt, self.fused = "slots", context, key, salt, fused
        self.streams = True
        self.S, self.G, self.E = S, G, G.E
        self.groups = committed(shape, fused, uncommitted)
        self.layout, self.bases, self.plan = plan.layout, {}, plan


class Pipelined(LayoutCommitter):
    """Overlapped: ``finish`` returns with the step's close in flight, and the side stream's reads stay at most ``LAG``
    groups behind the forward across steps. A step's root is taken (``Pending.result``) once ``depth`` later steps have
    finished, or by ``drain``."""

    #: the forward's host waits for the reads too, before it allocates the next group's tables: else at the HBM cap the
    #: allocator frees its cache for them, waiting on every stream (the side stream's pending reads included)
    host_lag = False

    def __init__(self, *args: Any, depth: int = 1, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._read, self.depth = [], depth
        self.inflight: list[tuple[int, Any]] = []
        self.roots: dict[int, bytes] = {}

    def _wait(self, read: Any) -> None:
        super()._wait(read)
        if self.host_lag:
            read.synchronize()

    def begin(self, st: CM.StepTables, salt_key: bytes | None = None, nonce0: int = 0) -> None:
        st.salt_key = salt_key or (self.salt(st.step) if self.salt else S.fresh_salt_key())
        self._marks, self._start = [], None
        st.pending = G.StepCommitter(self.plan, st.step, st.batch, st.salt_key, marks=self._marks)
        st.pending.salts = G.E.SaltKey(st.salt_key, nonce0)

    def finish(self, st: CM.StepTables) -> CM.Roots:
        self.inflight.append((st.step, st.pending.finish()))
        st.pending = None
        while len(self.inflight) > self.depth:
            self._take()
        return CM.Roots(bytes(64))

    def _take(self) -> None:
        step, pending = self.inflight.pop(0)
        self.roots[step] = pending.result().root

    def drain(self) -> None:
        while self.inflight:
            self._take()
        self._read = []


class Threaded(Pipelined):
    """`Pipelined` with every ``add`` and ``finish`` issued in order by a host thread of its own. As a group is handed
    over, the forward's stream records an event the side stream waits on; ``LAG`` groups later the forward's stream waits
    on that group's reads, once the thread has issued them."""

    def __init__(self, *args: Any, depth: int = 1, **kw: Any) -> None:
        super().__init__(*args, depth=depth, **kw)
        self._q: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._serve, daemon=True)
        self._worker.start()

    def _serve(self) -> None:
        with torch.cuda.stream(self.plan.stream):
            while (item := self._q.get()) is not None:
                fn, fut = item
                try:
                    fut.set_result(fn())
                except BaseException as e:  # noqa: BLE001  (raised where the future is read)
                    fut.set_exception(e)
                del item, fn, fut

    def _submit(self, fn: Any) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put((fn, fut))
        return fut

    def group(self, st: CM.StepTables, i: int) -> None:
        tensors = []
        for t in self.groups[i]:
            codes, word = st.tables[t.op]
            tensors += [codes, word] if t.quant else [codes]
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.plan.device))
        pending, stream = st.pending, self.plan.stream

        def add() -> Any:
            stream.wait_event(ev)
            return pending.add(tensors)
        self._read.append(self._submit(add))
        if len(self._read) > CM.LAG:
            done = self._read.pop(0).result()
            if done is not None:
                self._wait(done)

    def finish(self, st: CM.StepTables) -> CM.Roots:
        self.inflight.append((st.step, self._submit(st.pending.finish)))
        st.pending = None
        while len(self.inflight) > self.depth:
            self._take()
        return CM.Roots(bytes(64))

    def _take(self) -> None:
        step, fut = self.inflight.pop(0)
        self.roots[step] = fut.result().result().root

    def drain(self) -> None:
        for fut in self._read:
            fut.result()
        super().drain()

    def close(self) -> None:
        self._q.put(None)
        self._worker.join()


def prompts_for(run_id: str, batch: int, p: int) -> np.ndarray:
    """`demos/h200/run.py`'s prompts."""
    h = hashlib.sha512(b"e2e-h200/prompts-fixture\0" + run_id.encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
    return rng.integers(32, 127, size=(batch, p), dtype=np.int64)


def committed(shape: Any, fused: frozenset[str], uncommitted: frozenset[str]) -> list[list[Any]]:
    """Each group's tables the commit adds: the served map's at the fusions ``fused`` (`e2e_h200.commit.tables`), less the
    ops ``uncommitted`` names (``softmax``: every ``l<i>.softmax``)."""
    return [[t for t in g if t.op.split(".")[-1] not in uncommitted] for g in CM.tables(shape, fused)]


def check_sources(groups: list[list[Any]], layout: S.Layout) -> None:
    """Refuses a layout whose groups' tensors aren't ``groups``' tables (each one's codes, then a quantizer's word)."""
    if len(groups) != len(layout.groups):
        raise SystemExit(f"the layout has {len(layout.groups)} groups and the served map {len(groups)}")
    for i, (g, lg) in enumerate(zip(groups, layout.groups, strict=True)):
        have = [n for t in g for n in ((t.nbytes, 4) if t.quant else (t.nbytes,))]
        if have != list(lg.sources):
            raise SystemExit(f"group {i}: the layout's tensors are {list(lg.sources)} bytes a slot and the served tables' "
                             f"{have}: --fused names the layout's fusions, --uncommitted the tables it doesn't hold")


def stand_in(shape: Any, batch: int, dev: torch.device, seed: int,
             groups: list[list[Any]] | None = None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Every table of a step (of ``groups``, by default the unfused map's), uniform bytes: the embedding's and the head's
    own, one layer's shared by every layer."""
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    bufs: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
    out = {}
    for i, group in enumerate(CM.tables(shape) if groups is None else groups):
        layer = 1 <= i <= shape.layers
        for t in group:
            k = ("layer", t.op.split(".", 1)[1]) if layer else (i, t.op)
            if k not in bufs:
                codes = torch.randint(-128, 128, (batch, t.nbytes), dtype=torch.int8, device=dev, generator=g)
                word = torch.rand(batch, dtype=torch.float32, device=dev, generator=g) if t.quant else \
                    torch.zeros(batch, dtype=torch.float32, device=dev)
                bufs[k] = (codes, word)
            out[t.op] = bufs[k]
    return out


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 4) if xs else None


def run_one(name: str, mode: str, steps: int, d: OverlapDriver, plan: G.Plan | None, a: argparse.Namespace,
            ctx: bytes, key: bytes, prompts: np.ndarray, smi: Smi, sms: int = 0, rest: bool = False,
            rest_of: G.SmPartition | None = None) -> dict[str, Any]:
    """``rest_of``: the partition whose other SMs a served step alone runs on."""
    dev, shape = d.device, d.shape
    salt = CM.fixture_salt(a.run_id)
    fwd = torch.cuda.current_stream(dev)
    part = plan.partition if plan is not None else rest_of
    every = torch.cuda.get_device_properties(dev).multi_processor_count
    if plan is not None:
        torch.cuda.synchronize(dev)
        prio = HIGH if mode.startswith("high") else 0
        plan.stream = part.stream(prio) if part else torch.cuda.Stream(dev, priority=prio)
        if not rest and mode.startswith("low"):
            fwd = torch.cuda.Stream(dev, priority=HIGH)
    if rest:
        fwd = part.rest_stream(HIGH if mode.startswith("low") else 0)
    fu = {"fused": a.fused, "uncommitted": a.uncommitted}
    committer: Any = Null(a.fused) if mode == "serve" else \
        Threaded(plan, ctx, key, shape, salt, **fu) if mode.endswith("-thread") else \
        Pipelined(plan, ctx, key, shape, salt, **fu) if mode in OVERLAPPED else \
        LayoutCommitter(plan, ctx, key, shape, salt, **fu)
    host_lag = isinstance(committer, Pipelined) and a.host_lag
    committer.host_lag = host_lag
    d.committer, d.results, d.device_sync = committer, [], mode not in OVERLAPPED
    tabs = stand_in(shape, a.batch, dev, a.table_seed, committer.groups) if mode == "commit" else None
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    m0, c0 = torch.cuda.memory_allocated(dev), torch.cuda.memory_stats(dev)
    ends, walls, secs, roots = [], [], [], []
    t0 = time.perf_counter()
    with torch.cuda.stream(fwd):
        toks = prompts[:, 0]
        for t in range(steps):
            s0 = time.perf_counter()
            if mode == "commit":
                st = CM.StepTables(t, 0, a.batch)
                st.tables = dict(tabs)
                committer.begin(st)
                for i in range(len(committer.groups)):
                    committer.group(st, i)
                r = committer.finish(st)
                secs.append({"commit_gpu": r.seconds})
                roots.append(r.root.hex())
            else:
                res = d.step(t, toks)
                toks = prompts[:, t + 1] if t + 1 < prompts.shape[1] else res.tokens_out
                secs.append(res.seconds)
                roots.append(res.root.hex())
            ends.append(time.perf_counter())
            walls.append(ends[-1] - s0)
        t_steps = time.perf_counter()
        if isinstance(committer, Pipelined):
            try:
                committer.drain()
            finally:
                if isinstance(committer, Threaded):
                    committer.close()
            roots = [committer.roots[t].hex() for t in range(steps)]
        if plan is not None:
            plan.stream.synchronize()
        fwd.synchronize()
        torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    intervals = [ends[0] - t0] + [b - e for e, b in itertools.pairwise(ends)]
    keys = sorted({k for s in secs for k in s})
    r: dict[str, Any] = {
        "run": name, "mode": mode, "layout": plan.layout.name if plan is not None else None, "steps": steps,
        "step_s": _median(intervals[1:] or intervals), "intervals_s": [round(x, 4) for x in intervals],
        "walls_s": [round(x, 4) for x in walls], "drain_s": round(t1 - t_steps, 4),
        "total_s": round(t1 - t0, 4), "per_step_with_drain_s": round((t1 - t0) / steps, 4),
        "seconds": {k: _median([s[k] for s in secs[1:] or secs if s.get(k) is not None]) for k in keys},
        "roots": roots, "priorities": {"forward": "high" if mode.startswith("low") else "0",
                                       "commit": "high" if mode.startswith("high") else "0"} if plan is not None else None,
        "leaf_minb": getattr(plan, "leaf_minb", None) if plan is not None else None,
        "sms": sms, "commit_sms": None if plan is None else part.sms if part else every,
        "forward_sms": None if mode == "commit" else part.rest if rest else every, "forward_on_rest": rest,
        "host_lag": host_lag,
        "hbm": {"before_gb": round(m0 / 1e9, 3), "peak_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 3),
                "peak_over_before_gb": round((torch.cuda.max_memory_allocated(dev) - m0) / 1e9, 3)},
        "allocator": {k: torch.cuda.memory_stats(dev).get(k, 0) - c0.get(k, 0) for k in ALLOCATOR},
        "telemetry": smi.stats(t0 + a.skip, t1),
    }
    del tabs, committer
    d.committer = Null(a.fused)
    torch.cuda.empty_cache()
    return r


def derive(runs: list[dict], equal: dict[str, bool]) -> dict[str, Any]:
    """Per layout: the served step alone, the commit alone at each SMS, and today's loop and each overlapped run, with its
    hidden fraction (of the commit alone at its SMS, and of the commit alone on every SM) and its overhead a step against
    `TARGET_S`; ``best``, the run of least overhead whose roots are its reference's."""
    serve = next((r["step_s"] for r in runs if r["mode"] == "serve" and not r["sms"]), None)
    out: dict[str, Any] = {"serve_s": serve, "target_s": TARGET_S, "layouts": {},
                           "serve_s_on_rest": {str(r["forward_sms"]): r["step_s"] for r in runs if r["mode"] == "serve" and r["sms"]}}
    for lay in sorted({r["layout"] for r in runs if r["layout"]}):
        rs = [r for r in runs if r["layout"] == lay]
        alone = {r["sms"]: r["step_s"] for r in rs if r["mode"] == "commit"}
        v: dict[str, Any] = {"commit_s": alone.get(0), "commit_s_at_sms": {str(k): s for k, s in sorted(alone.items())},
                             "runs": {}}
        for r in rs:
            if r["mode"] == "commit" or serve is None:
                continue
            step, commit, full = r["step_s"], alone.get(r["sms"]), alone.get(0)
            e = {"commit_sms": r["commit_sms"], "forward_sms": r["forward_sms"], "step_s": step,
                 "overhead_s": round(step - serve, 4), "overhead_share": round((step - serve) / serve, 4),
                 "ratio_to_target": round((step - serve) / TARGET_S, 3), "meets_target": step - serve <= TARGET_S,
                 "commit_s": commit}
            if commit:
                e["hidden_s"] = round(serve + commit - step, 4)
                e["hidden_fraction"] = round((serve + commit - step) / commit, 4)
            if full:
                e["hidden_fraction_of_full_commit"] = round((serve + full - step) / full, 4)
            v["runs"][r["run"]] = e
        ok = [k for k in v["runs"] if equal.get(k) or k == _reference_run(rs, "serial")]
        if ok:
            v["best"] = min(ok, key=lambda k: v["runs"][k]["overhead_s"])
        out["layouts"][lay] = v
    return out


def _reference_run(runs: list[dict], mode: str) -> str | None:
    return next((x["run"] for x in runs if x["mode"] == mode and not x["sms"]), None)


def _power(t: dict, limits: dict) -> str:
    lim, top = limits.get("power_limit_w"), limits.get("sm_mhz_max")
    p, pi = t.get("power_w"), t.get("power_instant_w_max")
    return (f"power {p} W median against the {lim} W limit (ratio {_ratio(p, lim)}), instant max {pi} W (ratio "
            f"{_ratio(pi, lim)}); SM {t.get('sm_mhz')} MHz median, min {t.get('sm_mhz_min')}, against {top} MHz max (ratio "
            f"{_ratio(t.get('sm_mhz_min'), top)}); clock event reasons {t.get('reasons')}")


def _ratio(x: float | None, y: float | None) -> float | None:
    return round(x / y, 3) if x is not None and y else None


def verdicts(doc: dict) -> list[str]:
    """A ``VERDICT`` line a run, and the best."""
    dv, limits, out = doc["derived"], doc["env"].get("limits", {}), []
    every = doc["env"].get("sms")
    for r in doc["runs"]:
        t, name = r["telemetry"], r["run"]
        lay = dv["layouts"].get(r["layout"] or "", {})
        n = _checked(doc["runs"], r)
        eq = doc["roots_equal"].get(name)
        if r["mode"] == "serve" and r["sms"]:
            base, f = dv.get("serve_s"), r["forward_sms"]
            scaled = "" if not base else (f", {_ratio(r['step_s'], base)}x the served step on every SM ({base:.3f} s; "
                                          f"{base * every / f:.3f} s if it scaled as its SMs)")
            out.append(f"VERDICT {name}: served step alone on the other {f} of {every} SMs {r['step_s']:.3f} s{scaled}; "
                       f"{_power(t, limits)}")
        elif r["mode"] == "serve":
            out.append(f"VERDICT {name}: served step alone {r['step_s']:.3f} s, the overheads' base; {_power(t, limits)}")
        elif r["mode"] == "commit":
            full = lay.get("commit_s")
            line = f"VERDICT {name}: commit alone on {r['commit_sms']} of {every} SMs {r['step_s']:.3f} s"
            if r["sms"] and full:
                line += (f", {_ratio(r['step_s'], full)}x the commit alone on every SM ({full:.3f} s; "
                         f"{full * every / r['commit_sms']:.3f} s if it scaled as its SMs), roots equal the commit alone's on "
                         f"every SM at its {n} steps: {eq}")
            out.append(f"{line}; {_power(t, limits)}")
        else:
            e = lay.get("runs", {}).get(name)
            if e is None:
                continue
            ref = "" if name == _reference_run(doc["runs"], "serial") else \
                f", roots equal today's loop's at its {n} steps: {eq}"
            hid = "" if e.get("hidden_s") is None else \
                (f", commit alone at its SMs {e['commit_s']:.3f} s, hidden {e['hidden_s']:.3f} s "
                 f"({100 * e['hidden_fraction']:.1f}% of it; {100 * e.get('hidden_fraction_of_full_commit', 0):.1f}% of the "
                 f"commit alone on every SM)")
            lag = ", its host waiting on the reads" if r.get("host_lag") else ""
            out.append(f"VERDICT {name}: commit on {e['commit_sms']} SMs, forward on {e['forward_sms']}{lag}: step "
                       f"{e['step_s']:.3f} s, overhead {e['overhead_s']:+.3f} s against <= {TARGET_S} s (ratio "
                       f"{e['ratio_to_target']:.3f}, target met {e['meets_target']}){hid}{ref}; {_power(t, limits)}; HBM "
                       f"peak {r['hbm']['peak_gb']} GB")
    for lay, v in dv["layouts"].items():
        if v.get("best"):
            e = v["runs"][v["best"]]
            out.append(f"VERDICT best on {lay}: {v['best']}, overhead {e['overhead_s']:+.3f} s against <= {TARGET_S} s "
                       f"(ratio {e['ratio_to_target']:.3f}, target met {e['meets_target']})")
    return out


def summary(doc: dict) -> str:
    lines = [(f"{doc['env'].get('name')} {doc['env'].get('uuid', '')} batch {doc['batch']} micro-batch {doc['micro_batch']}; "
              f"limits {doc['env'].get('limits')}; fused {doc.get('fused')}, uncommitted {doc.get('uncommitted')}")]
    for lay in doc["layouts"].values():
        lines.append(f"layout {lay['name']} {lay['digest'][:16]}: {lay['units_a_slot']:,} units, {lay['rows_a_slot']:,} rows, "
                     f"{lay['bytes_a_slot']:,} B a slot")
    for k, p in doc.get("plans", {}).items():
        lines.append(f"plan {k}: {p}")
    for r in doc["runs"]:
        t, h = r["telemetry"], r["hbm"]
        lines.append(f"{r['run']:20s} step {r['step_s']:8.3f} s (intervals {r['intervals_s']}, drain {r['drain_s']} s) "
                     f"{json.dumps(r['seconds'])}; SM {t.get('sm_mhz')} MHz (min {t.get('sm_mhz_min')}), "
                     f"{t.get('power_w')} W (instant max {t.get('power_instant_w_max')}), {t.get('temp_c')} C, "
                     f"{t.get('reasons')}; HBM peak {h['peak_gb']} GB (+{h['peak_over_before_gb']} over the run's start); "
                     f"allocator {r.get('allocator')}")
    for r in doc.get("errors", []):
        lines.append(f"{r['run']:20s} not run: {r['error']}")
    dv = doc["derived"]
    lines.append(f"served step alone {dv['serve_s']} s (on the other SMs of a partition, by their count: "
                 f"{dv.get('serve_s_on_rest')}); target: overhead <= {dv['target_s']} s a step")
    for lay, v in dv["layouts"].items():
        lines.append(f"{lay}: commit alone {v.get('commit_s')} s on every SM; at each SMS {v.get('commit_s_at_sms')}")
        for name, e in v["runs"].items():
            lines.append(f"  {name:20s} step {e['step_s']:.3f} s, overhead {e['overhead_s']:+.3f} s "
                         f"({100 * e['overhead_share']:.1f}% of the step), hidden {e.get('hidden_s')} s "
                         f"({None if 'hidden_fraction' not in e else round(100 * e['hidden_fraction'], 1)}% of the commit alone "
                         f"at its SMs), target met {e['meets_target']}")
    lines.append(f"roots equal to their reference's: {doc['roots_equal']}")
    lines += verdicts(doc)
    return "\n".join(lines) + "\n"


def _reference(runs: list[dict], r: dict) -> dict | None:
    """``r``'s reference: the commit alone on every SM for a capped commit alone, else today's loop on every SM; none for
    those two themselves and the served step."""
    if r["mode"] == "serve" or (r["mode"] in ("serial", "commit") and not r["sms"]):
        return None
    mode = "commit" if r["mode"] == "commit" else "serial"
    return next((x for x in runs if x["mode"] == mode and x["layout"] == r["layout"] and not x["sms"]), None)


def _checked(runs: list[dict], r: dict) -> int:
    ref = _reference(runs, r)
    return 0 if ref is None else min(len(ref["roots"]), len(r["roots"]))


def roots_equal(runs: list[dict]) -> dict[str, bool]:
    """Each run's roots against its reference's (`_reference`) at every step both ran."""
    out = {}
    for r in runs:
        ref, n = _reference(runs, r), _checked(runs, r)
        if ref is not None:
            out[r["run"]] = n > 0 and ref["roots"][:n] == r["roots"][:n]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shared", type=Path, help="circuits' shared-row Layout JSON")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), help="MODE:LAYOUT[:STEPS], MODE one of " + ", ".join(MODES))
    ap.add_argument("--shape", default="h200")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=1, help="served steps before the runs")
    ap.add_argument("--prompt-bytes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", default="demo", help="the weights' seed: test, demo or a text seed")
    ap.add_argument("--table-seed", type=int, default=20261026, help="the commit-alone run's stand-in tables")
    ap.add_argument("--run-id", default="e2e-h200-overlap")
    ap.add_argument("--skip", type=float, default=1.0, help="seconds of each run the telemetry statistics skip")
    ap.add_argument("--leaf-minb", type=int, default=0, help="the plan's leaf kernel CTAs an SM (0: the plan's default)")
    ap.add_argument("--host-lag", action="store_true",
                    help="overlapped runs: the forward's host waits for the side stream's reads LAG groups back (Pipelined.host_lag)")
    ap.add_argument("--fused", nargs="*", default=[],
                    help="the fusions the forward serves (e2e_h200.plan.FUSIONS), the layout's")
    ap.add_argument("--uncommitted", nargs="*", default=[],
                    help="tables the forward writes and the layout doesn't hold (softmax: every layer's)")
    a = ap.parse_args(argv)
    a.fused, a.uncommitted = frozenset(a.fused), frozenset(a.uncommitted)
    a.out.mkdir(parents=True, exist_ok=True)
    from e2e_h200 import served_weights as SW
    dev = torch.device("cuda")
    shape = SHAPES[a.shape]
    if a.shared:
        check_sources(committed(shape, a.fused, a.uncommitted), S.load(a.shared))
    smi = Smi(SHARED.GPU_LEASE_UUID or "0")
    props = torch.cuda.get_device_properties(dev)
    doc: dict[str, Any] = {"env": {"name": props.name, "uuid": SHARED.GPU_LEASE_UUID, "sms": props.multi_processor_count,
                                   "hbm_gb": round(props.total_memory / 1e9, 2), "torch": torch.__version__,
                                   "cuda": torch.version.cuda, "smi_fields": list(smi.fields), "limits": smi.limits()},
                           "shape": shape.name, "batch": a.batch, "micro_batch": a.micro_batch, "fused": sorted(a.fused),
                           "uncommitted": sorted(a.uncommitted), "layouts": {}, "runs": []}

    def write() -> None:
        doc["roots_equal"] = roots_equal(doc["runs"])
        doc["derived"] = derive(doc["runs"], doc["roots_equal"])
        (a.out / "overlap.json").write_text(json.dumps(doc, indent=1) + "\n")
        (a.out / "summary.txt").write_text(summary(doc))

    seed = {"test": SW.TEST_SEED, "demo": SW.DEMO_SEED}.get(a.seed, a.seed.encode())
    t0 = time.perf_counter()
    params = Params(shape, GeneratorStore(Weights(shape, seed), a.workers), dev, True)
    doc["params"] = params.describe() | {"seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps({"params": doc["params"]}), flush=True)
    ctx, key = CM.fixture_secrets(a.run_id)
    prompts = prompts_for(a.run_id, a.batch, a.prompt_bytes)
    d = OverlapDriver(shape, params, a.batch, Null(a.fused), "cuda", keep=False, micro_batch=a.micro_batch)
    for t in range(a.warm):
        d.step(t, prompts[:, t])
    d.results = []
    plans: dict[tuple[str, int], G.Plan] = {}
    current: tuple[str, S.Layout] | None = None

    def drop(capped_only: bool) -> None:
        """Frees the plans (or the capped ones), and their partitions' streams and green contexts."""
        torch.cuda.synchronize(dev)
        parts = []
        for k in [k for k, p in plans.items() if not capped_only or p.partition is not None]:
            parts.append(plans.pop(k).partition)
        gc.collect()
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        for part in parts:
            if part is not None:
                part.close()

    def plan_for(lay: str, sms: int) -> G.Plan:
        nonlocal current
        if current is None or current[0] != lay:
            drop(False)
            layout = S.load(a.shared) if lay == "shared" else CM.layout(shape, a.fused) if lay == "families" else None
            if layout is None:
                raise SystemExit(f"--runs: {lay!r} is shared or families")
            check_sources(committed(shape, a.fused, a.uncommitted), layout)
            current = (lay, layout)
            doc["layouts"][lay] = describe(layout)
            print(json.dumps({"layout": doc["layouts"][lay]}), flush=True)
        if (lay, sms) not in plans:
            if sms:
                drop(True)
            torch.cuda.synchronize(dev)
            m0, p0 = torch.cuda.memory_allocated(dev), time.perf_counter()
            kw = ({"leaf_minb": a.leaf_minb} if a.leaf_minb else {}) | ({"sms": sms} if sms else {})
            plan = plans[(lay, sms)] = G.Plan(ctx, current[1], device=dev, **kw)
            p = plan.partition
            doc.setdefault("plans", {})[f"{lay}:{sms}"] = {
                "asked_sms": sms, "commit_sms": p.sms if p else None, "rest_sms": p.rest if p else None,
                "plan_s": round(time.perf_counter() - p0, 1),
                "plan_gb": round((torch.cuda.memory_allocated(dev) - m0) / 1e9, 3)}
            print(json.dumps({"plan": {f"{lay}:{sms}": doc["plans"][f"{lay}:{sms}"]}}), flush=True)
        return plans[(lay, sms)]

    for spec in a.runs:
        plan = None
        mode, lay, *n = spec.split(":") + [""] * (spec.count(":") == 0)
        if mode not in MODES:
            raise SystemExit(f"--runs: {spec!r}'s mode is one of {MODES}")
        steps = int(n[0]) if n and n[0] else a.steps
        cap = n[1] if len(n) > 1 else ""
        rest, sms = cap.endswith("r"), int(cap.removesuffix("r") or 0)
        if (sms or rest) and (not sms or (mode == "serve" and not rest) or (rest and mode not in OVERLAPPED + ("serve",))):
            raise SystemExit(f"--runs: {spec!r}: SMS is N, or Nr for an overlapped mode or serve")
        alone = None
        try:
            plan = plan_for(lay, sms) if mode != "serve" else None
            if mode == "serve" and rest:
                drop(True)
                alone = G.SmPartition(sms, dev)
            r = run_one(spec, mode, steps, d, plan, a, ctx, key, prompts, smi, sms, rest, alone)
        except Exception as e:  # noqa: BLE001  (recorded; the next run starts clean)
            r = {"run": spec, "mode": mode, "layout": lay, "error": f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"}
            d.committer, d.results = Null(a.fused), []
            torch.cuda.empty_cache()
        if alone is not None:
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
            alone.close()
        if "error" in r:
            doc.setdefault("errors", []).append(r)
        else:
            doc["runs"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("roots",)}), flush=True)
        write()
    smi.stop()
    write()
    print(summary(doc), flush=True)
    return 0 if doc["runs"] and all(doc["roots_equal"].values()) and not doc.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
