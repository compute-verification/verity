"""On one CUDA device: how much of the serve-time commit (`slots_gpu`, this directory's call) hides behind ALLP-H200's
served step (`benchmarks/e2e_h200`'s `Driver` at the frozen shape, resident weights), as JSON to ``--out``/overlap.json
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
  priority-0 side stream; ``high``: the commit's stream high, the forward's 0; ``equal``: both 0.
* ``low-thread``, ``high-thread``: ``low`` and ``high`` with the commit's adds and closes issued by a host thread of
  their own, so the forward's launches never wait behind the commit's on the host (an ``add`` holds its caller while
  the launch queue is full, and on a pageable upload's synchronization).

A run is ``MODE:LAYOUT[:STEPS]``, LAYOUT ``shared`` (circuits' shared-row Layout JSON, ``--shared``) or ``families``
(today's, `e2e_h200.commit.layout`). A step's time is the median interval between step ends after the first; the hidden
fraction is (serve + commit - step) / commit, the overhead step - serve. Salt keys are the fixture's (`fixture_salt`), so
every overlapped step's root must equal ``serial``'s at that step or the exit is 1. An nvidia-smi thread samples the
device through every run (SM clock, power, temperature, memory, clock event reasons), each run's statistics skipping its
first ``--skip`` seconds; HBM is torch's peak allocation in each run. ``--leaf-minb`` is `slots_gpu.Plan`'s
``leaf_minb`` (a tree whose Plan takes it).
"""
from __future__ import annotations

import argparse
import concurrent.futures
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

HERE = Path(__file__).resolve().parent
E2E = HERE.parents[1] / "e2e_h200"
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
#: CMP-149: the commit's share of the 3.78 s served step at 4,096 slots, at most 25%
TARGET_S = 0.95
#: an out-of-range priority: CUDA clamps it to the device's highest
HIGH = -(1 << 20)
SMI = ("clocks.sm", "power.draw.average", "power.draw.instant", "temperature.gpu", "memory.used",
       "clocks_event_reasons.active")
SMI_MIN = ("clocks.sm", "power.draw", "temperature.gpu", "memory.used")
#: NVML's clock event reasons, by bit
REASONS = {0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
           0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}


class Smi:
    """nvidia-smi's samples of one device every ``ms`` ms on a thread, each stamped with `time.perf_counter`."""

    def __init__(self, device: str, ms: int = 100) -> None:
        self.samples: list[tuple[float, dict[str, Any]]] = []
        self.fields, self.error = SMI, None
        base = ["nvidia-smi", "-i", device, "--format=csv,noheader,nounits"]
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

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()


def load_layout(path: Path) -> S.Layout:
    """circuits' Layout JSON (`slots._layout_json`), equal groups shared (CIR-113's loader)."""
    doc, memo = json.loads(path.read_text()), []

    def group(g: dict) -> S.Group:
        for raw, built in memo:
            if raw == g:
                return built
        built = S.Group(tuple(g["sources"]),
                        tuple(tuple(S.Row(tuple(S.Part(*p) for p in r)) for r in u) for u in g["units"]))
        memo.append((g, built))
        return built
    return S.Layout(doc["name"], tuple(group(g) for g in doc["groups"]))


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

    def begin(self, st: CM.StepTables, salt_key: bytes | None = None, nonce0: int = 0) -> None:
        pass

    def group(self, st: CM.StepTables, i: int) -> None:
        pass

    def finish(self, st: CM.StepTables) -> CM.Roots:
        return CM.Roots(bytes(64))


class LayoutCommitter(CM.Committer):
    """`e2e_h200.commit.Committer`'s slots backend on a given plan (its layout), unchanged otherwise: today's loop."""

    def __init__(self, plan: G.Plan, context: bytes, key: bytes, shape: Any, salt: Any) -> None:
        self.backend, self.context, self.key, self.salt = "slots", context, key, salt
        self.streams = True
        self.S, self.G, self.E = S, G, G.E
        self.groups = CM.tables(shape)
        self.layout, self.bases, self.plan = plan.layout, {}, plan


class Pipelined(LayoutCommitter):
    """Overlapped: ``finish`` returns with the step's close in flight, and the side stream's reads stay at most ``LAG``
    groups behind the forward across steps. A step's root is taken (``Pending.result``) once ``depth`` later steps have
    finished, or by ``drain``."""

    def __init__(self, *args: Any, depth: int = 1) -> None:
        super().__init__(*args)
        self._read, self.depth = [], depth
        self.inflight: list[tuple[int, Any]] = []
        self.roots: dict[int, bytes] = {}

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

    def __init__(self, *args: Any, depth: int = 1) -> None:
        super().__init__(*args, depth=depth)
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
                torch.cuda.current_stream(self.plan.device).wait_event(done)

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
    """`benchmarks/e2e_h200/run.py`'s prompts."""
    h = hashlib.sha512(b"e2e-h200/prompts-fixture\0" + run_id.encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
    return rng.integers(32, 127, size=(batch, p), dtype=np.int64)


def stand_in(shape: Any, batch: int, dev: torch.device, seed: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Every table of a step, uniform bytes: the embedding's and the head's own, one layer's shared by every layer."""
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    bufs: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
    out = {}
    for i, group in enumerate(CM.tables(shape)):
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
            ctx: bytes, key: bytes, prompts: np.ndarray, smi: Smi) -> dict[str, Any]:
    dev, shape = d.device, d.shape
    salt = CM.fixture_salt(a.run_id)
    fwd = torch.cuda.current_stream(dev)
    if plan is not None:
        torch.cuda.synchronize(dev)
        plan.stream = torch.cuda.Stream(dev, priority=HIGH if mode.startswith("high") else 0)
        if mode.startswith("low"):
            fwd = torch.cuda.Stream(dev, priority=HIGH)
    committer: Any = Null() if mode == "serve" else \
        Threaded(plan, ctx, key, shape, salt) if mode.endswith("-thread") else \
        Pipelined(plan, ctx, key, shape, salt) if mode in OVERLAPPED else LayoutCommitter(plan, ctx, key, shape, salt)
    d.committer, d.results, d.device_sync = committer, [], mode not in OVERLAPPED
    tabs = stand_in(shape, a.batch, dev, a.table_seed) if mode == "commit" else None
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    m0 = torch.cuda.memory_allocated(dev)
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
        "hbm": {"before_gb": round(m0 / 1e9, 3), "peak_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 3),
                "peak_over_before_gb": round((torch.cuda.max_memory_allocated(dev) - m0) / 1e9, 3)},
        "telemetry": smi.stats(t0 + a.skip, t1),
    }
    del tabs, committer
    d.committer = Null()
    torch.cuda.empty_cache()
    return r


def derive(runs: list[dict]) -> dict[str, Any]:
    """Per layout: the served step alone, the commit alone, today's loop and each overlapped run, with the hidden fraction
    and the overhead a step against `TARGET_S`."""
    serve = next((r["step_s"] for r in runs if r["mode"] == "serve"), None)
    out: dict[str, Any] = {"serve_s": serve, "target_s": TARGET_S, "layouts": {}}
    for lay in sorted({r["layout"] for r in runs if r["layout"]}):
        rs = {r["mode"]: r for r in runs if r["layout"] == lay}
        commit = rs.get("commit", {}).get("step_s")
        v: dict[str, Any] = {"commit_s": commit}
        for mode in ("serial", *OVERLAPPED):
            if mode not in rs or serve is None:
                continue
            step = rs[mode]["step_s"]
            e = {"step_s": step, "overhead_s": round(step - serve, 4), "overhead_share": round((step - serve) / serve, 4),
                 "meets_target": step - serve <= TARGET_S}
            if commit:
                e["hidden_s"] = round(serve + commit - step, 4)
                e["hidden_fraction"] = round((serve + commit - step) / commit, 4)
            v[mode] = e
        out["layouts"][lay] = v
    return out


def summary(doc: dict) -> str:
    lines = [f"{doc['env'].get('name')} {doc['env'].get('uuid', '')} batch {doc['batch']} micro-batch {doc['micro_batch']}"]
    for lay in doc["layouts"].values():
        lines.append(f"layout {lay['name']} {lay['digest'][:16]}: {lay['units_a_slot']:,} units, {lay['rows_a_slot']:,} rows, "
                     f"{lay['bytes_a_slot']:,} B a slot; plan {lay.get('plan_gb')} GB, built in {lay.get('plan_s')} s")
    for r in doc["runs"]:
        t, h = r["telemetry"], r["hbm"]
        lines.append(f"{r['run']:18s} step {r['step_s']:8.3f} s (intervals {r['intervals_s']}, drain {r['drain_s']} s) "
                     f"{json.dumps(r['seconds'])}; SM {t.get('sm_mhz')} MHz (min {t.get('sm_mhz_min')}), "
                     f"{t.get('power_w')} W (instant max {t.get('power_instant_w_max')}), {t.get('temp_c')} C, "
                     f"{t.get('reasons')}; HBM peak {h['peak_gb']} GB (+{h['peak_over_before_gb']} over the run's start)")
    dv = doc["derived"]
    lines.append(f"served step alone {dv['serve_s']} s; target: overhead <= {dv['target_s']} s a step")
    for lay, v in dv["layouts"].items():
        lines.append(f"{lay}: commit alone {v.get('commit_s')} s")
        for mode in ("serial", *OVERLAPPED):
            if mode in v:
                e = v[mode]
                lines.append(f"  {mode:7s} step {e['step_s']:.3f} s, overhead {e['overhead_s']:+.3f} s "
                             f"({100 * e['overhead_share']:.1f}% of the step), hidden "
                             f"{e.get('hidden_s')} s ({None if 'hidden_fraction' not in e else round(100 * e['hidden_fraction'], 1)}%"
                             f" of the commit), target met {e['meets_target']}")
    lines.append(f"roots equal to serial's: {doc['roots_equal']}")
    return "\n".join(lines) + "\n"


def roots_equal(runs: list[dict]) -> dict[str, bool]:
    out = {}
    for r in runs:
        if r["mode"] not in OVERLAPPED:
            continue
        ref = next((x for x in runs if x["mode"] == "serial" and x["layout"] == r["layout"]), None)
        if ref is not None:
            n = min(len(ref["roots"]), len(r["roots"]))
            out[r["run"]] = n > 0 and ref["roots"][:n] == r["roots"][:n]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shared", type=Path, help="circuits' shared-row Layout JSON (CIR-113)")
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
    ap.add_argument("--leaf-minb", type=int, default=0, help="the plan's leaf kernel blocks an SM (0: the plan's default)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    from verity_torch_loop import weights as SW
    dev = torch.device("cuda")
    shape = SHAPES[a.shape]
    smi = Smi(os.environ.get("GPU_LEASE_UUID") or "0")
    props = torch.cuda.get_device_properties(dev)
    doc: dict[str, Any] = {"env": {"name": props.name, "uuid": os.environ.get("GPU_LEASE_UUID"), "sms": props.multi_processor_count,
                                   "hbm_gb": round(props.total_memory / 1e9, 2), "torch": torch.__version__,
                                   "cuda": torch.version.cuda, "smi_fields": list(smi.fields)},
                           "shape": shape.name, "batch": a.batch, "micro_batch": a.micro_batch, "layouts": {}, "runs": []}

    def write() -> None:
        doc["derived"], doc["roots_equal"] = derive(doc["runs"]), roots_equal(doc["runs"])
        (a.out / "overlap.json").write_text(json.dumps(doc, indent=1) + "\n")
        (a.out / "summary.txt").write_text(summary(doc))

    seed = {"test": SW.TEST_SEED, "demo": SW.DEMO_SEED}.get(a.seed, a.seed.encode())
    t0 = time.perf_counter()
    params = Params(shape, GeneratorStore(Weights(shape, seed), a.workers), dev, True)
    doc["params"] = params.describe() | {"seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps({"params": doc["params"]}), flush=True)
    ctx, key = CM.fixture_secrets(a.run_id)
    prompts = prompts_for(a.run_id, a.batch, a.prompt_bytes)
    d = OverlapDriver(shape, params, a.batch, Null(), "cuda", keep=False, micro_batch=a.micro_batch)
    for t in range(a.warm):
        d.step(t, prompts[:, t])
    d.results = []
    plans: dict[str, G.Plan] = {}
    current: str | None = None
    for spec in a.runs:
        mode, lay, *n = spec.split(":") + [""] * (spec.count(":") == 0)
        if mode not in MODES:
            raise SystemExit(f"--runs: {spec!r}'s mode is one of {MODES}")
        if mode != "serve" and lay != current:
            plans.clear()
            torch.cuda.empty_cache()
            layout = load_layout(a.shared) if lay == "shared" else CM.layout(shape) if lay == "families" else None
            if layout is None:
                raise SystemExit(f"--runs: {spec!r}'s layout is shared or families")
            torch.cuda.synchronize(dev)
            m0, p0 = torch.cuda.memory_allocated(dev), time.perf_counter()
            plans[lay] = G.Plan(ctx, layout, device=dev, **({"leaf_minb": a.leaf_minb} if a.leaf_minb else {}))
            doc["layouts"][lay] = describe(layout) | {"plan_s": round(time.perf_counter() - p0, 1),
                                                      "plan_gb": round((torch.cuda.memory_allocated(dev) - m0) / 1e9, 3)}
            current = lay
            print(json.dumps({"layout": doc["layouts"][lay]}), flush=True)
        steps = int(n[0]) if n and n[0] else a.steps
        try:
            r = run_one(spec, mode, steps, d, plans.get(lay) if mode != "serve" else None, a, ctx, key, prompts, smi)
        except Exception as e:  # noqa: BLE001  (recorded; the next run starts clean)
            r = {"run": spec, "mode": mode, "layout": lay, "error": f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"}
            d.committer, d.results = Null(), []
            torch.cuda.empty_cache()
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
