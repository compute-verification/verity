"""MEM-41's probe (temporary): one bandwidth arm's engine lifetime as verity-bench's block mode runs it (install, ready,
then for each point a warm-up and REPS timed runs), with every engine step of every run recorded: its host time, the
tokens it scheduled, the stack's host calls inside it, and what changed across it (allocator cudaMalloc/cudaFree, GC,
cgroup CPU throttling, filler progress, the stack's lazy state). Writes $OUT/probe-<arm>.json.

    python -m benchmarks.vllm_encoded.first_rep_probe --arm audit|plain --out DIR [--points b32,b8] [--reps 4]
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import time
from pathlib import Path
from typing import Any

from benchmarks.vllm_encoded import bench_arm

PRESET = dict(
    id="llama31-8b-fp8-1gpu-bandwidth",
    model="/workspace/pouw/mvp-e2e/models/llama-3.1-8b-instruct",
    model_digest="98e25b0ecbb3eff924ea9fad4673feea9ea6457d2f45db5b5b60e88205de1daa",
    precision="fp8",
    gpu_id="rtx-pro-6000",
    gpu_count=1,
    vllm_pin="d9105ea8001e0a6d77a96327d17515bb5791fb36",
    engine_settings={"cuda_graphs": True, "capture_sizes": [1, 8, 32, 128], "max_model_len": 8384,
                     "resident_gpu_memory_utilization": 0.40, "block_gpu_memory_utilization": 0.80,
                     "max_num_batched_tokens": 16384, "block_kv_cache_memory_bytes": 60129542144},
)
POINTS = {"b32": (32, 512, "mixed", 64), "b8": (8, 512, "mixed", 64)}
MEM = ("num_device_alloc", "num_device_free", "num_alloc_retries", "num_sync_all_streams")


def _cgroup_stat() -> Path | None:
    try:
        line = next(x for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    except (OSError, StopIteration):
        return None
    path = Path("/sys/fs/cgroup") / line[3:].lstrip("/") / "cpu.stat"
    return path if path.is_file() else None


class Recorder:
    def __init__(self, installed: Any, target: Any) -> None:
        import torch

        from verity_vllm.engine.arrivals import scheduler_of
        from verity_vllm.engine.engine_profile import get_model_runner

        self.torch, self.inst = torch, installed
        self.st = installed.stack
        self.att = self.st.attention
        self.steps: list[dict] | None = None
        self.cur: dict | None = None
        self.sched: tuple[int, int] | None = None
        self.acc = {"gc_s": 0.0, "gc0": 0, "gc1": 0, "gc2": 0}
        self.gc_t0 = 0.0
        self.cpu_stat = _cgroup_stat()
        self.progress = (ctypes.c_uint64 * 2)()
        llm = target.engine.vllm
        hooks = target.hooks
        hooks.wrap(llm.llm_engine, "step", self._step)
        sched = scheduler_of(llm)
        if sched is not None:
            hooks.wrap(sched, "schedule", self._schedule)
        hooks.append(gc.callbacks, self._gc)
        for owner, names in ((self.att, ("finish_pass", "join", "plan", "gathered", "flush_eager", "flush_graph")),
                             (self.st, ("commit", "begin_pass"))):
            for name in names:
                hooks.wrap(owner, name, self._timed(name))
        head = get_model_runner(llm).get_model().lm_head
        hooks.wrap(head.quant_method, "apply", self._timed("lm_head"))

    def _gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self.gc_t0 = time.perf_counter()
        else:
            self.acc["gc_s"] += time.perf_counter() - self.gc_t0
            self.acc[f"gc{info.get('generation', 0)}"] += 1

    def _timed(self, name: str):
        def make(orig):
            def fn(*a, **kw):
                t = time.perf_counter()
                try:
                    return orig(*a, **kw)
                finally:
                    if self.cur is not None:
                        c = self.cur.setdefault(name, [0, 0.0])
                        c[0] += 1
                        c[1] += time.perf_counter() - t
            return fn
        return make

    def _schedule(self, orig):
        def fn(*a, **kw):
            out = orig(*a, **kw)
            self.sched = (int(getattr(out, "total_num_scheduled_tokens", -1)),
                          len(getattr(out, "num_scheduled_tokens", None) or {}))
            return out
        return fn

    def counters(self) -> dict[str, float]:
        c: dict[str, float] = dict(self.acc)
        stats = self.torch.cuda.memory_stats()
        for key in MEM:
            c[key] = stats.get(key, 0)
        c["reserved"] = stats.get("reserved_bytes.all.current", 0)
        c["cpu_s"] = time.process_time()
        if self.cpu_stat is not None:
            for line in self.cpu_stat.read_text().splitlines():
                key, _, value = line.partition(" ")
                if key in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec"):
                    c["cg_" + key] = int(value)
        self.inst.topup.so.tu_progress(self.progress)
        c["filler_bytes"], c["filler_chunks"] = self.progress[0], self.progress[1]
        c["passes"], c["lm_graphs"] = self.st.passes, len(self.st.lm.graphs)
        c["kv_tag_capacity"] = self.att.kv_tag_capacity
        c["gather_blocks"] = 0 if self.att.gather_buf is None else self.att.gather_buf.shape[0]
        c.update({f"att_{k}": v for k, v in self.att.calls.items()})
        return c

    @staticmethod
    def delta(after: dict, before: dict) -> dict:
        return {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}

    def _step(self, orig):
        def fn(*a, **kw):
            if self.steps is None:
                return orig(*a, **kw)
            before, self.cur, self.sched = self.counters(), {}, None
            t = time.perf_counter()
            out = orig(*a, **kw)
            s = time.perf_counter() - t
            calls, self.cur = self.cur, None
            self.steps.append({"t": t - self.t0, "s": s, "outputs": len(out), "sched": self.sched,
                               "calls": {k: [v[0], round(v[1], 6)] for k, v in calls.items()},
                               "d": self.delta(self.counters(), before)})
            return out
        return fn

    def begin(self) -> None:
        self.steps, self.t0, self.c0 = [], time.perf_counter(), self.counters()

    def end(self) -> dict:
        wall = time.perf_counter() - self.t0
        steps, self.steps = self.steps, None
        return {"wall": wall, "steps": steps, "d": self.delta(self.counters(), self.c0)}


class _ProbeInstalled(bench_arm._Installed):
    def ready(self, target: Any) -> None:
        super().ready(target)
        self.recorder = Recorder(self, target)


class _ProbeArm(bench_arm._ArmModule):
    def install(self, target: Any) -> bench_arm._Installed:
        self.state = _ProbeInstalled(target, encoded=self.encoded)
        return self.state


PLAIN = _ProbeArm("bandwidth-plain", encoded=False, control=None)
AUDIT = _ProbeArm("bandwidth-audit", encoded=True, control=PLAIN.id)


def _summary(label: str, run: dict, measured: dict | None) -> str:
    steps = run["steps"]
    top = sorted(range(len(steps)), key=lambda i: -steps[i]["s"])[:5]
    slow = [{"i": i, "t": round(steps[i]["t"], 3), "s": round(steps[i]["s"], 4), "sched": steps[i]["sched"],
             "calls": {k: v for k, v in steps[i]["calls"].items() if v[1] > 1e-3},
             "d": {k: v for k, v in steps[i]["d"].items() if k not in ("cpu_s", "cg_usage_usec", "cg_nr_periods",
                                                                       "filler_bytes", "passes")}}
            for i in top]
    head = {"label": label, "wall": round(run["wall"], 3), "steps": len(steps),
            "step_s": round(sum(s["s"] for s in steps), 3),
            "first": [round(s["s"], 4) for s in steps[:6]]}
    if measured:
        m = measured["metrics"]
        head |= {"wall_s": round(measured["wall_s"], 3), "ttft_ms": round(m["time_to_first_token_s"] * 1e3, 1),
                 "tpot_ms": round(m["time_per_output_token_s"] * 1e3, 3)}
    return json.dumps(head) + "\n  slowest " + json.dumps(slow) + "\n  run " + json.dumps(
        {k: round(v, 4) if isinstance(v, float) else v for k, v in run["d"].items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("audit", "plain"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", default="b32,b8")
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()

    from verity_bench.vllm.arms import experimental_arm
    from verity_bench.vllm.presets import Preset, SweepPoint, served
    from verity_bench.vllm.runner import _VllmEngine

    arm = experimental_arm(f"benchmarks.vllm_encoded.first_rep_probe:{a.arm.upper()}")
    preset, eager = served(Preset(**PRESET), False)
    t = time.perf_counter()
    engine = _VllmEngine(arm, preset, False, eager)
    rec = engine.obj.state.recorder
    doc: dict[str, Any] = {"arm": arm.id, "startup_s": time.perf_counter() - t, "identity": engine.identity,
                           "cgroup_cpu_stat": str(rec.cpu_stat), "runs": []}
    print(f"probe {arm.id}: engine ready in {doc['startup_s']:.1f} s", flush=True)
    try:
        for name in a.points.split(","):
            point = SweepPoint(*POINTS[name])
            for k in range(a.reps + 1):
                label = f"{point.id} " + ("warmup" if k == 0 else f"rep{k - 1}")
                rec.begin()
                measured = None if k == 0 else engine.measure(point)
                if k == 0:
                    engine.warmup(point)
                run = rec.end()
                doc["runs"].append({"label": label, "measured": measured, **run})
                print(_summary(label, run, measured), flush=True)
    finally:
        engine.close()
        out = Path(a.out) / f"probe-{a.arm}.json"
        out.write_text(json.dumps(doc, default=str))
        print(f"probe {arm.id}: wrote {out}", flush=True)


if __name__ == "__main__":
    main()
