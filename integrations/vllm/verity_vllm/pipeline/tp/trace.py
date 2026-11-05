"""`tp-commit --trace-dir DIR`: after the timed pairs, one more pair under `torch.profiler` (CPU ops, CUDA runtime calls, kernels and
copies, every thread) on every rank, never in `runs.jsonl` or the summary's timing (a profiler slows what it observes): a control run,
then an instrumented one (`tp2_profiler_start` / `tp2_profiler_stop`), DIR/torch.<arm>/census.rank<r>.json, with the gzipped Kineto
trace of the `--trace-ranks`.  No sampler from outside the process: `py-spy record --nonblocking` (0.4.2) looped in `process_vm_readv`
on every loaded rank and never wrote a profile; `thread_cpu` stands in for it on every timed run.

The instrumented run is finalized for its roots only (`tp2_commit_finalize(checks=False)`) and reset; every run's walls go to
DIR/traced.jsonl.  ``release``: the GPUs go back after the last run (`lease.stop`), the Commit's last device work."""
from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

from verity_vllm.pipeline.tp import lease as TPL


def thread_cpu(worker: Any) -> dict[str, Any]:
    """On a rank (`LLM.collective_rpc` with a callable): the CPU seconds (user + system, `/proc/self/task/<tid>/stat`) each thread of
    its process has used so far, summed by the thread's Python name (``MainThread`` serves the steps, ``cmt-worker`` absorbs them,
    ``cmt-hash_<i>``, ``pinned-populate_<i>``), a thread Python did not start by ``native:<comm>``."""
    names = {t.native_id: t.name for t in threading.enumerate()}
    tick = os.sysconf("SC_CLK_TCK")
    cpu: dict[str, float] = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            stat = Path(f"/proc/self/task/{tid}/stat").read_text()
        except OSError:
            continue
        head, tail = stat.rsplit(")", 1)
        f = tail.split()
        key = names.get(int(tid)) or f"native:{head.split('(', 1)[1]}"
        cpu[key] = cpu.get(key, 0.0) + (int(f[11]) + int(f[12])) / tick
    return {"rank": int(worker._tp2_commit["rank"]), "cpu_s": cpu}


def cpu_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Per rank, the CPU seconds each thread name used between two `thread_cpu` readings (the names that used any)."""
    b = {int(x["rank"]): x["cpu_s"] for x in before}
    return {int(x["rank"]): {k: round(v - b.get(int(x["rank"]), {}).get(k, 0.0), 3) for k, v in sorted(x["cpu_s"].items())
                             if v - b.get(int(x["rank"]), {}).get(k, 0.0) > 0} for x in after}


def run(llm: Any, run_once: Callable[[], dict[str, Any]], d0: Path, ranks: list[int], timeout: float, *, release: bool = False) -> list[dict[str, Any]]:
    rows = []
    for arm in ("control", "instrumented"):
        d = d0 / f"torch.{arm}"
        d.mkdir(parents=True, exist_ok=True)
        gc.collect()
        if arm != "control":
            llm.collective_rpc("tp2_commit_install", timeout=timeout)
        llm.collective_rpc("tp2_profiler_start", timeout=timeout)
        r = run_once()
        row = {"arm": arm, "wall_s": r["wall_s"], "step_wall_s": r["step_wall_s"], "step_tokens_out": r["step_tokens_out"],
               "profiler": llm.collective_rpc("tp2_profiler_stop", timeout=timeout, kwargs={"out_dir": str(d), "keep_trace": True, "keep_ranks": ranks})}
        if arm != "control":
            fin = llm.collective_rpc("tp2_commit_finalize", timeout=timeout, kwargs={"checks": False})
            row["run_roots"] = {int(f["rank"]): f["run_root"] for f in fin}
            TPL.reset(llm, timeout)
        rows.append(row)
        with (d0 / "traced.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(f"[tp2commit] traced torch {arm}: {r['wall_s']:.3f}s steps {len(r['step_wall_s'])} -> {d}", flush=True)
    if release:
        TPL.stop(llm, timeout, "work.trace")
    return rows


__all__ = ["thread_cpu", "cpu_delta", "run"]
