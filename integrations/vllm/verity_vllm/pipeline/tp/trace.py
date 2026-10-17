"""`tp-commit --trace-dir DIR`: after the timed pairs, one more pair per profiler in `--trace-tools`, never in `runs.jsonl` or the
summary's timing (a profiler slows what it observes).  Each tool profiles a control run, then an instrumented one, over every rank:

* ``torch``: `torch.profiler` (CPU ops, CUDA runtime calls, kernels and copies, every thread) on every rank
  (`tp2_profiler_start` / `tp2_profiler_stop`): DIR/torch.<arm>/census.rank<r>.json, with the gzipped Kineto trace of the
  `--trace-ranks`;
* ``pyspy``: `py-spy record` of every thread, idle ones included, of each `--trace-ranks` process, sampled from outside it
  (`--trace-pyspy`, the py-spy binary, under ``sudo -n``: a rank is not a descendant of the driver's children):
  DIR/pyspy.<arm>/rank<r>.speedscope.json.

The instrumented run is finalized for its roots only (`tp2_commit_finalize(checks=False)`) and reset; every run's walls go to
DIR/traced.jsonl."""
from __future__ import annotations

import gc
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Callable

from verity_vllm.pipeline.tp import lease as TPL

TOOLS = ("torch", "pyspy")


def rank_pid(worker: Any) -> dict[str, int]:
    """On a rank (`LLM.collective_rpc` with a callable): its rank and process id."""
    return {"rank": int(worker._tp2_commit["rank"]), "pid": os.getpid()}


def tools_of(spec: str, pyspy: str | None) -> list[str]:
    tools = [t for t in spec.split(",") if t]
    bad = sorted(set(tools) - set(TOOLS))
    if bad or not tools:
        raise SystemExit(f"--trace-tools {spec!r}: each of {list(TOOLS)}")
    if "pyspy" in tools and not pyspy:
        raise SystemExit("--trace-tools pyspy needs --trace-pyspy, the py-spy binary (`pip install --target DIR py-spy`: DIR/bin/py-spy)")
    return tools


def _torch(llm: Any, d: Path, ranks: list[int], timeout: float) -> Callable[[], Any]:
    llm.collective_rpc("tp2_profiler_start", timeout=timeout)
    return lambda: llm.collective_rpc("tp2_profiler_stop", timeout=timeout, kwargs={"out_dir": str(d), "keep_trace": True, "keep_ranks": ranks})


def _pyspy(exe: str, pids: dict[int, int], d: Path, ranks: list[int], rate: int) -> Callable[[], Any]:
    logs = {r: open(d / f"rank{r}.pyspy.log", "w") for r in ranks}
    procs = {r: subprocess.Popen(["sudo", "-n", exe, "record", "--pid", str(pids[r]), "--rate", str(rate), "--nonblocking",
                                  "--threads", "--idle", "--format", "speedscope", "--output", str(d / f"rank{r}.speedscope.json")],
                                 stdout=subprocess.DEVNULL, stderr=logs[r])
             for r in ranks}

    def stop() -> dict[int, int]:
        for p in procs.values():
            p.send_signal(signal.SIGINT)
        rcs = {r: p.wait() for r, p in procs.items()}
        for f in logs.values():
            f.close()
        return rcs
    return stop


def run(llm: Any, run_once: Callable[[], dict[str, Any]], d0: Path, tools: list[str], ranks: list[int], timeout: float, *,
        pyspy: str | None = None, rate: int = 500) -> list[dict[str, Any]]:
    d0.mkdir(parents=True, exist_ok=True)
    pids = {int(p["rank"]): int(p["pid"]) for p in llm.collective_rpc(rank_pid, timeout=timeout)}
    rows = []
    for tool in tools:
        for arm in ("control", "instrumented"):
            d = d0 / f"{tool}.{arm}"
            d.mkdir(parents=True, exist_ok=True)
            gc.collect()
            if arm != "control":
                llm.collective_rpc("tp2_commit_install", timeout=timeout)
            stop = _torch(llm, d, ranks, timeout) if tool == "torch" else _pyspy(str(pyspy), pids, d, ranks, rate)
            r = run_once()
            row = {"tool": tool, "arm": arm, "wall_s": r["wall_s"], "step_wall_s": r["step_wall_s"], "step_tokens_out": r["step_tokens_out"],
                   "pids": pids, "profiler": stop()}
            if arm != "control":
                fin = llm.collective_rpc("tp2_commit_finalize", timeout=timeout, kwargs={"checks": False})
                row["run_roots"] = {int(f["rank"]): f["run_root"] for f in fin}
                TPL.reset(llm, timeout)
            rows.append(row)
            with (d0 / "traced.jsonl").open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(f"[tp2commit] traced {tool} {arm}: {r['wall_s']:.3f}s steps {len(r['step_wall_s'])} -> {d}", flush=True)
    return rows


__all__ = ["TOOLS", "rank_pid", "tools_of", "run"]
