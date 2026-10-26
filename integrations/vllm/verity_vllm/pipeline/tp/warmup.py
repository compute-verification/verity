"""`tp-commit`'s warm-up instrumented run: on every rank, the learn-only pass of bounded staging, as the single-rank Commit runs it
(`pipeline.commit`, `acquire.sources.taps.warmup`).  Every shape class of the workload gets its plan here (same lengths, same schedule as
the record runs: `throwaway_requests`); a learning step over the transient budget records its plan and discards its windows, so the
record runs hit every plan and stage bounded.  The run is never a record: its ranks bind, attribute, compare and replay nothing of it
(`pipeline.tp.rank_worker.tp2_commit_finalize`)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.pipeline.c2_replay import rss_kb


def warming(worker: Any, on: bool) -> dict[str, Any]:
    """On a rank (`LLM.collective_rpc` with a callable): the warm-up on or off for the rank's committer, and the rank's host memory."""
    st = worker._tp2_commit
    st["warmup"] = on
    TAPS.warmup(st["com"], on)
    return {"rank": st["rank"], "learn_only": bool(getattr(st["com"], "learn_only", False)), "host_mem_kb": dict(zip(("rss", "hwm"), rss_kb()))}


def run(llm: Any, run_once: Callable[[], dict[str, Any]], out: Path, kind: str, rpc_timeout: float) -> dict[str, Any]:
    """The warm-up instrumented run over every rank: warm-up on, install, `run_once()`, finalize, warm-up off, reset.  Writes
    `finalize_warmup.json` and, under bounded staging, `bounded_warmup_<kind>.json` (one entry per rank); returns `run_once`'s result."""
    llm.collective_rpc(warming, args=(True,), timeout=rpc_timeout)
    llm.collective_rpc("tp2_commit_install", timeout=rpc_timeout)
    r = run_once()
    fin = llm.collective_rpc("tp2_commit_finalize", timeout=rpc_timeout)
    off = {o["rank"]: o for o in llm.collective_rpc(warming, args=(False,), timeout=rpc_timeout)}
    (out / "finalize_warmup.json").write_text(json.dumps(fin, indent=1, default=str) + "\n")
    print(f"[tp2commit] warm-up instrumented: {r['wall_s']:.3f}s; ranks {[(f['rank'], f['leaves'], f['errors_n'], f['errors'][:2]) for f in fin]}", flush=True)
    bounded = [{"rank": f["rank"], **f["bounded_staging"], "pinned_pool_bytes": f.get("pinned_pool_bytes"), "plans_cached": f.get("plans_cached"),
                "host_mem_kb": off.get(f["rank"], {}).get("host_mem_kb"), "errors": f.get("errors")}
               for f in fin if (f.get("bounded_staging") or {}).get("enabled")]
    for b in bounded:
        print(f"[tp2commit] rank {b['rank']}: bounded staging warm-up ({kind}): learned-only steps {b.get('learned_only_steps')} "
              f"({(b.get('learned_only_bytes') or 0) >> 20} MiB discarded), transient peak {(b.get('transient_peak_bytes') or 0) >> 20} MiB "
              f"(budget {(b.get('learn_budget_bytes') or 0) >> 20} MiB), plan mismatches {b.get('plan_mismatches')}, plans cached "
              f"{b.get('plans_cached')}, pinned pool {(b.get('pinned_pool_bytes') or 0) >> 20} MiB", flush=True)
    if bounded:
        (out / f"bounded_warmup_{kind}.json").write_text(json.dumps(bounded, indent=1, default=str) + "\n")
    llm.collective_rpc("tp2_commit_reset", timeout=rpc_timeout)
    return r


__all__ = ["warming", "run"]
