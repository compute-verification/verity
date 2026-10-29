"""`verity-vllm telemetry-admission` -- the empirical admission planner's subcommands (`pipeline.telemetry.admission`).

    verity-vllm telemetry-admission plan --workload wl.json [--row ROW] [--out admission.json]     the row's advisory plan line
    verity-vllm telemetry-admission compare <attempt_dir>... [--md]                                predicted vs observed peaks
    verity-vllm telemetry-admission predict --model M --requests N ...                             reservation per stage
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from verity_vllm.config import option, workload_arg
from verity_vllm.pipeline.telemetry.admission import (GiB, RowConfig, cgroup_memory_max, compare, plan_row, predict,
                                                      render_compare_md, render_points_md, retain_exclude_families,
                                                      row_config_from_attempt, row_config_from_workload)


@dataclass(frozen=True, kw_only=True)
class PlanOptions:
    """ADVISORY admission plan for a sweep row from its workload file (row_pod.sh hook): writes admission.json, prints one line, exit 0"""
    workload: str = option("--workload", type=workload_arg)
    row: str | None = option("--row", default=None)
    out: str | None = option("--out", default=None)
    stages: str = option("--stages", default="build,match,commit")
    pairs: int = option("--pairs", type=int, default=3)
    replay_workers: int | None = option("--replay-workers", default=None, type=int)
    snapshot_steps: str | None = option("--snapshot-steps", default=None)
    snapshot_max_bytes: int | None = option("--snapshot-max-bytes", default=None, type=int)
    host_limit_bytes: int | None = option("--host-limit-bytes", default=None, type=int, help="default: this cgroup's memory.max")
    largest_vu_in_parent: bool = option("--largest-vu-in-parent", default=False, action="store_true")
    retain: str | None = option("--retain", default=None, choices=["host", "device", "none", "sample"], help="the run's --retain (default: the workload's / host)")
    bounded_staging: bool = option("--bounded-staging", default=False, action="store_true", help="the run uses --bounded-staging (with --retain host --retain-exclude hidden_m1)")
    retain_exclude: str | None = option("--retain-exclude", default=None, help="the run's --retain-exclude families (hidden_m1 = fa2_hidden_m1,fa3_hidden_m1)")
    no_fork_gc_freeze: bool = option("--no-fork-gc-freeze", default=False, action="store_true", help="model a forked pool without gc.freeze before the fork (two children may copy the C2 heap)")
    cmd: str = field(default="plan", init=False)


@dataclass(frozen=True, kw_only=True)
class CompareOptions:
    """predicted vs observed per stage for attempt dirs"""
    attempt_dirs: list[str] = option("attempt_dirs", nargs="+")
    md: bool = option("--md", default=False, action="store_true")
    points: bool = option("--points", default=False, action="store_true", help="also print the commit-point tables")
    largest_vu_in_parent: bool = option("--largest-vu-in-parent", default=False, action="store_true", help="predict with the FIX: the largest strata evaluated in the parent, forked workers are 250 MiB workers")
    no_fork_gc_freeze: bool = option("--no-fork-gc-freeze", default=False, action="store_true", help="predict with a forked pool without gc.freeze (as records made without it ran)")
    json_out: str | None = option("--json-out", default=None)
    cmd: str = field(default="compare", init=False)


@dataclass(frozen=True, kw_only=True)
class PredictOptions:
    """reservation per stage for a row config"""
    model: str = option("--model")
    requests: int = option("--requests", type=int, default=1)
    prompt_tokens: int = option("--prompt-tokens", type=int, default=1024)
    decode_rows: int = option("--decode-rows", type=int, default=129)
    steps: int = option("--steps", type=int, default=129)
    max_context: int = option("--max-context", type=int, default=1153)
    pairs: int = option("--pairs", type=int, default=3)
    replay_workers: int | None = option("--replay-workers", default=None, type=int)
    host_limit_gib: float | None = option("--host-limit-gib", default=None, type=float)
    n_instances: int | None = option("--n-instances", default=None, type=int)
    raw_bytes: int | None = option("--raw-bytes", default=None, type=int)
    largest_vu_in_parent: bool = option("--largest-vu-in-parent", default=False, action="store_true")
    retain: str = option("--retain", default="host", choices=["host", "device", "none", "sample"])
    bounded_staging: bool = option("--bounded-staging", default=False, action="store_true")
    retain_exclude: str | None = option("--retain-exclude", default=None)
    no_fork_gc_freeze: bool = option("--no-fork-gc-freeze", default=False, action="store_true")
    cmd: str = field(default="predict", init=False)


COMMANDS = {"plan": PlanOptions, "compare": CompareOptions, "predict": PredictOptions}


def main(a: PlanOptions | CompareOptions | PredictOptions) -> int:
    if a.cmd == "plan":
        # advisory: never a non-zero exit, never an exception out of the hook -- the failure is written into the line instead
        try:
            limit = a.host_limit_bytes if a.host_limit_bytes is not None else cgroup_memory_max()
            cfg = row_config_from_workload(a.workload, a.row, pairs=a.pairs, replay_workers=a.replay_workers, host_limit_bytes=limit,
                                           snapshot_steps=a.snapshot_steps, snapshot_max_bytes=a.snapshot_max_bytes, stages=a.stages)
            cfg.largest_vu_in_parent = a.largest_vu_in_parent
            cfg.fork_gc_freeze = not a.no_fork_gc_freeze
            cfg.bounded_staging = a.bounded_staging
            cfg.retain_exclude = retain_exclude_families(a.retain_exclude)
            if a.retain:
                cfg.retain = a.retain
            print(plan_row(cfg, a.out)["line"])
        except Exception as e:  # noqa: BLE001 -- advisory hook: report, never stop the row
            print(f"ADMISSION advisory unavailable: {type(e).__name__}: {e}")
        return 0
    if a.cmd == "predict":
        cfg = RowConfig(model=a.model, requests=a.requests, prompt_tokens=a.prompt_tokens, decode_rows=a.decode_rows, steps=a.steps, max_context=a.max_context, pairs=a.pairs,
                        replay_workers=a.replay_workers, host_limit_bytes=int(a.host_limit_gib * GiB) if a.host_limit_gib else None, n_instances=a.n_instances,
                        scheduled_raw_bytes=a.raw_bytes, largest_vu_in_parent=a.largest_vu_in_parent, retain=a.retain,
                        fork_gc_freeze=not a.no_fork_gc_freeze, bounded_staging=a.bounded_staging, retain_exclude=retain_exclude_families(a.retain_exclude))
        print(json.dumps(predict(cfg), indent=1, default=str))
        return 0
    results = []
    for d in a.attempt_dirs:
        cfg = row_config_from_attempt(d)
        if a.largest_vu_in_parent:
            cfg.largest_vu_in_parent = True
        cfg.fork_gc_freeze = not a.no_fork_gc_freeze
        results.append(compare(d, cfg))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(results, indent=1, default=str))
    if a.md:
        print(render_compare_md(results))
        if a.points:
            for r in results:
                if any(k.startswith("commit/") for k in r["points"]):
                    print()
                    print(render_points_md(r))
    else:
        print(json.dumps(results, indent=1, default=str))
    return 0
