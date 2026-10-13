"""Run the shared vLLM overhead benchmark.

    verity-vllm bench --preset llama31-8b-fp8-1gpu \
        --arms stock,stock-fp8 --sweep quick --reps 3 --out "$OUT"
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from verity_vllm.bench.arms import resolve_arms
from verity_vllm.bench.presets import PRESETS, expand_sweep, get_preset
from verity_vllm.bench.result import write
from verity_vllm.bench.runner import ProcessEngineFactory, run_benchmark
from verity_vllm.bench.table import render
from verity_vllm.config import UsageError, option


@dataclass(frozen=True, kw_only=True)
class Options:
    preset: str = option("--preset", choices=tuple(PRESETS))
    arms: str = option("--arms", help="comma-separated arm ids; include every arm's control")
    arm_module: list[str] = option("--arm-module", action="append", default=[],
                                   help="experimental pkg.mod:NAME exposing install(target); repeatable")
    sweep: str = option("--sweep", choices=("quick", "full"), default="quick")
    reps: int = option("--reps", type=int, default=3)
    out: Path = option("--out", type=Path)
    eager: bool = option("--eager", action="store_true", default=False,
                         help="run eagerly instead of the preset's default CUDA graphs")


def main(a: Options) -> int:
    if a.reps < 1:
        raise UsageError("--reps must be positive")
    try:
        preset = get_preset(a.preset)
        preset = replace(
            preset,
            engine_settings={
                **preset.engine_settings,
                "cuda_graphs": not a.eager,
                "enforce_eager": a.eager,
            },
        )
        arms = resolve_arms(a.arms, tuple(a.arm_module))
        points = expand_sweep(a.sweep)
    except ValueError as e:
        raise UsageError(str(e)) from e
    a.out.mkdir(parents=True, exist_ok=True)
    doc = run_benchmark(preset, arms, points, sweep=a.sweep, reps=a.reps, factory=ProcessEngineFactory(eager=a.eager))
    write(doc, a.out / "result.json")
    table = render(doc)
    (a.out / "table.md").write_text(table)
    print(table, end="")
    return 0 if all(g["passed"] for g in doc["gates"]) else 3
