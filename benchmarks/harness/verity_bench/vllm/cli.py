"""Run the shared vLLM overhead benchmark.

    verity-bench vllm --preset llama31-8b-fp8-1gpu \
        --arms stock,stock-fp8 --sweep quick --reps 3 --out "$OUT"
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from verity_bench.vllm.arms import resolve_arms
from verity_bench.vllm.presets import PRESETS, expand_sweep, get_preset, local_checkpoint
from verity_bench.vllm.result import write
from verity_bench.vllm.runner import ProcessEngineFactory, run_benchmark
from verity_bench.vllm.table import render


def effective_eager(preset: Any, requested: bool) -> bool:
    """A preset may require eager execution even when the CLI default uses graphs."""

    return bool(requested or preset.engine_settings.get("enforce_eager", False))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--model-path", type=Path, default=None,
                        help="machine-local checkpoint override; every file is hashed into the result identity")
    parser.add_argument("--arms", required=True, help="comma-separated arm ids; include every arm's control")
    parser.add_argument("--arm-module", action="append", default=[],
                        help="experimental pkg.mod:NAME exposing install(target); repeatable")
    parser.add_argument("--sweep", choices=("quick", "full"), default="quick")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eager", action="store_true",
                        help="run eagerly instead of the preset's default CUDA graphs")


def run(a: Any) -> int:
    if a.reps < 1:
        raise ValueError("--reps must be positive")
    preset = get_preset(a.preset)
    if a.model_path is not None:
        preset = local_checkpoint(preset, a.model_path)
    eager = effective_eager(preset, a.eager)
    preset = replace(
        preset,
        engine_settings={
            **preset.engine_settings,
            "cuda_graphs": not eager,
            "enforce_eager": eager,
        },
    )
    arms = resolve_arms(a.arms, tuple(a.arm_module))
    points = expand_sweep(a.sweep)
    a.out.mkdir(parents=True, exist_ok=True)
    doc = run_benchmark(preset, arms, points, sweep=a.sweep, reps=a.reps, factory=ProcessEngineFactory(eager=eager))
    write(doc, a.out / "result.json")
    table = render(doc)
    (a.out / "table.md").write_text(table)
    print(table, end="")
    return 0 if all(g["passed"] for g in doc["gates"]) else 3
