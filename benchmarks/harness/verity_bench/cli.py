"""The command line for Verity's shared benchmark harnesses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verity_bench import jobs
from verity_bench.vllm import cli as vllm_cli
from verity_bench.vllm import result
from verity_bench.vllm.arms import resolve_arms
from verity_bench.vllm.presets import served
from verity_bench.vllm.result import write
from verity_bench.vllm.runner import ProcessEngineFactory, run_benchmark
from verity_bench.vllm.table import render


def _spec_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec", help="a spec under benchmarks/specs, or its path")


def _plan(args: Any) -> int:
    print(json.dumps(jobs.plan_json(jobs.make_plan(args.spec)), indent=1, sort_keys=True))
    return 0


def _check(args: Any) -> int:
    plan = jobs.make_plan(args.spec)
    done = sum(item.passing_run is not None for item in plan.points)
    print(f"{plan.spec.id}: {done}/{len(plan.points)} points have a passing result")
    for item in plan.points:
        print(f"{item.point.id}\t{item.identity}\t{item.passing_run or 'missing'}")
    return 0 if done == len(plan.points) else 1


def _run(args: Any) -> int:
    plan = jobs.make_plan(args.spec)
    return jobs.dispatch(plan, args.spec)


def _point(args: Any) -> int:
    root = jobs.repository_root()
    plan = jobs.make_plan(args.spec, root=root, lookup=lambda _identity, _commit: None)
    item = next((candidate for candidate in plan.points if candidate.point.id == args.point), None)
    if item is None:
        raise ValueError(f"point {args.point!r} is not in spec {args.spec!r}")
    if item.identity != args.identity:
        raise ValueError(f"point identity changed: expected {item.identity}, invocation named {args.identity}")
    spec = plan.spec
    preset, eager = served(spec.preset, spec.eager)
    arms = resolve_arms(spec.arms, spec.arm_modules)
    doc = run_benchmark(
        preset,
        arms,
        (item.point,),
        sweep=spec.sweep.id,
        reps=spec.reps,
        factory=ProcessEngineFactory(eager=eager),
        git_commit=plan.toolchain.commit,
        config_identity={
            "resolved_spec_sha256": plan.spec_sha256,
            "point_identity": item.identity,
            "uv_lock_sha256": plan.toolchain.uv_lock_sha256,
        },
    )
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR", ".")).resolve()
    out = Path(os.environ.get("OUT", run_dir / "out")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write(doc, out / "benchmark.json")
    (out / "table.md").write_text(render(doc), encoding="utf-8")
    jobs.write_resolved(plan, out / "resolved-spec.json")
    passed = all(g["passed"] for g in doc["gates"])
    envelope = {
        "schema": "research/result/v0.1",
        "kind": "verity-vllm-benchmark/v1",
        "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
        "workload_fingerprint": {
            "identity": item.identity,
            "resolved_spec_sha256": plan.spec_sha256,
            "point": item.point.to_json(),
        },
        "validation": {
            "status": "passed" if passed else "failed",
            "checker": "verity-bench token and arm gates",
            "detail": f"{sum(g['passed'] for g in doc['gates'])}/{len(doc['gates'])} gates passed",
            "evidence": ["out/benchmark.json"],
        },
        "measurements": [],
        "measurement_files": ["out/benchmark.json"],
        "artifacts": ["out/table.md", "out/resolved-spec.json"],
    }
    target = run_dir / "result.json"
    temporary = run_dir / f"result.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(envelope, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(render(doc), end="")
    return 0 if passed else 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verity-bench")
    commands = parser.add_subparsers(dest="command", required=True, metavar="BACKEND")

    vllm = commands.add_parser("vllm", help="measure serving overhead through the vLLM integration")
    vllm_cli.add_arguments(vllm)
    vllm.set_defaults(run=vllm_cli.run, command_parser=vllm)

    plan = commands.add_parser("plan", help="expand a spec and report passing points without running")
    _spec_argument(plan)
    plan.set_defaults(run=_plan, command_parser=plan)

    run = commands.add_parser("run", help="launch one recorded research run per incomplete point")
    _spec_argument(run)
    run.set_defaults(run=_run, command_parser=run)

    check = commands.add_parser("check", help="report whether every point has a passing result")
    _spec_argument(check)
    check.set_defaults(run=_check, command_parser=check)

    compare = commands.add_parser("compare", help="compare results only when their identities agree")
    result.add_judge_arguments(compare)
    compare.set_defaults(run=result.run_judge, command_parser=compare)

    point = commands.add_parser("_vllm-point", help=argparse.SUPPRESS)
    point.add_argument("--spec", required=True)
    point.add_argument("--point", required=True)
    point.add_argument("--identity", required=True)
    point.set_defaults(run=_point, command_parser=point)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.run(args))
    except ValueError as error:
        args.command_parser.error(str(error))
        raise AssertionError("argparse.error returned") from error


if __name__ == "__main__":
    sys.exit(main())
