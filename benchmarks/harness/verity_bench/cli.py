"""The command line for Verity's shared benchmark harnesses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verity_bench import (
    breakdown,
    jobs,
    lean_verify,
    overhead,
    overhead_records,
    prove_records,
    served_table,
    settings,
    steady_plan,
    steady_spec,
)
from verity_bench.vllm import cli as vllm_cli
from verity_bench.vllm import result
from verity_bench.vllm.arms import resolve_arms
from verity_bench.vllm.presets import local_checkpoint, served
from verity_bench.vllm.result import write
from verity_bench.vllm.runner import ProcessEngineFactory, run_benchmark
from verity_bench.vllm.table import render


def _spec_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec", help="a spec under benchmarks/harness/specs, or its path")
    parser.add_argument("--local-only", action="store_true",
                        help="find passing runs in this machine's catalog alone, not the store's remote")


def _make_plan(args: Any) -> jobs.Plan:
    """The spec's plan, by its schema: a steady-state spec's points (`steady_plan`), or a vLLM spec's."""

    root = jobs.repository_root()
    if steady_spec.is_steady(args.spec):
        return steady_plan.make_plan(args.spec, root=root, local_only=args.local_only)
    return jobs.make_plan(args.spec, root=root, lookup=jobs.StoreCompletion(root=root, local_only=args.local_only))


def _plan(args: Any) -> int:
    print(json.dumps(jobs.plan_json(_make_plan(args)), indent=1, sort_keys=True))
    return 0


def _check(args: Any) -> int:
    plan = _make_plan(args)
    done = sum(item.passing_run is not None for item in plan.points)
    print(f"{plan.spec.id}: {done}/{len(plan.points)} points have a passing result")
    for item in plan.points:
        print(f"{item.point.id}\t{item.identity}\t{item.passing_run or 'missing'}" + (f"\t{item.why}" if item.why else ""))
    return 0 if done == len(plan.points) else 1


def _run(args: Any) -> int:
    plan = _make_plan(args)
    if isinstance(plan.spec, steady_spec.SteadySpec):
        return steady_plan.dispatch(plan, args.spec, direct=args.direct)
    return jobs.dispatch(plan, args.spec, direct=args.direct)


def _point(args: Any) -> int:
    root = jobs.repository_root()
    plan = jobs.make_plan(args.spec, root=root, lookup=lambda _identity, _commit: None)
    item = next((candidate for candidate in plan.points if candidate.point.id == args.point), None)
    if item is None:
        raise ValueError(f"point {args.point!r} is not in spec {args.spec!r}")
    if item.identity != args.identity:
        raise ValueError(f"point identity changed: expected {item.identity}, invocation named {args.identity}")
    run_dir = Path(settings.RESEARCH_RUN_DIR).resolve()
    # A requeued (preempted) point reruns in its run directory: an earlier attempt's verdict must not outlive it.
    (run_dir / "result.json").unlink(missing_ok=True)
    spec = plan.spec
    jobs.check_point_environment(spec, settings.environ())
    if spec.preset.model_digest is not None:
        found = local_checkpoint(spec.preset, spec.preset.model).model_digest
        if found != spec.preset.model_digest:
            raise ValueError(f"checkpoint {spec.preset.model} hashes to {found}, not the spec's model_digest "
                             f"{spec.preset.model_digest}")
    preset, eager = served(spec.preset, spec.eager)
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    doc = run_benchmark(
        preset,
        arms,
        (item.point,),
        sweep=spec.sweep.id,
        reps=spec.reps,
        warmups=spec.warmups,
        max_warmups=spec.max_warmups,
        factory=ProcessEngineFactory(eager=eager),
        git_commit=plan.toolchain.commit,
        config_identity={
            "resolved_spec_sha256": plan.spec_sha256,
            "point_identity": item.identity,
            "uv_lock_sha256": plan.toolchain.uv_lock_sha256,
        },
    )
    out = Path(settings.OUT or run_dir / "out").resolve()
    out.mkdir(parents=True, exist_ok=True)
    write(doc, out / "benchmark.json")
    (out / "table.md").write_text(render(doc), encoding="utf-8")
    jobs.write_resolved(plan, out / "resolved-spec.json")
    (out / "settings.json").write_text(json.dumps(settings.record(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    passed = all(g["passed"] for g in doc["gates"])
    envelope = {
        "schema": "research/result/v0.1",
        "kind": "verity-vllm-benchmark/v1",
        "run_id": settings.RESEARCH_RUN_ID,
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
        "artifacts": ["out/table.md", "out/resolved-spec.json", "out/settings.json"],
    }
    target = run_dir / "result.json"
    temporary = run_dir / f"result.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(envelope, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(render(doc), end="")
    return 0 if passed else 3


def _table(args: Any) -> int:
    doc = result.merge([json.loads(path.read_text()) for path in args.results])
    if args.out is not None:
        write(doc, args.out)
    print(render(doc), end="")
    return 0


def _served_table(args: Any) -> int:
    table = served_table.build(local_only=args.local_only)
    if args.json:
        print(json.dumps(table, indent=1, sort_keys=True))
    else:
        print(served_table.render(table), end="")
    return 0


def _breakdown(args: Any) -> int:
    doc = breakdown.from_results(args.results) if args.results else breakdown.build(local_only=args.local_only)
    if args.json:
        print(json.dumps(doc, indent=1, sort_keys=True))
    else:
        print(breakdown.render(doc), end="")
    return 0


def _overhead_store(args: Any) -> Any:
    completion = jobs.StoreCompletion(root=jobs.repository_root(), local_only=args.local_only)
    if not args.local_only:
        completion.pull_labels()
    return completion.store


def _overhead_node(args: Any) -> str:
    node = args.on or overhead_records.NODES.get(args.card)
    if node is None:
        raise ValueError(f"--on NODE: the {args.card} has no default node")
    return node


def _overhead_points(args: Any) -> int:
    if args.store:
        print(overhead_records.store_points(_overhead_store(args), args.store, by=args.by))
        return 0
    return overhead_records.launch_points(jobs.repository_root(), _overhead_node(args), args.card, args.model,
                                          args.precision, args.minutes)


def _overhead_run(args: Any) -> int:
    if args.record:
        for why in overhead_records.record_run(_overhead_store(args), args.record, by=args.by):
            print(f"not of record at {why}")
        print(f"{args.record}: {served_table.RUN_OF_RECORD}={overhead_records.RUN_RECORD}")
        return 0
    if args.scheme is None:
        raise ValueError("--scheme S, or --record RUN")
    return overhead_records.launch_run(_overhead_store(args), jobs.repository_root(), _overhead_node(args), args.card,
                                       args.scheme, args.point, args.points, args.minutes, cubin=args.cubin)


def _overhead_table(args: Any) -> int:
    table = overhead_records.build(local_only=args.local_only)
    if args.json:
        print(json.dumps(table, indent=1, sort_keys=True))
    else:
        print(overhead_records.render(table), end="")
    return 0


def _overhead_parser(commands: Any) -> None:
    door = commands.add_parser("overhead", help="the one front door for overhead: operating points, timed γ, the table")
    sub = door.add_subparsers(dest="overhead_command", required=True, metavar="STEP")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--card", default="h200", choices=list(overhead.CARDS))
        p.add_argument("--on", metavar="NODE", help="the GPU node (default: the card's, overhead_records.NODES)")
        p.add_argument("--minutes", type=float, help="the run's bound, which its GPU lease enforces")
        p.add_argument("--as", dest="by", default=overhead_records.BY, help="who labels the store")
        p.add_argument("--local-only", action="store_true", help="read this machine's catalog alone, not the store's remote")

    points = sub.add_parser("points", help="sweep m on the card for C and L (a research run), or --store a finished sweep")
    common(points)
    points.add_argument("--model", default=overhead.KERNEL_MODEL, choices=list(overhead.MODELS))
    points.add_argument("--precision", default="fp8")
    points.add_argument("--store", metavar="RUN", help="store RUN's points as the next version of the card's, of record")
    points.set_defaults(run=_overhead_points, command_parser=points)

    run = sub.add_parser("run", help="time a scheme against stock at the stored points (a research run), or --record one")
    common(run)
    run.add_argument("--scheme", choices=list(overhead.SCHEMES))
    run.add_argument("--point", default="all", choices=[*overhead.POINTS, "all"])
    run.add_argument("--points", metavar="ART", help="time at this points art instead of the card's of record")
    run.add_argument("--cubin", metavar="ART", help="the art of the cubin the scheme's arm loads, sent with the run (its "
                                                    "arm reads $VERITY_BENCH_CUBIN); without it the run is development only")
    run.add_argument("--record", metavar="RUN", help="label RUN a run of record, once it stands as a whole")
    run.set_defaults(run=_overhead_run, command_parser=run)

    table = sub.add_parser("table", help="γ per scheme, point and card from labelled runs of record alone")
    table.add_argument("--json", action="store_true", help="the rows as JSON (each with of_record and why)")
    table.add_argument("--local-only", action="store_true", help="read this machine's catalog alone, not the store's remote")
    table.set_defaults(run=_overhead_table, command_parser=table)


def _prove(args: Any) -> int:
    doc = prove_records.build(args.system, args.workload, sampling=args.sampling, provers=args.provers,
                              cores=args.cores, local_only=args.local_only)
    if args.steady_rows:
        args.steady_rows.write_text(json.dumps(prove_records.P.steady_rows(doc), indent=1) + "\n")
    if args.json:
        print(json.dumps(doc, indent=1, sort_keys=True))
    else:
        print(prove_records.render(doc, args.rows), end="")
    return 0


def _prove_parser(commands: Any) -> None:
    prove = commands.add_parser("prove", help="a proof system's prover on a workload, a block type a row, from the "
                                              "store's labelled records")
    prove.add_argument("--system", required=True, choices=prove_records.SYSTEMS,
                       help="flock: every block proved; spot: Flock at a sampling plan of record")
    prove.add_argument("--workload", required=True, choices=list(prove_records.WORKLOADS))
    prove.add_argument("--sampling", metavar="ART", help="spot's sampling plan (default: the workload's plan of record); "
                                                         "refused unless it is labelled the plan of record")
    prove.add_argument("--provers", type=int, default=1, metavar="N",
                       help="the prover count each type is priced at: only steady-state rows at N provers measure a "
                            "type (default: 1)")
    prove.add_argument("--cores", type=int, default=prove_records.P.HOST_CORES, metavar="N",
                       help="the CPU cores a card's provers hold on the host each type is priced for: a type is priced "
                            "from steady-state rows timed at N cores, else from those timed at the fewest above N, and "
                            f"never from one timed at fewer (default: {prove_records.P.HOST_CORES}{prove_records.HOST_CORES_ARE})")
    prove.add_argument("--rows", type=int, metavar="N", help="show the N block types that take the most proving time, "
                                                             "the rest in one row (default: every block type)")
    prove.add_argument("--json", action="store_true", help="the table as JSON, every row whole")
    prove.add_argument("--steady-rows", type=Path, metavar="FILE",
                       help="also write FILE: the block types measured at steady state, a row each as the break-even "
                            "planner reads them (benchmarks/spot_figures/breakeven: `--steady FILE`)")
    prove.add_argument("--local-only", action="store_true", help="read this machine's catalog alone, not the store's remote")
    prove.set_defaults(run=_prove, command_parser=prove)

    timed = commands.add_parser("lean-verify", help="the Lean verifier's seconds on one kept proof and the proof's bytes, "
                                                    "as the record `prove` reads")
    lean_verify.add_arguments(timed)
    timed.set_defaults(run=lean_verify.run, command_parser=timed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verity-bench")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    vllm = commands.add_parser("vllm", help="measure serving overhead through the vLLM integration")
    vllm_cli.add_arguments(vllm)
    vllm.set_defaults(run=vllm_cli.run, command_parser=vllm)

    plan = commands.add_parser("plan", help="expand a spec and report passing points without running")
    _spec_argument(plan)
    plan.set_defaults(run=_plan, command_parser=plan)

    run = commands.add_parser("run", help="launch one recorded research run per incomplete point")
    _spec_argument(run)
    run.add_argument("--direct", metavar="NODE", default=None,
                     help="launch each point with `research run --on NODE` through a GPU slot of that node, not the queue")
    run.set_defaults(run=_run, command_parser=run)

    check = commands.add_parser("check", help="report whether every point has a passing result")
    _spec_argument(check)
    check.set_defaults(run=_check, command_parser=check)

    compare = commands.add_parser("compare", help="compare results only when their identities agree")
    result.add_judge_arguments(compare)
    compare.set_defaults(run=result.run_judge, command_parser=compare)

    table = commands.add_parser("table", help="one table from one spec's per-point results, when their identities agree")
    table.add_argument("results", type=Path, nargs="+")
    table.add_argument("--out", type=Path, default=None, help="also write the merged result here")
    table.set_defaults(run=_table, command_parser=table)

    served = commands.add_parser("served-table", help="every protocol's served overhead, from the store's runs of record")
    served.add_argument("--json", action="store_true", help="the rows as JSON, not markdown")
    served.add_argument("--local-only", action="store_true",
                        help="read this machine's catalog alone, not the store's remote")
    served.set_defaults(run=_served_table, command_parser=served)

    split = commands.add_parser("breakdown", help="where the bandwidth audit's time goes, by span, from its breakdown spec")
    split.add_argument("results", type=Path, nargs="*",
                       help="result files (benchmark.json) to read instead of the store's newest passing runs")
    split.add_argument("--json", action="store_true", help="the breakdown as JSON, not markdown")
    split.add_argument("--local-only", action="store_true",
                       help="read this machine's catalog alone, not the store's remote")
    split.set_defaults(run=_breakdown, command_parser=split)

    _overhead_parser(commands)
    _prove_parser(commands)

    point = commands.add_parser("_vllm-point", help=argparse.SUPPRESS)
    point.add_argument("--spec", required=True)
    point.add_argument("--point", required=True)
    point.add_argument("--identity", required=True)
    point.set_defaults(run=_point, command_parser=point)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings.validate(of_record=args.command == "_vllm-point")
        return int(args.run(args))
    except ValueError as error:
        args.command_parser.error(str(error))
        raise AssertionError("argparse.error returned") from error


if __name__ == "__main__":
    sys.exit(main())
