"""Evidence-store declarations for the harness's points: one vLLM benchmark sweep point, one steady-state point."""

import logging

from research.store.tool import Tool

log = logging.getLogger(__name__)

POINT_FLAGS = ("--spec", "--point", "--identity")


def _launched(argv: list[str]) -> dict[str, str]:
    """The point a `verity-bench run SPEC [--point P] [--again P]` launch runs, planned in the launching checkout (whose
    commit `research run --source` ships) as the run will plan it; nothing where it names no one point."""

    from verity_bench import cli, jobs, steady_plan, steady_spec

    start = next(i for i, token in enumerate(argv) if token.endswith("verity-bench")) + 2
    try:
        args = cli._parser().parse_args(["run", *argv[start:]])
        root = jobs.repository_root()
        plan = (steady_plan.make_plan(args.spec, root=root, lookup=lambda *_: (None, None))
                if steady_spec.is_steady(args.spec) else jobs.make_plan(args.spec, root=root, lookup=lambda *_: None))
        item = jobs.to_run(jobs.again(plan, args.again), args.point)
    except (ValueError, SystemExit) as e:
        log.info("verity-bench run %s names no one point, so its record keys none: %s", " ".join(argv[start:]), e)
        return {}
    return {"spec": args.spec, "point": item.point.id, "identity": item.identity}


def parse_point(argv: list[str]) -> dict[str, str]:
    """The point's flags wherever they sit, or the point a `verity-bench run SPEC` launch runs (`_launched`): `identity`
    is the key the resume lookup matches on, which the generic parse would never record."""

    if any(token.endswith("verity-bench") and argv[i + 1:i + 2] == ["run"] for i, token in enumerate(argv)):
        return _launched(argv)
    out = {}
    for i, token in enumerate(argv[:-1]):
        if token in POINT_FLAGS:
            out[token[2:]] = argv[i + 1]
    return out


VLLM_BENCH_POINT = Tool(
    name="vllm_bench_point",
    version="0",
    command=["bash", "benchmarks/harness/run_point.sh"],
    closure=[
        "benchmarks/harness/**",
        "benchmarks/harness/specs/**",
        "adapters/vllm/**",
        "tools/config/**",
        "uv.lock",
    ],
    key_params=["identity"],
    key_conditions=["hw.gpu.family"],
    produces=["verity-vllm-benchmark/v1", "run-files/v1"],
    parse=parse_point,
)

#: A steady-state point (`verity_bench.steady_spec`). Its identity is the one key: it already digests the files the
#: point's time depends on, so the lookup matches it across commits and on any card of the spec's model.
STEADY_POINT = Tool(
    name="steady_point",
    version="0",
    command=["bash", "benchmarks/harness/steady_point.sh"],
    closure=[
        "benchmarks/harness/**",
        "benchmarks/harness/specs/**",
        "benchmarks/one_stage/server_rate.py",
        "benchmarks/served_units/stage.py",
        "tools/config/**",
        "verity/protocols/proofs/flock/prover/pod/**",
        "verity/protocols/proofs/flock/prover/**",
    ],
    key_params=["identity"],
    produces=["verity-steady-point/v1", "run-files/v1"],
    parse=parse_point,
)
