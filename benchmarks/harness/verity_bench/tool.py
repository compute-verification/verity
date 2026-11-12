"""Evidence-store declarations for the harness's points: one vLLM benchmark sweep point, one steady-state point."""

from research.store.tool import Tool

POINT_FLAGS = ("--spec", "--point", "--identity")


def parse_point(argv: list[str]) -> dict[str, str]:
    """The point's flags wherever they sit: a queued run records its wrapper's argv (`gpu-lease 1 --wait ... -- bash
    benchmarks/harness/run_point.sh --spec S --point P --identity I`), which the generic parse reads as positionals after
    `--`, so `identity`, the key the resume lookup matches on, would never be recorded."""

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
        "benchmarks/specs/**",
        "adapters/vllm/**",
        "verity/configuration/config/**",
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
        "benchmarks/specs/**",
        "benchmarks/one_stage/server_rate.py",
        "benchmarks/served_units/stage.py",
        "verity/configuration/config/**",
        "verity/protocols/flock/pod/**",
        "verity/protocols/flock/prover/**",
    ],
    key_params=["identity"],
    produces=["verity-steady-point/v1", "run-files/v1"],
    parse=parse_point,
)
