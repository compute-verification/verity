"""Evidence-store declaration for one vLLM benchmark sweep point."""

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
        "integrations/vllm/**",
        "tools/config/**",
        "uv.lock",
    ],
    key_params=["identity"],
    key_conditions=["hw.gpu.family"],
    produces=["verity-vllm-benchmark/v1", "run-files/v1"],
    parse=parse_point,
)
