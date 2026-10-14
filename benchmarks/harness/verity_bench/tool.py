"""Evidence-store declaration for one vLLM benchmark sweep point."""

from research.store.tool import Tool

VLLM_BENCH_POINT = Tool(
    name="vllm_bench_point",
    version="0",
    command=["python", "-m", "verity_bench", "_vllm-point"],
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
)
