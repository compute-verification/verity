"""Evidence-store declarations for a prover point (``job.sh``) and its verifier point (``verify.sh``)."""
from research.store.tool import Tool

KEYS = ("WORKLOAD", "B", "WARM", "RUNS", "IDENTITY", "BUNDLE", "PROVE_RUN")


def parse_point(argv: list[str]) -> dict[str, str]:
    """The point's ``KEY=VALUE`` arguments wherever they sit (a queued run records its wrapper's argv), lower-cased:
    ``identity`` is the key the resume lookup matches on, ``workload`` and ``b`` what ``table --store`` selects by."""
    out = {}
    for token in argv:
        key, sep, val = token.partition("=")
        if sep and key in KEYS:
            out[key.lower()] = val
    return out


CLOSURE = ["benchmarks/proof_throughput/**", "benchmarks/numerical/python/verity_numerical/bench/**",
           "verity/ml/flock/**", "verity/ml/circuits/**", "uv.lock"]

PROVER_POINT = Tool(
    name="proof_throughput_prover",
    version="1",
    command=["bash", "benchmarks/proof_throughput/job.sh"],
    closure=CLOSURE,
    key_params=["identity"],
    key_conditions=["hw.gpu.family"],
    produces=["research/result/v0.1", "run-files/v1"],
    parse=parse_point,
)

VERIFIER_POINT = Tool(
    name="proof_throughput_verifier",
    version="1",
    command=["bash", "benchmarks/proof_throughput/verify.sh"],
    closure=[*CLOSURE, "verity/core/proof_system/flock/**"],
    key_params=["bundle"],
    key_conditions=[],
    produces=["research/result/v0.1", "run-files/v1"],
    parse=parse_point,
)
