"""Store declaration for the verifier-of-record agreement job (research.store README §6.1): ``ci-pod.sh`` with its settings as
``K=V`` arguments, the shipped bundle as its input.

    research run --on M --project verity --source . --send ci-bundle.tar.gz --tool flock_agreement --campaign flock-verifier \\
        --require-result -- bash verity/ml/flock/verifier/ci-pod.sh SEED=20260926 FUZZ=20 JOBS=4
"""
from __future__ import annotations

from research.store.tool import Tool

KEYS = ("SEED", "FUZZ", "JOBS", "SETS")


def parse(argv: list[str]) -> dict:
    out: dict = {"SEED": "20260926", "FUZZ": "20", "JOBS": "4", "SETS": ""}
    for t in argv:
        k, eq, v = t.partition("=")
        if eq and k in KEYS:
            out[k] = v
    return out


FLOCK_AGREEMENT = Tool(
    name="flock_agreement", version="1",
    command=["bash", "verity/ml/flock/verifier/ci-pod.sh"],
    closure=["verity/ml/flock/verifier/**", "verity/core/service/flock/**"],
    key_params=["SEED", "FUZZ", "SETS"],
    key_conditions=[],
    caches=[],
    produces=["verification-verdict/v1", "run-files/v1"],
    parse=parse,
)
