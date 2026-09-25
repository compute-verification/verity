"""Store declaration for the Flock backend's runs (research.store README §6.1): ``backends/flock/pod/20-pure.sh`` with its
settings as ``K=V`` arguments.

    research run --on M --project verity --source . --cwd source --tool flock_pure --campaign flock-backend -- \\
        bash backends/flock/pod/20-pure.sh REL=bf16-hopper "MODE=selftest sweep" START=1024 RUNS=5
"""
from __future__ import annotations

from research.store.tool import Tool

KEYS = ("REL", "MODE", "SELF_VUS", "START", "MAX", "RUNS", "VERIFIER", "SERVE_VUS")


def parse(argv: list[str]) -> dict:
    out: dict = {"REL": "bf16-hopper", "MODE": "selftest sweep"}
    for t in argv:
        k, eq, v = t.partition("=")
        if eq and k in KEYS:
            out[k] = v
    return out


FLOCK_PURE = Tool(
    name="flock_pure", version="1",
    command=["bash", "backends/flock/pod/20-pure.sh"],
    closure=["backends/flock/**", "backends/numerical/**", "backends/direct/ligero/relations.py", "backends/direct/ligero/hashauth.py",
             "backends/direct/ligero/auth.py", "backends/direct/ligero/timing_guard.py", "packages/verity/src/verity/**"],
    key_params=["REL", "MODE", "SELF_VUS", "START", "MAX", "RUNS", "VERIFIER", "SERVE_VUS"],
    key_conditions=["hw.family"],
    caches=[],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse,
)
