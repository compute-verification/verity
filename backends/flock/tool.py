"""Store declaration for the Flock backend's runs (research.store README §6.1): ``backends/flock/pod/20-pure.sh`` with its
settings as ``K=V`` arguments.

    research run --on M --project verity --source . --cwd source --tool flock_pure --campaign flock-backend -- \\
        bash backends/flock/pod/20-pure.sh REL=bf16-hopper "MODE=selftest sweep" START=1024 RUNS=5
"""
from __future__ import annotations

from research.pythonpath import package_paths, workspace_root
from research.store.tool import Tool

TREE = workspace_root(__file__)

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
             "backends/direct/ligero/auth.py", "backends/direct/ligero/timing_guard.py", *package_paths(TREE, "verity", "**")],
    key_params=["REL", "MODE", "SELF_VUS", "START", "MAX", "RUNS", "VERIFIER", "SERVE_VUS"],
    key_conditions=["hw.family"],
    caches=[],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse,
)


CLASS_SWEEP_KEYS = ("GRAPH", "DEFS", "SHAPES", "SELECT", "MODULES", "BATCH", "WARM", "RUNS", "SELFTEST", "SELFTEST_GPU", "SELFTEST_CASES", "GPU", "PROTOCOL",
                    "SELFTEST_LARGEST", "BATCH_ANDS", "MAX_STATEMENT_BITS")


def parse_class_sweep(argv: list[str]) -> dict:
    out: dict = {"BATCH": "auto", "WARM": "1", "RUNS": "3", "GPU": "1", "PROTOCOL": "sweep-v0"}
    for t in argv:
        k, eq, v = t.partition("=")
        if eq and k in CLASS_SWEEP_KEYS:
            out[k] = v
    return out


#: ``backends/flock/pod/70-class-sweep.sh``: unit shapes of a served row's Q_word partition (its program graph, or a list of Call
#: Definitions), each a verity/flock-circuit statement proved on the pod (verity_flock.class_statement); GRAPH, DEFS and SHAPES name
#: content (their files' digests key the run).
FLOCK_CLASS_SWEEP = Tool(
    name="flock_class_sweep", version="1",
    command=["bash", "backends/flock/pod/70-class-sweep.sh"],
    closure=["backends/flock/**", "backends/numerical/python/**", *package_paths(TREE, "verity", "**"), "integrations/vllm/verity_vllm/**",
             "tools/circuit_check/src/**"],
    key_params=list(CLASS_SWEEP_KEYS),
    param_artifacts=["GRAPH", "DEFS", "SHAPES"],
    key_conditions=["hw.family"],
    caches=[],
    produces=["class-sweep/v1", "run-files/v1"],
    parse=parse_class_sweep,
)
