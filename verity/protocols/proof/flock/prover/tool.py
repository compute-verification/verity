"""Store declarations for the Flock backend's runs (research.store README §6.1)."""
from __future__ import annotations

from research.pythonpath import package_paths, workspace_root
from research.store.tool import Tool

TREE = workspace_root(__file__)
#: core and the catalog, the kernels and the verifiers beside it, wherever the tree keeps them
CORE = [p for n in ("verity", "verity.circuits", "verity.circuits.kernels") for p in package_paths(TREE, n, "**")]

CLASS_SWEEP_KEYS = ("GRAPH", "DEFS", "SHAPES", "SELECT", "MODULES", "BATCH", "WARM", "RUNS", "SELFTEST", "SELFTEST_GPU", "SELFTEST_CASES", "GPU", "PROTOCOL",
                    "SELFTEST_LARGEST", "BATCH_ANDS", "MAX_STATEMENT_BITS")


def parse_class_sweep(argv: list[str]) -> dict:
    out: dict = {"BATCH": "auto", "WARM": "1", "RUNS": "3", "GPU": "1", "PROTOCOL": "sweep-v0"}
    for t in argv:
        k, eq, v = t.partition("=")
        if eq and k in CLASS_SWEEP_KEYS:
            out[k] = v
    return out


#: ``verity/protocols/proof/flock/prover/pod/70-class-sweep.sh``: unit shapes of a served row's Q_word partition (its program graph, or a list of Call
#: Definitions), each a verity/flock-circuit statement proved on the pod (verity_flock.class_statement); GRAPH, DEFS and SHAPES name
#: content (their files' digests key the run).
FLOCK_CLASS_SWEEP = Tool(
    name="flock_class_sweep", version="1",
    command=["bash", "verity/protocols/proof/flock/prover/pod/70-class-sweep.sh"],
    closure=["verity/protocols/proof/flock/prover/**", "verity/protocols/proof/flock/prover/pod/**", "verity/protocols/proof/flock/prover/agreement/**", "verity/protocols/proof/flock/prover/tool.py", "verity/protocols/proof/flock/verifier/**", "benchmarks/numerical/python/**", *CORE,
             "adapters/vllm/verity_vllm/**", "tools/verity/circuit_check/src/**"],
    key_params=list(CLASS_SWEEP_KEYS),
    param_artifacts=["GRAPH", "DEFS", "SHAPES"],
    key_conditions=["hw.family"],
    caches=[],
    produces=["class-sweep/v1", "run-files/v1"],
    parse=parse_class_sweep,
)
