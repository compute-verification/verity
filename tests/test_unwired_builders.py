"""The unwired population builders (`population_program`, `mixed_population_program`, `population`, at v0 or v1) rebuild a
recorded run's Program at the version its record names; every Program for a new run comes from `wired_population_program`.
`CALLERS` is every file outside `tests/` that names one today, and only shrinks."""

from __future__ import annotations

import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("adapters", "benchmarks", "calibration", "infra", "tools", "verity")
NAMES = frozenset({"population_program", "mixed_population_program"})
HOME = "verity/circuits/population.py"
KINDS = {
    "replay": "rebuilds a recorded run's Program at the version or digest its record names",
    "both": "replays a recorded run and builds for a new run; its new-run use still to move",
    "new-run": "builds a Program for a new run; still to move to `wired_population_program`",
    "re-export": "names the builders for its own callers, who are listed",
    "check": "a circuit-check binding of the builders' Definitions",
}

CALLERS: dict[str, str] = {
    "adapters/vllm/verity_vllm/pipeline/boundary_rows.py": "new-run",
    "adapters/vllm/verity_vllm/pipeline/interior_rows.py": "new-run",
    "benchmarks/flock_s8_block/program.py": "new-run",
    "benchmarks/one_stage/a0.py": "new-run",
    "benchmarks/one_stage/a2.py": "new-run",
    "benchmarks/one_stage/a3.py": "new-run",
    "benchmarks/one_stage/a5_network_attester.py": "new-run",
    "benchmarks/one_stage/a5_topp.py": "new-run",
    "benchmarks/one_stage/a5_tp_gates.py": "new-run",
    "benchmarks/one_stage/a6.py": "new-run",
    "benchmarks/one_stage/a7.py": "new-run",
    "benchmarks/one_stage/forged_proofs.py": "new-run",
    "benchmarks/one_stage/form_refusals.py": "new-run",
    "benchmarks/one_stage/r2_population.py": "both",
    "benchmarks/one_stage/replay.py": "replay",
    "benchmarks/one_stage/served_cases.py": "replay",
    "benchmarks/one_stage/toy.py": "new-run",
    "benchmarks/proof_throughput/verity_proof_throughput/population.py": "replay",
    "verity/circuits/ops/boolean/gemm.circuit-check.py": "check",
    "verity/circuits/population.circuit-check.py": "check",
    "verity/commitments/periods/population.py": "new-run",
    "verity/commitments/periods/rows.py": "new-run",
    "verity/protocols/proofs/flock/prover/python/verity_flock/rec_bindings.py": "replay",
    "verity/protocols/proofs/flock/prover/python/verity_flock/rec_outer.py": "replay",
    "verity/protocols/proofs/spot/verifier/one_stage/partition.py": "re-export",
    "verity/protocols/proofs/spot/verifier/service.py": "new-run",
    "verity/protocols/resource_accounting/compute/pouw/prover/circuit/h1tile.circuit-check.py": "check",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk/served_commit.py": "replay",
    "verity/protocols/resource_accounting/network/prover/network_attester_zk.py": "new-run",
    "verity/protocols/resource_accounting/network/prover/scan_zk.py": "new-run",
    "verity/protocols/resource_accounting/network/verifier/service.py": "new-run",
    "verity/protocols/resource_accounting/network/verifier/tie_service.py": "new-run",
    "verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts/record_difftest_vectors.py":
        "new-run",
}


def _names_a_builder(text: str, path: str) -> bool:
    """Whether a module names an unwired builder in code (not prose): an import, a name or an attribute of one, `population`
    as imported from `verity.circuits.population`."""
    for n in ast.walk(ast.parse(text, path)):
        if isinstance(n, ast.Name) and n.id in NAMES or isinstance(n, ast.Attribute) and n.attr in NAMES:
            return True
        if isinstance(n, ast.ImportFrom) and any(a.name in NAMES or a.name == "population"
                                                 and n.module == "verity.circuits.population" for a in n.names):
            return True
    return False


def callers() -> set[str]:
    out = set()
    for top in SCANNED:
        for d, dirs, files in os.walk(ROOT / top):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("tests", "__pycache__", "node_modules")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = Path(d) / f
                rel = p.relative_to(ROOT).as_posix()
                text = p.read_text(encoding="utf-8")
                if rel != HOME and "population" in text and _names_a_builder(text, rel):
                    out.add(rel)
    return out


def test_only_the_listed_files_call_the_unwired_population_builders():
    """A new caller builds its Program with `wired_population_program`, or, rebuilding a recorded run's, goes on `CALLERS`
    as ``replay``; a listed file that no longer names a builder leaves the list."""
    assert set(CALLERS.values()) <= set(KINDS), sorted(set(CALLERS.values()) - set(KINDS))
    got = callers()
    new, gone = sorted(got - set(CALLERS)), sorted(set(CALLERS) - got)
    assert not new, (f"{new} name an unwired population builder: build a new run's Program with "
                     "`verity.circuits.population.wired_population_program`; only a recorded run's replay rebuilds one, "
                     "and is listed in `CALLERS` as replay")
    assert not gone, f"{gone} no longer name an unwired population builder: delete their entries from `CALLERS`"
