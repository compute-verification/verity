"""Shared pieces for the bench-contract tests: a conforming result and the paths of the workload scripts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from verity_numerical.bench import contract

REPO = Path(__file__).resolve().parents[4]
JUDGE = REPO / "benchmarks" / "judge.py"

CONFORMING_FINGERPRINT: dict[str, Any] = {
    "profile": contract.PROFILE,
    "domain": contract.DOMAIN,
    "K": 16,
    "B": 1,
    "proof_class": "NON_ZK_PROOF",
    "authentication": "included",
    "zk_mode": "none",
    "security": {"target": None, "note": "synthetic"},
    "hardware": {"gpu": {"name": "NVIDIA GeForce RTX 4090", "driver": "580.159.04", "cuda": None}, "cpu": {"model": "test cpu"}},
    "software": {"backend": {"name": "synthetic", "version": "0", "commit": None}, "toolchain": {"python": "3.12"}},
    "preprocessing": "none",
    "retention": "retained",
    # null: not measured on the shared input set (bench-instances/v1); see contract.instances_ref
    "instances": None,
}


def conforming_result(run_id: str = "r-test", **overrides: Any) -> dict[str, Any]:
    """A result that passes ``contract.validate``: buckets split, one joint, everything summing to ``t.total``."""

    fp = copy.deepcopy(CONFORMING_FINGERPRINT)
    fp.update(overrides)
    measurements = [
        contract.measurement("t.witness", 1.0),
        contract.measurement(contract.joint("encoding_commitment", "arithmetic"), 2.0),
        contract.measurement("t.lookup", 0.5),
        contract.measurement("t.zk_additional", 0.0),
        contract.measurement("t.serialization", 0.25),
        contract.measurement("t.total", 4.0),
        contract.measurement("vu.count", 1),
        contract.measurement("vu.seconds_per_vu", 4.0),
        contract.measurement("vu.per_second", 0.25),
        contract.measurement("transcript.bytes", 1000),
        contract.measurement("verifier.seconds", 0.1),
        contract.measurement("fixed.setup_seconds", 10.0),
        contract.measurement("own_name", 3, "cycles"),
    ]
    return {
        "schema": contract.RESULT_SCHEMA,
        "run_id": run_id,
        "workload_fingerprint": fp,
        "validation": {"status": "passed", "checker": "synthetic", "detail": "ok", "evidence": []},
        "measurements": measurements,
        "measurement_files": [],
        "artifacts": [],
    }


def write_run(root: Path, run_id: str, result: dict[str, Any]) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return d


@pytest.fixture
def conforming() -> dict[str, Any]:
    return conforming_result()
