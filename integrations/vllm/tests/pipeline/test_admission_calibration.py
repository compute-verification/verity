"""The admission planner's facts live in `engine.profiles.admission_calibration`, and the planner plans exactly as it did
with them inline: the tables, the unknown-model notes and the plans here are pinned from the planner before the move."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity_vllm.engine.profiles import admission_calibration as C
from verity_vllm.pipeline.telemetry import admission as A

UNKNOWN = "granite-3b__bf16__h100__tp1__b2__i64__o8__mixed__t1__eager"
KNOWN = "llama32-1b__bf16__h100__tp1__b2__i64__o8__mixed__t1__eager"


def _digest(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _workload(row: str) -> dict:
    return {"sweep": {"row_id": row, "batch": 2, "max_input": 64, "max_output": 8}, "requests": [{"prompt_len": 64, "max_tokens": 8}] * 2}


def test_the_planner_reads_the_profile_tables():
    assert (A.MODELS, A.DENSE_DEFAULT, A.CALIBRATION) == (C.MODELS, C.DENSE_DEFAULT, C.CALIBRATION)
    assert _digest({"MODELS": C.MODELS, "DENSE_DEFAULT": C.DENSE_DEFAULT, "CALIBRATION": C.CALIBRATION}) == "118a90b99cd0cf91"
    assert C.DEFAULT_MODEL in C.MODELS


def test_unknown_model_notes_name_the_dense_default(tmp_path: Path):
    (tmp_path / "commit.log").write_text(f"x --workload workloads/{UNKNOWN}.json\n")
    assert A.row_config_from_attempt(str(tmp_path)).notes == [
        "modelled without gc.freeze before the fork (pre-R19 record; --no-fork-gc-freeze equivalent)",
        "model 'granite-3b' not in MODELS: dense defaults (phi3 coefficients) used",
        "host limit inferred from the sampler's cgroup peak 0 MiB -> 119209 MiB class"]
    assert A.row_config_from_workload(_workload(UNKNOWN)).notes == [
        "no calibration for model 'granite-3b': dense default coefficients (phi3's, the largest measured dense set) -- every term is an estimate"]


def test_plans_are_pinned():
    plans = {"unknown": A.predict(A.row_config_from_workload(_workload(UNKNOWN))), "known": A.predict(A.row_config_from_workload(_workload(KNOWN)))}
    assert _digest(plans) == "8c07793f04cfc08d"
