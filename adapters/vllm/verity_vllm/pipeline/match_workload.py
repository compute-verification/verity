"""What the Match driver (`verity_vllm.pipeline.match`) reads of its workload file: the longest `max_tokens` and the engine
steps the workload expects, its engine arguments, the MoE construction and the execution label it declares, with the
`enforce_eager` that label implies.  `fold_construction_args` gives the flags a fold of the workload's log is run with, from
the workload and the Build's Programs."""

from __future__ import annotations

import json
from pathlib import Path

from verity_vllm.config import RowEngine, row_execution
from verity_vllm.engine.served import ENFORCE_EAGER
from verity_vllm.observe.fold.supplied import program_registered_values
from verity_vllm.program.sampling_event import program_sampler_construction

__all__ = ["fold_construction_args"]


def _workload_doc(path: str | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, RecursionError):   # unreadable, not JSON, or nested past what the parser reads
        return None


def _workload_max_tokens(path: str | None) -> int | None:
    doc = _workload_doc(path)
    if not doc:
        return None
    mt = [int(r["max_tokens"]) for r in doc.get("requests", []) if r.get("max_tokens") is not None]
    return max(mt) if mt else None


def _workload_expected_steps(path: str | None) -> int | None:
    """The workload's declared `expected_engine_steps` (an upper bound: every request runs to max_tokens under the declared
    arrivals), else max(max_tokens)."""
    doc = _workload_doc(path)
    if doc and doc.get("expected_engine_steps"):
        return int(doc["expected_engine_steps"])
    return _workload_max_tokens(path)


def _workload_engine_args(path: str | None) -> dict:
    return dict(RowEngine.of(_workload_doc(path) or {}).args)


# torch-free mirror of vllm_adapter.execution_of_workload (this driver runs on the laptop): the workload's
# declared execution label (explicit `execution` / `sweep.execution`, else the row-id suffix) and the enforce_eager it implies
# (`engine.served.ENFORCE_EAGER`).  The GPU phases derive the same from the workload themselves and refuse a contradicting engine arg
# (build_engine).


def _workload_moe_construction(path: str | None) -> str | None:
    """The workload's DECLARED MoE construction (`engine.target.moe_construction`; None = undeclared = the default),
    validated by the frontend's own table so a misspelling fails here by name and is never folded as MOE-01 silently."""
    mc = (RowEngine.of(_workload_doc(path) or {}).target or {}).get("moe_construction")
    if mc is None:
        return None
    from verity_vllm.program.frontend.target_profile import TargetProfile
    if mc not in TargetProfile.MOE_CONSTRUCTIONS:
        raise SystemExit(f"workload engine.target.moe_construction {mc!r} is not one of {sorted(TargetProfile.MOE_CONSTRUCTIONS)}")
    return str(mc)


def fold_construction_args(workload: str | None, program_dirs) -> list[str]:
    """The fold worker's construction flags: the MoE scope under the workload's DECLARED construction (manifest
    target.moe_construction), the top-p select under the one the Build's Programs declare (a single-request select has no shared-greedy
    form) and the values they register; never off the trace.  The single-rank driver passes them from --program-dir, the TP fold Match
    (`tp.fold_match`) from each rank's Build dirs; a fold without them binds the profile's default select."""
    mc, sc, rv = _workload_moe_construction(workload), program_sampler_construction(program_dirs), program_registered_values(program_dirs)
    return (["--moe-construction", mc] if mc else []) + (["--sampler-construction", sc] if sc else []) + (["--registered-values"] if rv else [])


def _workload_execution(path: str | None) -> str | None:
    doc = _workload_doc(path) or {}
    sw = doc.get("sweep")
    if not isinstance(sw, dict):
        sw = {}
    for v in (doc.get("execution"), sw.get("execution")):
        if isinstance(v, str) and v:
            return v
    rid = sw.get("row_id") or doc.get("row_id")
    return row_execution(rid) if isinstance(rid, str) else None


def _workload_enforce_eager(path: str | None) -> bool:
    ex = _workload_execution(path)
    if ex is None:
        return True                                   # undeclared execution: the historical eager default (recorded as declared null)
    if ex not in ENFORCE_EAGER:
        raise SystemExit(f"workload {path}: execution label {ex!r} is not modelled by the harness (known: {sorted(ENFORCE_EAGER)})")
    return ENFORCE_EAGER[ex]
