"""A sealed bundle's Programs of record, request labels and arrivals, as its serving rows read them (`programs_of_record`,
`request_labels`, `arrivals`).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.log import init_logger
from verity_vllm.pipeline import commit_plan as CP

# The replay driver, its Definition check and codes, the executed prefix, the serving rows and the manifest's format are imported
# inside the functions that read them, as `vu_store` is: the driver's import puts the catalog's Definitions into the Definition
# table, and a process that only imports this module (the Commit before its replay, a deferred Commit) is to hold none of them.

__all__ = ["ProgramSources", "ReplaySources", "arrivals", "programs_of_record", "request_labels"]

logger = init_logger(__name__)


@dataclass
class ReplaySources:
    """Everything the C2 replay of one instrumented run reads.  `weights_of(prefix)`: the weights provider for the Programs' field
    prefix; `weights_check()`: the proof that the slices it serves are the committed weights ({"equal": bool, ...});
    `after_replay`: the VU export hook (in-process only), called as VUX.after_replay is.  `reads`: when a dict, the replay's sample is taken
    again after it, read only, in this process (`plan`), and its record lands here: a forked evaluator's reads of `com` never reach this
    process, so this is what makes `commit.store_dump.touched` see every pick's."""
    args: Any
    out: Path
    pair: int
    com: Any
    rc: Any
    row: dict[str, Any]
    reqs: list[dict[str, Any]]
    sampling: dict[str, Any]
    tokens: dict[str, Any]
    manifest: Any
    stop_of_record: dict[str, Any]
    ids_full: Any
    versions: dict[str, Any]
    components: dict[str, Any]
    vc_result: Any
    weights_of: Callable[[str], Callable]
    weights_check: Callable[[], dict[str, Any]]
    tl: Any
    after_replay: Callable | None = None
    protocols: Any = None
    reads: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProgramSources:
    """What `programs_of_record` reads of a replay's sources, for a caller that holds no `ReplaySources` (the slim bundle's planner)."""
    args: Any
    manifest: Any
    reqs: list[dict[str, Any]]
    sampling: dict[str, Any]


def programs_of_record(s: ReplaySources | ProgramSources, _rl: dict[str, str], *, definitions: bool = True) -> tuple[dict[str, dict], dict[str, str]]:
    """The population is derived from the Program OF RECORD: every loaded Program's digest must be the Build digest of
    record (--program-digest, B=1) or one of the manifest's component digests (B>=2), and the Definitions this tree evaluates their
    rows by must be the ones their Builds bound (`check.replay.definitions`); else REFUSED by name.  ``definitions=False`` for a
    caller that evaluates no row (the replay's plan, the serving rows): the replay that grades the picks checks them."""
    from verity_vllm.check.replay import definitions as DF
    from verity_vllm.query.manifest import format as MF

    args, required_manifest, reqs, sampling = s.args, s.manifest, s.reqs, s.sampling
    _of_record = CP.digests_of_record(args, required_manifest if isinstance(required_manifest, dict) else None)
    if getattr(args, "programs_root", None):
        _fx = {"requests": [{"request_id": _rl[str(r.get("request_id"))], "prompt_len": len(r["prompt_token_ids"]),
                             "max_tokens": int(r.get("max_tokens") or sampling.get("max_tokens") or 1), "cached": int(r.get("cached") or 0)}
                            for r in reqs]}
        _loaded = MF.programs_for(_fx, args.programs_root)
    else:
        _l1 = MF.load_program(args.program_dir)
        _loaded = {_rl[str(r.get("request_id"))]: _l1 for r in reqs}
    _digs = {k: CP.program_digest(v) for k, v in _loaded.items()}
    if not _of_record:
        raise RuntimeError("no Program digest of record to check the replay Programs against (pass --program-digest / --required-manifest)")
    _bad = {k: d for k, d in _digs.items() if d not in _of_record}
    if _bad:
        _named = ", ".join(f"{k}: {d[:16]} ({_loaded[k].get('dir')})" for k, d in _bad.items())
        raise RuntimeError(f"replay Program(s) are not the Program of record: {_named} vs of-record {sorted(x[:16] for x in _of_record)} -- refused")
    if definitions:
        DF.of_record({k: v["dir"] for k, v in _loaded.items()}, _digs, {k: v["instances"]["rows"] for k, v in _loaded.items()}, what="replay")
    return _loaded, _digs


def request_labels(reqs: list[dict[str, Any]]) -> dict[str, str]:
    """Request id -> the Programs' request label (``r<index>``)."""
    return {str(r.get("request_id")): f"r{int(r.get('index', i))}" for i, r in enumerate(reqs)}


def schedule_of_record(row: dict[str, Any]) -> dict[str, Any]:
    """The schedule a run's Commit placed its population on (`commit.binding.coverage.schedule`); an empty one where the row
    records none."""
    commit = row.get("commit")
    binding = commit.get("binding") if commit else None
    coverage = binding.get("coverage") if binding else None
    schedule = coverage.get("schedule") if coverage else None
    return schedule if schedule else {}


def arrivals(s: Any, rl: dict[str, str]) -> tuple[dict[str, int], bool]:
    """Each request's arrival step, and whether it moved off the declared one.  One schedule of record for every consumer: when the
    population was placed on the engine's committed-step axis (binding.coverage.schedule.use == observed), the replay's, the
    linkage's and the serving rows' arrivals are the same observed ones."""
    from verity_vllm.check.replay.linkage import arrivals_of_record

    declared = {rl[str(r.get("request_id"))]: int(r.get("arrive_step") or 0) for r in s.reqs}
    return arrivals_of_record(declared, schedule_of_record(s.row))
