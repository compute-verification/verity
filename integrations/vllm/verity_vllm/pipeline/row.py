"""`verity-vllm row`: run one sweep row's stages on a pod.

    verity-vllm row run   ROW ROLE REPO REV [--stages build,match,commit] [stage flags]
    verity-vllm row stage STAGE ROW ROLE REPO REV [stage flags]           one stage as one research Attempt
                                                                          (plan: the Commit's plan, on CPU, before the GPU;
                                                                          replay: a config run's deferred C2, on CPU)
    verity-vllm row chain ROW ROLE REPO REV [--stages ...] [--build-run RUN] [--match-run RUN] [--campaign C] [--project P]
                          [--ref REF] [--by LANE] [--tier T] [--no-publish] [stage flags]
                                                                          a row whose Build, Match and Commit pass publishes
                                                                          its record (`pipeline.research.chain`)
    verity-vllm row shed  [--force]

A row id with ``__tp<N>__``, N >= 2, runs the tensor-parallel stages (``row_tp``; world ``--world``, else the manifest's
``sweep.tp``), any other the single-GPU stages (``row_stages``).  The stage flags are the fields of ``BuildConfig``,
``MatchConfig``, ``CommitConfig`` and ``RunEnv``; each also reads the environment variable it names.  ``--protocols``
(``PROTOCOLS``) runs the row with protocol options beside sampled proofs (`verity_vllm.protocols`): a set it cannot
run is refused here, before any stage, for single-GPU and TP rows alike (exit 3), and so is a row of record that is not done only at
its public cap (``PRECHECK_FAIL_DONE_SOURCE``, ``config.done_source_refusal``).

The row's config is its file (``config.RowConfig``); the flags and variables change how its stages run, never it.  Every run first
writes what it used, ``<row dir>/effective_config.json`` (``effective_config``), which the row's config record carries.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from importlib import metadata
from typing import Any

from verity_vllm import protocols as PO
from verity_vllm.config import (ROOT, BuildConfig, CommitConfig, MatchConfig, RowConfig, RunEnv, UsageError, done_source_refusal,
                                group, option, row_file, row_path, row_world)
from verity_vllm.pipeline import config_record as CR
from verity_vllm.pipeline import research
from verity_vllm import settings

STAGES = "build,match,commit"
EFFECTIVE_SCHEMA = "verity-vllm/effective-config/v1"
EFFECTIVE_NAME = "effective_config.json"
#: The environment a row's engines inherit that can change their kernels, recorded beside the config (never a credential).
ENGINE_ENV = ("VLLM_", "NCCL_", "CUBLAS", "TORCH_", "TORCHINDUCTOR_", "TRITON_", "FLASHINFER_", "CUDA_VISIBLE_DEVICES")


@dataclass(frozen=True, kw_only=True)
class _Row:
    row: str = option("row", help="row id (workloads/<row>.json, or in a directory $VERITY_WORKLOADS names: config.row_file)")
    role: str = option("role", help="checkpoint role (manifests/checkpoints.json)")
    repo: str = option("repo", help="HF repo")
    rev: str = option("rev", help="revision")
    gpu_util: str | None = option("--gpu-util", env="GPU_UTIL", default=None,
                                  help="gpu_memory_utilization (default 0.5; TP 0.6: a rank shard above 0.6 of the device memory "
                                       "needs more, e.g. 0.92 with --commit-gpu-util 0.80 for a 30B MoE at TP2 on 48 GB)")
    world: int | None = option("--world", env="WORLD", type=int, default=None, help="TP: ranks (default: the manifest's sweep.tp)")
    query: str = option("--query", env="VERITY_QUERY_ID", default="Q_word_v1{X=16,W=32,R=no-recompute}", help="verification query id (recorded)")
    protocols: str | None = option("--protocols", env="PROTOCOLS", default=None,
                                   help="the row's protocol set, sampled proofs included: 'pouw:<scheme>,pous:<scheme>,sampled-proofs:rows-sha512' "
                                        "(verity_vllm.protocols); default: the workload target's, else sampled proofs alone")
    build: BuildConfig = group(BuildConfig)
    match: MatchConfig = group(MatchConfig)
    commit: CommitConfig = group(CommitConfig)
    env: RunEnv = group(RunEnv)


@dataclass(frozen=True, kw_only=True)
class Run(_Row):
    """Run a row's stages in order, stopping at the first FAIL."""

    stages: str = option("--stages", default=STAGES, help="comma list of build, plan, match, commit, replay")


@dataclass(frozen=True, kw_only=True)
class _Research:
    research_bin: str | None = option("--research-bin", env="RESEARCH_BIN", default=None, help="the research CLI")
    research_py: str = option("--research-py", env="RESEARCH_PY", default="python3", help="interpreter of `-m research`")


@dataclass(frozen=True, kw_only=True)
class _StageName:
    stage: str = option("stage", choices=("build", "plan", "match", "commit", "replay"))


@dataclass(frozen=True, kw_only=True)
class Stage(_Row, _Research, _StageName):
    """One stage of a row as the workload of one research Attempt."""


@dataclass(frozen=True, kw_only=True)
class Chain(_Research):
    """One research Attempt per stage, each citing the earlier stages' artifacts; a row whose Build, Match and Commit pass publishes
    its record in the circuit database."""

    row: str = option("row")
    role: str = option("role")
    repo: str = option("repo")
    rev: str = option("rev")
    stages: str = option("--stages", default=STAGES)
    build_run: str | None = option("--build-run", default=None,
                                   help="a passing Build Attempt the chain cites in place of its build stage (--stages without build); "
                                        "its row dir must be the chain's, whose Match precheck holds it to the cited Program digest")
    match_run: str | None = option("--match-run", default=None,
                                   help="a passing Match Attempt the chain cites in place of its match stage (--stages without match); "
                                        "its row dir, with the Match's outputs, must be the chain's")
    campaign: str | None = option("--campaign", env="RESEARCH_CAMPAIGN", default=None)
    project: str = option("--project", env="RESEARCH_PROJECT", default="verity")
    publish: bool = option("--publish", action="boolean_optional", default=True,
                           help="publish the row's record once its Build, Match and Commit pass (`verity-vllm records publish`)")
    ref: str | None = option("--ref", default=None, help="the branch of the record's tree REF@SHA (default the tree's EXPORT.json or git branch)")
    by: str | None = option("--by", env="RESEARCH_LANE", default=None, help="the lane the record's row label is by")
    tier: str | None = option("--tier", choices=CR.TIERS, default=None, help="the tier of the row's rebuild; a row new to records.json needs one")
    flags: list[str] = option("flags", nargs="...", default=[], help="stage flags, passed to every stage")
    env: RunEnv = group(RunEnv)


@dataclass(frozen=True, kw_only=True)
class Shed(_Research):
    """Terminate the pod once every Attempt it registered is preserved."""

    force: bool = option("--force", action="store_true", default=False, help="terminate even when not preserved")
    env: RunEnv = group(RunEnv)


@dataclass(frozen=True, kw_only=True)
class FaVersion:
    """The FlashAttention version a workload's row executes on this device (its declared target over the device's)."""

    workload: str = option("workload")
    target: str | None = option("--target", default=None, help="the row's declared TargetProfile JSON (the Build's --target); with a capability, no device is read")


COMMANDS = {"run": Run, "stage": Stage, "chain": Chain, "shed": Shed, "fa-version": FaVersion}


def _overrides(cfg: Any, prefix: str) -> dict[str, Any]:
    return {f"{prefix}{f.name}": getattr(cfg, f.name) for f in fields(cfg)
            if f.name != "environ" and getattr(cfg, f.name) != (f.default_factory() if callable(f.default_factory) else f.default)}


def effective_config(o: Run | Stage, stages: str, protocols: tuple[str, ...] | None) -> dict[str, Any]:
    """What a run of a row used: its config (the row file without its requests, and the file's digest), the stage configs in full with
    the options a flag or variable set away from its default, the run's positionals, the installed vLLM and the engine environment
    the stages inherit (``ENGINE_ENV``)."""
    wl = row_file(o.row)
    cfg = RowConfig.load(wl) if wl.is_file() else None
    try:
        vllm = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        vllm = None
    run = {"role": o.role, "repo": o.repo, "rev": o.rev, "world": o.world, "gpu_util": o.gpu_util, "query": o.query,
           "protocols": list(protocols) if protocols else None, "stages": stages}
    differs = [] if cfg is None else [k for k, want in (("role", cfg.case), ("repo", cfg.model), ("rev", cfg.engine.revision))
                                      if want is not None and str(run[k]) != str(want)]
    environ = o.env.environ
    return {"schema": EFFECTIVE_SCHEMA, "row": o.row,
            "config": None if cfg is None else {k: v for k, v in cfg.to_json().items() if k != "requests"},
            "file": None if cfg is None else {"path": row_path(o.row), "sha256": hashlib.sha256(wl.read_bytes()).hexdigest()},
            "run": run, "run_differs_from_config": differs,
            "stages": {"build": asdict(o.build), "match": asdict(o.match), "commit": asdict(o.commit)},
            "overrides": {**_overrides(o.build, "build."), **_overrides(o.match, "match."), **_overrides(o.commit, "commit."),
                          **{k: run[k] for k in ("world", "gpu_util", "protocols") if run[k] is not None}},
            "vllm_installed": vllm,
            "engine_env": {k: environ[k] for k in sorted(environ) if k.startswith(ENGINE_ENV)
                           and not settings.is_secret(k)}}


def run_row(o: Run | Stage, stages: str) -> int:
    wl = row_file(o.row)
    doc = json.loads(wl.read_text()) if wl.is_file() else {}
    try:
        protocols = PO.at_row(o.protocols, doc, world=o.world if o.world is not None and row_world(o.row) > 1 else row_world(o.row))
    except PO.Refused as e:
        print(f"[row] REFUSE (fail-closed): {e}", flush=True)
        return 3
    if o.env.verity_tree and os.path.realpath(o.env.verity_tree) != os.path.realpath(ROOT):   # refuse before writing anything
        print(f"PRECHECK_FAIL_SOURCE_IDENTITY: row at {os.path.realpath(ROOT)} but $VERITY_TREE={o.env.verity_tree} names another tree",
              file=sys.stderr)
        return 3
    if why := done_source_refusal(doc):
        print(f"PRECHECK_FAIL_DONE_SOURCE: {why}", file=sys.stderr)
        return 3
    d = os.path.join(o.env.sweep_dir, o.row)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, EFFECTIVE_NAME), "w") as f:
        json.dump(effective_config(o, stages, protocols), f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    if row_world(o.row) > 1:
        from verity_vllm.pipeline import row_records as R
        from verity_vllm.pipeline.row_tp import TpRow
        from verity_vllm.pipeline.tp.c2 import DEFER_WORLDS
        world = o.world if o.world is not None else R.tp_world(_workload(o.row))
        if world < 2:
            raise UsageError(f"--world {world}: a tensor-parallel row needs a world of at least 2 ranks")
        if o.commit.group:
            raise UsageError(f"--commit-group: {o.row} is a tensor-parallel row; a grouped Commit is a single-GPU row's")
        if o.commit.replay_deferred == "slim" and world in DEFER_WORLDS:     # at any other world the TP Commit replays in process
            raise UsageError(f"--replay-deferred slim: {o.row} is a tensor-parallel row; the slim plan is a single-GPU Commit's (TP defers with 1)")
        return TpRow(row=o.row, role=o.role, repo=o.repo, rev=o.rev, stages=stages, env=o.env, world=world, gpu_util=o.gpu_util,
                     build=o.build, match=o.match, commit=o.commit).run()
    from verity_vllm.pipeline.row_stages import SingleRow
    return SingleRow(row=o.row, role=o.role, repo=o.repo, rev=o.rev, stages=stages, env=o.env, gpu_util=o.gpu_util or "0.5",
                     build=o.build, match=o.match, commit=o.commit, protocols=protocols).run()


def _workload(row: str) -> str:
    return str(row_file(row))


def fa_version(workload: str, target: str | None = None) -> int:
    from verity_vllm.program.frontend.target_profile import TargetProfile
    t = json.loads(target) if target else dict(RowConfig.load(workload).engine.target or {})
    d = (TargetProfile.default() if t.get("compute_capability") else TargetProfile.from_device()).to_json()
    d.update({k: v for k, v in t.items() if v is not None})
    return TargetProfile.from_json(d).effective_flash_attn_version()


def main(o: Any) -> int:
    if isinstance(o, Stage):
        return research.stage(o, run_row)
    if isinstance(o, Run):
        return run_row(o, o.stages)
    if isinstance(o, Chain):
        stray = [f for f in ("--stages", "--build-run", "--match-run", "--campaign", "--project", "--publish", "--no-publish", "--ref", "--by", "--tier")
                 if f in o.flags]
        if stray:
            raise UsageError(f"{', '.join(stray)} must come before ROW (the flags after the positionals go to every stage)")
        from verity_vllm.pipeline import cli
        first = o.stages.split(",")[0]
        st = cli.parse("row", ["stage", first, o.row, o.role, o.repo, o.rev, *o.flags], o.env.environ)
        return research.chain(o, st.env.sweep_dir)
    if isinstance(o, Shed):
        return research.shed(o)
    print(fa_version(o.workload, o.target))
    sys.stdout.flush()
    return 0
