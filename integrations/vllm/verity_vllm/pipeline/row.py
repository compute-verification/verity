"""`verity-vllm row`: run one sweep row's stages on a pod.

    verity-vllm row run   ROW ROLE REPO REV [--stages build,match,commit] [stage flags]
    verity-vllm row stage STAGE ROW ROLE REPO REV [stage flags]           one stage as one research Attempt
                                                                          (plan: the Commit's plan, on CPU, before the GPU;
                                                                          replay: a config run's deferred C2, on CPU)
    verity-vllm row chain ROW ROLE REPO REV [--stages ...] [--campaign C] [--project P] [stage flags]
    verity-vllm row shed  [--force]

A row id with ``__tp<N>__``, N >= 2, runs the tensor-parallel stages (``row_tp``; world ``--world``, else the manifest's
``sweep.tp``), any other the single-GPU stages (``row_stages``).  The stage flags are the fields of ``BuildConfig``,
``MatchConfig``, ``CommitConfig`` and ``RunEnv``; each also reads the environment variable it names.  ``--protocols``
(``PROTOCOLS``) runs the row with protocol options beside sampled proofs (`verity_vllm.protocol_options`): a set it cannot
run is refused here, before any stage, for single-GPU and TP rows alike (exit 3).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from verity_vllm import protocol_options as PO
from verity_vllm.config import WORKLOADS, BuildConfig, CommitConfig, MatchConfig, RunEnv, UsageError, group, option, row_world
from verity_vllm.pipeline import research

STAGES = "build,match,commit"


@dataclass(frozen=True, kw_only=True)
class _Row:
    row: str = option("row", help="row id (workloads/<row>.json)")
    role: str = option("role", help="checkpoint role (manifests/checkpoints.json)")
    repo: str = option("repo", help="HF repo")
    rev: str = option("rev", help="revision")
    gpu_util: str | None = option("--gpu-util", env="GPU_UTIL", default=None,
                                  help="gpu_memory_utilization (default 0.5; TP 0.6: a rank shard above 0.6 of the device memory "
                                       "needs more, e.g. 0.92 with --commit-gpu-util 0.80 for a 30B MoE at TP2 on 48 GB)")
    world: int | None = option("--world", env="WORLD", type=int, default=None, help="TP: ranks (default: the manifest's sweep.tp)")
    nccl_p2p_disable: str = option("--nccl-p2p-disable", env="NCCL_P2P_DISABLE", default="1", help="TP: NCCL_P2P_DISABLE of the stages")
    query: str = option("--query", env="VERITY_QUERY_ID", default="Q_word_v1{X=16,W=32,R=no-recompute}", help="verification query id (recorded)")
    protocols: str | None = option("--protocols", env="PROTOCOLS", default=None,
                                   help="the row's protocol set, sampled proofs included: 'pouw:<scheme>,pous:<scheme>,sampled-proofs:vllm-v1' "
                                        "(verity_vllm.protocol_options); default: the workload target's, else sampled proofs alone")
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
    """One research Attempt per stage, each citing the earlier stages' artifacts."""

    row: str = option("row")
    role: str = option("role")
    repo: str = option("repo")
    rev: str = option("rev")
    stages: str = option("--stages", default=STAGES)
    campaign: str | None = option("--campaign", env="RESEARCH_CAMPAIGN", default=None)
    project: str = option("--project", env="RESEARCH_PROJECT", default="verity")
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


def run_row(o: Run | Stage, stages: str) -> int:
    wl = WORKLOADS / f"{o.row}.json"
    try:
        protocols = PO.at_row(o.protocols, json.loads(wl.read_text()) if wl.is_file() else {},
                              world=o.world if o.world is not None and row_world(o.row) > 1 else row_world(o.row))
    except PO.Refused as e:
        print(f"[row] REFUSE (fail-closed): {e}", flush=True)
        return 3
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
                     nccl_p2p_disable=o.nccl_p2p_disable, build=o.build, match=o.match, commit=o.commit).run()
    from verity_vllm.pipeline.row_stages import SingleRow
    return SingleRow(row=o.row, role=o.role, repo=o.repo, rev=o.rev, stages=stages, env=o.env, gpu_util=o.gpu_util or "0.5",
                     build=o.build, match=o.match, commit=o.commit, protocols=protocols).run()


def _workload(row: str) -> str:
    from verity_vllm.config import WORKLOADS
    return str(WORKLOADS / f"{row}.json")


def fa_version(workload: str, target: str | None = None) -> int:
    from verity_vllm.program.frontend.target_profile import TargetProfile
    t = json.loads(target) if target else (json.load(open(workload)).get("target") or {})
    d = (TargetProfile.default() if t.get("compute_capability") else TargetProfile.from_device()).to_json()
    d.update({k: v for k, v in t.items() if v is not None})
    return TargetProfile.from_json(d).effective_flash_attn_version()


def main(o: Any) -> int:
    if isinstance(o, Stage):
        return research.stage(o, run_row)
    if isinstance(o, Run):
        return run_row(o, o.stages)
    if isinstance(o, Chain):
        stray = [f for f in ("--stages", "--campaign", "--project") if f in o.flags]
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
