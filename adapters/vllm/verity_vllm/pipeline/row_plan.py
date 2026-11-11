"""The plan stage of a single-GPU row (`row_stages.SingleRow`): what the Commit would otherwise derive on the GPU without needing the
engine, derived on a 0-GPU task before the GPU is held.  The manifest of record and the weights of record beside the component Program,
as the Commit stage would derive them, and the producer facts (`commit-plan`, `pipeline.commit_plan`) in `plan/commit_plan.json`, which
the Commit stage passes (`--plan`) and takes only when the plan's key is its own.  Nothing seeded by the run root is planned here.  The
call-boundary plan is the Build's (`call_boundaries_cmd`, beside `call_boundaries.json`), passed the same way (`--call-boundary-plan`).
"""
from __future__ import annotations

import os
import time
from typing import Any

from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop

PLAN_TIMEOUT = 14400


def path(row: Any) -> str:
    return f"{row.d}/plan/commit_plan.json"


def boundary_path(row: Any) -> str:
    return f"{row.d}/call_boundary_plan.json"


def programs_args(row: Any) -> list[str]:
    """The Build's Programs the Commit reads (sampled replay, producer facts): the per-request Programs under the row dir at B >= 2,
    the request Program at B = 1; none without the Build's instance set."""
    pd_req = f"{row.d}/build_request"
    if not os.path.isfile(f"{pd_req}/instances.json.gz"):
        return []
    return ["--programs-root", row.d] if row.B > 1 else ["--program-dir", pd_req]


def commit_flags(row: Any) -> list[str]:
    """The Commit's flags for what the plan stage covers: the Build's Programs (C2 of record: the sampled exact VU replay and the
    boundary linkage) with the row's replay flags and the weights of record, and the plan when there is one."""
    flags, progs = [], programs_args(row)
    if progs:
        flags += [*progs, *(row.ck_.replay_args or "").split()]
        row.log(f"C2 of record armed: sampled exact VU replay + boundary linkage ({' '.join(progs)} {row.ck_.replay_args or ''})")
        flags += row.weights_of_record()
    else:
        row.log(f"WARNING: C2 sampled replay NOT armed (no {row.d}/build_request/instances.json.gz): C2 of record not run")
    if os.path.isfile(path(row)):
        flags += ["--plan", path(row)]
        row.log(f"plan: {path(row)} (the Commit takes its producer facts from it when the plan's key is its own, else plan_recomputed)")
    if os.path.isfile(boundary_path(row)):
        flags += ["--call-boundary-plan", boundary_path(row)]
        row.log(f"call-boundary plan: {boundary_path(row)} (the Commit takes it when the plan's key is its own, else plan_recomputed)")
    return flags


def call_boundaries_cmd(row: Any) -> list[str] | None:
    """The Build's call-boundary check over the Programs the Commit reads, writing the Commit's call-boundary plan on a PASS
    (`boundary-plan`, `pipeline.boundary_plan`); None (`gate call-boundaries` over the row dir, no plan) without the Build's manifest or
    Programs.  A plan of an earlier attempt is removed either way."""
    if os.path.exists(boundary_path(row)):
        os.remove(boundary_path(row))
    man, progs = f"{row.d}/manifest.json", programs_args(row)
    if not (os.path.isfile(man) and progs):
        return None
    pdig = row.program_digest()
    return row.cli("boundary-plan", "--workload", row.wl, "--required-manifest", man, *(["--program-digest", pdig] if pdig else []), *progs,
                   "--gate-out", f"{row.d}/call_boundaries.json", "--out", boundary_path(row))


def plan(row: Any, *, required: bool = True) -> None:
    """The plan stage; a FAIL stops the row (exit 15) when it is `required`, else it leaves no plan and the Commit derives what the plan
    would have held (the plan a Build task ends with, `SingleRow.run`)."""
    d, out = row.d, path(row)
    row.tl("stage.plan", "start")
    t0 = int(time.time())
    if os.path.exists(out):
        os.remove(out)
    man, progs = row.manifest_of_record("plan"), programs_args(row)
    if not (man and progs):
        row.verdict("plan", "PASS", "nothing to plan: " + ("no manifest of record" if not man else "no Build Programs (instances.json.gz)"))
        row.tl("stage.plan", "end", ("ok", "PASS"), ("planned", "false"))
        return
    row.weights_of_record("plan")
    pdig = row.program_digest()
    rc = row.run_tl("plan.commit_plan", PLAN_TIMEOUT, f"{d}/plan.log", row.cli(
        "commit-plan", "--workload", row.wl, "--required-manifest", man, *(["--program-digest", pdig] if pdig else []), *progs,
        "--out", out))
    wall = int(time.time()) - t0
    ok = "PASS" if rc == 0 and os.path.isfile(out) else "FAIL"
    row.verdict("plan", ok, f"rc={rc} wall={wall}s {R.tail(f'{d}/plan.log', 240)}")
    row.tl("stage.plan", "end", ("ok", ok), ("wall_s", wall))
    if ok != "PASS" and not required:
        if os.path.exists(out):
            os.remove(out)
        row.log("PLAN FAIL -> no plan: the Commit derives its producer facts")
    elif ok != "PASS":
        row.log("PLAN FAIL -> stop")
        row.tl("row", "end", ("ok", "false"), ("stopped_at", "plan"))
        raise Stop(15)
