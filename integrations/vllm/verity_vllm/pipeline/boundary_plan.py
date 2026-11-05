"""The Commit's call-boundary plan: the call-boundary source's host plan (`call_boundary_plan.plan`: every `call_boundaries` identity of
the required manifest as a target of a module body's row template), derived once at the Build's call-boundary check rather than again
in the Commit with its GPU held (3085 s of a Gemma-2-2B B64 Commit).

    verity-vllm boundary-plan --workload W --required-manifest M [--program-digest D] (--program-dir P | --programs-root R)
                              --gate-out call_boundaries.json --out PLAN

The check is `gate call-boundaries` over the Programs the Commit reads (its `--program-dir` / `--programs-root`): the same record, line
and exit code.  On a PASS the plan is written, keyed as the producer-facts plan is (`commit_plan.key`: the Program digest, the sha256 of
the required manifest and of the workload, the Programs it reads, the code identity, the vLLM version).  The Commit
(`--call-boundary-plan PLAN`) takes it only when the key is its own; a plan it cannot read, keyed differently or that does not load
(`call_boundary_plan.of_json`), and it derives the plan itself and records `plan_recomputed` with the reason (`use`).
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.acquire import gate as G
from verity_vllm.acquire.sources import call_boundary_plan as CBP
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.config import option, workload_arg
from verity_vllm.pipeline import commit_plan as CP

SCHEMA = "verity-vllm/call-boundary-plan/v1"


def roots(args: Any) -> tuple[str | None, str | None]:
    """The Programs the Commit `args` describes reads, as `taps.attach` is given them."""
    return getattr(args, "program_dir", None), getattr(args, "programs_root", None)


def use(args: Any, manifest: Mapping[str, Any] | None, *, tl: Any) -> tuple[CBP.Plan | None, dict[str, Any] | None]:
    """The call-boundary plan of the Commit `args` describes over `manifest` and the plan's record: from `--call-boundary-plan` when its
    key is the Commit's and it loads, else derived here (`plan_recomputed` with the reason); (None, None) for a manifest naming no
    identity the source evaluates (the record says so when a plan was given).  `tl` times it (`prep.call_boundary_plan`, with its source)."""
    path = getattr(args, "call_boundary_plan", None)
    if not CB.manifest_sites(manifest):
        return None, (None if not path else {"path": str(path), "schema": SCHEMA, "used": False, "plan_recomputed": False,
                                             "why": "the manifest names no call-boundary identity the source evaluates"})
    assert manifest is not None
    tl.start("prep.call_boundary_plan")
    t0 = time.monotonic()
    plan, rec = None, None
    if path:
        doc, rec = CP.keyed(path, args, SCHEMA)
        try:
            plan = CBP.of_json(doc["plan"]) if doc is not None else None
        except Exception as e:  # noqa: BLE001  -- the plan is derived here: the reason is on the record
            rec = {**rec, "used": False, "plan_recomputed": True, "why": f"plan does not load: {e!r}"[:300]}
        if plan is not None:
            rec = {**rec, "used": True, "plan_recomputed": False, "identities": plan.identities, "bodies": len(plan.bodies),
                   "load_s": round(time.monotonic() - t0, 3)}
        print(f"[delta] call-boundary plan {path}: " + (f"{plan.identities} identities in {len(plan.bodies)} bodies from the plan (planned "
                                                       f"{rec.get('planned_utc')})" if plan is not None else f"plan_recomputed: {rec['why']}"),
              flush=True)
    source = "plan" if plan is not None else "derived"
    if plan is None:
        plan = CB.plan_of(manifest, *roots(args))
        if rec is not None:
            rec["derive_s"] = round(time.monotonic() - t0, 1)
    tl.end("prep.call_boundary_plan", source=source, identities=plan.identities, bodies=len(plan.bodies))
    return plan, rec


def verdict_entry(args: Any, rec: dict[str, Any] | None) -> dict[str, Any]:
    """The Commit verdict's `call_boundary_plan` when it was given one: the plan's record, or why none was read."""
    if not getattr(args, "call_boundary_plan", None):
        return {}
    return {"call_boundary_plan": rec or {"path": args.call_boundary_plan, "used": False, "plan_recomputed": False,
                                          "why": "no call-boundary plan read (no manifest or no committer)"}}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """The call-boundary check of a row's Build over the Programs its Commit reads, and on a PASS the Commit's call-boundary plan."""

    workload: str = option("--workload", type=workload_arg, help="the row's workload (the Commit's --workload)")
    required_manifest: str = option("--required-manifest", help="the Commit's required-value manifest")
    program_digest: str | None = option("--program-digest", default=None, help="the Commit's Program digest of record")
    program_dir: str | None = option("--program-dir", default=None, help="B=1: the Build's request Program dir")
    programs_root: str | None = option("--programs-root", default=None, help="B>=2: the directory holding the per-request Programs")
    gate_out: str | None = option("--gate-out", default=None, help="the check's record (call_boundaries.json)")
    out: str = option("--out", help="where the plan is written (the Commit's --call-boundary-plan)")


def main(o: Options) -> int:
    """The check's record, line (the last one printed) and exit code; the plan beside it on a PASS, or the reason none was written (the
    Commit then derives it)."""
    t0 = time.monotonic()
    if os.path.exists(o.out):
        os.remove(o.out)
    man = json.loads(Path(o.required_manifest).read_text())
    plan = CB.plan_of(man, *roots(o)) if CB.manifest_sites(man) else None
    rec = G.call_boundaries_of(man, plan)
    rec["seconds"] = round(time.monotonic() - t0, 1)
    if o.gate_out:
        Path(o.gate_out).write_text(json.dumps(rec, indent=1) + "\n")
    if plan is not None and rec["pass"]:
        try:
            k = CP.key(o)
            CP.write(o.out, {"schema": SCHEMA, "key": k, "plan": CBP.to_json(plan), "wall_s": round(time.monotonic() - t0, 1),
                             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"[plan] {o.out}: {plan.covered} identities in {len(plan.bodies)} bodies, {os.path.getsize(o.out)} bytes; key " +
                  " ".join(f"{x}={str(v)[:16]}" for x, v in k.items()), flush=True)
        except Exception as e:  # noqa: BLE001  -- no plan: the Commit derives it, the check stands
            print(f"[plan] {o.out} not written: {e!r}"[:400], flush=True)
    print(G.call_boundaries_line(rec), flush=True)
    return 0 if rec["pass"] else 1
