"""The CPU gate: v1's hook set vs the plan's site set on one row, as identical (module path, kind, slot) sets, differences named with
the table that caused them.

    python -m verity_vllm.acquire.gate --manifest <manifest.json> --runtime <runtime_tree.json> --out <dir> [--row-id ID]

`--runtime` is a `RuntimeTree` document: written by the GPU stage under ACQUIRE_ENGINE=v2 (`m1_capture` -> DIR/runtime_tree.json) or by
`python -m verity_vllm.acquire.runtime_tree` from a model instantiated on the meta device as Build does (pod: needs vLLM).  Writes
`acquisition_plan.json`, `v1_decision.json`, `site_compare.json`; exit 0 when there is no UNEXPLAINED difference, 1 otherwise.

Projection: a v1 forward hook takes every returned tensor (slot `*`), so plan producer_output sites are compared per module (their slots
are listed beside); consumer inputs, runner returns and taps compare exactly; the plan's registered inputs (every `weights` Value) meet
v1's `register_weights` (every parameter + buffer) as one site (`weights`, registered_input, `*`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.acquire.correspondence import AnnotationCorrespondence, RuntimeTree
from verity_vllm.acquire.plan import AcquisitionPlan, acquisition_plan
from verity_vllm.acquire.v1_decision import V1Decision, v1_decision

COMPARE_SCHEMA = "verity-vllm/acquisition-site-compare/v1"


def project_plan(plan: AcquisitionPlan) -> tuple[set[tuple[str, str, str]], dict[str, list[str]]]:
    sites: set[tuple[str, str, str]] = set()
    slots: dict[str, list[str]] = {}
    for s in plan.chosen_sites():
        if s.kind == "producer_output":
            sites.add((s.module, s.kind, "*"))
            slots.setdefault(s.module, []).append(s.slot)
        elif s.kind == "registered_input":
            sites.add(("weights", "registered_input", "*"))
        elif s.kind == "analytic_table":
            continue                                          # a check beside the registered input, not an acquisition hook
        elif s.kind == "native_tap":
            sites.add((s.module, s.kind, s.slot))
        else:
            sites.add(s.key())
    return sites, {k: sorted(v) for k, v in slots.items()}


def explain_v1_only(key: tuple[str, str, str], plan: AcquisitionPlan, d: V1Decision, rt: RuntimeTree) -> str:
    module, kind, _slot = key
    cause = d.causes.get(f"{module}|{kind}", "?")
    res = [r for r in plan.residuals if r.module == module and r.table in cause] or [r for r in plan.residuals if r.module == module]
    if res:
        return f"residual {res[0].table}: {res[0].reason}"
    return f"v1 cause: {cause}; the plan derives no Value here"


def explain_plan_only(key: tuple[str, str, str], plan: AcquisitionPlan, d: V1Decision, rt: RuntimeTree, tables: Mapping[str, Any]) -> str:
    module, kind, _slot = key
    vals = [v for v in plan.values if v.chosen is not None and v.chosen.module == module and v.chosen.kind == kind]
    classes = rt.classes(module)
    if classes & set(tables.get("SKIP_CLASSES", ())):
        return f"SKIP_CLASSES: v1 skips class {sorted(classes & set(tables['SKIP_CLASSES']))[0]} by table; the manifest requires {[v.value_id for v in vals][:3]}"
    return f"the manifest requires {[v.value_id for v in vals][:3]} ({len(vals)} Value class(es)) and v1's tables name no hook here (class {rt.class_name(module)})"


def compare_sites(plan: AcquisitionPlan, d: V1Decision, rt: RuntimeTree, *, tables: Mapping[str, Any] | None = None, row_id: str | None = None) -> dict[str, Any]:
    from verity_vllm.acquire.plan import _v1_tables
    t = dict(tables or _v1_tables())
    p_sites, p_slots = project_plan(plan)
    v_sites = d.sites()
    only_v1 = sorted(v_sites - p_sites)
    only_plan = sorted(p_sites - v_sites)
    diffs = []
    unexplained = []
    for k in only_v1:
        why = explain_v1_only(k, plan, d, rt)
        diffs.append({"site": list(k), "in": "v1", "cause": why})
        if not why.startswith("residual"):
            unexplained.append(list(k))
    for k in only_plan:
        why = explain_plan_only(k, plan, d, rt, t)
        diffs.append({"site": list(k), "in": "plan", "cause": why})
        if not (why.startswith("SKIP_CLASSES") or "v1's tables name no hook" in why):
            unexplained.append(list(k))
    by_kind_common: dict[str, int] = {}
    for k in p_sites & v_sites:
        by_kind_common[k[1]] = by_kind_common.get(k[1], 0) + 1
    return {"schema": COMPARE_SCHEMA, "row_id": row_id, "program_digest": plan.program_digest, "manifest_digest": plan.manifest_digest,
            "plan_digest": plan.digest, "runtime_source": rt.source, "plan_complete": plan.complete,
            "counts": {"v1_sites": len(v_sites), "plan_sites": len(p_sites), "common": len(p_sites & v_sites), "only_v1": len(only_v1), "only_plan": len(only_plan),
                       "unexplained": len(unexplained), "common_by_kind": by_kind_common, "plan_values": len(plan.values), "plan_unsupported": len(plan.unsupported),
                       "plan_residuals": len(plan.residuals)},
            "identical": not only_v1 and not only_plan, "differences": diffs, "unexplained": unexplained,
            "plan_output_slots": p_slots, "unsupported": [{"value": v.value_id, "why": v.why} for v in plan.unsupported],
            "residuals": [r.to_json() for r in plan.residuals], "v1_problems": list(d.problems)}


def run(manifest: Mapping[str, Any], rt: RuntimeTree, *, row_id: str | None = None, inplace_facts: Mapping[str, Any] | None = None) -> tuple[AcquisitionPlan, V1Decision, dict[str, Any]]:
    corr = AnnotationCorrespondence(rt, wrapper_prefix=str(manifest.get("wrapper_prefix") or ""), aliases=manifest.get("op_path_aliases") or {}, inplace_facts=inplace_facts)
    plan = acquisition_plan(manifest, corr)
    d = v1_decision(rt, manifest)
    return plan, d, compare_sites(plan, d, rt, row_id=row_id)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="the row's required-value manifest (manifest.json)")
    ap.add_argument("--runtime", required=True, help="the runtime tree document (runtime_tree.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-id", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.manifest).read_text())
    rt = RuntimeTree.load(a.runtime)
    plan, d, cmp = run(man, rt, row_id=a.row_id)
    plan.save(out / "acquisition_plan.json")
    (out / "v1_decision.json").write_text(json.dumps(d.to_json(), indent=1) + "\n")
    (out / "site_compare.json").write_text(json.dumps(cmp, indent=1) + "\n")
    c = cmp["counts"]
    print(f"[gate] {a.row_id or ''} plan {plan.digest[:16]} values {c['plan_values']} (unsupported {c['plan_unsupported']}) sites v1 {c['v1_sites']} plan {c['plan_sites']} "
          f"common {c['common']} only_v1 {c['only_v1']} only_plan {c['only_plan']} unexplained {c['unexplained']} residuals {c['plan_residuals']}", flush=True)
    for df in cmp["differences"]:
        print(f"[gate]   {df['in']:<4} {df['site']}: {df['cause'][:160]}", flush=True)
    return 0 if (not cmp["unexplained"] and plan.complete) else 1


if __name__ == "__main__":
    sys.exit(_main())
