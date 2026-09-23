"""The CPU gate: v1's hook set vs the plan's site set on one row, as identical (module path, kind, slot) sets, differences named with
the table that caused them.

    python -m verity_vllm.acquire.gate --manifest <manifest.json> --runtime <runtime_tree.json> --out <dir> [--row-id ID]
                                       [--programs-root <row dir> | --program <derive dir> ...] [--against <acquisition_plan.json>]
    python -m verity_vllm.acquire.gate compare-plans <a.json> <b.json> [--out plan_compare.json]

`--runtime` is a `RuntimeTree` document: written by the GPU stage under ACQUIRE_ENGINE=v2 (`m1_capture` -> DIR/runtime_tree.json) or by
`python -m verity_vllm.acquire.runtime_tree` from a model instantiated on the meta device as Build does (pod: needs vLLM).  `--programs-root`
/ `--program` name the Build's request Program dirs (`build_request*/`): the owning modules are read off their `runtime-correspondence/v1`
record when they carry it (`correspondence.source = correspondence`), else off the manifest's v1 annotations -- the source is recorded in
the plan and the compare document.  Writes `acquisition_plan.json`, `v1_decision.json`, `site_compare.json` (and `plan_compare.json`
against another plan: every Value whose chosen site differs, listed); exit 0 when there is no UNEXPLAINED difference, 1 otherwise.

Projection: a v1 forward hook takes every returned tensor (slot `*`), so plan producer_output sites are compared per module (their slots
are listed beside); consumer inputs, runner returns and taps compare exactly; the plan's registered inputs (every `weights` Value) meet
v1's `register_weights` (every parameter + buffer) as one site (`weights`, registered_input, `*`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from verity_vllm.acquire.correspondence import AcquireCorrespondence, RuntimeTree
from verity_vllm.acquire.plan import AcquisitionPlan, acquisition_plan
from verity_vllm.acquire.v1_decision import V1Decision, v1_decision

COMPARE_SCHEMA = "verity-vllm/acquisition-site-compare/v1"
PLAN_COMPARE_SCHEMA = "verity-vllm/acquisition-plan-compare/v1"


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
            "plan_digest": plan.digest, "runtime_source": rt.source, "correspondence": dict(plan.correspondence), "plan_complete": plan.complete,
            "counts": {"v1_sites": len(v_sites), "plan_sites": len(p_sites), "common": len(p_sites & v_sites), "only_v1": len(only_v1), "only_plan": len(only_plan),
                       "unexplained": len(unexplained), "common_by_kind": by_kind_common, "plan_values": len(plan.values), "plan_unsupported": len(plan.unsupported),
                       "plan_residuals": len(plan.residuals)},
            "identical": not only_v1 and not only_plan, "differences": diffs, "unexplained": unexplained,
            "plan_output_slots": p_slots, "unsupported": [{"value": v.value_id, "why": v.why} for v in plan.unsupported],
            "residuals": [r.to_json() for r in plan.residuals], "v1_problems": list(d.problems)}


def run(manifest: Mapping[str, Any], rt: RuntimeTree, *, row_id: str | None = None, inplace_facts: Mapping[str, Any] | None = None,
        programs: Iterable[str | Path] = ()) -> tuple[AcquisitionPlan, V1Decision, dict[str, Any]]:
    kw = dict(wrapper_prefix=str(manifest.get("wrapper_prefix") or ""), aliases=manifest.get("op_path_aliases") or {}, inplace_facts=inplace_facts)
    programs = list(programs)
    corr = AcquireCorrespondence.from_programs(rt, programs, **kw) if programs else AcquireCorrespondence(rt, **kw)
    plan = acquisition_plan(manifest, corr)
    d = v1_decision(rt, manifest)
    return plan, d, compare_sites(plan, d, rt, row_id=row_id)


def request_program_dirs(root: str | Path) -> list[str]:
    """The Build's request Program dirs under a row dir: every `build_request*/` holding a `result.json` (the row's default request
    Program and one per further request shape)."""
    return sorted(d for d in glob.glob(os.path.join(str(root), "build_request*")) if os.path.isdir(d) and os.path.isfile(os.path.join(d, "result.json")))


def compare_plans(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Two `acquisition-plan/v1` documents Value by Value: the Values whose CHOSEN site (module, kind, slot) differs, the Values only one
    plan has, and the two digests beside each other with the source each plan read its owning modules from."""
    def by_value(p: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return {v["value"]: v for v in p.get("values") or []}

    def key(v: Mapping[str, Any]) -> tuple | None:
        c = v.get("chosen")
        return (c["module"], c["kind"], c["slot"]) if c else None

    va, vb = by_value(a), by_value(b)
    differing = []
    for val in sorted(set(va) & set(vb)):
        ka, kb = key(va[val]), key(vb[val])
        if ka != kb or va[val].get("status") != vb[val].get("status"):
            differing.append({"value": val, "a": {"chosen": list(ka) if ka else None, "status": va[val].get("status"), "why": va[val].get("why")},
                              "b": {"chosen": list(kb) if kb else None, "status": vb[val].get("status"), "why": vb[val].get("why")}})
    src = lambda p: (p.get("correspondence") or {}).get("source", "v1_annotations")  # noqa: E731
    return {"schema": PLAN_COMPARE_SCHEMA,
            "a": {"plan_digest": a.get("plan_digest"), "correspondence_source": src(a), "program_digest": a.get("program_digest"), "manifest_digest": a.get("manifest_digest"),
                  "runtime_source": a.get("runtime_source"), "values": len(va)},
            "b": {"plan_digest": b.get("plan_digest"), "correspondence_source": src(b), "program_digest": b.get("program_digest"), "manifest_digest": b.get("manifest_digest"),
                  "runtime_source": b.get("runtime_source"), "values": len(vb)},
            "plan_digest_equal": a.get("plan_digest") == b.get("plan_digest"), "common_values": len(set(va) & set(vb)),
            "only_a": sorted(set(va) - set(vb)), "only_b": sorted(set(vb) - set(va)), "differing": differing,
            "identical": a.get("plan_digest") == b.get("plan_digest") and not differing and set(va) == set(vb)}


def _main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["compare-plans"]:
        ap = argparse.ArgumentParser(prog="verity_vllm.acquire.gate compare-plans")
        ap.add_argument("a")
        ap.add_argument("b")
        ap.add_argument("--out", default=None)
        c = ap.parse_args(argv[1:])
        cmp = compare_plans(json.loads(Path(c.a).read_text()), json.loads(Path(c.b).read_text()))
        if c.out:
            Path(c.out).write_text(json.dumps(cmp, indent=1) + "\n")
        print(f"[gate] plans a {str(cmp['a']['plan_digest'])[:16]} ({cmp['a']['correspondence_source']}) b {str(cmp['b']['plan_digest'])[:16]} ({cmp['b']['correspondence_source']}): "
              f"digest_equal {cmp['plan_digest_equal']} common {cmp['common_values']} only_a {len(cmp['only_a'])} only_b {len(cmp['only_b'])} differing {len(cmp['differing'])}", flush=True)
        for dv in cmp["differing"][:40]:
            print(f"[gate]   {dv['value']}: a {dv['a']['chosen']} b {dv['b']['chosen']}", flush=True)
        return 0 if cmp["identical"] else 1
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="the row's required-value manifest (manifest.json)")
    ap.add_argument("--runtime", required=True, help="the runtime tree document (runtime_tree.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-id", default=None)
    ap.add_argument("--programs-root", default=None, help="the row dir: its build_request*/ Programs' correspondence records are read (else the v1 annotations)")
    ap.add_argument("--program", action="append", default=[], help="a request Program dir (repeatable); alternative to --programs-root")
    ap.add_argument("--against", default=None, help="another acquisition_plan.json to compare Value by Value (-> plan_compare.json)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.manifest).read_text())
    rt = RuntimeTree.load(a.runtime)
    programs = list(a.program) + (request_program_dirs(a.programs_root) if a.programs_root else [])
    plan, d, cmp = run(man, rt, row_id=a.row_id, programs=programs)
    plan.save(out / "acquisition_plan.json")
    (out / "v1_decision.json").write_text(json.dumps(d.to_json(), indent=1) + "\n")
    (out / "site_compare.json").write_text(json.dumps(cmp, indent=1) + "\n")
    c = cmp["counts"]
    cp = plan.correspondence
    print(f"[gate] {a.row_id or ''} plan {plan.digest[:16]} values {c['plan_values']} (unsupported {c['plan_unsupported']}) sites v1 {c['v1_sites']} plan {c['plan_sites']} "
          f"common {c['common']} only_v1 {c['only_v1']} only_plan {c['only_plan']} unexplained {c['unexplained']} residuals {c['plan_residuals']}", flush=True)
    print(f"[gate]   correspondence source={cp.get('source')} programs={len(cp.get('programs') or {})} absent={len(cp.get('absent') or [])} module_paths={cp.get('module_paths')} "
          f"modules confirmed={(cp.get('modules') or {}).get('confirmed')} unconfirmed={len((cp.get('modules') or {}).get('unconfirmed') or [])} problems={len(cp.get('problems') or [])}", flush=True)
    for u in ((cp.get("modules") or {}).get("unconfirmed") or [])[:20]:
        print(f"[gate]   UNCONFIRMED {u['value']} ({u['role']} {u['module']}): {u['why']}", flush=True)
    for p in (cp.get("problems") or [])[:10]:
        print(f"[gate]   correspondence problem: {p[:200]}", flush=True)
    for df in cmp["differences"]:
        print(f"[gate]   {df['in']:<4} {df['site']}: {df['cause'][:160]}", flush=True)
    rc = 0 if (not cmp["unexplained"] and plan.complete) else 1
    if a.against:
        pc = compare_plans(plan.to_json(), json.loads(Path(a.against).read_text()))
        (out / "plan_compare.json").write_text(json.dumps(pc, indent=1) + "\n")
        print(f"[gate]   vs {a.against}: digest_equal {pc['plan_digest_equal']} ({pc['a']['correspondence_source']} vs {pc['b']['correspondence_source']}) "
              f"common {pc['common_values']} only_this {len(pc['only_a'])} only_other {len(pc['only_b'])} differing {len(pc['differing'])}", flush=True)
        for dv in pc["differing"][:40]:
            print(f"[gate]   DIFFERS {dv['value']}: this {dv['a']['chosen']} other {dv['b']['chosen']}", flush=True)
        if not pc["identical"]:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(_main())
