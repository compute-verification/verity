"""The CPU gate: the acquisition plan of one row, off-pod (the runtime tree read off the meta-instantiated model or a live engine), and
plan-vs-plan comparison.

    python -m verity_vllm.acquire.gate --manifest <manifest.json> --runtime <runtime_tree.json> --out <dir> [--row-id ID]
                                       [--programs-root <row dir> | --program <derive dir> ...] [--against <acquisition_plan.json>]
    python -m verity_vllm.acquire.gate compare-plans <a.json> <b.json> [--out plan_compare.json]

`--runtime` is a `RuntimeTree` document: written by the GPU stage (`m1_capture` -> DIR/runtime_tree.json) or by
`python -m verity_vllm.observe.runtime_tree` from a model instantiated on the meta device as Build does (pod: needs vLLM).  `--programs-root`
/ `--program` name the Build's request Program dirs (`build_request*/`): the owning modules are read off their `runtime-correspondence/v1`
record when they carry it (`correspondence.source = correspondence`), else off the manifest's v1 annotations -- the source is recorded in
the plan.  Writes `acquisition_plan.json` (and `plan_compare.json` against another plan: every Value whose chosen site differs, listed);
exit 0 when the plan is complete (and identical to `--against`), 1 otherwise.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from verity_vllm.correspondence.reader_for_acquire import AcquireCorrespondence, RuntimeTree
from verity_vllm.acquire.plan import AcquisitionPlan, acquisition_plan

PLAN_COMPARE_SCHEMA = "verity-vllm/acquisition-plan-compare/v1"


def run(manifest: Mapping[str, Any], rt: RuntimeTree, *, inplace_facts: Mapping[str, Any] | None = None, programs: Iterable[str | Path] = ()) -> AcquisitionPlan:
    kw = dict(wrapper_prefix=str(manifest.get("wrapper_prefix") or ""), aliases=manifest.get("op_path_aliases") or {}, inplace_facts=inplace_facts)
    programs = list(programs)
    corr = AcquireCorrespondence.from_programs(rt, programs, **kw) if programs else AcquireCorrespondence(rt, **kw)
    return acquisition_plan(manifest, corr)


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
    plan = run(man, rt, programs=programs)
    plan.save(out / "acquisition_plan.json")
    cp = plan.correspondence
    print(f"[gate] {a.row_id or ''} plan {plan.digest[:16]} values {len(plan.values)} (unsupported {len(plan.unsupported)}) sites {len(plan.chosen_sites())} "
          f"hooks {len(plan.hooks())} residuals {len(plan.residuals)}", flush=True)
    print(f"[gate]   correspondence source={cp.get('source')} programs={len(cp.get('programs') or {})} absent={len(cp.get('absent') or [])} module_paths={cp.get('module_paths')} "
          f"modules confirmed={(cp.get('modules') or {}).get('confirmed')} unconfirmed={len((cp.get('modules') or {}).get('unconfirmed') or [])} problems={len(cp.get('problems') or [])}", flush=True)
    for u in ((cp.get("modules") or {}).get("unconfirmed") or [])[:20]:
        print(f"[gate]   UNCONFIRMED {u['value']} ({u['role']} {u['module']}): {u['why']}", flush=True)
    for p in (cp.get("problems") or [])[:10]:
        print(f"[gate]   correspondence problem: {p[:200]}", flush=True)
    for v in plan.unsupported[:20]:
        print(f"[gate]   UNSUPPORTED {v.value_id}: {v.why[:160]}", flush=True)
    rc = 0 if plan.complete else 1
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
