"""The plan at a GPU stage's entry (behind `ACQUIRE_ENGINE=v2`): runtime tree of the built engine -> plan from the row's required-value
manifest -> the CPU gate's comparison, all written beside the stage's outputs; the caller installs (`install_from_plan`) or adapts the
committer (`apply_plan_to_committer`).

    docs = plan_at_stage(model, runner, manifest_path, out_dir, source="m1_capture")
    docs.plan / docs.runtime / docs.compare / docs.paths

Writes DIR/runtime_tree.json (always: the CPU gate's off-pod input), and with a manifest DIR/acquisition_plan.json, DIR/v1_decision.json,
DIR/site_compare.json.  Fails closed by name (`IncompletePlan`) when the plan has an UNSUPPORTED Value; an UNEXPLAINED site difference to v1
is recorded (the compare document, the log), never hidden.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.acquire.correspondence import RuntimeTree, runtime_tree_of
from verity_vllm.acquire.gate import run as gate_run
from verity_vllm.acquire.install import IncompletePlan
from verity_vllm.acquire.plan import AcquisitionPlan
from verity_vllm.acquire.switch import ENV, engine

MANIFEST_ENV = "ACQUIRE_MANIFEST"       # the row's required-value manifest path when the stage's CLI has no flag for it


@dataclass
class StageDocs:
    engine: str
    runtime: RuntimeTree
    plan: AcquisitionPlan | None = None
    compare: dict[str, Any] | None = None
    paths: dict[str, str] = field(default_factory=dict)

    def log_lines(self, tag: str) -> list[str]:
        out = [f"[{tag}] {ENV}={self.engine}: runtime tree {len(self.runtime.modules)} modules, {len(self.runtime.buffers)} buffers -> {self.paths.get('runtime_tree')}"]
        if self.plan is not None and self.compare is not None:
            c = self.compare["counts"]
            out.append(f"[{tag}] acquisition plan {self.plan.digest[:16]}: {c['plan_values']} Value classes, {c['plan_unsupported']} unsupported, "
                       f"{len(self.plan.hooks())} hooks, {c['plan_residuals']} residuals -> {self.paths.get('acquisition_plan')}")
            out.append(f"[{tag}] site compare vs v1: common {c['common']} only_v1 {c['only_v1']} only_plan {c['only_plan']} UNEXPLAINED {c['unexplained']} -> {self.paths.get('site_compare')}")
            for df in self.compare["differences"][:12]:
                out.append(f"[{tag}]   {df['in']:<4} {df['site']}: {df['cause'][:150]}")
        return out


def plan_at_stage(model: Any, runner: Any, manifest: str | Path | Mapping[str, Any] | None, out_dir: str | Path, *, source: str,
                  row_id: str | None = None, refuse_incomplete: bool = True) -> StageDocs:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rt = runtime_tree_of(model, runner, source=source)
    rt.save(out / "runtime_tree.json")
    docs = StageDocs(engine=engine(), runtime=rt, paths={"runtime_tree": str(out / "runtime_tree.json")})
    if manifest is None:
        return docs
    man = manifest if isinstance(manifest, Mapping) else json.loads(Path(manifest).read_text())
    plan, d, cmp = gate_run(man, rt, row_id=row_id or ((man.get("workload") or {}).get("row_id") if isinstance(man.get("workload"), dict) else None))
    plan.save(out / "acquisition_plan.json")
    (out / "v1_decision.json").write_text(json.dumps(d.to_json(), indent=1) + "\n")
    (out / "site_compare.json").write_text(json.dumps(cmp, indent=1) + "\n")
    docs.plan, docs.compare = plan, cmp
    docs.paths.update({"acquisition_plan": str(out / "acquisition_plan.json"), "v1_decision": str(out / "v1_decision.json"), "site_compare": str(out / "site_compare.json")})
    if plan.unsupported and refuse_incomplete:
        raise IncompletePlan(f"{len(plan.unsupported)} unsupported Value(s): " + "; ".join(f"{v.value_id}: {v.why}" for v in plan.unsupported[:6]))
    return docs
