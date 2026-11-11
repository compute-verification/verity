"""The plan at a GPU phase's entry: runtime tree of the built engine -> plan from the row's required-value manifest, both written beside
the stage's outputs; the caller installs (`install_from_plan`) or adapts the committer (`apply_plan_to_committer`).

    docs = plan_at_stage(model, runner, manifest_path, out_dir, source="m1_capture")
    docs.plan / docs.runtime / docs.paths

Writes DIR/runtime_tree.json (always: the CPU gate's off-pod input), and with a manifest DIR/acquisition_plan.json.  Fails closed by name
(`IncompletePlan`) when the plan has an UNSUPPORTED Value.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.observe.runtime_tree import RuntimeTree, runtime_tree_of
from verity_vllm.acquire.gate import run as gate_run
from verity_vllm.acquire.install import IncompletePlan
from verity_vllm.acquire.plan import AcquisitionPlan

MANIFEST_ENV = "ACQUIRE_MANIFEST"       # the row's required-value manifest path when the stage's CLI has no flag for it


@dataclass
class StageDocs:
    runtime: RuntimeTree
    plan: AcquisitionPlan | None = None
    paths: dict[str, str] = field(default_factory=dict)

    def log_lines(self, tag: str) -> list[str]:
        out = [f"[{tag}] runtime tree {len(self.runtime.modules)} modules, {len(self.runtime.buffers)} buffers -> {self.paths.get('runtime_tree')}"]
        if self.plan is not None:
            p = self.plan
            out.append(f"[{tag}] acquisition plan {p.digest[:16]}: {len(p.values)} Value classes, {len(p.unsupported)} unsupported, "
                       f"{len(p.hooks())} hooks, {len(p.residuals)} residuals -> {self.paths.get('acquisition_plan')}")
        return out


def plan_at_stage(model: Any, runner: Any, manifest: str | Path | Mapping[str, Any] | None, out_dir: str | Path, *, source: str,
                  refuse_incomplete: bool = True) -> StageDocs:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rt = runtime_tree_of(model, runner, source=source)
    rt.save(out / "runtime_tree.json")
    docs = StageDocs(runtime=rt, paths={"runtime_tree": str(out / "runtime_tree.json")})
    if manifest is None:
        return docs
    man = manifest if isinstance(manifest, Mapping) else json.loads(Path(manifest).read_text())
    plan = gate_run(man, rt)
    plan.save(out / "acquisition_plan.json")
    docs.plan = plan
    docs.paths["acquisition_plan"] = str(out / "acquisition_plan.json")
    if plan.unsupported and refuse_incomplete:
        raise IncompletePlan(f"{len(plan.unsupported)} unsupported Value(s): " + "; ".join(f"{v.value_id}: {v.why}" for v in plan.unsupported[:6]))
    return docs
