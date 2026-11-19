"""The plan at a GPU phase's entry: runtime tree of the built engine -> plan of the capture plan's classes, both written beside
the stage's outputs; the caller installs (`install_from_plan`) or adapts the committer (`apply_plan_to_committer`).

    docs = plan_at_stage(model, runner, classes, out_dir, source="commit serve")
    docs.plan / docs.runtime / docs.paths

Writes DIR/runtime_tree.json (always: the CPU gate's off-pod input), and with classes DIR/acquisition_plan.json.  Fails closed by name
(`IncompletePlan`) when the plan has an UNSUPPORTED Value.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.acquire.gate import run_plan as gate_run_plan
from verity_vllm.acquire.install import IncompletePlan
from verity_vllm.acquire.plan import AcquisitionPlan
from verity_vllm.records.runtime_tree import RuntimeTree, runtime_tree_of


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


def plan_at_stage(model: Any, runner: Any, capture: Sequence[Mapping[str, Any]] | None, out_dir: str | Path, *,
                  source: str) -> StageDocs:
    """The runtime tree and, when `capture` is given, the acquisition plan of the capture plan's classes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rt = runtime_tree_of(model, runner, source=source)
    rt.save(out / "runtime_tree.json")
    docs = StageDocs(runtime=rt, paths={"runtime_tree": str(out / "runtime_tree.json")})
    if capture is None:
        return docs
    plan = gate_run_plan(capture, rt)
    plan.save(out / "acquisition_plan.json")
    docs.plan = plan
    docs.paths["acquisition_plan"] = str(out / "acquisition_plan.json")
    if plan.unsupported:
        raise IncompletePlan(f"{len(plan.unsupported)} unsupported Value(s): " + "; ".join(f"{v.value_id}: {v.why}" for v in plan.unsupported[:6]))
    return docs


__all__ = ["plan_at_stage"]
