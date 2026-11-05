"""Install EXACTLY the plan's hooks (module path -> hook kind -> slot) on a built module tree, and the committer adapter.

    inst = install_from_plan(plan, model, runner, sink=...)     forward hooks (producer_output), forward pre-hooks (consumer_input), the
                                                               runner `sample()` wrap (runner_return); `inst.installed` lists what went in,
                                                               `inst.deferred` the plan's sites of other kinds (taps, registered inputs,
                                                               runner state, launch proxies, prescribed): those belong to their sources
    inst.remove()

    apply_plan_to_committer(committer, plan)                    the `NativeHostCommitter` (and its native_collect subclass) takes its
                                                               module set and consumer-input set from the plan; the rest of the committer
                                                               (arena, hashing, roots, sources) is untouched
    plan_selection(plan, model)                                 that module set and consumer-input set, shared with the compiled workloads'
                                                               record ops (`CompiledGraphSource.wrap_model`)

A plan with an UNSUPPORTED Value is refused by name (`IncompletePlan`) unless the caller says `refuse_incomplete=False` -- fail closed.
`sink(site_key, tensor)` receives the observed tensor at each site; `OccurrenceSink` counts occurrences and records shapes only (no
value reads, no synchronisation): the GPU gate's non-interference observer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from verity_vllm.engine import hooks as engine_hooks
from verity_vllm.observe.runtime_tree import RUNNER_PATH
from verity_vllm.acquire.plan import HOOK_KINDS, AcquisitionPlan, Site, hooks_of_doc

Sink = Callable[[tuple[str, str, str], Any], None]


class IncompletePlan(RuntimeError):
    """The plan names UNSUPPORTED Values: nothing is installed."""


def _sites_of(plan: AcquisitionPlan | Mapping[str, Any]) -> tuple[list[Site], list[Site], list[dict[str, Any]]]:
    """(hook sites, deferred sites, unsupported) of a plan object or its serialised document."""
    if isinstance(plan, AcquisitionPlan):
        chosen = plan.chosen_sites()
        unsupported = [{"value": v.value_id, "why": v.why} for v in plan.unsupported]
    else:
        chosen = [Site(s["kind"], s["module"], s["slot"], s.get("lifetime", "n/a"), s.get("derived_from", "")) for s in (plan.get("sites") or [])]
        unsupported = list(plan.get("unsupported") or [])
        assert [s.key() for s in chosen if s.kind in HOOK_KINDS] == hooks_of_doc(plan)
    return [s for s in chosen if s.kind in HOOK_KINDS], [s for s in chosen if s.kind not in HOOK_KINDS], unsupported


def _slot_of_output(output: Any, slot: str) -> Any:
    if slot == "out":
        return output
    if slot.isdigit():
        k = int(slot)
        return output[k] if isinstance(output, (tuple, list)) and k < len(output) else None
    return None


def _slot_of_args(args: tuple, slot: str) -> Any:
    if slot.startswith("in[") and slot.endswith("]"):
        i = int(slot[3:-1])
        return args[i] if i < len(args) else None
    return None


@dataclass
class Installation:
    installed: list[Site] = field(default_factory=list)
    deferred: list[Site] = field(default_factory=list)
    missing: list[Site] = field(default_factory=list)        # hook sites whose module is not in the built tree (fail: the plan was made for another runtime)
    handles: list = field(default_factory=list)
    _runner: Any = None
    _sample_patch: Any = None

    def keys(self) -> list[tuple[str, str, str]]:
        return sorted(s.key() for s in self.installed)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        if self._sample_patch is not None:
            self._sample_patch.uninstall()
            self._runner, self._sample_patch = None, None


class OccurrenceSink:
    """Counts runtime occurrences per site and records (shape, dtype) of what was seen -- each tensor of a tuple, as a module's whole
    return (slot `out`) is when it returns several; never reads a value."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str, str], int] = {}
        self.shapes: dict[tuple[str, str, str], list[list[int]]] = {}
        self.dtypes: dict[tuple[str, str, str], set[str]] = {}
        self.non_tensor: dict[tuple[str, str, str], int] = {}

    def __call__(self, key: tuple[str, str, str], t: Any) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        ts = [x for x in (t if isinstance(t, (tuple, list)) else (t,)) if getattr(x, "shape", None) is not None]
        if not ts:
            self.non_tensor[key] = self.non_tensor.get(key, 0) + 1
            return
        lst = self.shapes.setdefault(key, [])
        for x in ts:
            sh = [int(n) for n in x.shape]
            if sh not in lst and len(lst) < 64:
                lst.append(sh)
            self.dtypes.setdefault(key, set()).add(str(getattr(x, "dtype", "?")).replace("torch.", ""))

    def to_json(self) -> dict[str, Any]:
        keys = sorted(set(self.counts) | set(self.non_tensor))
        return {"schema": "verity-vllm/acquisition-observed/v1", "sites": [
            {"module": k[0], "kind": k[1], "slot": k[2], "occurrences": self.counts.get(k, 0), "shapes": self.shapes.get(k, []),
             "dtypes": sorted(self.dtypes.get(k, ())), "non_tensor": self.non_tensor.get(k, 0)} for k in keys]}


def install_from_plan(plan: AcquisitionPlan | Mapping[str, Any], model: Any, runner: Any = None, *, sink: Sink | None = None,
                      refuse_incomplete: bool = True) -> Installation:
    hooks, deferred, unsupported = _sites_of(plan)
    if unsupported and refuse_incomplete:
        raise IncompletePlan(f"{len(unsupported)} unsupported Value(s): " + "; ".join(f"{u.get('value')}: {u.get('why')}" for u in unsupported[:6]))
    sink = sink or (lambda key, t: None)
    inst = Installation(deferred=list(deferred))
    mods = dict(model.named_modules()) if model is not None else {}
    for s in hooks:
        key = s.key()
        if s.kind == "producer_output":
            m = mods.get(s.module)
            if m is None:
                inst.missing.append(s)
                continue

            def hook(_m, _args, output, *, _key=key, _slot=s.slot):
                t = _slot_of_output(output, _slot)
                if t is not None:
                    sink(_key, t)

            inst.handles.append(m.register_forward_hook(hook))
            inst.installed.append(s)
        elif s.kind == "consumer_input":
            m = mods.get(s.module)
            if m is None:
                inst.missing.append(s)
                continue

            def pre(_m, args, *, _key=key, _slot=s.slot):
                t = _slot_of_args(args, _slot)
                if t is not None:
                    sink(_key, t)

            inst.handles.append(m.register_forward_pre_hook(pre))
            inst.installed.append(s)
        elif s.kind == "runner_return":
            if s.module != RUNNER_PATH or runner is None or not hasattr(runner, "sample"):
                inst.missing.append(s)
                continue
            if inst._runner is None:
                inst._runner = runner
                attrs: list[tuple[tuple[str, str, str], str]] = []

                def sample(*a, _orig=runner.sample, _attrs=attrs, **kw):
                    res = _orig(*a, **kw)
                    so = res[0] if isinstance(res, tuple) else res
                    for k, attr in _attrs:
                        t = getattr(so, attr, None)
                        if t is not None:
                            sink(k, t)
                    return res

                inst._sample_patch = engine_hooks.patch(runner, "sample", sample)
                inst._attrs = attrs  # type: ignore[attr-defined]
            inst._attrs.append((key, s.slot.lstrip(".")))  # type: ignore[attr-defined]
            inst.installed.append(s)
    if inst.missing:
        inst.remove()
        raise IncompletePlan("plan hook sites not in the built runtime: " + ", ".join(f"{s.module}[{s.kind}/{s.slot}]" for s in inst.missing[:8]))
    return inst


# ---------------------------------------------------------------------------------------------------------------------------------
# the committer adapter: the committer takes its hook sets from the plan
# ---------------------------------------------------------------------------------------------------------------------------------

def plan_selection(plan: AcquisitionPlan | Mapping[str, Any], model: Any) -> tuple[list[str], list[str]]:
    """The module boundaries the plan hooks on `model`: (producer_output modules in runtime order, consumer_input modules sorted) -- the
    ONE selection both the committer (`apply_plan_to_committer`) and the compiled workloads' record ops (`CompiledGraphSource.wrap_model`)
    take.  Refuses (`IncompletePlan`) a plan with UNSUPPORTED Values or with hook sites that are not modules of `model`."""
    hooks, _deferred, unsupported = _sites_of(plan)
    if unsupported:
        raise IncompletePlan(f"{len(unsupported)} unsupported Value(s): " + "; ".join(f"{u.get('value')}: {u.get('why')}" for u in unsupported[:6]))
    if model is None:
        raise IncompletePlan("no model to install on")
    order = {name: i for i, (name, _m) in enumerate(model.named_modules())}
    out_mods = {s.module for s in hooks if s.kind == "producer_output"}
    in_mods = {s.module for s in hooks if s.kind == "consumer_input"}
    absent = sorted(m for m in out_mods if m not in order) + sorted(m for m in in_mods if m not in order)
    if absent:
        raise IncompletePlan(f"plan hook sites not in the built model: {absent[:8]}")
    return sorted(out_mods, key=lambda n: order[n]), sorted(in_mods)


def _values_at(plan: AcquisitionPlan | Mapping[str, Any], site: Site) -> list[str]:
    if isinstance(plan, AcquisitionPlan):
        return plan.values_at(site)
    return next((list(h.get("values") or []) for h in (plan.get("hooks") or []) if (h["module"], h["kind"], h["slot"]) == site.key()), [])


def apply_plan_to_committer(committer: Any, plan: AcquisitionPlan | Mapping[str, Any]) -> dict[str, Any]:
    """The committer's selection from the plan (`plan_selection`): `_modules` = the producer_output sites' modules
    (runtime order), `plan_inputs` = the consumer_input sites' modules.  The promoted-member arming (`acquire_inputs` bytes per token)
    stays the manifest's (commit_delta sets it); the plan must name every armed consumer, else the plan is refused.
    A VERITY_SKIP ablation (`committer.skip_classes`) is applied to THIS selection (`apply_skip`, refused by name when it cannot be); the
    record names the dropped modules and the Values left unacquired.  Returns a record."""
    hooks, deferred, _unsupported = _sites_of(plan)
    model = getattr(committer, "model", None)
    if model is None:
        raise IncompletePlan("the committer has no model to install on")
    out_mods, in_mods = plan_selection(plan, model)
    armed = set(getattr(committer, "acquire_inputs", {}) or {})
    if not armed <= set(in_mods):
        raise IncompletePlan(f"the manifest arms consumer inputs the plan does not name: {sorted(armed - set(in_mods))[:8]}")
    mods = dict(model.named_modules())
    committer._modules = [(m, mods[m]) for m in out_mods]
    committer.plan_inputs = set(in_mods)
    committer.name = f"{committer.name}+plan"
    rec = {"engine": "v2", "modules_after": list(out_mods), "plan_inputs": list(in_mods),
           "deferred_sites": [s.key() for s in deferred], "runner_return_sites": [s.key() for s in hooks if s.kind == "runner_return"]}
    if getattr(committer, "skip_classes", None):
        dropped = committer.apply_skip()
        gone = {n for names in dropped.values() for n in names}
        rec["modules_after"] = [n for n, _m in committer._modules]
        rec["skip"] = {"classes": sorted(dropped), "modules": dropped,
                       "values_unacquired": sorted(v for s in hooks if s.kind == "producer_output" and s.module in gone for v in _values_at(plan, s))}
    return rec
