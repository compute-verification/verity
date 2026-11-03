"""A survey of a derivation's refusals (`with_holes`): every node the rules refuse becomes a hole, its outputs fields of the
Program's `run` input, so one derivation lists every refusal (`holes`) and its Program counts everything else.

    with_holes(ruleset) -> RuleSet                     the same rules, each refused node a hole
    holes(report) -> list[dict]                        a survey's refusals
"""

from __future__ import annotations

from verity_vllm.program.frontend.derive import _no_rule
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.outcomes import Refuse
from verity_vllm.program.frontend.rules.base import ANY, Rule, RuleInfo, RuleSet
from verity_vllm.program.frontend.value_types import type_of_tensor



class Hole(Rule):
    """Survey only: a node every rule refuses takes its outputs from fields of the `run` input, and the refusal is recorded."""
    info = RuleInfo("Hole[train]", 1, "op-binding", applies_to="any node the other rules refuse (survey only)",
                    preconditions=("every other candidate refused",), evidence_domain="none: a hole is no derivation")

    def targets(self):
        return ANY

    def run_inputs(self, ep, mode):
        return True

    def applies(self, node, ctx):
        return True

    def emit(self, node, ctx, error: Exception | None = None):
        if error is None:
            reasons = []
            for r in ctx.ruleset.for_target(node.target, ctx.mode):
                if r is not self:
                    v = r.applies(node, ctx)
                    reasons.append(f"{r.rule_id}: {'accepted, then refused' if v is True else v}")
            try:
                _no_rule(ctx, node, reasons)
            except Refuse as e:
                u = e.u
                reason, hint = u.reason, u.hint
        elif isinstance(error, Refuse):
            reason, hint = error.u.reason, error.u.hint
        else:
            reason, hint = "unsupported_view", f"{getattr(error, 'code', '')}: {error}"
        meta = node.meta.get("val")
        k = len(ctx.state.setdefault("train_holes", []))
        ctx.state["train_holes"].append(node.name)

        def hole(i, m):
            return ctx.run_value(f"hole{k}.{i}:{node.name}", type_of_tensor(m), node)

        ctx.define(node, [hole(i, m) for i, m in enumerate(meta)] if isinstance(meta, (tuple, list)) else hole(0, meta))
        ctx.record(self, node, {}, emitted="run input", reason=reason, hint=str(hint))


def _guarded(rule: Rule, hole: Hole) -> Rule:
    """`rule`, whose refusals while emitting become holes (the node's partial emission is removed first)."""
    cls = type(rule)

    def applies(self, node, ctx):
        try:
            return cls.applies(self, node, ctx)
        except (Refuse, FrontendError) as e:
            return f"refused while deciding: {getattr(e, 'u', e)}"

    def emit(self, node, ctx):
        k0, a0, e0 = len(ctx.B.nodes), len(ctx.applications), len(ctx.env)
        try:
            cls.emit(self, node, ctx)
        except (Refuse, FrontendError) as e:
            del ctx.B.nodes[k0:]
            del ctx.B.offsets[k0 + 1:]
            del ctx.applications[a0:]
            for d in (ctx.node_hints, ctx.node_sources, ctx.node_fx):
                for k in [k for k in d if k >= k0]:
                    del d[k]
            for key in list(ctx.env)[e0:]:
                del ctx.env[key]
            hole.emit(node, ctx, e)

    g = object.__new__(type(f"Guarded{cls.__name__}", (cls,), {"applies": applies, "emit": emit}))
    g.__dict__.update(rule.__dict__)
    return g


def with_holes(ruleset: RuleSet) -> RuleSet:
    """`ruleset` for a survey: every refused node becomes a hole (`Hole`), so one derivation lists every refusal."""
    hole = Hole()
    return RuleSet(f"{ruleset.name}+holes", [_guarded(r, hole) for r in ruleset.rules] + [hole], list(ruleset.families))


def holes(report) -> list[dict]:
    """The holes of a survey's report: node, op, reason and hint of each refusal."""
    prov = report.provenance
    apps = prov["applications"] if "applications" in prov else prov.get("applications_before_refusal", [])
    return [{"node": a["node"], "op": a["op"], "reason": a.get("reason"), "hint": a.get("hint")} for a in apps
            if a["rule"] == Hole.info.rule_id]



__all__ = ["Hole", "holes", "with_holes"]
