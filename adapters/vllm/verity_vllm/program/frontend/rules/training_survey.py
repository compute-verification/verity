"""A survey of a derivation's refusals (`survey`): every node the rules refuse becomes a hole (`with_holes`), its outputs fields of
the Program's `run` input, so one derivation lists every refusal and its Program counts everything else.

A hole is a value no gate checks, which a prover would choose: a Program with holes is no statement. Its holes are in its
annotations (`annotations["survey_holes"]`), `check_identity` refuses it, and `derive_report`'s tail, which runs `check_identity`,
returns no Program for a derivation that left holes. `survey` builds it from the flat body for a count, without that tail.

    with_holes(ruleset) -> RuleSet                     the same rules, each refused node a hole
    survey(graph, example_inputs, *, mode, rules, ...) -> (holes, Program | Unsupported)
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from verity.circuits.program import Program
from verity_vllm.program.frontend.derive import _no_rule
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.outcomes import _CODE_TO_REASON, Refuse, Unsupported
from verity_vllm.program.frontend.rules.base import ANY, Rule, RuleInfo, RuleSet
from verity_vllm.program.frontend.value_types import type_of_tensor



class Hole(Rule):
    """Survey only: a node every rule refuses takes its outputs from fields of the `run` input, and the refusal is recorded
    (`ctx.state["survey_holes"]`: node, op, reason, hint and the fields)."""
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
        holes = ctx.state.setdefault("survey_holes", [])
        k, fields = len(holes), []

        def hole(i, m):
            fields.append(f"hole{k}.{i}:{node.name}")
            return ctx.run_value(fields[-1], type_of_tensor(m), "survey_hole", node)

        ctx.define(node, [hole(i, m) for i, m in enumerate(meta)] if isinstance(meta, (tuple, list)) else hole(0, meta))
        holes.append({"node": node.name, "op": str(node.target), "reason": reason, "hint": str(hint), "fields": fields})
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


def survey(graph: torch.fx.GraphModule, example_inputs: Sequence[Any], *, mode: str, rules: RuleSet, profile: str | None = None,
           name: str | None = None, input_names: Sequence[str] | None = None,
           theta: Sequence[str] = ()) -> tuple[list[dict], Program | Unsupported]:
    """`derive_graph`'s derivation of `graph` with every refused node a hole (`with_holes(rules)`), to its flat body: the holes, and
    the Program of the rest marked with them (`annotations["survey_holes"]`, which `check_identity` refuses); or, with the holes
    before it, the `Unsupported` of a refusal no hole stands in for (one outside the rules; a graph with no refusal at all is
    `side_input`, its `run` input unread: `derive_graph` derives it). No correspondence, digest or identity check: a count needs
    none of them."""
    from verity_vllm.program.frontend.derive import _derive_flat
    from verity_vllm.program.frontend.derive_graph import graph_export
    from verity_vllm.program.frontend.rules import vocabulary_for

    ep, source = graph_export(graph, input_names, theta, name)
    holder: list = []
    try:
        flat, ctx = _derive_flat(source, example_inputs, mode=mode, profile=profile, config={}, allowed_dims={}, name=source.__name__,
                                 ruleset=with_holes(rules), vocab=vocabulary_for(mode, profile), _ctx_holder=holder, exported=ep)
    except Refuse as e:
        return (holder[0].state.get("survey_holes", []) if holder else []), e.u
    except FrontendError as e:
        u = Unsupported(_CODE_TO_REASON.get(e.code, "shape"), None, e.where, f"{e.code}: {e.message}")
        return (holder[0].state.get("survey_holes", []) if holder else []), u
    holes = ctx.state.get("survey_holes", [])
    return holes, Program(flat, annotations={"survey_holes": holes}, theta=tuple(theta))



__all__ = ["Hole", "survey", "with_holes"]
