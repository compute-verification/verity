"""The norm scale as a required value: one 32-bit word per norm Call per row, the one committed interior value `Q_word_v1{16,32}`
derives for a norm.  Opt-in (`manifest build --norm-scales`, the row's NORM_TAP): with it off nothing here runs, and no manifest,
Program, root or verdict changes.

Two kinds of norm carry the scale:

* kernel norms (`KERNEL_SCALES`): `RMSNormTriton_v1` (vLLM's batch-invariant Triton `_rms_norm_kernel`, whose `inv_rms` is the Call's
  `inv` gate) and `RMSNormFusedCuda_v2` (vLLM's generic `fused_add_rms_norm_kernel`, whose `s_variance` is the `RsqrtApprox` gate).
  The scale is interior to the Call, so no Program Value carries it: `kernel_rows` adds one `norm_scales` identity per (step, norm
  module), ranging over the module's Call rows at that step, and the norm-scale tap acquires it (`acquire.sources.norm_scale_source`);
* an ATen-chain norm (`CALL_SCALES`): the unfused chain's `RsqrtF32_v1{N=1}` Call, whose output is the row's scale and is interior to
  the norm's module body.  `population` makes it protocol-required and spells it `norm_scale` in the `norm_scales` family.

Either way the committed tensor is `<norm module>/norm_scale`: one f32 per row, in the norm input's row order.  The Match record holds
no value for it, so form (B) does not compare it (`check.oracle_compare`, by that member).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Mapping

FAMILY = "norm_scales"
MEMBER = "norm_scale"

#: the norm Definitions whose scale is interior to the Call -> the kernel the tap replaces for it
KERNEL_SCALES: dict[str, str] = {"RMSNormTriton_v1": "triton", "RMSNormFusedCuda_v2": "cuda"}
#: the Definitions whose output is a norm's scale when bound per row (`N` = 1) -> where the source takes it
CALL_SCALES: dict[str, str] = {"RsqrtF32_v1": "aten"}
#: `Q_word_v1` committed interior words per Call that the tap acquires, by Definition family
TAP_WORDS: dict[str, int] = {fam: 1 for fam in KERNEL_SCALES}

POLICY_TEXT = ("norm scales: one f32 per norm Call per row, `<norm module>/norm_scale` (family norm_scales): the interior scale of "
               + ", ".join(sorted(KERNEL_SCALES)) + " (norm-scale tap) and the output of " + ", ".join(sorted(CALL_SCALES))
               + " bound per row (protocol-required)")


def site_of(family: str) -> str | None:
    """Where the scale of a norm Definition family is taken: `triton` | `cuda` (a tapped kernel) | `aten` (a Call's output) | None."""
    return KERNEL_SCALES.get(family) or CALL_SCALES.get(family)


def scale_calls(P: Any, literals: frozenset[int] = frozenset()) -> list[Any]:
    """The Calls of `P` whose output is a norm scale (`CALL_SCALES` bound per row)."""
    return [c for c in P.calls if c.family in CALL_SCALES and c.id not in literals and int(c.statics.get("N", 0)) == 1]


@lru_cache(maxsize=None)
def _population_class() -> type:
    from verity.proofs.query import Policy, RequiredValues, ValueRef
    from verity_vllm.query.required import DTYPE_OF_WIDTH, CallMember, Population, Residual

    residual = Residual("*", "<norm scale>", MEMBER, "a norm's scale is the one f32 per row its rsqrt produces inside the norm's module "
                        "body; under the norm-scale policy it is required as `<norm module>/norm_scale`, family norm_scales", kind="encoding")

    class NormScalePopulation(Population):
        """`Population` under the norm-scale policy: the outputs of `scale_calls` are protocol-required and spelt `norm_scale`."""

        RESIDUAL = residual

        def __init__(self, P: Any, corr: Any, **kw: Any) -> None:
            super().__init__(P, corr, **kw)
            base = self.required.values
            extra = frozenset(ValueRef(c.id, p.name) for c in scale_calls(P, self.literals) for p in c.port_table) - base
            self.norm_scale_values = extra
            self.policy = Policy(prescribed_inputs=self.policy.prescribed_inputs, protocol_required=self.policy.protocol_required | extra)
            self.required = RequiredValues(self.B.values, self.policy.prescribed_inputs, self.policy.protocol_required)
            self._ports_of = None

        def member(self, v: Any) -> Any:
            if v not in self.norm_scale_values:
                return super().member(v)
            port = self.P.call(v.call).port(v.port)
            return CallMember(port.name, MEMBER, port.elems, DTYPE_OF_WIDTH.get(port.width, f"w{port.width}"), residual.structural,
                              residual, FAMILY)

    return NormScalePopulation


def population(P: Any, corr: Any, *, return_slots: bool = True, tp: Mapping[str, Any] | None = None) -> Any:
    """`required.population` under the norm-scale policy."""
    return _population_class()(P, corr, return_slots=return_slots, tp=tp)


def kernel_rows(pop: Any, add: Callable[..., None], impl: Callable[[str], str], aliases: Mapping[str, tuple[str, int]]) -> dict[str, Any]:
    """Add one `norm_scales` identity per (step, norm module) of the kernel norms (`KERNEL_SCALES`): its range is the module's Call rows at
    that step, one scale each.  `add` is `request_manifest`'s row writer.  Returns the words by Definition family, the ATen-chain
    scales (`pop.norm_scale_values`) included."""
    groups: dict[tuple[int, str], list] = {}
    for c in pop.P.calls:
        if c.family not in KERNEL_SCALES:
            continue
        key = (pop.step_of(c.id), pop.program_path(c.id))
        g = groups.setdefault(key, [c.family, 0])
        if g[0] != c.family:
            raise ValueError(f"norm scales: module {impl(key[1])} step {key[0]} runs two norm kernels ({g[0]}, {c.family})")
        g[1] += 1
    words: Counter = Counter()
    for (step, mod), (fam, rows) in sorted(groups.items()):
        canon, k = aliases.get(mod, (mod, 0))
        add(FAMILY, step, k, impl(canon), MEMBER, 0, rows, rows=rows, width=1, dtype="f32", spec=fam)
        words[fam] += rows
    for v in getattr(pop, "norm_scale_values", ()):
        words[pop.P.call(v.call).family] += pop.P.call(v.call).port(v.port).elems
    return {"words": sum(words.values()), "words_by_family": dict(sorted(words.items()))}


def header(q: dict[str, Any], stats: Mapping[str, Any] | None) -> dict[str, Any]:
    """The manifest's query header under the norm-scale policy (`q` itself when the policy is off, `stats` None)."""
    if stats is None:
        return q
    text = f"{q['query_text']}; {POLICY_TEXT}"
    out = dict(q, query_text=text, query_id_digest=hashlib.sha256(text.encode()).hexdigest(),
               required_families=[*q.get("required_families", []), FAMILY])
    out["named_residuals"] = [*q.get("named_residuals", []), _population_class().RESIDUAL.as_dict()]
    out["norm_scales"] = {"family": FAMILY, "member": MEMBER, "kernel_scales": dict(KERNEL_SCALES), "call_scales": dict(CALL_SCALES), **stats}
    return out


class Policy:
    """The norm-scale policy as `required.request_manifest(norm_scales=POLICY)` applies it (the query imports nothing from here)."""

    population = staticmethod(population)
    kernel_rows = staticmethod(kernel_rows)
    header = staticmethod(header)


POLICY = Policy()


def policy_of(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest was built under the norm-scale policy (its query header says so)."""
    q = (manifest or {}).get("query")
    return isinstance(q, dict) and isinstance(q.get("norm_scales"), dict)


def manifest_modules(manifest: Mapping[str, Any] | None) -> dict[str, str]:
    """op_path -> the norm Definition family of every `norm_scales` identity of a manifest; a module named under two families is refused."""
    out: dict[str, str] = {}
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") != FAMILY:
            continue
        op, fam = str(r["op_path"]), str(r.get("spec") or "").split("{", 1)[0]
        if site_of(fam) is None:
            raise ValueError(f"norm scales: {op} is named with {fam!r}, which carries no norm scale")
        if out.setdefault(op, fam) != fam:
            raise ValueError(f"norm scales: {op} is named with two norm families ({out[op]}, {fam})")
    return out


def words_per_token(manifest: Mapping[str, Any] | None) -> int:
    """The most scale words any step of the manifest commits per token row (q/k norms commit one per head): the source's arena bound."""
    tokens: Counter = Counter()
    words: Counter = Counter()
    for r in (manifest or {}).get("identities") or []:
        key = (str(r.get("request_id")), str(r.get("engine_step", r.get("step"))))
        if r.get("family") == "input_token_ids":
            tokens[key] = max(tokens[key], int(r.get("rows") or 0))
        elif r.get("family") == FAMILY:
            words[key] += int(r.get("rows") or 0)
    return max((-(-w // tokens[k]) for k, w in words.items() if tokens.get(k)), default=0)


__all__ = ["FAMILY", "MEMBER", "KERNEL_SCALES", "CALL_SCALES", "TAP_WORDS", "POLICY_TEXT", "POLICY", "Policy", "site_of", "scale_calls",
           "population", "kernel_rows", "header", "policy_of", "manifest_modules", "words_per_token"]
