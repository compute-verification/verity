"""Lowering rules (the class names `ViewRule` / `OpRule` / ... keep the older name 'Rule').
`DEFAULT_RULES` is the versioned default rule set: view rules (no gates), reference-mode op rules, profile-mode
op / in-place / state rules, and the registered program families (`family.py`) whose identity a derivation may claim after verified
isomorphism.  Each rule application emits one Call of a registered Definition; the (rule, node, module path) record is the runtime
correspondence's Build-side half.

The rule modules import `..derive` (for `Ctx`, `refuse`, ...), and `derive` imports `rules.base`; the concrete
rule sets are therefore resolved lazily (PEP 562) to keep the package import acyclic.

PROTOCOL SUMMARY (Build phase: the lowering rules that turn vLLM's exported ATen graph into a Program over registered primitives)
  Purpose: `DEFAULT_RULES` is the versioned rule set `verity_vllm.program.frontend.derive` applies node by node to the `torch.export(strict=False)`
    graph of the vLLM model class: ViewRules (reference rearrangements, no gates), reference-mode and profile-mode OpRules / InPlaceRules /
    StateRules (each emits a Call of a REGISTERED Definition, e.g. rules/vllm_bindings/ for the Triton / unified-attention custom ops),
    PaddingRules, and FamilyRules that may claim a registered program identity after a verified isomorphism.
  Inputs: the exported graph nodes + `Ctx` (mode, profile, kind vocabulary from rules/vocab.py: PROFILE_B1_EAGER | REFERENCE_CPU_F32).
  Outputs: lowering-rule applications recorded in the Program's provenance (derive-report.json: rule, node, module path, emitted kind) and the
    Program itself (descriptor.json.gz + instances.json.gz via derive_step); a node no rule accepts is `Unsupported(reason)`, never a gate.
  Invariants: rules decide structurally (`applies(node, ctx)`), never by op name alone; exactly one applicable rule translates a node;
    a refusal names one of derive.REASONS (mutation, decomposition, custom_op_without_binding, side_input, dynamic_control_flow,
    numerics_unregistered, hidden_state, unsupported_view, shape, export, identity); the rule set + versions go into the Program's
    provenance (`annotations["derive"]`), which `verity.ml.circuits.codec.program_digest` EXCLUDES -- the digest covers the dataflow only."""

from __future__ import annotations

from verity_vllm.program.frontend.rules.base import ANY, FamilyRule, InPlaceRule, ModuleRule, OpRule, Rule, RuleInfo, RuleSet, StateRule, ViewRule
from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER, REFERENCE_CPU_F32, Vocabulary, vocabulary_for

_LAZY = ("DEFAULT_RULES", "VIEW_RULES", "REFERENCE_RULES", "PROFILE_RULES", "PADDING_RULES", "DEFAULT_FAMILIES")
_cache: dict[str, object] = {}


def _load() -> None:
    if _cache:
        return
    from verity_vllm.program.frontend.rules.views import VIEW_RULES
    from verity_vllm.program.frontend.rules.reference import REFERENCE_RULES
    from verity_vllm.program.frontend.rules.profile import PROFILE_RULES
    from verity_vllm.program.frontend.rules.padding import PADDING_RULES
    from verity_vllm.program.frontend.rules.family import DEFAULT_FAMILIES
    _cache.update(VIEW_RULES=VIEW_RULES, REFERENCE_RULES=REFERENCE_RULES, PROFILE_RULES=PROFILE_RULES, PADDING_RULES=PADDING_RULES,
                  DEFAULT_FAMILIES=DEFAULT_FAMILIES,
                  DEFAULT_RULES=RuleSet("default", list(VIEW_RULES) + list(REFERENCE_RULES) + list(PROFILE_RULES) + list(PADDING_RULES),
                                        list(DEFAULT_FAMILIES)))


def __getattr__(name: str):
    if name in _LAZY:
        _load()
        return _cache[name]
    raise AttributeError(name)


__all__ = ["DEFAULT_RULES", "RuleSet", "Rule", "RuleInfo", "OpRule", "ViewRule", "InPlaceRule", "StateRule", "ModuleRule", "FamilyRule", "ANY",
           "Vocabulary", "vocabulary_for", "PROFILE_B1_EAGER", "REFERENCE_CPU_F32", "VIEW_RULES", "REFERENCE_RULES", "PROFILE_RULES",
           "DEFAULT_FAMILIES"]
