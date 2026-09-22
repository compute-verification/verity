"""Derivation rules (design §4, GUIDANCE §43.3). `DEFAULT_RULES` is the versioned default rule set: view rules
(no gates), reference-mode op rules, profile-mode op / in-place / state rules, and the registered program
families (`family.py`) whose identity a derivation may claim after verified isomorphism.

The rule modules import `..derive` (for `Ctx`, `refuse`, ...), and `derive` imports `rules.base`; the concrete
rule sets are therefore resolved lazily (PEP 562) to keep the package import acyclic."""

from __future__ import annotations

from verity_vllm.frontend.rules.base import ANY, FamilyRule, InPlaceRule, ModuleRule, OpRule, Rule, RuleInfo, RuleSet, StateRule, ViewRule
from verity_vllm.frontend.rules.vocab import PROFILE_B1_EAGER, REFERENCE_CPU_F32, Vocabulary, vocabulary_for

_LAZY = ("DEFAULT_RULES", "VIEW_RULES", "REFERENCE_RULES", "PROFILE_RULES", "PADDING_RULES", "DEFAULT_FAMILIES")
_cache: dict[str, object] = {}


def _load() -> None:
    if _cache:
        return
    from verity_vllm.frontend.rules.views import VIEW_RULES
    from verity_vllm.frontend.rules.reference import REFERENCE_RULES
    from verity_vllm.frontend.rules.profile import PROFILE_RULES
    from verity_vllm.frontend.rules.padding import PADDING_RULES
    from verity_vllm.frontend.rules.family import DEFAULT_FAMILIES
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
