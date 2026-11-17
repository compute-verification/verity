"""`verity_torch.frontend.rules`' list: every rule of `TORCH_RULES` targets ATen ops only, and the list keeps the
sequence of its four parts (views, reference, and the PyTorch halves of profile and padding)."""
from __future__ import annotations

from verity_torch.frontend import rules as R
from verity_torch.frontend.rules import padding, profile, reference, views
from verity_torch.frontend.rules.base import ANY


def _namespaces(rule) -> set[str]:
    targets = rule.targets()
    assert targets is not ANY, f"{rule.info.rule_id} takes any target, so it belongs to no list"
    return {t.namespace for t in targets}


def _ids(rules) -> list[str]:
    return [r.info.rule_id for r in rules]


def test_every_torch_rule_targets_aten_ops_only():
    assert R.TORCH_RULES
    assert {r.info.rule_id: _namespaces(r) for r in R.TORCH_RULES if _namespaces(r) != {"aten"}} == {}


def test_torch_rules_keep_the_sequence_of_their_four_parts():
    ids = _ids(R.TORCH_RULES)
    assert len(set(ids)) == len(ids)
    assert ids == (_ids(views.VIEW_RULES) + _ids(reference.REFERENCE_RULES) + _ids(profile.PROFILE_TORCH_RULES)
                   + _ids(padding.PADDING_TORCH_RULES))
