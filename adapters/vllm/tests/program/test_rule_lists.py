"""`verity_vllm.program.frontend.rules`' two named lists: every rule of `TORCH_RULES` targets ATen ops only, every rule of
`VLLM_RULES` targets an op that is not ATen's, and the default rule set holds exactly their members, in the sequence of its
four lists (views, reference, profile, padding)."""
from __future__ import annotations

from verity_vllm.program.frontend import rules as R
from verity_vllm.program.frontend.rules import padding, profile
from verity_vllm.program.frontend.rules.base import ANY


def _namespaces(rule) -> set[str]:
    targets = rule.targets()
    assert targets is not ANY, f"{rule.info.rule_id} takes any target, so it belongs to neither list"
    return {t.namespace for t in targets}


def _ids(rules) -> list[str]:
    return [r.info.rule_id for r in rules]


def test_every_torch_rule_targets_aten_ops_only():
    assert R.TORCH_RULES
    assert {r.info.rule_id: _namespaces(r) for r in R.TORCH_RULES if _namespaces(r) != {"aten"}} == {}


def test_every_vllm_rule_targets_an_op_that_is_not_atens():
    assert R.VLLM_RULES
    assert _ids(r for r in R.VLLM_RULES if not _namespaces(r) - {"aten"}) == []


def test_the_default_holds_the_two_lists_members_each_list_in_its_own_order():
    default = R.DEFAULT_RULES.rules
    torch_rules, vllm_rules = {id(r) for r in R.TORCH_RULES}, {id(r) for r in R.VLLM_RULES}
    assert not torch_rules & vllm_rules
    assert len(default) == len(R.TORCH_RULES) + len(R.VLLM_RULES)
    assert [r for r in default if id(r) in torch_rules] == R.TORCH_RULES
    assert [r for r in default if id(r) in vllm_rules] == R.VLLM_RULES


def test_the_default_keeps_the_sequence_of_its_four_lists():
    ids = _ids(R.DEFAULT_RULES.rules)
    assert len(set(ids)) == len(ids)
    assert ids == _ids(R.VIEW_RULES) + _ids(R.REFERENCE_RULES) + _ids(R.PROFILE_RULES) + _ids(R.PADDING_RULES)
    assert profile.PROFILE_RULES == profile.PROFILE_TORCH_RULES + profile.PROFILE_VLLM_RULES
    assert padding.PADDING_RULES == padding.PADDING_TORCH_RULES + padding.PADDING_VLLM_RULES
