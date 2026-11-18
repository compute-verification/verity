"""`verity_vllm.program.frontend.rules`' lists: every rule of `VLLM_RULES` targets an op that is not ATen's, the package's
`TORCH_RULES` is the PyTorch adapter's, the default rule set holds exactly the two lists' members, in the sequence of
its four lists (views, reference, profile, padding), and a training rule set holds none of `VLLM_RULES`."""
from __future__ import annotations

import pytest
from verity_torch.frontend import rules as T
from verity_torch.frontend.rules.base import ANY
from verity_torch.frontend.rules.training import training_rules

from verity_vllm.definitions.registry import cublas_bf16 as CB
from verity_vllm.program.frontend import rules as R
from verity_vllm.program.frontend.rules import padding, profile


def _namespaces(rule) -> set[str]:
    targets = rule.targets()
    assert targets is not ANY, f"{rule.info.rule_id} takes any target, so it belongs to neither list"
    return {t.namespace for t in targets}


def _ids(rules) -> list[str]:
    return [r.info.rule_id for r in rules]


def test_torch_rules_are_the_pytorch_adapters():
    assert R.TORCH_RULES is T.TORCH_RULES


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


@pytest.mark.parametrize("kw", [{}, {"cc": CB.TRAIN_CC}, {"cc": CB.TRAIN_CC, "tf32": True},
                                {"cc": CB.TRAIN_CC, "deterministic": True}, {"cc": CB.TRAIN_CC, "structure": True}],
                         ids=["cpu", "cc", "cc-tf32", "cc-deterministic", "cc-structure"])
def test_a_training_rule_set_holds_no_vllm_rule(kw: dict) -> None:
    """A training step's rule set (`training_rules`) names no op of vLLM's."""
    assert not {r.rule_id for r in training_rules(**kw).rules} & {r.rule_id for r in R.VLLM_RULES}
