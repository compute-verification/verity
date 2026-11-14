"""`derive` and `derive_report` with the vLLM adapter's rules: the PyTorch adapter's derivation
(`verity_torch.frontend.derive`) under the default rule set `rules.DEFAULT_RULES` where the caller names none, the
vocabulary of vLLM's profiles (`rules.vocabulary_for`) and, after the walk, the two served rules: the seed rule
(`seed_rule.in_derive`), then the length rule under the config's ``stop`` (`stop_rule.in_derive`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity_torch.frontend import derive as _derive
from verity_torch.frontend.outcomes import Unsupported
from verity_torch.frontend.report import DeriveReport
from verity_vllm.program.frontend import seed_rule, stop_rule
from verity.circuits.program import Program


def _seed(ctx: Any, outs: list, at: Any) -> None:
    seed_rule.in_derive(ctx, at)


def _stop(ctx: Any, outs: list, at: Any) -> None:
    stop_rule.in_derive(ctx, outs, at)


END_RULES = (_seed, _stop)


def derive_report(source, example_inputs: Sequence[Any], *, rules=None, **kw: Any) -> DeriveReport:
    """`verity_torch.frontend.derive.derive_report` with the vLLM adapter's rule set, vocabulary and served rules."""
    from verity_vllm.program.frontend.rules import DEFAULT_RULES, vocabulary_for
    return _derive.derive_report(source, example_inputs, rules=rules if rules is not None else DEFAULT_RULES,
                                 vocabulary=vocabulary_for, end_rules=END_RULES, **kw)


def derive(source, example_inputs: Sequence[Any], **kw: Any) -> Program | Unsupported:
    """`derive_report`'s result: the `Program`, or the `Unsupported` that refuses it."""
    return derive_report(source, example_inputs, **kw).result


__all__ = ["END_RULES", "derive", "derive_report"]
