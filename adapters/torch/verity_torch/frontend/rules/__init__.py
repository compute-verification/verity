"""The lowering rules on ATen ops: `TORCH_RULES`, the rules whose targets are ATen ops only (views, reference, then profile's
and padding's rules on ATen ops, each list in its own order), and `vocabulary_for`, the kinds a rule may emit under a mode and
profile. An adapter's rules for its own ops, and its default rule set, are the adapter's (the vLLM adapter's
`verity_vllm.program.frontend.rules`).

The rule modules import `..derive`, and `derive` imports `rules.base`, so these load at first use (PEP 562) to keep the
package import acyclic."""

from __future__ import annotations

_LAZY = ("TORCH_RULES", "vocabulary_for")
_cache: dict[str, object] = {}


def _load() -> None:
    if _cache:
        return
    from verity_torch.frontend.rules.padding import PADDING_TORCH_RULES
    from verity_torch.frontend.rules.profile import PROFILE_TORCH_RULES
    from verity_torch.frontend.rules.reference import REFERENCE_RULES
    from verity_torch.frontend.rules.views import VIEW_RULES
    from verity_torch.frontend.rules.vocab import vocabulary_for
    _cache.update(TORCH_RULES=list(VIEW_RULES) + list(REFERENCE_RULES) + PROFILE_TORCH_RULES + PADDING_TORCH_RULES,
                  vocabulary_for=vocabulary_for)


def __getattr__(name: str):
    if name in _LAZY:
        _load()
        return _cache[name]
    raise AttributeError(name)


__all__ = ["TORCH_RULES", "vocabulary_for"]
