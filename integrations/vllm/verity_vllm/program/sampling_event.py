"""[R15 stoch] The sampling-event families: the Program instances that END an engine step for one request (one sampled token per
step) — the step boundary `cb_a.compare` / `batch_decomp` / `global_match` segment at and the `sampled_token_ids` identity
the required-value manifest derives.  Torch-free; the commit and match tooling import it.

    TokenSelect_v1{V}                     greedy: first maximum of the bf16 logits row (torch.argmax / greedy _gumbel_sample_kernel)
    GumbelTopPTokenSelect_v1{V,S}         temperature + top-p + seeded Gumbel-max (registry.sampling; vLLM d9105ea80 stochastic path)
    GumbelTokenSelect_v1{V}               temperature + seeded Gumbel-max, top_p == 1 (registry.sampling, X-06)

Greedy is not stochastic: the families are distinct Definitions with distinct digests; this module only says which families are
"the sampling event of a step".  A spec may carry a `prefix:` (the fold's `spec` strings) and statics `{...}`.
"""
from __future__ import annotations

SAMPLING_EVENT_FAMILIES = ("TokenSelect_v1", "GumbelTopPTokenSelect_v1", "GumbelTokenSelect_v1")


def family_of(spec: str) -> str:
    """`prefix:Family_v1{K=..}` / `Family_v1{..}` / `Family_v1` -> `Family_v1`."""
    return spec.split(":", 1)[-1].split("{", 1)[0]


def is_sampling_event(spec: str) -> bool:
    return family_of(spec) in SAMPLING_EVENT_FAMILIES


__all__ = ["SAMPLING_EVENT_FAMILIES", "family_of", "is_sampling_event"]
