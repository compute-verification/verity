"""[R15 stoch] The sampling-event families: the Program instances that END an engine step for one request (one sampled token per
step) — the step boundary `cb_a.compare` / `batch_decomp` / `global_match` segment at and the `sampled_token_ids` identity
the required-value manifest derives.  Torch-free; the commit and match tooling import it.

    TokenSelect_v1{V}                     greedy: first maximum of the bf16 logits row (torch.argmax / greedy _gumbel_sample_kernel)
    GumbelTopPTokenSelect_v1{V,S}         temperature + top-p + seeded Gumbel-max (registry.sampling; vLLM d9105ea80 stochastic path)
    GumbelTokenSelect_v1{V}               temperature + seeded Gumbel-max, top_p == 1 (registry.sampling, X-06)
    Gumbel[TopP]TokenSelectSharedGreedy_v1{V}   the two above with `temp == 0` computed once (`sampler_construction = shared-greedy`)
    GumbelTopPTokenSelect_v2{V,S}         GumbelTopPTokenSelect_v1 at a constant split count S, its keep word as word gates
                                          (registry.topp_words; a single-request workload)

Greedy is not stochastic: the families are distinct Definitions with distinct digests; this module only says which families are
"the sampling event of a step".  A spec may carry a `prefix:` (the fold's `spec` strings) and statics `{...}`.
"""
from __future__ import annotations

SAMPLING_EVENT_FAMILIES = ("TokenSelect_v1", "GumbelTopPTokenSelect_v1", "GumbelTokenSelect_v1", "GumbelTopPTokenSelect_v2",
                           "GumbelTopPTokenSelectSharedGreedy_v1", "GumbelTokenSelectSharedGreedy_v1")


def family_of(spec: str) -> str:
    """`prefix:Family_v1{K=..}` / `Family_v1{..}` / `Family_v1` -> `Family_v1`."""
    return spec.split(":", 1)[-1].split("{", 1)[0]


def is_sampling_event(spec: str) -> bool:
    return family_of(spec) in SAMPLING_EVENT_FAMILIES


#: the `TargetProfile.sampler_construction` a stochastic select family is bound under (`registry.sampling_vocab.select_for`): a shared-greedy
#: restatement is `shared-greedy`; a plain select, and the one specialised on a constant split count, which has no shared-greedy form, are
#: `greedy-check-per-stage` (tests/check/test_fold_sampler_construction.py pins this against the registry)
CONSTRUCTION_OF = {"GumbelTopPTokenSelect_v1": "greedy-check-per-stage", "GumbelTokenSelect_v1": "greedy-check-per-stage",
                   "GumbelTopPTokenSelect_v2": "greedy-check-per-stage",
                   "GumbelTopPTokenSelectSharedGreedy_v1": "shared-greedy", "GumbelTokenSelectSharedGreedy_v1": "shared-greedy"}


def sampler_construction_of(specs) -> str | None:
    """The sampler construction the stochastic selects among `specs` declare (`CONSTRUCTION_OF`); None when there are none, ValueError
    naming them when they declare two."""
    found = {family_of(s): CONSTRUCTION_OF[family_of(s)] for s in specs if family_of(s) in CONSTRUCTION_OF}
    if len(set(found.values())) > 1:
        raise ValueError(f"the stochastic selects {sorted(found)} declare two sampler constructions {sorted(set(found.values()))}")
    return next(iter(found.values()), None)


def program_sampler_construction(program_dirs) -> str | None:
    """The sampler construction the Build's Program dirs declare, read off each one's root instances (`result.json` spec_histogram, the
    nodes the Match compares); None when none has a stochastic select.  A dir without the histogram is refused by name (SystemExit)."""
    import json
    import os
    specs: list[str] = []
    for d in program_dirs:
        try:
            specs += list(json.load(open(os.path.join(d, "result.json")))["spec_histogram"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise SystemExit(f"{d}: no spec_histogram in result.json ({type(e).__name__}): the Program's sampler construction is unknown") from None
    try:
        return sampler_construction_of(specs)
    except ValueError as e:
        raise SystemExit(f"{list(program_dirs)}: {e}") from None


__all__ = ["SAMPLING_EVENT_FAMILIES", "family_of", "is_sampling_event", "CONSTRUCTION_OF", "sampler_construction_of",
           "program_sampler_construction"]
