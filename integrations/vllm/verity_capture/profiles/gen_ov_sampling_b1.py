"""[GEN ov-sampling] Experimental B1 profile: the pinned Qwen2.5-1.5B sm89 profile (`vllm_d9105ea80_sm89_eager_qwen15`) plus
the `TokenSampleGumbel` recognizer (`gen_ov_sampling_patterns.py`) that explains the sampler's `_gumbel_sample_kernel` at
temperature 1 with the quarantine definition `GumbelArgmaxBf16_v1{V, SEED, POS}` (`verity_vllm/registry/quarantine/ov_sampling`).

NOT an accepted profile (`accepted=False` inherited): it exists to put the sampled-token contract through the sealed
pipeline (fold -> G2 -> G4 Tier A -> G5) on the lane's captures, so the report can state what passes and what the gates
still cannot see (the seed's provenance, out/gen/lanes/ov-sampling/REPORT.md).  Runs whose header records greedy
sampling resolve exactly as under the base profile (the lane pattern returns None for them).

    python -m verity_capture.bench.run_config ... --profile gen_ov_sampling_b1 --phase cpu
"""

from __future__ import annotations

import dataclasses

from . import vllm_d9105ea80_sm89_eager_qwen15 as base
from .gen_ov_sampling_patterns import ov_sampling_patterns

PROFILE_ID = base.PROFILE_ID + "+ov-sampling"
C = base.C
weight_field = base.weight_field
root_class = base.root_class
check_header = base.check_header


def build_profile():
    p = base.PROFILE
    notes = dict(p.notes)
    notes["ov_sampling"] = {"pattern": "gen_ov_sampling_patterns.TokenSampleGumbel (_gumbel_sample_kernel at temperature 1, no top-k/top-p)",
                            "definition": "quarantine ov-sampling/GumbelArgmaxBf16@v1 {V, SEED, POS}: argmax_v(f32(logits[v]) + g(SEED, POS)[v])",
                            "seed_provenance": "SEED static from the run header's request record (children: parent seed + index); "
                                               "the device seeds buffer is not a checked fact -- PROPOSED Root(role=sampling_seed)"}
    return dataclasses.replace(p, id=PROFILE_ID, patterns=ov_sampling_patterns(C) + list(p.patterns), notes=notes)


PROFILE = build_profile()
