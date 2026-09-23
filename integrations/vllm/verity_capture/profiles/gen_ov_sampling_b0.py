"""[COV cov-difr, D87 temp=1 column] Experimental B0 profile: the pinned SmolLM2-135M sm89 profile
(`vllm_d9105ea80_sm89_eager`) plus the `TokenSampleGumbel` recognizer (`gen_ov_sampling_patterns.py`) that explains the
sampler's `_gumbel_sample_kernel` for seeded sampled rows with the quarantine Definitions `GumbelArgmaxBf16_v1{V, SEED, POS}`
(temperature exactly 1: vLLM launches no `_temperature_kernel`, the kernel adds Gumbel noise to the bf16 logits' exact f32
widening) and `GumbelArgmaxTempBf16_v1{V, SEED, POS, T}` (T not in {0, 1}); greedy rows stay the registered `TokenSelect_v1{V}`.

The same one-line derivation as `gen_ov_sampling_b1.py` for the other role: the pinned profile is PROTECTED and is not edited;
the D87 `_t1` workloads (`workload_cov_b0_*_t1.json`, `sampling = {variant: t1, seed: N}`) are run with

    bash cov_pod.sh run B0 cov_b0_b1_c256_t1 --profile gen_ov_sampling_b0        # run.json records the profile; the CPU phase reuses it

NOT an accepted profile (`accepted=False` inherited).  A run whose header records greedy sampling resolves exactly as under the
base profile (the lane pattern returns None for greedy headers).  Evidence: out/gen/lanes/cov-difr/ (this lane's bundles) and
out/gen/lanes/ov-sampling/REPORT.md (the Definitions' admission on Qwen2.5-1.5B).
"""

from __future__ import annotations

import dataclasses

from . import vllm_d9105ea80_sm89_eager as base
from .gen_ov_sampling_patterns import ov_sampling_patterns

PROFILE_ID = base.PROFILE_ID + "+ov-sampling"
C = base.C
weight_field = base.weight_field
root_class = base.root_class
check_header = base.check_header


def build_profile():
    p = base.PROFILE
    notes = dict(p.notes)
    notes["ov_sampling"] = {"pattern": "gen_ov_sampling_patterns.TokenSampleGumbel (_gumbel_sample_kernel; seeded rows at T=1 and T not in {0,1}, no top-k/top-p)",
                            "definition": "quarantine ov-sampling/GumbelArgmaxBf16@v1 {V, SEED, POS}: argmax_v(f32(logits[v]) + g(SEED, POS)[v]); "
                                          "GumbelArgmaxTempBf16@v1 {V, SEED, POS, T} for tempered rows (div.full.f32 quotient)",
                            "seed_provenance": "SEED static from the run header's request record (children: parent seed + index); "
                                               "the device seeds buffer is not a checked fact -- PROPOSED Root(role=sampling_seed)"}
    return dataclasses.replace(p, id=PROFILE_ID, patterns=ov_sampling_patterns(C) + list(p.patterns), notes=notes)


PROFILE = build_profile()
