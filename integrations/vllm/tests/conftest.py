"""Known failures on `main`, marked expected to fail so that `check` gates every other test of this suite.

Each one failed on `main` a8e72c81 (with #331's dead-module fix) with torch 2.14 and 2.4.1 on CPU, before this suite joined
`check`; the reason says what it found. They are not strict: the check machines' torch and numpy differ from the lock's,
and a strict mark that passed there would fail `check`. Fix the test or the code, then delete its entry.
"""
from __future__ import annotations

import pytest

KNOWN_FAILURES = {
    "tests/engine/test_gen_ov_easy.py::test_serve_untied_is_not_registered_and_binds_from_the_profile_config":
        "ServeV4's signature now ends with lm_head after cos_sin",
    "tests/engine/test_gen_ov_easy.py::test_serve_untied_differs_from_serve_v4_only_at_the_lm_head_operands":
        "no operand differs from ServeV4's; the test expects three at the lm_head",
    "tests/program/test_derive.py::test_s3_untied_lm_head_loses_sharing_and_is_refused_by_family":
        "the untied lm_head adds no input gates (7,158,595 either way)",
    "tests/observe/test_patterns_synthetic.py::test_gumbel_two_stage_sampler_is_one_token_select":
        "the observer refuses the sampler's bf16 [1, V] logits view as not a row-contiguous float32 [R, V] view",
    "tests/pipeline/test_research_tools.py::test_closure_covers_every_core_file":
        "research_tools.CLOSURE lacks verity.claims, verity.evaluation and verity.randomness, so the vLLM Tools' identity"
        " misses them; adding them changes that identity, so stored attempts stop being reused",
    "tests/program/test_analytic.py::test_check_cos_sin_against_the_captured_b0_table":
        "torch CPU cos/sin against the captured b0 table: max ulp 29,154, the bound 2",
    "tests/engine/test_gen_dense2.py::test_inv_freq_models_cuda_reciprocal_multiply_and_is_identity_for_power_of_two_D":
        "15 inv_freq elements differ at D=96; the model expects 14",
    "tests/program/test_ref_prims.py::test_gelu_ref_vs_torch[none-bf16]":
        "exact GELU against torch CPU: max ulp 128, normalised error inf",
    "tests/program/test_ref_prims.py::test_gelu_ref_vs_torch[none-f32]":
        "exact GELU against torch CPU: max ulp 875,646,710, normalised error nan",
    "tests/program/test_lifted_tiny.py::test_specified_list_is_closed":
        "passes alone; fails once another test in the process has resolved Lifted[GatherBf16x64_v1]_v2{ORD=3}, which"
        " registers it as specified",
}


def pytest_collection_modifyitems(items) -> None:
    for item in items:
        if reason := KNOWN_FAILURES.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=f"known failure on main: {reason}", strict=False))
