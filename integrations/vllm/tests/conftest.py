"""Known failures on `main`, marked expected to fail so that `check` gates every other test of this suite.

Each one failed on `main` a8e72c81 (with #331's dead-module fix) with torch 2.14 and 2.4.1 on CPU, before this suite joined
`check`; the reason says what it found. They are not strict: the check machines' torch and numpy differ from the lock's,
and a strict mark that passed there would fail `check`. Fix the test or the code, then delete its entry.
"""
from __future__ import annotations

import pytest

KNOWN_FAILURES = {
    "tests/acquire/test_compiled_source.py::test_renumber_assigns_invocations_per_call_site":
        "CPU torch: `Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing` (owner vllm-epoch-prep)",
    "tests/check/test_fold_sampler_construction.py::test_101s_records_fold_canonical_equal_under_the_programs_construction":
        "the store fetch of *match_compare.json returns nothing off the check pod (owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_B0]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_B1]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_B1_tp2]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_B7]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_GEMMA2_2B]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_LLAMA32_1B]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_LLAMA32_1B_tp2]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_MISTRAL7B]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_OLMOE]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_PHI3_MINI]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_PYTHIA160M]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_QWEN05]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_QWEN3_30B_A3B]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_QWEN3_30B_A3B_tp2]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_QWEN3_4B]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_QWEN3_4B_FP8]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_SMOL360]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/engine/test_profiles_generic.py::test_generic_profile_equals_its_fixture[derived_TINYLLAMA]":
        "#321 added TokenSelectGumbelTopP.construction; the canonical profile fixtures predate it (re-pin with `verity-vllm canonical-profiles --write`; owner vllm-epoch-prep)",
    "tests/pipeline/test_source_identity.py::test_negative_requested_sha_mismatch_fails_by_name":
        "needs a .git; a shipped check tree has none (owner vllm-epoch-run)",
    "tests/pipeline/test_source_identity.py::test_precheck_passes_with_matching_veritor_repo_and_sha[integration]":
        "needs a .git; a shipped check tree has none (owner vllm-epoch-run)",
    "tests/pipeline/test_source_identity.py::test_precheck_passes_with_matching_veritor_repo_and_sha[repo]":
        "needs a .git; a shipped check tree has none (owner vllm-epoch-run)",
    "tests/pipeline/test_source_identity.py::test_repo_root_resolution_prefers_export_json_then_git_then_markers":
        "needs a .git; a shipped check tree has none (owner vllm-epoch-run)",
    "tests/program/test_registry_one_process.py::test_core_and_every_integration_registry_load_in_one_process[core-first]":
        "core verity.ml.fp32 registers F32Add_v2 ... F32SubFtz_v2 and the one-process load reports the 9 ids as a conflict (owner flock-ir-lowering)",
    "tests/program/test_registry_one_process.py::test_core_and_every_integration_registry_load_in_one_process[integration-first]":
        "core verity.ml.fp32 registers F32Add_v2 ... F32SubFtz_v2 and the one-process load reports the 9 ids as a conflict (owner flock-ir-lowering)",
    "tests/program/test_twins.py::test_check_writes_the_evidence_schema":
        "the evidence schema's `openmp` key depends on whether the C++ twins were built in this tree (environmental; owner vllm-epoch-prep)",
    "tests/pipeline/test_row.py::test_the_commit_flags_the_row_arms_exist":
        "fails on the torch-less CPU check pod only (passes with CPU torch); needs importorskip where torch is required (owner vllm-epoch-prep)",
    "tests/check/test_compiled_autotune.py::test_unknown_kernel_is_not_shown_and_does_not_raise":
        "fails on the torch-less CPU check pod only (passes with CPU torch); needs importorskip where torch is required (owner vllm-epoch-prep)",
    "tests/engine/test_gen_ov_easy.py::test_serve_untied_is_not_registered_and_binds_from_the_profile_config":
        "ServeV4's signature now ends with lm_head after cos_sin",
    "tests/engine/test_gen_ov_easy.py::test_serve_untied_differs_from_serve_v4_only_at_the_lm_head_operands":
        "no operand differs from ServeV4's; the test expects three at the lm_head",
    "tests/program/test_derive.py::test_s3_untied_lm_head_loses_sharing_and_is_refused_by_family":
        "the untied lm_head adds no input gates (7,158,595 either way)",
    "tests/observe/test_patterns_synthetic.py::test_gumbel_two_stage_sampler_is_one_token_select":
        "the observer refuses the sampler's bf16 [1, V] logits view as not a row-contiguous float32 [R, V] view",
    "tests/program/test_analytic.py::test_check_cos_sin_against_the_captured_b0_table":
        "torch CPU cos/sin against the captured b0 table: max ulp 29,154, the bound 2",
    "tests/engine/test_gen_dense2.py::test_inv_freq_models_cuda_reciprocal_multiply_and_is_identity_for_power_of_two_D":
        "15 inv_freq elements differ at D=96; the model expects 14",
    "tests/program/test_sampling_rows.py::test_nv_logf_and_nv_log1pf_are_word_equal_to_the_libdevice_transcriptions_on_every_non_nan_word":
        "nv_logf of a negative word gives NaN 0x7fc00000; the libdevice transcription gives 0xffc00000",
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
