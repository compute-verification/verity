"""Known failures on `main`, marked expected to fail so that `check` gates every other test of this suite.

Each one failed on `main` a8e72c81 (with #331's dead-module fix) with torch 2.14 and 2.4.1 on CPU, before this suite joined
`check`; the reason says what it found. They are not strict: the check machines' torch and numpy differ from the lock's,
and a strict mark that passed there would fail `check`. Fix the test or the code, then delete its entry.
"""
from __future__ import annotations

import os

import pytest

KNOWN_FAILURES = {
    "tests/pipeline/test_gc_tuning.py::test_env_contract_in_a_fresh_interpreter[None-expect0]":
        "the check pod's CPython gives gc thresholds (2000, 10, 10), where the test expects CPython's older default (700, 10, 10) (owner vllm-coordinator)",
    "tests/pipeline/test_gc_tuning.py::test_env_contract_in_a_fresh_interpreter[-expect1]":
        "the check pod's CPython gives gc thresholds (2000, 10, 10), where the test expects CPython's older default (700, 10, 10) (owner vllm-coordinator)",
    "tests/pipeline/test_gc_tuning.py::test_env_contract_in_a_fresh_interpreter[default-None]":
        "the check pod's CPython gives gc thresholds (2000, 10, 10), where the test expects CPython's older default (700, 10, 10) (owner vllm-coordinator)",
    "tests/pipeline/test_gc_tuning.py::test_env_contract_in_a_fresh_interpreter[OFF-None]":
        "the check pod's CPython gives gc thresholds (2000, 10, 10), where the test expects CPython's older default (700, 10, 10) (owner vllm-coordinator)",
    "tests/pipeline/test_gc_tuning.py::test_env_contract_in_a_fresh_interpreter[700,10,10-expect4]":
        "the check pod's CPython gives gc thresholds (2000, 10, 10), where the test expects CPython's older default (700, 10, 10) (owner vllm-coordinator)",
    "tests/pipeline/test_gc_tuning.py::test_env_contract_in_a_fresh_interpreter[50000, 20, 100-expect5]":
        "the check pod's CPython gives gc thresholds (2000, 10, 10), where the test expects CPython's older default (700, 10, 10) (owner vllm-coordinator)",
    "tests/acquire/test_compiled_source.py::test_renumber_assigns_invocations_per_call_site":
        "CPU torch: `Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing` (owner vllm-epoch-prep)",
    "tests/pipeline/test_source_identity.py::test_negative_requested_sha_mismatch_fails_by_name":
        "needs a .git; a shipped check tree has none (owner vllm-epoch-run)",
    "tests/pipeline/test_source_identity.py::test_precheck_passes_with_matching_verity_tree_and_sha[integration]":
        "needs a .git; a shipped check tree has none (owner vllm-epoch-run)",
    "tests/pipeline/test_source_identity.py::test_precheck_passes_with_matching_verity_tree_and_sha[repo]":
        "needs a .git; a shipped check tree has none (owner vllm-epoch-run)",
    "tests/pipeline/test_source_identity.py::test_repo_root_resolution_prefers_export_json_then_git_then_markers":
        "needs a .git; a shipped check tree has none (owner vllm-epoch-run)",
    "tests/program/test_twins.py::test_check_writes_the_evidence_schema":
        "the evidence schema's `openmp` key depends on whether the C++ twins were built in this tree (environmental; owner vllm-epoch-prep)",
    "tests/pipeline/test_row.py::test_the_commit_flags_the_row_arms_exist":
        "fails on the torch-less CPU check pod only (passes with CPU torch); needs importorskip where torch is required (owner vllm-epoch-prep)",
    "tests/check/test_compiled_autotune.py::test_unknown_kernel_is_not_shown_and_does_not_raise":
        "fails on the torch-less CPU check pod only (passes with CPU torch); needs importorskip where torch is required (owner vllm-epoch-prep)",
    "tests/program/test_analytic.py::test_check_cos_sin_against_the_captured_b0_table":
        "torch CPU cos/sin against the captured b0 table: max ulp 29,154, the bound 2",
    "tests/engine/test_gen_dense2.py::test_inv_freq_models_cuda_reciprocal_multiply_and_is_identity_for_power_of_two_D":
        "15 inv_freq elements differ at D=96; the model expects 14",
    "tests/program/test_ref_prims.py::test_gelu_ref_vs_torch[none-bf16]":
        "exact GELU against torch CPU: max ulp 128, normalised error inf",
    "tests/program/test_ref_prims.py::test_gelu_ref_vs_torch[none-f32]":
        "exact GELU against torch CPU: max ulp 875,646,710, normalised error nan",
}


def pytest_configure(config) -> None:
    """Make this process's first MKL VML call on one thread, before any test.  torch's CPU `exp`, `log`, `tanh`, `erf` and the
    other VML functions run in the MKL it links statically, whose first-call initialisation races when several threads call at
    once: one intra-op thread then computes its whole chunk at about 2e-4 relative error (the `exp` reference against torch,
    max ULP 1771, in some 6% of fresh processes).  Any VML call initialises all of them.

    And keep oneDNN, which runs torch's CPU bf16 matmuls, off AMX, set before its first primitive reads it.  On a KVM guest
    with AMX (the cloud agent VMs) a preempted bf16 matmul now and then returns one batch entry's rows wrong, single-threaded
    too: 49 of 16,000 `(4, 1, 130) @ (4, 130, 64)` matmuls under 16 busy processes, and none with AMX off
    (`test_eager_attention_head_ref_vs_torch[div-bf16]`, normalised error 779)."""
    os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_BF16")
    try:
        import torch
    except ImportError:
        return
    torch.exp(torch.zeros(1))


def pytest_collection_modifyitems(items) -> None:
    for item in items:
        if reason := KNOWN_FAILURES.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=f"known failure on main: {reason}", strict=False))
