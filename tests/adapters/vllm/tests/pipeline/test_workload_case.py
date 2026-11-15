"""`workload.case_for`, the `case` a generated workload records: an explicit case wins, else the adapter's fixed case of the repo
(`engine.profiles.cases.CASES`, the first in table order), else the manifest's registry role at that revision, else None."""
from __future__ import annotations

from verity_vllm.pipeline import workload as W


def test_an_explicit_case_wins():
    assert W.case_for("HuggingFaceTB/SmolLM2-135M", "0" * 40, "X") == "X"


def test_a_fixed_cases_repo_is_its_first_case():
    assert W.case_for("HuggingFaceTB/SmolLM2-135M", "0" * 40, None) == "B0"      # B2 runs the same repo
    assert W.case_for("Qwen/Qwen1.5-MoE-A2.7B-Chat", "0" * 40, None) == "M1"     # so do M2 and M0
    assert W.case_for("microsoft/phi-2", "0" * 40, None) == "B5"


def test_a_registry_role_is_the_manifests_at_that_revision():
    assert W.case_for("unsloth/Llama-3.2-1B", "9535bd9b1d", None) == "LLAMA32_1B"
    assert W.case_for("unsloth/Llama-3.2-1B", "0" * 40, None) is None


def test_an_unknown_repo_has_no_case():
    assert W.case_for("org/unknown", "0" * 40, None) is None
