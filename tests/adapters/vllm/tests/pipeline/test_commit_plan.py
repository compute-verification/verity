"""The Commit's plan (`pipeline.commit_plan`): the producer facts derived on CPU before the GPU is held, taken by the Commit only under
its own key, and the row's `plan` stage that writes it.  CPU only: the Programs are the synthetic Gemma-like step of the oracle tests."""
from __future__ import annotations

import types

import pytest


pytestmark = pytest.mark.usefixtures("code_versions_stood_in")








def _another_program(a: types.SimpleNamespace) -> None:
    a.program_digest = "q" * 64


def _programs_under_a_root(a: types.SimpleNamespace) -> None:
    a.programs_root, a.program_dir = a.program_dir, None










def test_the_commit_plan_s_parser_takes_its_inputs():
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline import commit_plan as CPL
    o = cli.parse(CPL, ["--workload", "w.json", "--required-manifest", "m.json", "--program-dir", "pd", "--out", "o.json"], {})
    assert (o.workload, o.required_manifest, o.program_dir, o.programs_root, o.out) == ("w.json", "m.json", "pd", None, "o.json")


# ---------------------------------------------------------------------------------------------------------------- the row's plan stage









