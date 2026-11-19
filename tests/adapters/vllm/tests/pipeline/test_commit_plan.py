"""The Commit's plan (`pipeline.commit_plan`): the producer facts derived on CPU before the GPU is held, taken by the Commit only under
its own key, and the row's `plan` stage that writes it.  CPU only: the Programs are the synthetic Gemma-like step of the oracle tests."""
from __future__ import annotations

import copy
import json
import types
from pathlib import Path

import pytest

from tests.check.test_oracle_compare_v2_producers import RES, gemma_like_program
from verity_vllm.check import oracle_compare as OC
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.pipeline import commit_plan as CP
from verity_vllm.query.required import request_manifest

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")








def _another_program(a: types.SimpleNamespace) -> None:
    a.program_digest = "q" * 64


def _programs_under_a_root(a: types.SimpleNamespace) -> None:
    a.programs_root, a.program_dir = a.program_dir, None










def test_the_commit_s_parser_takes_the_plan(tmp_path):
    pytest.importorskip("torch")
    from verity_vllm.pipeline import cli
    assert cli.parse("commit", ["--out", str(tmp_path), "--plan", "p.json"], {}).plan == "p.json"
    o = cli.parse("commit-plan", ["--workload", "w.json", "--required-manifest", "m.json", "--program-dir", "pd", "--out", "o.json"], {})
    assert (o.workload, o.required_manifest, o.program_dir, o.programs_root, o.out) == ("w.json", "m.json", "pd", None, "o.json")


# ---------------------------------------------------------------------------------------------------------------- the row's plan stage









