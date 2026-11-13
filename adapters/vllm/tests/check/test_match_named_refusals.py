"""What the global match refuses by name before it compares anything: `verity_vllm.check.match.attribution.attribute_instances`
on a view with no recorded rows, and `verity_vllm.check.match.declaration.from_gprog` on a manifest that declares no prompt."""
from __future__ import annotations

import json
import types

import pytest

from verity_vllm.check.match import batch_decomp
from verity_vllm.check.match.attribution import attribute_instances
from verity_vllm.check.match.declaration import from_gprog
from verity_vllm.check.match.match_run import MatchRun
from verity_vllm.records.fields import RecordError


def _run(fold) -> MatchRun:
    """A run as `attribute_instances` reads it: the fold's view and the record's steps; nothing else is read before the refusal."""
    return MatchRun(gp={}, oracle_dir="the-fold-dir", tokens_path="", within_step="dag", fast=False, ops=batch_decomp.record_ops(),
                    loaded={}, tokens=None, tokens_optional=True, rec={"header": {}, "steps": {}, "samples": {}}, res={}, fold=fold,
                    declared={})


def test_a_view_with_no_recorded_rows_is_refused_as_the_fold():
    """A derived Program's view has none (`instances.jsonl` is the fold's): as the fold of a global match it is refused by name,
    as `batch_decomp.decompose` refuses it."""
    view = types.SimpleNamespace(oracle_rows=None, n_input_nodes=3, inst=[])
    with pytest.raises(ValueError, match="the-fold-dir: the view passed as the fold has no instances.jsonl rows"):
        attribute_instances(_run(view), [])


def test_a_manifest_that_declares_lengths_alone_is_refused_at_its_first_request_without_a_prompt(tmp_path):
    """The Build artifact here has no descriptor beside it and names no request: the manifest is read, and refused, before either."""
    art = {"schema": "verity/workload-program/v1", "ok": True, "workload_digest": "00" * 32, "requests": [], "inputs": {"derived": {}}}
    manifest = {"schema": "verity-vllm/workload/v1", "model": "a/model", "variants": {},
                "requests": [{"request_id": "u0", "max_tokens": 4, "prompt_token_ids": [5, 6, 7]},
                             {"request_id": "u1", "max_tokens": 4, "prompt_len": 3}]}
    (tmp_path / "workload_program.json").write_text(json.dumps(art))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RecordError) as e:
        from_gprog(str(tmp_path / "workload_program.json"), str(tmp_path / "manifest.json"))
    assert (e.value.where, e.value.field) == (str(tmp_path / "manifest.json"), "requests[1].prompt_token_ids")
    assert e.value.why == "is absent (a lengths-only plan): the Program declares each prompt"
