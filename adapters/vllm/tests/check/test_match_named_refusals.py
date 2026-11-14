"""What the global match does before it compares anything with a fold, a manifest or a Build it cannot use:
`verity_vllm.check.match.attribution.attribute_instances` on a view with no recorded rows fails where it reads them, and
`verity_vllm.check.match.declaration.from_gprog` and `mock_global_program` refuse by name a manifest that declares no prompt, a
Build that did not succeed, a Build whose request is not the manifest's, and a Build whose record of its descriptor file is not
the file."""
from __future__ import annotations

import gzip
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
from verity.circuits.codec import canonical_json, program_digest

from verity_vllm.check.match import batch_decomp
from verity_vllm.check.match.attribution import attribute_instances
from verity_vllm.check.match.declaration import from_gprog, mock_global_program
from verity_vllm.check.match.match_run import MatchRun
from verity_vllm.program.workload import WorkloadRequest
from verity_vllm.records.fields import RecordError

MANIFEST: dict[str, Any] = {"schema": "verity-vllm/workload/v1", "model": "a/model", "variants": {},
                            "requests": [{"request_id": "u0", "max_tokens": 4, "prompt_token_ids": [5, 6, 7]},
                                         {"request_id": "u1", "max_tokens": 4, "prompt_token_ids": [8, 9]}]}
LENGTHS_ONLY = MANIFEST | {"requests": [MANIFEST["requests"][0], {"request_id": "u1", "max_tokens": 4, "prompt_len": 2}]}
COMPONENT: Any = SimpleNamespace(fn="Request_v1")       # a request's component, of which `static` reads the name alone


def _request(k: int, **changed) -> dict:
    """Request `k` of `MANIFEST` as the Build states a request (`WorkloadRequest.static`), `changed` laid over it."""
    q = MANIFEST["requests"][k]
    return WorkloadRequest(f"r{k}", k, len(q["prompt_token_ids"]), q["max_tokens"], 0, COMPONENT).static() | changed


def _build(d, requests: list[dict], manifest: dict = MANIFEST, **stated) -> tuple[str, str]:
    """A Build artifact that names `requests`, and the manifest beside it; returns the two paths."""
    art = {"schema": "verity/workload-program/v1", "ok": True, "workload_digest": "00" * 32, "requests": requests, "inputs": {"derived": {}}}
    (d / "workload_program.json").write_text(json.dumps(art | stated))
    (d / "manifest.json").write_text(json.dumps(manifest))
    return str(d / "workload_program.json"), str(d / "manifest.json")


def _run(fold) -> MatchRun:
    """A run as `attribute_instances` reads it: the fold's view and the record's steps; nothing else is read before the refusal."""
    return MatchRun(gp={}, oracle_dir="the-fold-dir", tokens_path="", within_step="dag", fast=False, ops=batch_decomp.record_ops(),
                    loaded={}, tokens=None, tokens_optional=True, rec={"header": {}, "steps": {}, "samples": {}}, res={}, fold=fold,
                    declared={})


class _NoRows:
    """A view no loader has given rows: `oracle_rows` is None, and every other attribute reads as empty."""

    oracle_rows, n_input_nodes = None, 3

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return []


def test_a_view_with_no_recorded_rows_fails_as_the_fold_where_its_rows_are_read():
    """A derived Program's view has none (`instances.jsonl` is the fold's): as the fold of a global match its rows are read after
    its weights' Input nodes are, and the error is Python's own."""
    with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
        attribute_instances(_run(_NoRows()), [])


def test_a_manifest_that_declares_lengths_alone_is_refused_at_its_first_request_without_a_prompt(tmp_path):
    """The Build artifact here has no descriptor beside it and names no request: the manifest is read, and refused, before either."""
    art, manifest = _build(tmp_path, [], LENGTHS_ONLY)
    with pytest.raises(RecordError) as e:
        from_gprog(art, manifest)
    assert (e.value.where, e.value.field) == (manifest, "requests[1].prompt_token_ids")
    assert e.value.why == "is absent (a lengths-only plan): the Program declares each prompt"


def test_a_build_of_the_manifest_is_read_as_its_requests_by_position(tmp_path):
    """What the refusals below change one field of."""
    gp = from_gprog(*_build(tmp_path, [_request(0), _request(1)]))
    assert [(r["request_id"], r["workload_request_id"], r["prompt_len"], r["max_tokens"]) for r in gp["requests"]] == [
        ("r0", "u0", 3, 4), ("r1", "u1", 2, 4)]
    assert gp["gprog_artifact"]["digest_verification"] == {"verified": False, "why": "descriptor.json.gz not beside the artifact"}


def test_a_build_that_did_not_succeed_is_refused_with_its_reason(tmp_path):
    art, manifest = _build(tmp_path, [], ok=False, reason="missing-derived", missing_derived=["3,3"])
    with pytest.raises(ValueError) as e:
        from_gprog(art, manifest)
    assert str(e.value) == f"{art}: builder result not ok (missing-derived: ['3,3'])"


@pytest.mark.parametrize("changed, why", [
    ({"request_id": "r0"}, "request 'r0' at index 1 is not the r<k> convention: rebuild the workload Program with the current global_program"),
    ({"workload_request_id": "u0"}, "r1 carries workload_request_id 'u0' but the manifest at position 1 has 'u1'"),
    ({"prefix": {"rows": 16, "producers": ["r0"]}},
     ("r1 declares the prefix-cache hit {'rows': 16, 'producers': ['r0']}, the manifest's arrival chronology and prompts give "
      "{'cached': 0, 'producers': []} (program.prefix_plan)")),
], ids=["id", "workload-id", "prefix"])
def test_a_build_whose_request_is_not_the_manifests_is_refused_by_name(tmp_path, changed, why):
    art, manifest = _build(tmp_path, [_request(0), _request(1, **changed)])
    with pytest.raises(ValueError) as e:
        from_gprog(art, manifest)
    assert str(e.value) == f"{art}: {why}"


def test_a_build_whose_record_of_its_descriptor_file_is_not_the_file_is_refused(tmp_path):
    """The digest the artifact claims is the descriptor's in both, so the Program is the one it names; the second artifact records
    another sha256 for the file."""
    desc = {"schema": "toy/descriptor", "definitions": [{"name": "D0"}], "root": {"body": []}}
    with gzip.open(tmp_path / "descriptor.json.gz", "wb") as f:
        f.write(canonical_json(desc))
    sha = hashlib.sha256(canonical_json(desc)).hexdigest()
    art, manifest = _build(tmp_path, [], workload_digest=program_digest(desc), descriptor={"sha256": sha})
    assert from_gprog(art, manifest)["gprog_artifact"]["digest_verification"] == {
        "verified": True, "recomputed_from": str(tmp_path / "descriptor.json.gz"), "descriptor_sha256": sha}
    art, manifest = _build(tmp_path, [], workload_digest=program_digest(desc), descriptor={"sha256": "0" * 64})
    with pytest.raises(ValueError) as e:
        from_gprog(art, manifest)
    assert str(e.value) == f"{art}: descriptor.sha256 {'0' * 16} != the descriptor file's {sha[:16]}"


def test_the_mock_program_refuses_a_lag_that_is_neither_0_nor_1_and_a_manifest_without_a_prompt(tmp_path):
    _, manifest = _build(tmp_path, [])
    gp = mock_global_program(manifest, [], lag=1)
    assert [r["request_id"] for r in gp["requests"]] == ["r0", "r1"] and gp["lag"]["value"] == 1
    with pytest.raises(ValueError, match="^lag must be 0 or 1, got 2$"):
        mock_global_program(manifest, [], lag=2)
    _, manifest = _build(tmp_path, [], LENGTHS_ONLY)
    with pytest.raises(RecordError) as e:
        mock_global_program(manifest, [])
    assert (e.value.where, e.value.field) == (manifest, "requests[1].prompt_token_ids")
    assert e.value.why == "is absent (a lengths-only plan): the Program declares each prompt"
