"""G1 of the global match (`verity_vllm.check.match.declaration.requests_declared`) on a manifest, the Program declared from it and
the header of a run that served it.  The three agree and G1 passes; then one field of one of them is changed, and G1 names the
difference.  G1 reads the declaration, the manifest and the header alone: no fold is decoded and the record has no steps."""
from __future__ import annotations

import json
from typing import Any

import pytest

from verity_vllm.check.match import batch_decomp
from verity_vllm.check.match.declaration import (
    mock_global_program,
    requests_declared,
    sha256_file,
)
from verity_vllm.check.match.match_run import MatchRun
from verity_vllm.records.workload import Workload

MANIFEST: dict[str, Any] = {"schema": "verity-vllm/workload/v1", "model": "a/model", "variants": {"hot": {"temperature": 0.7}},
                            "requests": [{"request_id": "u0", "max_tokens": 4, "prompt_token_ids": [5, 6, 7]},
                                         {"request_id": "u1", "max_tokens": 4, "prompt_token_ids": [8, 9]}]}
NO_FOLD: Any = None                                      # G1 reads none


def _first(**stated) -> dict:
    """`MANIFEST` with its first request stating more."""
    return MANIFEST | {"requests": [MANIFEST["requests"][0] | stated, MANIFEST["requests"][1]]}


def _declared(d, manifest: dict = MANIFEST) -> tuple[dict, dict]:
    """`manifest` written under `d`, the mock Program of it, and the header of a run that served it under the engine the Program models."""
    path = d / "manifest.json"
    path.write_text(json.dumps(manifest))
    served = [{"request_id": f"r{i}", "workload_request_id": r["request_id"], "prompt_token_ids": r["prompt_token_ids"], "max_tokens": r["max_tokens"]}
              for i, r in enumerate(manifest["requests"])]
    header = {"workload": {"sha256": Workload.load(path).doc_sha256, "requests": served},
              "profile": {"engine_kwargs": {"enforce_eager": True, "enable_prefix_caching": False}}}
    return mock_global_program(str(path), []), header


def _run(d, gp: dict, header: dict) -> MatchRun:
    """A run as G1 reads it: the declaration, its requests by id, the record's header and the document."""
    return MatchRun(gp=gp, oracle_dir=str(d), tokens_path="", within_step="dag", fast=False, ops=batch_decomp.record_ops(), loaded={}, tokens=None,
                    tokens_optional=True, rec={"header": header, "steps": {}, "samples": {}}, res={"checks": [], "request_ids": {"unknown_ids": []}},
                    fold=NO_FOLD, declared={r["request_id"]: r for r in gp["requests"]})


def _problems(d, gp: dict, header: dict) -> list[str]:
    """What G1 names, in the order it meets them; the check's status says the same."""
    run = _run(d, gp, header)
    requests_declared(run)
    [g1] = run.res["checks"]
    assert g1["id"] == "G1" and g1["status"] == ("FAIL" if g1["problems"] else "PASS") and g1["n_problems"] == len(g1["problems"])
    return g1["problems"]


def test_a_manifest_the_program_declared_from_it_and_the_header_of_its_run_agree(tmp_path):
    gp, header = _declared(tmp_path)
    run = _run(tmp_path, gp, header)
    facts, served, param_of, declared_ids = requests_declared(run)
    assert (param_of, declared_ids, [r["request_id"] for r in served]) == ({"r0": 0, "r1": 1}, ["r0", "r1"], ["r0", "r1"])
    assert facts["engine"] == {"enforce_eager": True, "enable_prefix_caching": False}
    [g1] = run.res["checks"]
    assert (g1["status"], g1["problems"], g1["n_declared"], g1["n_record"]) == ("PASS", [], 2, 2)
    assert g1["header_workload_sha256"] == g1["workload_doc_sha256"] and g1["workload_sha256"] == sha256_file(str(tmp_path / "manifest.json"))
    assert run.res["moe_construction"]["value"] is None


@pytest.mark.parametrize("field, served, problems", [
    ("workload_request_id", "zz", ["r0: record served workload request id 'zz', the Program declares 'u0'"]),
    ("max_tokens", 5, ["r0: record max_tokens 5 != declared 4"]),
    ("prompt_token_ids", [5, 6], ["r0: record prompt length 2 != declared 3", "r0: record prompt != declared prompt (input association)",
                                  "r0: record prompt != manifest prompt at position 0"]),
    ("prompt_token_ids", [5, 6, 8], ["r0: record prompt != declared prompt (input association)",
                                     "r0: record prompt != manifest prompt at position 0"]),
], ids=["workload-id", "max-tokens", "prompt-length", "prompt"])
def test_a_request_the_header_records_otherwise_than_the_program_declares_is_named_by_field(tmp_path, field, served, problems):
    gp, header = _declared(tmp_path)
    header["workload"]["requests"][0][field] = served
    assert _problems(tmp_path, gp, header) == problems


def test_a_program_request_whose_id_is_not_r_k_at_its_index_is_named(tmp_path):
    gp, header = _declared(tmp_path)
    gp["requests"][1]["request_id"] = "x1"
    assert _problems(tmp_path, gp, header) == [
        "global Program request ids are not r<k> at their manifest index: ['x1']",
        "manifest requests ['r0', 'r1'] != global Program requests ['r0', 'x1']",
        "record header requests != global Program requests: only in Program ['x1'], only in record ['r1']"]


def test_a_header_that_lists_a_request_twice_is_named(tmp_path):
    gp, header = _declared(tmp_path)
    header["workload"]["requests"].insert(1, dict(header["workload"]["requests"][0]))
    assert _problems(tmp_path, gp, header) == [
        "record header lists requests more than once: ['r0']",
        "record header requests != global Program requests: only in Program [], only in record []"]


def test_a_manifest_without_a_prompt_for_a_served_request_ties_the_records_prompt_to_none(tmp_path):
    """The manifest of record is a lengths-only plan at position 1, and the Program and the header name that manifest by its
    digests: the prompt the record served for r1 has no declared one to equal."""
    gp, header = _declared(tmp_path)
    lengths_only = MANIFEST | {"requests": [MANIFEST["requests"][0], {"request_id": "u1", "max_tokens": 4, "prompt_len": 2}]}
    (tmp_path / "manifest.json").write_text(json.dumps(lengths_only))
    assert _problems(tmp_path, gp, header)[-1] == (
        "r1: the manifest has no prompt at position 1 (a lengths-only plan): the record's prompt is tied to no declared one")
    gp["workload"]["sha256"] = sha256_file(str(tmp_path / "manifest.json"))
    header["workload"]["sha256"] = Workload.load(tmp_path / "manifest.json").doc_sha256
    assert _problems(tmp_path, gp, header) == [
        "r1: the manifest has no prompt at position 1 (a lengths-only plan): the record's prompt is tied to no declared one"]


def test_a_manifest_that_is_no_workload_record_is_named_and_nothing_is_compared_with_it(tmp_path):
    gp, header = _declared(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "another/record"}))
    [problem] = _problems(tmp_path, gp, header)
    assert problem.startswith(f"the workload manifest is not a workload record: {tmp_path / 'manifest.json'}: field 'schema' is 'another/record'")


def test_a_seed_the_manifest_declares_and_the_header_does_not_record_is_named(tmp_path):
    gp, header = _declared(tmp_path, _first(seed=7))
    assert _problems(tmp_path, gp, header) == ["r0: manifest declares seed=7 but the run header records no seed"]
    header["workload"]["requests"][0]["seed"] = 7
    assert _problems(tmp_path, gp, header) == []


@pytest.mark.parametrize("served, problems", [
    ({"variant": "hot", "temperature": 0.7, "top_p": 1.0}, []),
    ({"variant": "hot", "temperature": 0.5, "top_p": 1.0},
     ["r0: manifest declares sampling temperature=0.7 but the run header records temperature=0.5 for the request"]),
    # the table's entry states no top_p: the driver's default, 1.0, is what the manifest declares
    ({"variant": "hot", "temperature": 0.7, "top_p": 0.9},
     ["r0: manifest declares sampling top_p=1.0 but the run header records top_p=0.9 for the request"]),
    ({"variant": "cold", "temperature": 0.7},
     ["r0: manifest declares sampling variant='hot' but the run header records variant='cold' for the request"]),
], ids=["as-pinned", "temperature", "default-top-p", "variant"])
def test_a_request_the_manifest_pins_to_a_variant_is_compared_with_the_sampling_the_header_records_for_it(tmp_path, served, problems):
    gp, header = _declared(tmp_path, _first(variant="hot"))
    header["workload"]["requests"][0].update(served)
    assert _problems(tmp_path, gp, header) == problems


def test_the_programs_fused_moe_construction_is_reported_as_declared_and_never_as_attested(tmp_path):
    gp, header = _declared(tmp_path)
    gp["moe_construction"] = {"value": "padded", "moe_block_present": True, "population_label": "PADDED-TABLE", "declared_explicitly": True}
    run = _run(tmp_path, gp, header)
    requests_declared(run)
    assert run.res["checks"][0]["status"] == "PASS"
    assert run.res["moe_construction"] == gp["moe_construction"] | {
        "attested": False, "source": "GP-01 moe_construction block (TargetProfile.moe_construction, declared-and-bound; never attested by the record)"}
