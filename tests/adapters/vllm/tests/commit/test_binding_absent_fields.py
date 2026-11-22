"""What `verity_vllm.commit.binding` reads where a workload, a model structure, a required-value manifest, its loaded header, a launch or
a map entry leaves a field out: the field absent, null and empty read alike, a field that is given is read as given, and nothing is
refused that was not."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from verity_vllm.commit import binding as B
from verity_vllm.query.manifest import format as MF

PD = hashlib.sha256(b"program").hexdigest()
OP = "model.embed_tokens"
ABSENT = object()


def given(**fields: Any) -> dict[str, Any]:
    """`fields` less those left `ABSENT`."""
    return {k: v for k, v in fields.items() if v is not ABSENT}


# ---------------------------------------------------------------------- the expected population



ONE_MODULE = {"modules": [{"op_path": OP, "members": ["out"]}]}














# ---------------------------------------------------------------------- the required-value manifest

def row(request_id: str = "r0", member: str = "out") -> dict[str, Any]:
    return {"family": "instance_outputs", "program_digest": PD, "request_id": request_id, "step": 0, "invocation": 0, "op_path": OP,
            "output_member": member, "element_range": [0, 4], "rank": 0}


def load(tmp_path, digest_of: list[dict[str, Any]], **fields: Any) -> tuple[list[tuple], dict[str, Any]]:
    """The manifest of `fields` whose digest is that of the rows `digest_of`, loaded for the request r0 over one step."""
    doc = {"schema": MF.SCHEMA, "program_digest": PD, "manifest_digest": MF.manifest_digest(digest_of), **given(**fields)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc))
    return B.load_required_manifest(str(path), PD, requests=["r0"], steps=1)


def members(pop: list[tuple]) -> list[str]:
    return [i[5] for i in pop]


@pytest.mark.parametrize("spelling", ["entries", "rows", "identities"])
def test_the_manifests_rows_are_read_under_each_of_their_three_spellings(tmp_path, spelling):
    pop, hdr = load(tmp_path, [row()], **{spelling: [row()]})
    assert pop == [B.identity(PD, "r0", 0, 0, OP, "out", 0, 4, 0)] and hdr["rows_n"] == 1 and hdr["manifest_digest_verified"] is True


def test_the_first_spelling_that_holds_a_row_is_the_one_read(tmp_path):
    a, b, c = row(member="a"), row(member="b"), row(member="c")
    assert members(load(tmp_path, [a], entries=[a], rows=[b], identities=[c])[0]) == ["a"]
    assert members(load(tmp_path, [b], entries=[], rows=[b], identities=[c])[0]) == ["b"]
    assert members(load(tmp_path, [c], entries=None, rows=[], identities=[c])[0]) == ["c"]
    assert members(load(tmp_path, [c], rows=None, identities=[c])[0]) == ["c"]


@pytest.mark.parametrize("fields", [{}, {"entries": []}, {"rows": None}, {"identities": []}, {"entries": None, "rows": [], "identities": None}])
def test_a_manifest_that_carries_no_row_loads_as_no_identity(tmp_path, fields):
    pop, hdr = load(tmp_path, [], **fields)
    assert pop == [] and hdr["rows_n"] == 0 and hdr["manifest_digest_verified"] is True


def test_a_manifest_whose_rows_are_not_the_digested_ones_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="edited after digesting"):
        load(tmp_path, [row()])
    with pytest.raises(ValueError, match="edited after digesting"):
        load(tmp_path, [], identities=[row()])


@pytest.mark.parametrize("unmodelled", [ABSENT, None, {}])
def test_a_manifest_that_states_nothing_unmodelled_has_none(tmp_path, unmodelled):
    assert load(tmp_path, [row()], identities=[row()], unmodelled=unmodelled)[1]["unmodelled"] == {}


def test_what_a_manifest_states_as_unmodelled_is_on_its_header(tmp_path):
    assert load(tmp_path, [row()], identities=[row()], unmodelled={"ops": ["x"]})[1]["unmodelled"] == {"ops": ["x"]}


@pytest.mark.parametrize("program", [ABSENT, None, "a digest", [], {}, {"dataflow": None}, {"dataflow": {}}, {"digest": "d"}])
def test_a_manifest_whose_program_states_no_dataflow_has_none_on_its_header(tmp_path, program):
    hdr = load(tmp_path, [row()], identities=[row()], program=program)[1]
    assert "dataflow_in_digest" not in hdr and "dataflow" not in hdr


def test_a_manifest_whose_program_states_its_dataflow_has_its_counts_on_the_header(tmp_path):
    hdr = load(tmp_path, [row()], identities=[row()], program={"dataflow": {"n_edges": 3, "split_graph": True, "other": 1}})[1]
    assert hdr["dataflow_in_digest"] is True
    assert hdr["dataflow"] == {"n_edges": 3, "n_launch_edges": None, "n_undefined": None, "wiring_declared": None, "split_graph": True}


@pytest.mark.parametrize("declared", [ABSENT, None, []])
def test_a_manifest_that_declares_no_request_maps_none(tmp_path, declared):
    pop, hdr = load(tmp_path, [row()], identities=[row()], requests_declared=declared)
    assert hdr["requests_declared_n"] == 0 and [i[1] for i in pop] == ["r0"]


def test_a_declared_request_is_placed_under_its_index_and_one_without_an_id_is_passed_over(tmp_path):
    rows = [row(request_id="chat-7")]
    declared = [{"index": 4}, {"request_id": "chat-7", "index": 0}, {"request_id": "chat-9"}]
    pop, hdr = load(tmp_path, rows, identities=rows, requests_declared=declared)
    assert hdr["requests_declared_n"] == 2 and [i[1] for i in pop] == ["r0"]


@pytest.mark.parametrize("engine_steps", [ABSENT, None, []])
def test_a_manifest_that_states_no_engine_step_declares_no_composition_and_no_step_number(tmp_path, engine_steps):
    hdr = load(tmp_path, [row()], identities=[row()], engine_steps=engine_steps)[1]
    assert hdr["engine_steps"] == [] and hdr["engine_step_numbers"] == []


def test_a_manifests_engine_steps_are_kept_as_numbers_or_as_the_declared_composition(tmp_path):
    assert load(tmp_path, [row()], identities=[row()], engine_steps=[0, 1])[1]["engine_step_numbers"] == [0, 1]
    steps = [{"engine_step": 2, "rows": 4, "n_seq": 1, "members": [{"request_id": "r0", "rows": 4}]}]
    hdr = load(tmp_path, [row()], identities=[row()], engine_steps=steps)[1]
    assert hdr["engine_steps"] == steps and hdr["engine_step_numbers"] == [2]


@pytest.mark.parametrize("reference", [ABSENT, None, {}])
def test_a_manifest_that_names_no_composition_reference_has_an_empty_one(tmp_path, reference):
    assert load(tmp_path, [row()], identities=[row()], composition_reference=reference)[1]["composition_reference"] == {}


def test_a_manifests_composition_reference_is_on_its_header(tmp_path):
    reference = {"input_token_ids": {"op_path": OP, "output_member": "in", "per_row": 1}}
    assert load(tmp_path, [row()], identities=[row()], composition_reference=reference)[1]["composition_reference"] == reference


# ---------------------------------------------------------------------- a launch's declared rows

HIDDEN = "fa2.m1/L0"








# ---------------------------------------------------------------------- the composition check

TOKENS = {"op_path": OP, "output_member": "in", "per_row": 1}
SKIPPED = {"ok": True, "checked_steps": 0, "skipped": "manifest declares no engine-step composition (B=1 / non-global manifest)"}
NO_REFERENCE = {"ok": False, "checked_steps": 0, "error": "global manifest names no input_token_ids reference member: composition unverifiable"}


def one_step(**header: Any) -> dict[str, Any]:
    """A loaded header that declares request r0 with four rows at engine step 0."""
    return {"engine_steps": [{"engine_step": 0, "members": [{"request_id": "r0", "rows": 4}]}], **given(**header)}


def token_entry(**more: Any) -> dict[str, Any]:
    """The map's entry of r0's four token rows at step 0."""
    identity = {"step": 0, "op_path": OP, "output_member": "in", "invocation": 0, "request_id": "r0", "element_range": [0, 4]}
    return {"kind": "leaf", "identity": identity, **given(**more)}








