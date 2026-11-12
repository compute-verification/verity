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

def placed(structure: dict[str, Any], **workload: Any) -> list[tuple]:
    """(request, step, op_path, member) of each identity the structure requires of two requests over two steps."""
    pop = B.expected_population(structure, {"program_digest": PD, "requests": ["r0", "r1"], "steps": 2, **workload})
    return [(i[1], i[2], i[4], i[5]) for i in pop]


ONE_MODULE = {"modules": [{"op_path": OP, "members": ["out"]}]}


@pytest.mark.parametrize("request_steps", [ABSENT, None, {}])
def test_a_workload_that_states_no_request_steps_places_every_request_at_every_step(request_steps):
    assert placed(ONE_MODULE, **given(request_steps=request_steps)) == [(r, s, OP, "out") for r in ("r0", "r1") for s in (0, 1)]


def test_a_request_the_stated_request_steps_leave_out_is_placed_at_every_step():
    assert placed(ONE_MODULE, request_steps={"r1": [1]}) == [("r0", 0, OP, "out"), ("r0", 1, OP, "out"), ("r1", 1, OP, "out")]


@pytest.mark.parametrize("layers", [ABSENT, None, []])
def test_a_structure_that_names_no_moe_layer_requires_no_moe_plane(layers):
    assert placed({"modules": [], "moe_planes": ["up", "sum"], **given(moe_layers=layers)}) == []


@pytest.mark.parametrize("planes", [ABSENT, None, []])
def test_a_structure_whose_moe_layers_name_no_plane_requires_none(planes):
    assert placed({"modules": [], "moe_layers": [3, 5], **given(moe_planes=planes)}) == []


def test_a_structure_with_moe_layers_and_planes_requires_each_plane_of_each_layer():
    got = placed({"modules": [], "moe_layers": [3, 5], "moe_planes": ["up", "sum"]})
    assert got == [(r, s, f"moe/L{layer}", plane) for r in ("r0", "r1") for s in (0, 1) for layer in (3, 5) for plane in ("up", "sum")]


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


def shortfalls(launch: dict[str, Any] | None) -> list[str]:
    """What the launch of request r0 at step 0 lacks, bound to one committed tensor of eight words."""
    ident = B.LaunchIdentity(B.identity(PD, "r0", 0, 0, HIDDEN, "*", 0, -1), launch)
    idx = {("r0", 0, 0, HIDDEN, "g4x8x16/q", 0): [{"name": "t", "stream_offset": 0, "identity": {"element_range": [0, 8]}}]}
    return list(B._launch_shortfalls([ident], idx, B._op_key_index(idx)).values())


@pytest.mark.parametrize("launch", [None, {}, {"rows": None}, {"rows": {}}, {"geometry": {"HB": 4}, "rows": {}}])
def test_a_launch_that_declares_no_rows_declares_none_for_its_bound_request(launch):
    assert shortfalls(launch) == ["requests ['r0'] have no declared rows at engine step 0"]


def test_a_launch_that_declares_its_requests_rows_is_held_to_its_geometry():
    assert shortfalls({"rows": {"r0": 8}}) == ["the batch-level row carries no launch geometry"]
    assert shortfalls({"rows": {"r1": 8}}) == ["requests ['r0'] have no declared rows at engine step 0"]


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


@pytest.mark.parametrize("header", [{}, {"engine_steps": None}, {"engine_steps": []}, {"engine_steps": [0, 1]}])
def test_a_header_that_declares_no_composition_is_skipped_by_name(header):
    assert B.composition_check(header, {"entries": [token_entry()]}) == SKIPPED


@pytest.mark.parametrize("reference", [ABSENT, None, {}, {"sampled_token_ids": TOKENS}, {"input_token_ids": None}])
def test_a_composition_without_a_token_reference_member_fails_by_name(reference):
    assert B.composition_check(one_step(composition_reference=reference), {"entries": [token_entry()]}) == NO_REFERENCE


@pytest.mark.parametrize("encoding", [ABSENT, None, {}, {"dtype": "int64"}, {"shape": [None]}, {"shape": [0]}, {"shape": [4]}])
def test_a_tiled_member_passes_where_its_entry_states_no_row_count_or_the_tiled_one(encoding):
    got = B.composition_check(one_step(composition_reference={"input_token_ids": TOKENS}), {"entries": [token_entry(encoding=encoding)]})
    assert got == {"ok": True, "checked_steps": 1, "problem_steps_n": 0, "problems": [], "error": None,
                   "reference": {"input_token_ids": TOKENS, "sampled_token_ids": None}}


def test_rows_of_the_member_past_the_tiled_ones_are_bound_to_no_request():
    got = B.composition_check(one_step(composition_reference={"input_token_ids": TOKENS}), {"entries": [token_entry(encoding={"shape": [6]})]})
    assert not got["ok"] and got["problems"][0]["gaps"] == ["rows [4,6) bound to no request"]
    assert got["error"] == f"composition: engine step 0 composition on {OP}/in; gaps ['rows [4,6) bound to no request']"
