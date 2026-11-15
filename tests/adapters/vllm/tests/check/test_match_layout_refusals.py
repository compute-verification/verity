"""What the Match's readers refuse by name when a document states a layout its Program does not hold: a fold's `program.json`
(`verity_vllm.check.match.program_loaders.load_oracle`: where its parameters put the weights, its peers layout and its witness
layout), a supplied parameter (`verity_vllm.check.match.supplied_inputs.set_supplied`), the request-owned Input nodes and the
weights after them (`verity_vllm.check.match.input_layout`), the capture log (`verity_vllm.check.match.record.read_record`) and a
derived Program's prefix layout (`verity_vllm.check.match.prefix_reads.load_layout`).  Beside them, two things a loader passes
over without a word: a blank line of the fold's instance rows, and a supplied node that the view took for a seed.

The fold is the two-rank toy's (`test_tp2_d90_fixture`), rank 0's, as its producer writes it, with `program.json` or the rows
changed.  The other views hold only what their reader reads."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import types

import pytest

from tests.check import test_tp2_d90_fixture as T
from verity_vllm.check.match import input_layout as IL
from verity_vllm.check.match import prefix_reads as PX
from verity_vllm.check.match import program_loaders as PL
from verity_vllm.check.match import record as REC
from verity_vllm.check.match import supplied_inputs as SI

# --- a fold's program.json -----------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def written_fold(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy") / "fold"
    d.mkdir()
    T.persist_fold(T.fold_rank(0), d, 0)
    return d


@pytest.fixture
def fold(written_fold, tmp_path):
    """A copy of the toy's fold to change; what a load of it notes is taken out of the loaders' notes afterwards."""
    d = tmp_path / "fold"
    shutil.copytree(written_fold, d)
    yield d
    PL.LOAD_NOTES.pop(str(d), None)


def _change(fold, change) -> None:
    pj = json.loads((fold / "program.json").read_text())
    change(pj)
    (fold / "program.json").write_text(json.dumps(pj))


def test_the_fold_loads_as_its_producer_writes_it(fold):
    """What each case below changes: three parameters, the weights last and one Input node each; one peers parameter, the
    request's, of six fields; no witness."""
    pj = json.loads((fold / "program.json").read_text())
    assert [name for name, _type in pj["params"]] == ["prompt", "peers", "weights"]
    assert [(e["param"], e["name"], e["request_index"], len(e["fields"])) for e in pj["peers_layout"]] == [(1, "peers", 0, 6)]
    assert pj["witness_layout"] == []
    B = PL.load_oracle(str(fold), "fold")
    assert B.inputs == {0: ("Input32_v1", 2), 1: ("Input16_v1", 112), 2: ("Input16_v1", 1712)}
    assert B.peer_fields == {1: [(0, 0), (32, 1), (64, 2), (72, 3), (88, 4), (104, 5)]} and B.peer_index == {1: 0}
    assert (B.witness_fields, B.witness_index, B.input_base) == ({}, {}, {2: 0})


def _weights_before_the_peers(pj: dict) -> None:
    pj["params"] = [pj["params"][0], pj["params"][2], pj["params"][1]]


def _one_parameter_more(pj: dict) -> None:
    pj["params"].insert(1, ["seed_0", "Value<64>"])


def _peers_at_no_input_node(pj: dict) -> None:
    pj["peers_layout"][0]["param"] = 9


def _peers_of_no_fields(pj: dict) -> None:
    pj["peers_layout"][0]["fields"] = []


def _peers_without_their_fields(pj: dict) -> None:
    del pj["peers_layout"][0]["fields"]


def _two_fields_of_one_ordinal(pj: dict) -> None:
    pj["peers_layout"][0]["fields"][1]["ordinal"] = pj["peers_layout"][0]["fields"][0]["ordinal"]


def _a_witness_at_no_input_node(pj: dict) -> None:
    pj["witness_layout"] = [{"param": 9, "name": "schedule_0", "witness": "schedule", "request_index": 0}]


@pytest.mark.parametrize("change, refusal", [
    (_weights_before_the_peers, "program.json params end with 'peers', the fold contract puts `weights` LAST"),
    (_one_parameter_more, "instances read parameter 2 as the largest, program.json puts `weights` at 3"),
    (_peers_at_no_input_node, "peers parameter 'peers' (index 9) is not an Input node of the Program"),
    (_peers_of_no_fields, "peers parameter 'peers' has no fields"),
    (_peers_without_their_fields, "peers parameter 'peers' has no fields"),
    (_two_fields_of_one_ordinal, "peers parameter 'peers' has duplicate field ordinals"),
    (_a_witness_at_no_input_node, "witness parameter 'schedule_0' (index 9) is not an Input node of the Program"),
], ids=["the weights before the peers", "one parameter more", "peers at no Input node", "peers of no fields", "peers without their fields",
        "two fields of one ordinal", "a witness at no Input node"])
def test_a_fold_whose_program_json_states_a_layout_its_program_does_not_hold_is_refused_by_name(fold, change, refusal: str):
    _change(fold, change)
    with pytest.raises(ValueError, match="^" + re.escape(f"{fold}: {refusal}") + "$"):
        PL.load_oracle(str(fold), "fold")


def test_blank_lines_of_the_instance_rows_are_passed_over_and_the_rows_digest_is_of_the_bytes_as_read(fold):
    plain = PL.load_oracle(str(fold), "fold")
    noted = dict(PL.LOAD_NOTES[str(fold)])
    lines = (fold / "instances.jsonl").read_text().splitlines(keepends=True)
    spaced = (lines[0] + "\n" + lines[1] + "  \n" + "".join(lines[2:]) + "\n").encode()
    (fold / "instances.jsonl").write_bytes(spaced)
    B = PL.load_oracle(str(fold), "fold")
    assert len(B.oracle_rows) == len(lines) - 1 == 24 and B.oracle_rows == plain.oracle_rows
    assert (B.digest, B.oracle_wmap, B.input_base) == (plain.digest, plain.oracle_wmap, plain.input_base)
    assert PL.LOAD_NOTES[str(fold)] == {**noted, "instances_sha256": hashlib.sha256(spaced).hexdigest()}
    assert noted["instances_sha256"] == hashlib.sha256("".join(lines).encode()).hexdigest() != PL.LOAD_NOTES[str(fold)]["instances_sha256"]


# --- a supplied parameter ------------------------------------------------------------------------------------------------------------


def _supplied_view(inputs: dict[int, tuple[str, int]], seeds: tuple[int, ...] = ()) -> types.SimpleNamespace:
    """A view as `set_supplied` reads it: its Input nodes, the nodes it took for seeds in their order, no supplied parameter yet."""
    return types.SimpleNamespace(inputs=inputs, supplied_fields={}, supplied_owner={}, seed_nodes=set(seeds),
                                 seed_index={k: i for i, k in enumerate(seeds)})


def _entry(name: str, widths: list[int]) -> dict:
    return {"name": name, "fields": [{"name": f"f{i}", "width": w, "offset": i} for i, w in enumerate(widths)]}


def test_a_supplied_parameter_at_no_input_node_or_of_two_widths_is_refused():
    P = _supplied_view({0: ("Input32_v1", 4), 1: ("Input32_v1", 2)})
    with pytest.raises(ValueError, match=r"^fold: supplied parameter 'values' \(node 3\) is not an Input node of the Program$"):
        SI.set_supplied(P, [(3, None, _entry("values", [32, 32]))], "fold")
    with pytest.raises(ValueError, match="^fold: supplied parameter 'values' mixes widths, so it is not one Input node$"):
        SI.set_supplied(P, [(1, None, _entry("values", [32, 16]))], "fold")
    assert (P.supplied_fields, P.supplied_owner) == ({}, {})
    SI.set_supplied(P, [(1, None, _entry("values", [32, 32]))], "fold")
    assert (P.supplied_fields, P.supplied_owner) == ({1: [(0, "f0", 32), (1, "f1", 32)]}, {1: None})


def test_a_supplied_node_is_no_seed_and_the_seeds_after_it_are_the_requests_in_order():
    """A view takes every Input node of one 64-bit value for a seed, the i-th for request i.  A supplied parameter of one such
    field is one too, and it is no seed: the seeds are counted again without it."""
    inputs = {0: ("Input32_v1", 4), 1: ("Input64_v1", 1), 2: ("Input64_v1", 1), 3: ("Input64_v1", 1), 4: ("Input16_v1", 8)}
    P = _supplied_view(inputs, seeds=(1, 2, 3))
    SI.set_supplied(P, [(2, 0, _entry("run_0", [64]))], "fold")
    assert (P.seed_nodes, P.seed_index) == ({1, 3}, {1: 0, 3: 1}) and P.supplied_owner == {2: 0}
    P = _supplied_view(inputs, seeds=(1, 2, 3))
    SI.set_supplied(P, [(4, None, _entry("values", [16] * 8))], "fold")
    assert (P.seed_nodes, P.seed_index) == ({1, 2, 3}, {1: 0, 2: 1, 3: 2})


# --- the request-owned Input nodes and the weights -----------------------------------------------------------------------------------


def _layout_view(n_inputs: int, **bound) -> types.SimpleNamespace:
    """A view as `input_layout` reads it: Input nodes 0..n-1 and what a loader bound of them."""
    return types.SimpleNamespace(inputs={k: ("Input16_v1", 1) for k in range(n_inputs)}, **bound)


REQS = [{"request_id": "r0"}, {"request_id": "r1"}]


def test_supplied_nodes_sit_after_the_request_owned_nodes_and_the_weights_right_after_both():
    assert IL.weights_input_nodes(_layout_view(5, supplied_fields={2: []}), 2) == {3, 4}
    with pytest.raises(ValueError, match=r"^supplied input nodes \[1, 3\] precede the 2 request-owned ones$"):
        IL.weights_input_nodes(_layout_view(5, supplied_fields={1: [], 3: []}), 2)
    with pytest.raises(ValueError, match=r"^first weights input node is 2, expected 3 \(request params 2, seeds 0, peers 0, witnesses 0\)$"):
        IL.weights_input_nodes(_layout_view(5, supplied_fields={4: []}), 2)
    with pytest.raises(ValueError, match=r"^first weights input node is 3, expected 4 \(request params 1, seeds 1, peers 1, witnesses 0\)$"):
        IL.weights_input_nodes(_layout_view(6, supplied_fields={5: []}), 1, 1, 1)
    apart = _layout_view(4)
    del apart.inputs[2]
    with pytest.raises(ValueError, match=r"^first weights input node is 3, expected 2 \(request params 2, seeds 0, peers 0, witnesses 0\)$"):
        IL.weights_input_nodes(apart, 2)


def test_the_request_owned_nodes_are_named_by_request_and_the_weights_follow_them():
    B = _layout_view(9, seed_index={2: 0, 3: 1}, peer_index={4: 0, 5: 1}, witness_index={6: 1})
    assert IL.request_inputs(B, REQS) == ({"r0": 2, "r1": 3}, {"r0": 4, "r1": 5}, [("r1", 6)], {7, 8})
    assert IL.request_inputs(_layout_view(3), REQS) == ({}, {}, [], {2})


@pytest.mark.parametrize("bound, refusal", [
    ({"seed_index": {2: 2}}, "program has a seed input node for request index 2 but the header has 2 requests"),
    ({"peer_index": {2: 0, 3: 5}}, "program has a peers input node for request index 5 but the header has 2 requests"),
    ({"witness_index": {2: 3}}, "program has a witness input node for request index 3 but the header has 2 requests"),
    ({"peer_index": {2: 0}}, "program has 1 peers input nodes for 2 requests (a TP rank fold declares one per request)"),
    ({"peer_index": {2: 0, 3: 0}}, "program has 1 peers input nodes for 2 requests (a TP rank fold declares one per request)"),
], ids=["a seed of a third request", "peers of a sixth request", "a witness of a fourth request", "peers of one request of two",
        "two peers nodes of one request"])
def test_an_input_node_of_a_request_the_header_does_not_have_or_peers_that_are_not_one_a_request_are_refused(bound: dict, refusal: str):
    with pytest.raises(ValueError, match="^" + re.escape(refusal) + "$"):
        IL.request_inputs(_layout_view(9, **bound), REQS)


# --- the capture log -----------------------------------------------------------------------------------------------------------------


def _log(tmp_path, *records: dict) -> str:
    p = tmp_path / "capture.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(p)


def test_a_capture_log_with_two_starts_of_one_step_or_with_no_header_is_refused(tmp_path):
    start = {"$e": "StepStart", "step": 3, "rows": [["u0", 0]], "logits_rows": [["u0", 0]]}
    with pytest.raises(ValueError, match="^two StepStart records for step 3$"):
        REC.read_record(_log(tmp_path, {"$e": "RunHeader"}, start, {"$e": "StepStart", "step": 4, "rows": [], "logits_rows": []}, start))
    log = _log(tmp_path, start, {"$e": "SampleEnd", "step": 3, "sampled_token_ids": [7]})
    with pytest.raises(ValueError, match="^" + re.escape(f"{log}: no RunHeader") + "$"):
        REC.read_record(log)
    rec = REC.read_record(_log(tmp_path, start, {"$e": "RunHeader", "profile": {"name": "toy"}}))
    assert rec["header"] == {"$e": "RunHeader", "profile": {"name": "toy"}} and list(rec["steps"]) == [3]


# --- a derived Program's prefix layout -----------------------------------------------------------------------------------------------


def test_a_prefix_layout_whose_layer_names_no_number_is_refused_before_its_input_node_is_read(tmp_path):
    P = types.SimpleNamespace(inputs={})
    layout = {"input": 1, "C": 2, "row_leaves": 4, "layers": ["model.layers.3.self_attn", "model.norm", "model.head"]}
    (tmp_path / PX.PREFIX_FILE).write_text(json.dumps(layout))
    with pytest.raises(ValueError, match="^" + re.escape(f"{tmp_path / PX.PREFIX_FILE}: layer 'model.norm' names no layer number") + "$"):
        PX.load_layout(P, str(tmp_path))
    assert not hasattr(P, "pfx")
    (tmp_path / PX.PREFIX_FILE).write_text(json.dumps({**layout, "layers": ["model.layers.3.self_attn", "model.layers.5.self_attn"]}))
    with pytest.raises(ValueError, match=r"user input 1 is not one Input node of 32 values \(L=2, C=2, KVH\*D=4\)$"):
        PX.load_layout(P, str(tmp_path))
    P.inputs[1] = ("Input16_v1", 32)
    PX.load_layout(P, str(tmp_path))
    assert P.pfx == {1: (2, 4, [3, 5])}
