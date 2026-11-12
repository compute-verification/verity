"""What the Match record's reader makes of a field one of its files leaves out.

`MatchOracle` reads four files of a Match record: the capture's snapshots, its log and its tokens, and the fold's
instances.  Each test takes the complete record of `tests/check/test_oracle_compare.py`, leaves one field out of one
file (absent, null, empty, or any other value that tests false: `left_out`) and states what the oracle then
answers: which requests, steps and ids it knows, and what it says of a member it cannot read.  A workload and a
required manifest's row, which a caller hands in, are held the same way.
"""
from __future__ import annotations

import copy
import gzip
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.check.test_absent_commit_fields import put
from tests.check.test_absent_link_fields import left_out
from tests.check.test_oracle_compare import U0, U1, H, make_match
from verity_vllm.check import match_oracle as MO

EMBED, NORM = "model.embed_tokens", "model.layers.0.post_attention_layernorm"
#: the record's files: its path under the Match directory, and whether it holds a document a line
FILES = {"snapshots": ("capture/snapshots.json", False), "tokens": ("capture/tokens.json", False),
         "instances": ("instances.jsonl", True), "log": ("capture/log.jsonl.gz", True)}
RUN_HEADER, STEP_START, SAMPLE_END = 0, 1, 2      # the log's first three lines; step 0's
DECLARED = {"requests": [{"request_id": U0, "index": 0, "prompt_len": 5, "max_tokens": 1}, {"request_id": U1, "index": 1, "prompt_len": 3, "max_tokens": 1}]}
NO_VIEW = "instance has no view for member {!r}"
NOT_RESOLVED = "no fold instance of producer X_v1 whose output the consumers ['model.embed_tokens'] read at this step (promoted member)"


def oracle(tmp_path: Path, file: str | None = None, change: Callable[[Any], object] | None = None) -> MO.MatchOracle:
    """The oracle of the complete record, after `change` edited one file's document (its list of documents, where it holds one a line)."""
    d = make_match(str(tmp_path))[0]
    if file is not None and change is not None:
        rel, by_line = FILES[file]
        path = Path(d) / rel
        opener = gzip.open if rel.endswith(".gz") else open
        with opener(path, "rt") as f:
            doc = [json.loads(line) for line in f] if by_line else json.load(f)
        change(doc)
        with opener(path, "wt") as f:
            f.write("".join(json.dumps(x) + "\n" for x in doc) if by_line else json.dumps(doc))
    return MO.MatchOracle(d)


def embed_instances(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The fold's instances of the embedding among the instances file's lines (its first line is the header)."""
    return [m for m in lines[1:] if m["hint"].startswith(EMBED + "/")]


def without_view(key: str, stated: Any) -> Callable[[list[dict[str, Any]]], None]:
    def change(lines: list[dict[str, Any]]) -> None:
        for m in embed_instances(lines):
            put(m, (key,), stated)
    return change


# ------------------------------------------------------------------------------------------------------------ the snapshots

@left_out([])
def test_a_record_whose_snapshots_name_no_step_has_no_step_to_read(tmp_path, stated):
    o = oracle(tmp_path, "snapshots", lambda doc: put(doc, ("steps",), stated))
    assert o.snapshot_steps == set()
    assert o.member_bytes(0, EMBED, "out", "r0") == (None, "step not snapshotted by the Match", 0)
    assert o.provenance()["snapshot_steps"] == []


@left_out([])
def test_a_record_whose_snapshots_hold_no_entry_names_the_bytes_it_does_not_have(tmp_path, stated):
    o = oracle(tmp_path, "snapshots", lambda doc: put(doc, ("entries",), stated))
    assert o.snapshot_steps == {0}
    assert o.member_bytes(0, EMBED, "out", "r0") == (None, "no snapshot of alloc 11 bytes [24,32) at event 50 (0 entries of that alloc)", 5)


# ------------------------------------------------------------------------------------------------------------ the fold's instances

@left_out([])
def test_a_fold_header_that_names_no_request_knows_no_request_by_its_label(tmp_path, stated):
    o = oracle(tmp_path, "instances", lambda lines: put(lines[0], ("requests",), stated))
    assert o.requests == {} and o.request_index == {}
    assert o.member_bytes(0, EMBED, "out", "r0") == (None, "request r0 not in the Match record's request table", 0)
    assert o.member_bytes(0, EMBED, "out", None)[1:] == (None, 8)      # every request's rows are still read
    assert o.sampled_id(0, "r0") == (None, f"request r0 not scheduled at step 0 in the Match record ({[U1, U0]})")
    assert o.workload_check(DECLARED)["problems"] == []                # the tokens file names both requests


@left_out([])
@pytest.mark.parametrize(("key", "member", "other"), [("operand_views", "in", "out"), ("output_views", "out", "in")])
def test_an_instance_that_states_no_view_has_no_bytes_for_the_member_that_reads_it(tmp_path, stated, key, member, other):
    o = oracle(tmp_path, "instances", without_view(key, stated))
    assert o.member_bytes(0, EMBED, member, "r0") == (None, NO_VIEW.format(member), 5)
    assert o.member_bytes(0, EMBED, other, "r0")[1:] == (None, 5)
    assert o.member_bytes(0, NORM, "0", "r0")[1:] == (None, 5)


@left_out([])
@pytest.mark.parametrize("key", ["operand_views", "output_views"])
def test_every_instance_the_oracle_indexes_states_both_lists_of_views(tmp_path, stated, key):
    o = oracle(tmp_path, "instances", without_view(key, stated))
    indexed = [m for insts in o._inst.values() for m in insts]
    assert len(indexed) == 24
    assert all(isinstance(m["operand_views"], list) and isinstance(m["output_views"], list) for m in indexed)
    assert [m[key] for m in o._inst[(0, EMBED)]] == [[]] * 8


@left_out([])
def test_an_instance_without_an_output_view_is_no_producers_output_at_its_step(tmp_path, stated):
    o = oracle(tmp_path, "instances", without_view("output_views", stated))
    assert sorted(o._step_outputs(0)) == [12, 13, 14]                  # the embedding's allocation 11 is not among them
    assert o.member_bytes(0, EMBED, "out", "r0", producer={"spec": "X_v1"}) == (None, NO_VIEW.format("out"), 5)
    assert o.member_bytes(0, EMBED, "in", "r0", producer={"spec": "X_v1"})[1:] == (None, 5)


@left_out([])
def test_a_promoted_member_whose_consumers_read_no_view_finds_no_producer(tmp_path, stated):
    o = oracle(tmp_path, "instances", without_view("operand_views", stated))
    producer = {"promoted": True, "spec": "X_v1", "consumers": [EMBED]}
    assert o.member_bytes(0, NORM, "out", "r0", producer=producer) == (None, NOT_RESOLVED, 0)


def two_producers_a_row(stated: Any) -> Callable[[list[dict[str, Any]]], None]:
    """Each instance of the embedding twice in its row's group, the second without its output views."""
    def change(lines: list[dict[str, Any]]) -> None:
        for m in embed_instances(lines):
            lines.insert(lines.index(m) + 1, put(dict(copy.deepcopy(m), event=m["event"] + 1), ("output_views",), stated))
    return change


@left_out([])
def test_of_two_producers_a_row_one_without_an_output_view_is_fed_by_nothing_and_spans_no_bytes(tmp_path, stated):
    o = oracle(tmp_path, "instances", two_producers_a_row(stated))
    group = o._inst[(0, EMBED)]
    assert o._member_span(group[:1], "out") == 2 * H and o._member_span(group, "out") is None
    fed = o.member_bytes(0, EMBED, "out", "r0", producer={"spec": "X_v1", "operand": "X_v1"}, nbytes=5 * 2 * H)
    assert fed == (None, ("producer_operand X_v1: 0 of 2 X_v1 instances in a row's group under model.embed_tokens consume a X_v1 output "
                          "(member 'out'; want exactly 1)"), 10)
    assert o.member_bytes(0, EMBED, "out", "r0", producer={"spec": "X_v1", "ordinal": 0}, nbytes=5 * 2 * H)[1:] == (None, 5)
    assert o.member_bytes(0, EMBED, "out", "r0", producer={"spec": "X_v1", "ordinal": 1}, nbytes=5 * 2 * H) == (None, NO_VIEW.format("out"), 5)
    unnamed = o.member_bytes(0, EMBED, "out", "r0", producer={"spec": "X_v1"}, nbytes=5 * 2 * H)
    assert unnamed[0] is None and unnamed[1].startswith("ambiguous producer: 2 instances of X_v1 per row under model.embed_tokens") and unnamed[2] == 10


def test_an_op_the_fold_has_no_instance_of_at_a_step_has_no_bytes(tmp_path):
    o = oracle(tmp_path)
    assert o.member_bytes(0, "no.such.module", "out", "r0") == (None, "no fold instance of this op at this step", 0)
    assert o.member_bytes(0, "no.such.module", "out", None) == (None, "no fold instance of this op at this step", 0)


# ------------------------------------------------------------------------------------------------------------ a view

@left_out([])
def test_a_view_that_states_no_stride_is_read_as_one_run_of_bytes(tmp_path, stated):
    whole = oracle(tmp_path / "whole").member_bytes(0, EMBED, "out", "r0")
    o = oracle(tmp_path / "left_out", "instances", lambda lines: [put(m["output_views"][0], ("strides",), stated) for m in embed_instances(lines)])
    assert whole[0] is not None and len(whole[0]) == 5 * 2 * H
    assert o.member_bytes(0, EMBED, "out", "r0") == whole


@left_out([])
def test_the_range_of_a_view_that_states_no_shape_is_one_element(stated):
    view = {"alloc": 1, "offset": 2, "dtype": "bfloat16", "shape": [3, 2], "strides": [4, 1]}
    assert MO.MatchOracle._view_bytes_range(view) == (1, 4, 24)        # as stated: strided, a span of 10 elements
    assert MO.MatchOracle._view_bytes_range(put(view, ("shape",), stated)) == (1, 4, 6)


@left_out([])
def test_the_range_of_a_view_that_states_no_stride_is_its_whole_shape(stated):
    view = {"alloc": 1, "offset": 2, "dtype": "bfloat16", "shape": [3, 2], "strides": [4, 1]}
    assert MO.MatchOracle._view_bytes_range(put(view, ("strides",), stated)) == (1, 4, 16)


# ------------------------------------------------------------------------------------------------------------ the capture's log

@left_out([])
def test_a_step_whose_start_names_no_row_schedules_no_request(tmp_path, stated):
    o = oracle(tmp_path, "log", lambda lines: put(lines[STEP_START], ("rows",), stated))
    assert o.step_requests == {0: [], 1: [U1, U0]} and o.step_rows[0] == [] and o.n_steps == 2
    assert o.ids_bytes(0, "r0", 4) == (None, "request r0 has no rows at step 0 in the Match record")
    assert o.sampled_id(0, "r0") == (None, "request r0 not scheduled at step 0 in the Match record ([])")
    assert o.sampled_id(1, "r0") == (2, None)


@left_out([])
def test_a_sample_end_that_carries_no_id_has_none_for_a_request(tmp_path, stated):
    o = oracle(tmp_path, "log", lambda lines: put(lines[SAMPLE_END], ("sampled_token_ids",), stated))
    assert o.sampled == {0: [], 1: [1, 2]}
    assert o.sampled_id(0, "r1") == (None, "SampleEnd carries 0 ids for 2 requests")
    assert o.sampled_id(1, "r1") == (1, None)


@left_out({})
def test_a_run_header_that_states_no_profile_names_no_vllm_version(tmp_path, stated):
    assert oracle(tmp_path / "whole").header["vllm"] == "t"
    o = oracle(tmp_path / "left_out", "log", lambda lines: put(lines[RUN_HEADER], ("profile",), stated))
    assert o.header == {"created_utc": "2026-09-19T20:00:00Z", "profile_id": "abc", "schema": "verity-capture/log/v1", "vllm": None}
    assert o.provenance()["vllm"] is None


# ------------------------------------------------------------------------------------------------------------ the capture's tokens

@left_out([])
def test_a_tokens_file_that_names_no_request_has_no_ids(tmp_path, stated):
    o = oracle(tmp_path, "tokens", lambda doc: put(doc, ("requests",), stated))
    assert o.tokens == {}
    assert o.ids_bytes(0, "r0", 4) == (None, "request r0 not in the Match record's tokens.json")
    assert o.workload_check(DECLARED) == {"ok": True, "problems": [], "compared": 2, "oracle_requests": 2, "run_requests": 2}


@left_out([])
def test_a_request_whose_tokens_state_no_prompt_id_has_its_generated_ids_alone(tmp_path, stated):
    o = oracle(tmp_path, "tokens", lambda doc: put(doc["requests"][1], ("prompt_token_ids",), stated))
    assert o.tokens[U0] == ([], [555]) and o.tokens[U1] == ([100, 101, 102], [777])
    assert o.ids_bytes(0, "r0", 4) == (None, "position 1 beyond prompt 0 + generated 1 of r0")
    assert o.ids_bytes(0, "r1", 4)[1] is None
    assert o.workload_check(DECLARED)["problems"] == ["request r0: oracle prompt has 0 ids != declared prompt_len 5"]


@left_out([])
def test_a_request_whose_tokens_state_no_generated_id_has_its_prompt_alone(tmp_path, stated):
    o = oracle(tmp_path, "tokens", lambda doc: put(doc["requests"][1], ("generated_token_ids",), stated))
    assert o.tokens[U0] == ([103, 104, 105, 106, 107], [])
    assert o.ids_bytes(0, "r0", 4) == (b"".join(i.to_bytes(4, "little") for i in range(103, 108)), None)
    assert o.ids_bytes(1, "r0", 4) == (None, "position 5 beyond prompt 5 + generated 0 of r0")
    assert o.workload_check(DECLARED)["problems"] == []


# ------------------------------------------------------------------------------------------------------------ what a caller hands in

@left_out([])
def test_a_workload_that_declares_no_request_is_not_checked(tmp_path, stated):
    workload = put({"name": "w", "requests": [{"request_id": U0}]}, ("requests",), stated)
    assert oracle(tmp_path).workload_check(workload) == {"ok": None, "problems": ["workload declares no requests: oracle identity not checked"], "compared": 0}


@left_out([])
def test_a_promoted_member_whose_row_names_no_consumer_is_not_resolvable(tmp_path, stated):
    producer = put({"promoted": True, "spec": "X_v1", "consumers": [EMBED]}, ("consumers",), stated)
    assert oracle(tmp_path).member_bytes(0, NORM, "out", "r0", producer=producer) == (
        None, "promoted member without producer spec / consumers in the manifest row: not resolvable", 0)


def test_a_promoted_member_whose_consumers_have_no_instance_at_the_step_says_so(tmp_path):
    o = oracle(tmp_path)
    producer = {"promoted": True, "spec": "X_v1", "consumers": ["no.such.module"]}
    assert o.member_bytes(0, NORM, "out", "r0", producer=producer) == (None, "no fold instance of the promoted member's consumers ['no.such.module'] at this step", 0)
    assert o.member_bytes(0, NORM, "out", None, producer=producer) == (None, "no fold instance of the promoted member's consumers ['no.such.module'] at this step", 0)
    assert o.member_bytes(0, NORM, "out", "r0", producer={"promoted": True, "spec": "X_v1", "consumers": [EMBED]}) == (None, NOT_RESOLVED, 0)
