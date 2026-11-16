"""What the Match record's reader makes of a field one of its files leaves out.

`MatchOracle` reads four files of a Match record: the capture's snapshots, its log and its tokens, and the fold's
instances.  Each test takes the complete record of `tests/check/test_oracle_compare.py`, leaves one field out of one
file (absent, null, empty, or any other value that tests false: `left_out`) and states what the oracle then
answers: which requests, steps and ids it knows, and what it says of a member it cannot read.  A workload and a
required manifest's row, which a caller hands in, are held the same way.  A view's own fields are held beside them:
where the oracle reads a member's bytes through a view, one that states no allocation, offset, shape or dtype, or states
there what is no number, no list of numbers or no dtype the recorder writes, is refused by name, and so are strides that
are no list of numbers.  Every dtype the recorder writes is read.
"""
from __future__ import annotations

import copy
import gzip
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.check.test_absent_commit_fields import ABSENT, put
from tests.check.test_absent_link_fields import NOTHING, left_out
from tests.check.test_oracle_compare import U0, U1, H, make_bmap, make_match, reader
from verity_vllm.check import match_oracle as MO
from verity_vllm.check import oracle_compare as OC
from verity_vllm.observe.events import DTYPE_BYTES
from verity_vllm.records.fields import RecordError

EMBED, NORM = "model.embed_tokens", "model.layers.0.post_attention_layernorm"
#: the record's files: its path under the Match directory, and whether it holds a document a line
FILES = {"snapshots": ("capture/snapshots.json", False), "tokens": ("capture/tokens.json", False),
         "instances": ("instances.jsonl", True), "log": ("capture/log.jsonl.gz", True)}
RUN_HEADER, STEP_START, SAMPLE_END = 0, 1, 2      # the log's first three lines; step 0's
DECLARED = {"requests": [{"request_id": U0, "index": 0, "prompt_len": 5, "max_tokens": 1}, {"request_id": U1, "index": 1, "prompt_len": 3, "max_tokens": 1}]}
NO_VIEW = "instance has no view for member {!r}"
NOT_RESOLVED = "no fold instance of producer X_v1 whose output the consumers ['model.embed_tokens'] read at this step (promoted member)"
#: what a view states under `alloc` or `offset` that is no number: the first three test false
NO_NUMBER = ["", [], {}, "thirteen", "2.5", [1], {"a": 1}, float("nan"), float("inf")]


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


# ------------------------------------------------------------------------------------------------------------ a view's allocation and offset

def refused(e: pytest.ExceptionInfo[RecordError], field: str, stated: Any, kind: str = "a number") -> bool:
    """Whether the refusal names the view's `field`, what the view states there (None where it leaves the field out) and the
    `kind` of thing it is not."""
    said = None if stated is ABSENT else stated
    return (e.value.where, e.value.field) == (MO.VIEW, field) and e.value.why.startswith(f"is {said!r}, not {kind} (the view: {{")


def first_view(views: str, field: str, stated: Any) -> Callable[[list[dict[str, Any]]], object]:
    """The first view of `views` of the embedding's first instance (request r1's first row), with `stated` under `field`."""
    return lambda lines: put(embed_instances(lines)[0][views][0], (field,), stated)


@pytest.mark.parametrize("field", ["alloc", "offset"])
@pytest.mark.parametrize("stated", [ABSENT, None, *NO_NUMBER])
def test_a_member_whose_view_states_no_allocation_or_offset_or_one_that_is_no_number_is_refused_by_name(tmp_path, field, stated):
    o = oracle(tmp_path, "instances", first_view("output_views", field, stated))
    with pytest.raises(RecordError) as e:
        o.member_bytes(0, EMBED, "out", "r1")
    assert refused(e, field, stated)
    assert o.member_bytes(0, EMBED, "out", "r0")[1:] == (None, 5)      # the other request's rows are still read


def test_a_compare_leaves_by_the_refusal_of_a_view_one_of_its_members_reads(tmp_path):
    bmap, committed = make_bmap(make_match(str(tmp_path / "values"))[1])
    assert OC.oracle_compare(oracle(tmp_path / "whole"), bmap, reader(committed))["result"] is True
    o = oracle(tmp_path / "no_number", "instances", first_view("output_views", "alloc", "thirteen"))
    with pytest.raises(RecordError) as e:
        OC.oracle_compare(o, bmap, reader(committed))
    assert refused(e, "alloc", "thirteen")


@left_out(0)
def test_instances_whose_first_view_states_no_offset_are_ordered_by_their_position_alone(tmp_path, stated):
    whole = oracle(tmp_path / "whole").member_bytes(0, EMBED, "in", "r0")
    o = oracle(tmp_path / "left_out", "instances", lambda lines: [put(m["output_views"][0], ("offset",), stated) for m in embed_instances(lines)])
    assert whole[0] is not None and len(whole[0]) == 5 * 4
    assert o.member_bytes(0, EMBED, "in", "r0") == whole               # the ids are the operand view's bytes, and it states its offset


@pytest.mark.parametrize("stated", [x for x in NO_NUMBER if x])
def test_an_instance_whose_first_view_states_an_offset_that_is_no_number_is_refused_by_name_where_it_is_ordered(tmp_path, stated):
    o = oracle(tmp_path, "instances", first_view("output_views", "offset", stated))
    with pytest.raises(RecordError) as e:
        o.member_bytes(0, EMBED, "in", "r1")                           # the ids again: no line but the order reads the output view
    assert refused(e, "offset", stated)


def two_whole_producers_a_row(views: str | None = None, field: str = "", stated: Any = None) -> Callable[[list[dict[str, Any]]], None]:
    """Each instance of the embedding twice in its row's group; with `views`, the first view of `views` of the first one states
    `stated` under `field`."""
    def change(lines: list[dict[str, Any]]) -> None:
        for m in embed_instances(lines):
            lines.insert(lines.index(m) + 1, dict(copy.deepcopy(m), event=m["event"] + 1))
        if views is not None:
            first_view(views, field, stated)(lines)
    return change


@pytest.mark.parametrize("views", ["output_views", "operand_views"])
@pytest.mark.parametrize("field", ["alloc", "offset"])
@pytest.mark.parametrize("stated", [ABSENT, None, *NO_NUMBER])
def test_of_two_producers_a_row_the_one_an_operand_names_is_looked_up_by_views_that_state_their_allocation_and_offset(tmp_path, views, field, stated):
    fed_by = {"spec": "X_v1", "operand": "X_v1"}
    whole = oracle(tmp_path / "whole", "instances", two_whole_producers_a_row()).member_bytes(0, EMBED, "out", "r1", producer=fed_by)
    assert whole[0] is None and whole[1].startswith("producer_operand X_v1: 0 of 2 X_v1 instances in a row's group") and whole[2] == 6
    o = oracle(tmp_path / "stated", "instances", two_whole_producers_a_row(views, field, stated))
    with pytest.raises(RecordError) as e:
        o.member_bytes(0, EMBED, "out", "r1", producer=fed_by)
    assert refused(e, field, stated)


def no_operand_view(lines: list[dict[str, Any]]) -> None:
    """The embedding's first instance with a null in place of its operand's view, as the fold writes an operand that has no device
    view (a constant, a parameter, a weight's field, several pieces)."""
    embed_instances(lines)[0]["operand_views"] = [None]


def test_a_null_in_place_of_an_operands_view_is_refused_by_no_name(tmp_path):
    o = oracle(tmp_path / "plain", "instances", no_operand_view)
    with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
        o.member_bytes(0, EMBED, "in", "r1")                           # the ids are operand view 0's bytes
    assert o.member_bytes(0, EMBED, "out", "r1")[1:] == (None, 3) and o.member_bytes(0, EMBED, "in", "r0")[1:] == (None, 5)
    fed_by = {"spec": "X_v1", "operand": "X_v1"}
    whole = oracle(tmp_path / "whole", "instances", two_whole_producers_a_row()).member_bytes(0, EMBED, "out", "r1", producer=fed_by)
    def both(lines: list[dict[str, Any]]) -> None:
        two_whole_producers_a_row()(lines)
        no_operand_view(lines)

    two = oracle(tmp_path / "two", "instances", both)
    assert two.member_bytes(0, EMBED, "out", "r1", producer=fed_by) == whole and whole[0] is None     # fed by nothing, as an operand of no producer


def read_through_the_second_view(field: str | None = None, stated: Any = None) -> Callable[[list[dict[str, Any]]], None]:
    """The embedding's instances read the norm's second output view; with `field`, the norm's first output view states `stated`
    under it."""
    def change(lines: list[dict[str, Any]]) -> None:
        norms = [m for m in lines[1:] if m["hint"].startswith(NORM + "/")]
        for m, norm in zip(embed_instances(lines), norms):
            m["operand_views"] = [dict(norm["output_views"][1])]
            if field is not None:
                put(norm["output_views"][0], (field,), stated)
    return change


PROMOTED = {"promoted": True, "spec": "X_v1", "consumers": [EMBED]}


@pytest.mark.parametrize("field", ["alloc", "offset"])
@pytest.mark.parametrize("stated", [ABSENT, None])
def test_a_promoted_member_found_through_its_producers_second_view_is_refused_by_name_for_the_first_it_reads(tmp_path, field, stated):
    whole = oracle(tmp_path / "whole", "instances", read_through_the_second_view()).member_bytes(0, "top", "out", "r1", producer=PROMOTED)
    assert whole[0] is not None and len(whole[0]) == 3 * 2 * H and whole[1:] == (None, 3)
    o = oracle(tmp_path / "stated", "instances", read_through_the_second_view(field, stated))
    assert len(o._step_outputs(0)[13]) == 8 and 12 not in o._step_outputs(0)        # the first view is no storage; the second still is
    with pytest.raises(RecordError) as e:
        o.member_bytes(0, "top", "out", "r1", producer=PROMOTED)
    assert refused(e, field, stated)


@pytest.mark.parametrize("field", ["alloc", "offset"])
def test_a_producer_whose_first_view_leaves_its_allocation_or_offset_out_is_found_and_one_that_states_null_is_refused_by_name(tmp_path, field):
    left_out = oracle(tmp_path / "left_out", "instances", read_through_the_second_view(field, ABSENT))
    assert left_out.member_bytes(0, "top", "in", "r1", producer=PROMOTED) == (None, NO_VIEW.format("in"), 3)     # found; it has no operand
    null = oracle(tmp_path / "null", "instances", read_through_the_second_view(field, None))
    with pytest.raises(RecordError) as e:
        null.member_bytes(0, "top", "in", "r1", producer=PROMOTED)
    assert refused(e, field, None)


# ------------------------------------------------------------------------------------------------------------ a view's numbers, as `int` reads them

def test_a_member_is_read_through_a_view_that_states_its_allocation_or_offset_as_int_reads_it(tmp_path):
    whole = oracle(tmp_path / "whole").member_bytes(0, EMBED, "out", "r1")
    assert whole[0] is not None and len(whole[0]) == 3 * 2 * H and whole[1:] == (None, 3)
    for field, stated in [("alloc", "11"), ("alloc", 11.0), ("offset", "4"), ("offset", 4.9), ("offset", True)]:
        def second_row(lines: list[dict[str, Any]], field: str = field, stated: Any = stated) -> None:
            put(embed_instances(lines)[1]["output_views"][0], (field,), stated)      # request r1's second row: allocation 11, offset 4

        read = oracle(tmp_path / f"{field}-{stated}", "instances", second_row).member_bytes(0, EMBED, "out", "r1")
        if stated is True:
            assert read[0] is not None and read[0][2 * H:4 * H] == whole[0][2:2 + 2 * H] and read[1:] == (None, 3)      # the row at element 1
        else:
            assert read == whole


def test_instances_are_ordered_by_the_offset_their_first_view_states_as_int_reads_it(tmp_path):
    whole = oracle(tmp_path / "whole").member_bytes(0, EMBED, "out", None)
    assert whole[0] is not None and len(whole[0]) == 8 * 2 * H and whole[1:] == (None, 8)

    def as_text(lines: list[dict[str, Any]]) -> None:
        for m in embed_instances(lines):
            m["output_views"][0]["offset"] = str(m["output_views"][0]["offset"])      # "8" sorts after "20", and 8 before 20

    assert oracle(tmp_path / "text", "instances", as_text).member_bytes(0, EMBED, "out", None) == whole


def a_producer_a_row(text_at: str | None = None) -> Callable[[list[dict[str, Any]]], None]:
    """Before each instance of the embedding, in its row's group, a producer whose one output view is the embedding's operand view;
    with `text_at`, the views of `text_at` ("output_views": the producers'; "operand_views": the embeddings') state their allocation
    and their offset as text."""
    def change(lines: list[dict[str, Any]]) -> None:
        for m in embed_instances(lines):
            fed_by = dict(copy.deepcopy(m), operand_views=[], output_views=copy.deepcopy(m["operand_views"]))
            lines.insert(lines.index(m), fed_by)
            if text_at is not None:
                view = (fed_by if text_at == "output_views" else m)[text_at][0]
                view.update(alloc=str(view["alloc"]), offset=str(view["offset"]))
    return change


@pytest.mark.parametrize("text_at", [None, "output_views", "operand_views"])
def test_the_producer_an_operand_names_is_found_by_the_allocation_and_offset_as_int_reads_them(tmp_path, text_at):
    whole = oracle(tmp_path / "whole").member_bytes(0, EMBED, "out", "r1")
    fed = oracle(tmp_path / "fed", "instances", a_producer_a_row(text_at))
    assert fed.member_bytes(0, EMBED, "out", "r1", producer={"spec": "X_v1", "operand": "X_v1"}) == whole and whole[0] is not None


@pytest.mark.parametrize("field", ["alloc", "offset"])
def test_a_promoted_members_producer_is_one_instance_whichever_way_its_first_view_states_its_allocation_and_offset(tmp_path, field):
    whole = oracle(tmp_path / "whole", "instances", read_through_the_second_view()).member_bytes(0, "top", "out", "r1", producer=PROMOTED)
    assert whole[0] is not None and len(whole[0]) == 3 * 2 * H and whole[1:] == (None, 3)

    def twice(lines: list[dict[str, Any]]) -> None:
        read_through_the_second_view()(lines)
        for norm in [m for m in lines[1:] if m["hint"].startswith(NORM + "/")]:
            again = copy.deepcopy(norm)
            again["output_views"][0][field] = str(norm["output_views"][0][field])
            lines.insert(lines.index(norm) + 1, again)

    assert oracle(tmp_path / "twice", "instances", twice).member_bytes(0, "top", "out", "r1", producer=PROMOTED) == whole


# ------------------------------------------------------------------------------------------------------------ a view's shape, strides and dtype

#: what a view states under `shape` or `strides` that tests true and is no list of numbers
NO_NUMBERS = [1, 2.5, True, "thirteen", "24", {"a": 1}, ["two"], [None], [[2]], [float("inf")]]
#: what a view states under `dtype` that is no dtype the recorder writes
NO_DTYPE = [None, "", "thirteen", "BFLOAT16", "bf16", 2, ["bfloat16"], {"bfloat16": 2}]
A_LIST, A_DTYPE = "a list of numbers", "a dtype the recorder writes (observe.events.DTYPE_BYTES)"


def refuses_the_first_request(tmp_path: Path, field: str, stated: Any, kind: str) -> bool:
    """Whether the embedding's output for request r1, whose first row's view states `stated` under `field`, is refused by that
    field's name, and request r0's rows are still read."""
    o = oracle(tmp_path, "instances", first_view("output_views", field, stated))
    with pytest.raises(RecordError) as e:
        o.member_bytes(0, EMBED, "out", "r1")
    return refused(e, field, stated, kind) and o.member_bytes(0, EMBED, "out", "r0")[1:] == (None, 5)


@pytest.mark.parametrize("stated", [ABSENT, None, 0, False, "", {}, *NO_NUMBERS])
def test_a_member_whose_view_states_no_shape_or_one_that_is_no_list_of_numbers_is_refused_by_name(tmp_path, stated):
    assert refuses_the_first_request(tmp_path, "shape", stated, A_LIST)


@pytest.mark.parametrize("stated", NO_NUMBERS)
def test_a_member_whose_view_states_strides_that_are_no_list_of_numbers_is_refused_by_name(tmp_path, stated):
    assert refuses_the_first_request(tmp_path, "strides", stated, A_LIST)


@pytest.mark.parametrize("stated", [ABSENT, *NO_DTYPE])
def test_a_member_whose_view_states_no_dtype_or_one_the_recorder_does_not_write_is_refused_by_name(tmp_path, stated):
    assert refuses_the_first_request(tmp_path, "dtype", stated, A_DTYPE)


@pytest.mark.parametrize(("field", "stated"), [("shape", ["4"]), ("shape", [4.0]), ("shape", [4.9]), ("strides", ["1"]), ("strides", [1.0]), ("strides", [True])])
def test_a_views_shape_and_strides_are_numbers_as_int_reads_them(tmp_path, field, stated):
    whole = oracle(tmp_path / "whole").member_bytes(0, EMBED, "out", "r1")
    assert whole[0] is not None and len(whole[0]) == 3 * 2 * H and whole[1:] == (None, 3)
    assert oracle(tmp_path / "stated", "instances", first_view("output_views", field, stated)).member_bytes(0, EMBED, "out", "r1") == whole


def test_a_view_that_states_an_empty_shape_is_one_element(tmp_path):
    o = oracle(tmp_path, "instances", first_view("output_views", "shape", []))
    read, why, rows = o.member_bytes(0, EMBED, "out", "r1")
    assert read is not None and len(read) == 2 + 2 * 2 * H and (why, rows) == (None, 3)      # the first row's one element, then two rows whole


# ------------------------------------------------------------------------------------------------------------ a view's dtype

def in_dtype(dtype: str) -> Callable[[list[dict[str, Any]]], None]:
    """The norm's second output view of every instance, stated in `dtype` over the bytes it has: a row's `H` two-byte elements as
    `2 * H / size` elements of the dtype's size."""
    size = DTYPE_BYTES[dtype]

    def change(lines: list[dict[str, Any]]) -> None:
        for m in lines[1:]:
            if m["hint"].startswith(NORM + "/"):
                view = m["output_views"][1]
                view.update(dtype=dtype, shape=[2 * H // size], offset=view["offset"] * 2 // size)
    return change


def bound_in_dtype(bmap: dict[str, Any], dtype: str) -> dict[str, Any]:
    """`bmap` with the norm's second output bound in `dtype`, over the bytes its entries have."""
    size = DTYPE_BYTES[dtype]
    for e in bmap["entries"]:
        if e["name"] == NORM + "/1":
            e["identity"]["element_range"] = [x * 2 // size for x in e["identity"]["element_range"]]
            e["element_offset"] = e["element_offset"] * 2 // size
            e["encoding"] = {"dtype": dtype, "shape": [e["encoding"]["shape"][0], 2 * H // size]}
    return bmap


@pytest.mark.parametrize("dtype", sorted(DTYPE_BYTES))
def test_a_member_is_read_through_a_view_of_any_dtype_the_recorder_writes(tmp_path, dtype):
    whole = oracle(tmp_path / "whole").member_bytes(0, NORM, "1", "r1")
    assert whole[0] is not None and len(whole[0]) == 3 * 2 * H and whole[1:] == (None, 3)
    assert oracle(tmp_path / "stated", "instances", in_dtype(dtype)).member_bytes(0, NORM, "1", "r1") == whole
    size = DTYPE_BYTES[dtype]
    view = {"alloc": 1, "offset": 2, "dtype": dtype, "shape": [3, 2], "strides": [4, 1]}
    assert MO.MatchOracle._view_bytes_range(view) == (1, 2 * size, 12 * size)      # strided: a span of 10 elements


@pytest.mark.parametrize("dtype", sorted(DTYPE_BYTES))
def test_a_member_bound_in_any_dtype_the_recorder_writes_is_compared(tmp_path, dtype):
    bmap, committed = make_bmap(make_match(str(tmp_path / "values"))[1])
    whole = OC.oracle_compare(oracle(tmp_path / "whole"), bmap, reader(committed))
    assert (whole["result"], whole["complete"], whole["mismatch_n"]) == (True, True, 0)
    o, bmap = oracle(tmp_path / "stated", "instances", in_dtype(dtype)), bound_in_dtype(bmap, dtype)
    equal = OC.oracle_compare(o, bmap, reader(committed))
    assert (equal["result"], equal["complete"], equal["compared"], equal["mismatch_n"]) == (True, True, whole["compared"], 0)
    member = (0, NORM + "/1")
    committed[member] = bytes([committed[member][0] ^ 1]) + committed[member][1:]                # a byte of request r1's rows
    differs = OC.oracle_compare(o, bmap, reader(committed))
    assert (differs["result"], differs["compared"], differs["mismatch_n"]) == (False, whole["compared"], 1)


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

@pytest.mark.parametrize("workload", NOTHING)
def test_a_caller_that_hands_in_no_workload_is_not_checked(tmp_path, workload):
    assert oracle(tmp_path).workload_check(workload) == {"ok": None, "problems": ["no workload given: oracle identity not checked"], "compared": 0}


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
