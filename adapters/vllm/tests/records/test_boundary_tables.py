"""`records.boundary_tables`: a class's table assignment and rank 0's answer load as their typed records and dump back to the
same JSON; another format, an unknown, missing or mistyped field, and an answer with both or neither of its tables and its
error are refused by name."""
from __future__ import annotations

import pytest

from verity_vllm.records.boundary_tables import ASSIGN, ROOTS, TableAssignment, TableRoots
from verity_vllm.records.fields import MissingField, RecordError, UnknownField


def _assign() -> dict:
    return {"format": ASSIGN, "window": 2, "run": "a" * 128, "program": "b" * 128, "prompt": 8, "outputs": 4,
            "layers": [0, 3], "capacity": 16, "spans": [{"engine_id": "r1", "prompt": 5, "served": 4, "finish": "stop"},
                                                         {"engine_id": "r2", "prompt": 3, "served": 1, "finish": None}]}


def _roots(**kw) -> dict:
    entry = {"value": "history/x/model.layers.0.self_attn.latent", "root": "c" * 128, "leaves": 16, "words": 576,
             "domain": "d" * 128}
    return {"format": ROOTS, "window": 2, "run": "a" * 128, "prompt": 8, "outputs": 4, "tables": [entry], **kw}


def test_an_assignment_and_an_answer_load_and_dump_back_to_themselves():
    a = TableAssignment.of(_assign(), "w")
    assert a.dump() == _assign() and a.layers == (0, 3) and a.spans[1] == ("r2", 3, 1, None)
    r = TableRoots.of(_roots(), "w")
    assert r.dump() == _roots() and r.error is None
    failed = {k: v for k, v in _roots(error="ValueError: no rows").items() if k != "tables"}
    assert TableRoots.of(failed, "w").dump() == failed


def test_another_format_or_a_wrong_missing_or_unknown_field_is_refused_by_name():
    with pytest.raises(RecordError, match="'format' is 'other'"):
        TableAssignment.of(_assign() | {"format": "other"}, "w")
    with pytest.raises(UnknownField, match="'spans\\[0\\].slot'"):
        TableAssignment.of(_assign() | {"spans": [{**_assign()["spans"][0], "slot": 0}]}, "w")
    with pytest.raises(MissingField, match="'spans\\[0\\].finish'"):
        TableAssignment.of(_assign() | {"spans": [{"engine_id": "r", "prompt": 1, "served": 1}]}, "w")
    with pytest.raises(RecordError, match="'layers\\[1\\]' is a str"):
        TableAssignment.of(_assign() | {"layers": [0, "3"]}, "w")
    with pytest.raises(RecordError, match="rank 0 answers exactly one of them"):
        TableRoots.of(_roots(error="both"), "w")
    with pytest.raises(RecordError, match="rank 0 answers exactly one of them"):
        TableRoots.of({k: v for k, v in _roots().items() if k != "tables"}, "w")
