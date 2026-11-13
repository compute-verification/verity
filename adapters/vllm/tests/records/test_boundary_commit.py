"""`records.boundary_commit`: the ``commit.json`` `pipeline.boundary_rows` writes loads as the typed record and dumps back to
the same JSON, with a serve's stop rule and whether the class is a stop class, or without a rule; another format, a
missing, unknown or mistyped field is refused by name."""
from __future__ import annotations

import dataclasses

import pytest

from verity_vllm.records.boundary_commit import KINDS, OPTIONAL, SCHEMA, BoundaryCommit
from verity_vllm.records.fields import MissingField, RecordError, UnknownField


def _doc() -> dict:
    """The fields `boundary_rows.Committed.public` writes, for one class of three spans."""
    value = {"root": "a" * 128, "words": 64, "leaves": 3, "domain": "b" * 128}
    return {"format": SCHEMA, "window": 0, "run": "c" * 128, "prompt": 4, "outputs": 4, "check": "Embedding_v1{V=8,H=16}",
            "per_prompt": 0, "per_output": 1, "inline": False, "spans": 3, "replay": "TemplateInstances_v1",
            "program": "d" * 128, "partition": "e" * 128, "schema": "hm96-sha512/row/v2",
            "values": {"b.ingress": value, "b.egress": value}, "index": "f" * 128, "index_rule": "r", "seconds": 0.01}


def test_a_writers_record_loads_and_dumps_back_to_itself():
    rec = BoundaryCommit.of(_doc(), "commit.json")
    assert rec.dump() == _doc() and BoundaryCommit.of(rec.dump(), "w") == rec
    assert set(KINDS) | set(OPTIONAL) == {f.name for f in dataclasses.fields(BoundaryCommit)}
    stop = _doc() | {"stop": {"ids": [3, 7], "max_model_len": 64, "outputs": 4}, "stopped": False}
    held = BoundaryCommit.of(stop, "commit.json")
    assert held.dump() == stop and held.stop == stop["stop"] and held.stopped is False
    assert rec.stop is None and rec.stopped is None and not {"stop", "stopped"} & set(rec.dump())
    tables = _doc() | {"tables": {"layers": [0, 3], "capacity": 16}}
    assert BoundaryCommit.of(tables, "w").dump() == tables and rec.tables is None


@pytest.mark.parametrize("field", sorted(KINDS))
def test_a_missing_field_is_refused_by_name(field):
    doc = _doc()
    del doc[field]
    with pytest.raises(MissingField) as e:
        BoundaryCommit.of(doc, "commit.json")
    assert (e.value.where, e.value.field) == ("commit.json", field)


def test_another_format_an_unknown_field_and_a_wrong_kind_are_refused_by_name():
    with pytest.raises(RecordError, match="'format' is 'verity-vllm/two-stage/stage1/v0', not verity-vllm/boundary-rows/v1"):
        BoundaryCommit.of(_doc() | {"format": "verity-vllm/two-stage/stage1/v0"}, "w")
    with pytest.raises(RecordError, match="not a JSON object"):
        BoundaryCommit.of([], "w")
    with pytest.raises(UnknownField) as e:
        BoundaryCommit.of(_doc() | {"salts": "00"}, "w")
    assert e.value.field == "salts"
    with pytest.raises(RecordError, match="'spans' is a str, not a int"):
        BoundaryCommit.of(_doc() | {"spans": "3"}, "w")
    with pytest.raises(RecordError, match="'stop' is a list"):
        BoundaryCommit.of(_doc() | {"stop": [3, 7]}, "w")
    with pytest.raises(RecordError, match="'stopped' is a int"):
        BoundaryCommit.of(_doc() | {"stopped": 1}, "w")
