"""`records.predicted_row`: the record `predict row` writes loads as the typed record and dumps back to the same JSON; another
schema, a missing, unknown or mistyped field is refused by name."""
from __future__ import annotations

import dataclasses

import pytest

from verity_vllm.records.fields import MissingField, RecordError, UnknownField
from verity_vllm.records.predicted_row import KINDS, SCHEMA, PredictedRow


def _doc() -> dict:
    """The fields `program.predict.aot.write_row` writes, for a two-request row."""
    return {"schema": SCHEMA, "row": "r", "config": "/c/config.json", "code_version": "c" * 64,
            "requests": {"build_request_LP11_T8": "a" * 64, "build_request_LP256_T1": "b" * 64}, "workload_digest": "d" * 64,
            "lag": 0, "manifest_digest": "e" * 64, "manifest_sha256": "f" * 64, "program_digest": "d" * 64, "identities": 23230,
            "secs": 12.5}


def test_a_writers_record_loads_and_dumps_back_to_itself():
    rec = PredictedRow.of(_doc(), "predicted.json")
    assert rec.dump() == _doc() and PredictedRow.of(rec.dump(), "w") == rec
    assert set(KINDS) == {f.name for f in dataclasses.fields(PredictedRow)}


@pytest.mark.parametrize("field", sorted(KINDS))
def test_a_missing_field_is_refused_by_name(field):
    doc = _doc()
    del doc[field]
    with pytest.raises(MissingField) as e:
        PredictedRow.of(doc, "predicted.json")
    assert (e.value.where, e.value.field) == ("predicted.json", field)


def test_another_schema_an_unknown_field_and_a_wrong_kind_are_refused_by_name():
    with pytest.raises(RecordError, match="'schema' is 'verity-vllm/aot-comparison/v1', not verity-vllm/predicted-row/v1"):
        PredictedRow.of(_doc() | {"schema": "verity-vllm/aot-comparison/v1"}, "w")
    with pytest.raises(RecordError, match="not a JSON object"):
        PredictedRow.of([], "w")
    with pytest.raises(UnknownField) as e:
        PredictedRow.of(_doc() | {"manifest": {}}, "w")
    assert e.value.field == "manifest"
    with pytest.raises(RecordError, match="'identities' is a str, not a int"):
        PredictedRow.of(_doc() | {"identities": "23230"}, "w")
